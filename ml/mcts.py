"""
AlphaZero-Style Guided Lookahead Searcher for Mini Metro.

Uses the simulator's instantaneous in-memory cloning capability (env.clone())
to perform counterfactual forward simulations during overcrowding crises.
Ranks candidate actions via PPO policy priors + rollout reward + discounted critic leaf evaluation.
"""

from typing import Dict, Any, Tuple, Optional, List
import time
import numpy as np
import torch


class GuidedLookaheadSearcher:
    """
    Tactical crisis searcher combining neural policy priors with multi-step forward lookahead.
    Activated during high queue stress or overcrowding countdown timers.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        top_k: int = 6,
        lookahead_duration: float = 4.0,
        discount: float = 0.99,
        c_puct: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.top_k = top_k
        self.lookahead_duration = lookahead_duration
        self.discount = discount
        self.c_puct = c_puct
        self.device = device or next(model.parameters()).device

    def is_in_crisis(self, obs: Dict[str, Any]) -> bool:
        """
        Detects if any station has active overcrowding countdown or queue fill > 75%.
        """
        nodes = obs.get("nodes")
        if nodes is None:
            return False
        num_nodes = int(obs["num_nodes"][0]) if "num_nodes" in obs else len(nodes)
        for i in range(num_nodes):
            st = nodes[i]
            # st[22] is overcrowding circle progress (0.0 to 1.0)
            if float(st[22]) > 0.0:
                return True
            # st[12:22] is passenger demand counts; st[24] is interchange flag
            q_total = float(st[12:22].sum())
            cap = 18.0 if bool(st[24]) else 6.0
            if (q_total / cap) > 0.75:
                return True
        return False

    def select_action(
        self,
        env,
        obs: Dict[str, Any],
        mask: Optional[np.ndarray] = None,
        lstm_state=None,
        emergency_only: bool = True,
        deterministic: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Selects an action using neural policy, optionally upgrading to guided lookahead during crisis.
        Returns: (selected_action, search_info_dict)
        """
        info = {
            "search_used": False,
            "candidates_evaluated": 0,
            "search_latency_ms": 0.0,
            "best_candidate": 0,
            "prior_candidate": 0,
        }

        # Convert observation to PyTorch tensor dict on device
        obs_tensor = {
            k: torch.as_tensor(v).unsqueeze(0).to(self.device)
            if not isinstance(v, torch.Tensor)
            else v.unsqueeze(0).to(self.device) if v.ndim == (2 if k in ("nodes", "edge_attrs") else 1) else v.to(self.device)
            for k, v in obs.items()
        }

        if mask is None:
            mask = obs["action_mask"]
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device).view(1, -1)

        # Base network forward pass
        with torch.no_grad():
            logits, value, next_lstm = self.model(obs_tensor, lstm_state=lstm_state, mask=mask_tensor)
            if deterministic:
                if mask_tensor is not None:
                    prior_action = int(torch.argmax(logits.masked_fill(~mask_tensor, -1e9), dim=-1).squeeze().item())
                else:
                    prior_action = int(torch.argmax(logits, dim=-1).squeeze().item())
            else:
                masked_logits = logits.masked_fill(~mask_tensor, -1e9) if mask_tensor is not None else logits
                probs_dist = torch.distributions.Categorical(logits=masked_logits)
                prior_action = int(probs_dist.sample().squeeze().item())
            info["prior_candidate"] = prior_action

        # Check if crisis conditions warrant lookahead search
        crisis_active = self.is_in_crisis(obs)
        if emergency_only and not crisis_active:
            return prior_action, info

        # Perform Lookahead Search
        t0 = time.perf_counter()
        try:
            # Extract top-k valid actions from policy distribution
            valid_indices = np.where(mask)[0]
            if len(valid_indices) <= 1:
                return prior_action, info

            candidates = [prior_action]
            if 0 not in candidates and mask[0]:
                candidates.append(0)  # Always consider NoOp

            probs = torch.softmax(logits.squeeze(), dim=-1).cpu().numpy()
            probs[~mask] = -1.0
            sorted_valid = np.argsort(-probs)

            for idx in sorted_valid[: self.top_k]:
                a_idx = int(idx)
                if a_idx not in candidates and mask[a_idx]:
                    candidates.append(a_idx)
                    if len(candidates) >= self.top_k:
                        break

            best_score = -float("inf")
            best_action = prior_action
            candidate_scores = {}

            for cand_act in candidates:
                cand_prior = float(probs[cand_act]) if cand_act < len(probs) else 0.0
                cloned_sim = env.clone()
                try:
                    sim_obs, sim_r, sim_done, _, sim_info = cloned_sim.step(
                        cand_act, duration=self.lookahead_duration
                    )

                    if sim_done:
                        # Severe game over penalty
                        q_val = -300.0
                    else:
                        # Value critic leaf evaluation
                        sim_obs_t = {
                            k: torch.as_tensor(v).unsqueeze(0).to(self.device)
                            for k, v in sim_obs.items()
                        }
                        with torch.no_grad():
                            leaf_val = self.model.get_value(sim_obs_t, lstm_state=next_lstm)
                            leaf_v = float(leaf_val.squeeze().item())

                        q_val = sim_r + (self.discount * leaf_v)

                    # PUCT combination: Q + exploration prior bonus
                    u_val = self.c_puct * cand_prior
                    total_score = q_val + u_val
                    candidate_scores[cand_act] = total_score

                    if total_score > best_score:
                        best_score = total_score
                        best_action = cand_act

                finally:
                    cloned_sim.close()

            latency = (time.perf_counter() - t0) * 1000.0
            info["search_used"] = True
            info["candidates_evaluated"] = len(candidates)
            info["search_latency_ms"] = latency
            info["best_candidate"] = best_action
            info["candidate_scores"] = candidate_scores

            return best_action, info

        except Exception as e:
            # Resilient fallback to base neural actor if cloning or search encounters issue
            info["search_error"] = str(e)
            return prior_action, info

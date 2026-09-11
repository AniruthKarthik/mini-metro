"""
Strategic Network Intervention & Counterfactual Evaluation Module.

Provides modular candidate intervention evaluation, counterfactual short-horizon
simulation rollouts via Go state cloning, and strategic arbitration between:
  - KEEP (preserve topology)
  - LOCAL_EDIT (AddTrain, AddCarriage, ExtendLine, InsertStation, ShortenLine, Interchange)
  - MAJOR_REBUILD (RemoveLine + corridor redesign)

Ensures network editing occurs only when the expected improvement in network
performance justifies the operational disruption and resource cost.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Action Space Offsets matching simulator/engine/action_space.go
ACTION_NOOP = 0
ADD_LINE_START = 1
ADD_LINE_END = 436
EXTEND_LINE_START = 436
EXTEND_LINE_END = 856
INSERT_STATION_START = 856
INSERT_STATION_END = 4006
ADD_TRAIN_START = 4006
ADD_TRAIN_END = 4013
ADD_CARRIAGE_START = 4013
ADD_CARRIAGE_END = 4020
INTERCHANGE_START = 4020
INTERCHANGE_END = 4050
CHOOSE_REWARD_START = 4050
CHOOSE_REWARD_END = 4052
CLOSE_LOOP_START = 4052
CLOSE_LOOP_END = 4059
OPEN_LOOP_START = 4059
OPEN_LOOP_END = 4066
REMOVE_LINE_START = 4066
REMOVE_LINE_END = 4073
SHORTEN_LINE_START = 4073
SHORTEN_LINE_END = 4087


class InterventionTier(Enum):
    KEEP = "keep"                         # Tier 1: No change, preserve stability
    DISPATCH = "dispatch"                 # Tier 2: AddTrain, AddCarriage, Interchange (zero disruption)
    LOCAL_EDIT = "local_edit"             # Tier 3: ExtendLine, InsertStation, ShortenLine, Loop (low disruption)
    MAJOR_REBUILD = "major_rebuild"       # Tier 4: RemoveLine (high disruption)
    REWARD_CHOICE = "reward_choice"       # Meta: Weekly reward selection


def classify_action_tier(action_id: int) -> InterventionTier:
    """Classifies any discrete action ID into its strategic intervention tier."""
    if action_id == ACTION_NOOP:
        return InterventionTier.KEEP
    if ADD_TRAIN_START <= action_id < ADD_CARRIAGE_END or INTERCHANGE_START <= action_id < INTERCHANGE_END:
        return InterventionTier.DISPATCH
    if CHOOSE_REWARD_START <= action_id < CHOOSE_REWARD_END:
        return InterventionTier.REWARD_CHOICE
    if EXTEND_LINE_START <= action_id < INSERT_STATION_END or SHORTEN_LINE_START <= action_id < SHORTEN_LINE_END or CLOSE_LOOP_START <= action_id < OPEN_LOOP_END:
        return InterventionTier.LOCAL_EDIT
    if REMOVE_LINE_START <= action_id < REMOVE_LINE_END:
        return InterventionTier.MAJOR_REBUILD
    if ADD_LINE_START <= action_id < ADD_LINE_END:
        return InterventionTier.LOCAL_EDIT
    return InterventionTier.KEEP


class StrategicInterventionArbiter:
    """
    Evaluates candidate network interventions using fast Go state cloning.
    Estimates:
        InterventionValue = ExpectedFutureBenefit - EditCost - OperationalDisruption
    """

    def __init__(
        self,
        cf_duration: float = 4.0,
        rebuild_threshold: float = 1.5,
        local_edit_bias: float = 0.5,
        emergency_override: bool = True,
    ):
        """
        Args:
            cf_duration: Simulation seconds for short-horizon counterfactual evaluation.
            rebuild_threshold: Minimum expected marginal benefit required to justify RemoveLine over Keep.
            local_edit_bias: Preference bonus for local modifications vs full demolition.
            emergency_override: If True, bypasses stability hurdles during severe overcrowding.
        """
        self.cf_duration = cf_duration
        self.rebuild_threshold = rebuild_threshold
        self.local_edit_bias = local_edit_bias
        self.emergency_override = emergency_override

    def compute_network_utility(self, breakdown: Dict[str, float], done: bool) -> float:
        """
        Computes composite network utility from simulation reward breakdown:
            Utility = Delivery + 2.0*Connectivity + CrowdPenalty + TrackEfficiency + Disruption - GameOver
        """
        if done:
            return -200.0

        delivery = breakdown.get("delivery", 0.0)
        connectivity = breakdown.get("connectivity", 0.0)
        crowd = breakdown.get("crowd_penalty", 0.0)
        game_over = breakdown.get("game_over", 0.0)
        redundancy = breakdown.get("redundancy", 0.0)
        track_eff = breakdown.get("track_efficiency", 0.0)
        disruption = breakdown.get("disruption", 0.0)

        # Net utility combines throughput, crowding relief, network connectivity, and disruption
        utility = (
            delivery * 1.0
            + connectivity * 1.0
            + crowd * 1.0
            + game_over * 1.0
            + redundancy * 1.0
            + track_eff * 1.0
            + disruption * 1.0
        )
        return utility

    def evaluate_candidates(
        self,
        env: Any,
        candidate_actions: List[int],
        obs: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[int, Dict[int, float]]:
        """
        Runs short-horizon counterfactual simulations on candidate actions and selects
        the highest strategic utility intervention.

        Args:
            env: MiniMetroEnv instance (must support env.clone()).
            candidate_actions: List of valid candidate action IDs.
            obs: Current observation dict.

        Returns:
            best_action: Chosen action ID.
            candidate_utilities: Dict mapping candidate action ID -> marginal utility delta over KEEP.
        """
        if not candidate_actions:
            return ACTION_NOOP, {ACTION_NOOP: 0.0}

        # 1. Evaluate baseline KEEP (NoOp)
        keep_reward, keep_breakdown, keep_done, _ = env.simulate_candidate(
            ACTION_NOOP, duration=self.cf_duration
        )
        u_keep = self.compute_network_utility(keep_breakdown, keep_done)

        # Check for emergency state
        is_emergency = False
        if obs is not None and "nodes" in obs:
            nodes = obs["nodes"]
            num_nodes = int(obs["num_nodes"][0]) if "num_nodes" in obs else len(nodes)
            for i in range(num_nodes):
                overcrowd_timer_prog = float(nodes[i, 22])
                fill_ratio = float(nodes[i, 25]) if nodes.shape[-1] > 25 else 0.0
                if overcrowd_timer_prog > 0.0 or fill_ratio >= 0.85:
                    is_emergency = True
                    break

        candidate_utilities: Dict[int, float] = {ACTION_NOOP: 0.0}
        best_action = ACTION_NOOP
        best_net_benefit = 0.0

        for action_id in candidate_actions:
            if action_id == ACTION_NOOP:
                continue

            tier = classify_action_tier(action_id)

            # Evaluate candidate in isolated clone
            reward, breakdown, done, _ = env.simulate_candidate(
                action_id, duration=self.cf_duration
            )
            u_cand = self.compute_network_utility(breakdown, done)
            delta_u = u_cand - u_keep

            # Apply tier-specific strategic calibration
            if tier == InterventionTier.MAJOR_REBUILD:
                # Require significant improvement to justify tearing down a line
                threshold = 0.0 if (is_emergency and self.emergency_override) else self.rebuild_threshold
                net_benefit = delta_u - threshold
            elif tier == InterventionTier.LOCAL_EDIT:
                # Local edits are preferred over major reconfigurations
                net_benefit = delta_u + (0.1 if delta_u >= -0.05 else 0.0)
            elif tier == InterventionTier.DISPATCH:
                # Resource dispatch has 0 disruption
                net_benefit = delta_u + 0.2
            else:
                net_benefit = delta_u

            candidate_utilities[action_id] = float(delta_u)

            if net_benefit > best_net_benefit:
                best_net_benefit = net_benefit
                best_action = action_id

        return best_action, candidate_utilities

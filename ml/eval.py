"""
Rigorous Multi-Seed & Cross-Map Evaluation Suite (P3-1).

Evaluates trained models and baselines across:
- 10 fixed random seeds (1000..1009)
- All 3 official maps: London (0), New York City (1), Tokyo (2)
- 4 distinct policy modes:
  1. Model (Hierarchical Deterministic)
  2. Model (Stochastic Sampling)
  3. Greedy Heuristic Baseline
  4. Random Legal Baseline

Computes and tabulates full statistical dispersion metrics:
- Score (passengers delivered): Mean, StdDev, Median, Min, Max, Q25, Q75
- Survival duration: Macro-steps and simulated in-game seconds
- Cause of death: Station ID, overcrowding progress, and station kind
- Resource utilization: Lines, trains, tunnels, carriages, interchanges
"""

import os
import sys
import argparse
import glob
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import MiniMetroEnv
from model import MiniMetroActorCritic, ACTION_TYPE_SLICES
from intervention import StrategicInterventionArbiter, is_high_impact_structural_action

MAP_NAMES = {
    0: "London",
    1: "New York City",
    2: "Tokyo",
}

STATION_KINDS = [
    "Circle",
    "Triangle",
    "Square",
    "Star",
    "Pentagon",
    "Gem",
    "Sector",
    "Cross",
    "Drop",
    "Oval",
]

DEFAULT_SEEDS = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]


# ==============================================================================
# POLICIES
# ==============================================================================

class BasePolicy:
    def reset(self, seed: Optional[int] = None):
        pass

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        raise NotImplementedError


class ModelPolicy(BasePolicy):
    def __init__(self, model: MiniMetroActorCritic, deterministic: bool = True, device: torch.device = torch.device("cpu"), use_arbiter: bool = True):
        self.model = model
        self.deterministic = deterministic
        self.device = device
        self.lstm_state = None
        self.use_arbiter = use_arbiter
        self.arbiter = StrategicInterventionArbiter() if use_arbiter else None
        self.sim_time = 0.0

    def reset(self, seed: Optional[int] = None):
        self.lstm_state = None
        self.sim_time = 0.0
        if self.use_arbiter:
            self.arbiter = StrategicInterventionArbiter()

    def act(self, obs: Dict[str, np.ndarray], env: Optional[Any] = None) -> int:
        obs_tensor = {
            k: torch.as_tensor(v, device=self.device).unsqueeze(0)
            for k, v in obs.items()
        }
        mask = obs_tensor["action_mask"].bool()
        with torch.no_grad():
            action, _, _, _, self.lstm_state = self.model.get_action_and_value(
                obs_tensor,
                lstm_state=self.lstm_state,
                mask=mask,
                deterministic=self.deterministic,
            )
        action_id = int(action.item())

        if self.use_arbiter and self.arbiter is not None and env is not None:
            if is_high_impact_structural_action(action_id, obs):
                candidates = self.arbiter.generate_candidate_portfolio(env, obs, action_id, top_k=6)
                res = self.arbiter.evaluate_candidates(env, candidates, obs, sim_time=self.sim_time)
                action_id = res.best_action
            self.arbiter.record_executed_intervention(action_id, self.sim_time)

        return action_id


class RandomLegalPolicy(BasePolicy):
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        mask = obs["action_mask"]
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return 0
        return int(self.rng.choice(legal))


class GreedyHeuristicPolicy(BasePolicy):
    """
    Transparent, deterministic rule-based priority agent:
    1. Priority 1: If pending reward card exists, pick the most valuable reward (Line > Train > Interchange > Tunnel > Carriage).
    2. Priority 2: If a station is critically overcrowding (progress > 0.5) and can be upgraded to Interchange, upgrade it!
    3. Priority 3: If trains are available and an active line has high queue demand, dispatch AddTrain to that line.
    4. Priority 4: If any station is unconnected (degree 0), connect via AddLine or ExtendLine.
    5. Priority 5: Otherwise NoOp (action 0).
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        mask = obs["action_mask"]
        globals_feat = obs["globals"]
        nodes = obs["nodes"]
        edge_attrs = obs["edge_attrs"]

        # 1. Pending Reward Choices (Actions 4050, 4051)
        if mask[4050] or mask[4051]:
            card_priority = [10, 9, 5, 4, 7]  # line=10, train=9, tunnel=5, carriage=4, interchange=7
            val0 = 0
            if mask[4050] and len(globals_feat) >= 23:
                c0_type = int(np.argmax(globals_feat[13:18]))
                val0 = card_priority[c0_type] if c0_type < len(card_priority) else 0
            val1 = 0
            if mask[4051] and len(globals_feat) >= 23:
                c1_type = int(np.argmax(globals_feat[18:23]))
                val1 = card_priority[c1_type] if c1_type < len(card_priority) else 0

            if val0 >= val1 and mask[4050]:
                return 4050
            elif mask[4051]:
                return 4051

        # 2. Upgrade Interchange on critically overcrowding stations (progress > 0.5)
        interchange_slice = ACTION_TYPE_SLICES[6]  # 4020..4049 (30 actions)
        interchange_legal = np.where(mask[interchange_slice])[0]
        if len(interchange_legal) > 0:
            best_st = -1
            max_prog = 0.5
            for st_id in interchange_legal:
                prog = float(nodes[st_id, 22])
                if prog > max_prog:
                    max_prog = prog
                    best_st = st_id
            if best_st >= 0:
                return interchange_slice.start + best_st

        # 3. AddTrain to most crowded active line
        add_train_slice = ACTION_TYPE_SLICES[4]  # 4006..4012 (7 actions)
        add_train_legal = np.where(mask[add_train_slice])[0]
        if len(add_train_legal) > 0:
            best_line = add_train_legal[0]
            max_line_queue = -1.0
            edge_lines = edge_attrs[:, 0:7]
            for l_idx in add_train_legal:
                l_mask = edge_lines[:, l_idx] > 0
                q_sum = float(np.sum(edge_attrs[l_mask, 7]))
                if q_sum > max_line_queue:
                    max_line_queue = q_sum
                    best_line = l_idx
            return add_train_slice.start + best_line

        # 4. Connect unconnected alive stations (degree == 0) via AddLine or ExtendLine
        extend_slice = ACTION_TYPE_SLICES[2]  # 436..855 (420 actions)
        add_line_slice = ACTION_TYPE_SLICES[1]  # 1..435 (435 actions)

        unconnected_st = [i for i in range(30) if np.any(nodes[i, 2:12] > 0) and nodes[i, 23] == 0]

        for target_st in unconnected_st:
            for line_id in range(7):
                for end in [0, 1]:
                    ext_idx = extend_slice.start + (line_id * 30 + target_st) * 2 + end
                    if ext_idx < extend_slice.stop and mask[ext_idx]:
                        return ext_idx

            target_kind = int(np.argmax(nodes[target_st, 2:12]))
            triu_u, triu_v = np.triu_indices(30, k=1)
            for pair_idx in range(len(triu_u)):
                u, v = triu_u[pair_idx], triu_v[pair_idx]
                if u == target_st or v == target_st:
                    other_st = v if u == target_st else u
                    if np.any(nodes[other_st, 2:12] > 0):
                        other_kind = int(np.argmax(nodes[other_st, 2:12]))
                        if other_kind != target_kind:
                            action_id = add_line_slice.start + pair_idx
                            if mask[action_id]:
                                return action_id

        # 5. Default NoOp
        if mask[0]:
            return 0

        legal = np.where(mask)[0]
        return int(legal[0]) if len(legal) > 0 else 0


# ==============================================================================
# EPISODE EVALUATOR & METRICS
# ==============================================================================

def run_single_episode(
    env: MiniMetroEnv,
    policy: BasePolicy,
    seed: Optional[int] = None,
    max_steps: int = 1000,
) -> Dict[str, Any]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    policy.reset(seed=seed)

    done = False
    step = 0
    total_reward = 0.0

    last_obs = obs

    while not done and step < max_steps:
        if isinstance(policy, ModelPolicy):
            action = policy.act(obs, env=env)
        else:
            action = policy.act(obs)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        step += 1
        if isinstance(policy, ModelPolicy):
            policy.sim_time += step_info.get("simulation_seconds", 1.0)
        done = terminated or truncated
        last_obs = obs

    final_globals = last_obs["globals"]
    final_nodes = last_obs["nodes"]
    final_edge_attrs = last_obs["edge_attrs"]

    # Final score: passengers delivered
    score = int(round(float(final_globals[6]) * 500.0))

    # Cause of death: station with highest overcrowding progress
    max_prog = -1.0
    death_st_id = -1
    death_kind = "None"
    for i in range(30):
        prog = float(final_nodes[i, 22])
        if prog > max_prog:
            max_prog = prog
            death_st_id = i
            k_idx = int(np.argmax(final_nodes[i, 2:12]))
            death_kind = STATION_KINDS[k_idx] if k_idx < len(STATION_KINDS) else "Unknown"

    # Resources used
    # 1. Active lines: count lines with forward edges > 0
    active_lines = 0
    edge_lines = final_edge_attrs[:, 0:7]
    for l_idx in range(7):
        if np.any(edge_lines[:, l_idx] > 0):
            active_lines += 1

    trains_deployed = int(round(float(final_globals[7])))
    tunnels_used = int(np.sum((final_edge_attrs[:, 9] > 0.5) & (final_edge_attrs[:, 8] > 0)))
    interchanges_used = int(np.sum(final_nodes[:, 24] > 0.5))

    return {
        "score": score,
        "steps": step,
        "seconds": float(step * 1.0),
        "total_reward": float(total_reward),
        "death_station_id": death_st_id,
        "death_station_kind": death_kind,
        "death_progress": max_prog,
        "active_lines": active_lines,
        "trains_deployed": trains_deployed,
        "tunnels_used": tunnels_used,
        "interchanges_used": interchanges_used,
    }


# ==============================================================================
# STATISTICAL DISPERSION METRICS
# ==============================================================================

def compute_dispersion(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return {k: 0.0 for k in ["mean", "std", "median", "min", "max", "q25", "q75"]}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def aggregate_evaluation_results(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [ep["score"] for ep in episodes]
    steps = [ep["steps"] for ep in episodes]
    seconds = [ep["seconds"] for ep in episodes]
    rewards = [ep["total_reward"] for ep in episodes]
    lines = [ep["active_lines"] for ep in episodes]
    trains = [ep["trains_deployed"] for ep in episodes]
    tunnels = [ep["tunnels_used"] for ep in episodes]
    interchanges = [ep["interchanges_used"] for ep in episodes]

    # Cause of death counts
    death_counts: Dict[str, int] = {}
    for ep in episodes:
        k = ep["death_station_kind"]
        death_counts[k] = death_counts.get(k, 0) + 1

    return {
        "num_episodes": len(episodes),
        "score": compute_dispersion(scores),
        "steps": compute_dispersion(steps),
        "seconds": compute_dispersion(seconds),
        "reward": compute_dispersion(rewards),
        "lines": compute_dispersion(lines),
        "trains": compute_dispersion(trains),
        "tunnels": compute_dispersion(tunnels),
        "interchanges": compute_dispersion(interchanges),
        "death_distribution": death_counts,
    }


# ==============================================================================
# MARKDOWN REPORT GENERATOR
# ==============================================================================

def generate_markdown_report(
    benchmark_results: Dict[str, Dict[int, Dict[str, Any]]],
    seeds: List[int],
    model_path: Optional[str],
) -> str:
    lines: List[str] = []
    lines.append("# Mini Metro Empirical Evaluation Report (P3-1)\n")
    lines.append(f"**Evaluation Seeds ({len(seeds)})**: `{seeds}`  ")
    lines.append(f"**Model Checkpoint**: `{model_path if model_path else 'None'}`  ")
    lines.append(f"**Maps Evaluated**: London (Map 0), New York City (Map 1), Tokyo (Map 2)\n")
    lines.append("---\n")

    # Table 1: Score & Survival Summary
    lines.append("## 1. Performance Summary (Score & Survival Duration)\n")
    lines.append("| Map | Policy | Score (Mean ± Std) | Median [Q25 - Q75] | Min - Max | Survival Steps | Survival Time (s) |")
    lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|")

    for map_id in sorted(benchmark_results.keys()):
        map_name = MAP_NAMES.get(map_id, f"Map {map_id}")
        for pol_name, stats in benchmark_results[map_id].items():
            sc = stats["score"]
            st = stats["steps"]
            sec = stats["seconds"]
            lines.append(
                f"| **{map_name}** | {pol_name} | "
                f"{sc['mean']:.1f} ± {sc['std']:.1f} | "
                f"{sc['median']:.1f} [{sc['q25']:.1f} - {sc['q75']:.1f}] | "
                f"{int(sc['min'])} - {int(sc['max'])} | "
                f"{st['mean']:.1f} ± {st['std']:.1f} | "
                f"{sec['mean']:.1f}s |"
            )
    lines.append("\n---\n")

    # Table 2: Resource Utilization Summary
    lines.append("## 2. Resource Utilization (Mean ± Std)\n")
    lines.append("| Map | Policy | Active Lines | Trains Deployed | Tunnels Used | Interchanges Upgraded |")
    lines.append("|:---|:---|:---:|:---:|:---:|:---:|")

    for map_id in sorted(benchmark_results.keys()):
        map_name = MAP_NAMES.get(map_id, f"Map {map_id}")
        for pol_name, stats in benchmark_results[map_id].items():
            ln = stats["lines"]
            tr = stats["trains"]
            tu = stats["tunnels"]
            ic = stats["interchanges"]
            lines.append(
                f"| **{map_name}** | {pol_name} | "
                f"{ln['mean']:.1f} ± {ln['std']:.1f} | "
                f"{tr['mean']:.1f} ± {tr['std']:.1f} | "
                f"{tu['mean']:.1f} ± {tu['std']:.1f} | "
                f"{ic['mean']:.1f} ± {ic['std']:.1f} |"
            )
    lines.append("\n---\n")

    # Table 3: Cause of Death Distribution
    lines.append("## 3. Cause of Death Distribution (Overcrowded Station Kinds)\n")
    lines.append("| Map | Policy | Top Cause of Death | Full Breakdown (Shape: Count) |")
    lines.append("|:---|:---|:---|:---|")

    for map_id in sorted(benchmark_results.keys()):
        map_name = MAP_NAMES.get(map_id, f"Map {map_id}")
        for pol_name, stats in benchmark_results[map_id].items():
            dist = stats["death_distribution"]
            total = stats["num_episodes"]
            sorted_kinds = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            top_str = f"{sorted_kinds[0][0]} ({sorted_kinds[0][1] / total * 100:.0f}%)" if sorted_kinds else "N/A"
            breakdown_str = ", ".join([f"{k}: {c}" for k, c in sorted_kinds])
            lines.append(f"| **{map_name}** | {pol_name} | {top_str} | {breakdown_str} |")

    lines.append("\n---\n")
    return "\n".join(lines)


# ==============================================================================
# MAIN EVALUATION RUNNER
# ==============================================================================

def run_evaluation_suite(
    seeds: List[int],
    map_ids: List[int],
    policy_names: List[str],
    model_path: Optional[str] = None,
    output_md: Optional[str] = None,
    device_name: Optional[str] = None,
    max_steps: int = 500,
) -> Dict[str, Any]:
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 80)
    print("MINI METRO RIGOROUS EVALUATION SUITE (P3-1)")
    print("=" * 80)
    print(f"Device   : {device}")
    print(f"Maps     : {[MAP_NAMES.get(m, m) for m in map_ids]}")
    print(f"Seeds    : {seeds} ({len(seeds)} total)")
    print(f"Policies : {policy_names}")
    print("=" * 80, flush=True)

    # Initialize model if required
    model = None
    if any("model" in p.lower() or "deterministic" in p.lower() or "stochastic" in p.lower() for p in policy_names):
        model = MiniMetroActorCritic(hidden_dim=256).to(device)
        if model_path is None:
            model_files = glob.glob("runs/minimetro_ppo/model_*.pt") + glob.glob("ml/runs/minimetro_ppo/model_*.pt")
            if model_files:
                def get_ckpt_priority(f):
                    base = os.path.basename(f).replace("model_", "").replace(".pt", "")
                    if base == "final":
                        return float("inf")
                    try:
                        return float(base)
                    except ValueError:
                        return -1.0
                model_files.sort(key=get_ckpt_priority)
                model_path = model_files[-1]

        if model_path and os.path.exists(model_path):
            print(f"✓ Loaded model checkpoint: {model_path}")
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
        else:
            print(f"⚠️ No checkpoint found at '{model_path}', using randomly initialized model")
        model.eval()

    benchmark_results: Dict[int, Dict[str, Any]] = {}

    for map_id in map_ids:
        map_name = MAP_NAMES.get(map_id, f"Map {map_id}")
        benchmark_results[map_id] = {}
        print(f"\n>>> EVALUATING MAP {map_id}: {map_name.upper()} <<<", flush=True)

        for pol_name in policy_names:
            pol_key = pol_name.lower().strip()
            print(f"  Evaluating Policy: [{pol_name}] across {len(seeds)} seeds...", end="", flush=True)

            episodes_data: List[Dict[str, Any]] = []

            for seed in seeds:
                # Instantiate policy for this seed
                if pol_key in ("deterministic", "model_deterministic", "model (hierarchical deterministic)"):
                    assert model is not None, "Model required for deterministic policy"
                    policy = ModelPolicy(model, deterministic=True, device=device)
                    display_name = "Model (Deterministic)"
                elif pol_key in ("stochastic", "model_stochastic", "model (stochastic sampling)"):
                    assert model is not None, "Model required for stochastic policy"
                    policy = ModelPolicy(model, deterministic=False, device=device)
                    display_name = "Model (Stochastic)"
                elif pol_key in ("random", "random_legal", "random legal baseline"):
                    policy = RandomLegalPolicy(seed=seed)
                    display_name = "Random Legal"
                elif pol_key in ("greedy", "greedy_heuristic", "greedy heuristic baseline"):
                    policy = GreedyHeuristicPolicy(seed=seed)
                    display_name = "Greedy Heuristic"
                else:
                    raise ValueError(f"Unknown policy name: {pol_name}")

                # Create fresh environment with exact map and seed
                env = MiniMetroEnv(map_id=map_id, seed=seed)
                ep_result = run_single_episode(env, policy, seed=seed, max_steps=max_steps)
                episodes_data.append(ep_result)
                env.close()
                print(f" (s{seed}:{ep_result['score']}p/{ep_result['steps']}st)", end="", flush=True)

            summary = aggregate_evaluation_results(episodes_data)
            benchmark_results[map_id][display_name] = summary

            sc = summary["score"]
            st = summary["steps"]
            print(f"\n     -> Summary: Score={sc['mean']:.1f}±{sc['std']:.1f} (Med={sc['median']:.1f}) | Steps={st['mean']:.1f}±{st['std']:.1f}", flush=True)

        # Save intermediate report after each map
        if output_md:
            partial_report = generate_markdown_report(benchmark_results, seeds, model_path)
            with open(output_md, "w", encoding="utf-8") as f:
                f.write(partial_report)

    # Generate Markdown Report
    report_md = generate_markdown_report(benchmark_results, seeds, model_path)

    print("\n" + "=" * 80)
    print(report_md)
    print("=" * 80)

    if output_md:
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"✓ Saved markdown report to: {output_md}")

    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="Rigorous Multi-Seed & Cross-Map Evaluation Suite (P3-1)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="List of random seeds to evaluate")
    parser.add_argument("--maps", type=int, nargs="+", default=[0, 1, 2], help="List of map IDs: 0=London, 1=NYC, 2=Tokyo")
    parser.add_argument("--policies", type=str, nargs="+", default=["deterministic", "stochastic", "greedy", "random"],
                        help="List of policies: deterministic, stochastic, greedy, random")
    parser.add_argument("--model_path", type=str, default=None, help="Path to checkpoint model file")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per evaluation episode (default: 500)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: runs 2 seeds on specified maps")
    parser.add_argument("--output_md", type=str, default=None, help="Path to write markdown output report")
    parser.add_argument("--device", type=str, default=None, help="Torch device: cpu or cuda")
    args = parser.parse_args()

    seeds = [1000, 1001] if args.quick else args.seeds
    run_evaluation_suite(
        seeds=seeds,
        map_ids=args.maps,
        policy_names=args.policies,
        model_path=args.model_path,
        output_md=args.output_md,
        device_name=args.device,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()

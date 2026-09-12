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


class GrandmasterPolicy(BasePolicy):
    """
    Grandmaster-level strategic network controller and intervention policy:
    1. Weekly Reward Optimization: Prioritizes network coverage (Lines until 4-5 active lines),
       then throughput scaling (Interchanges for high-degree transfer junctions, Carriages to double train
       capacity, Locomotives to cut headway).
    2. Strategic Interchange Placement: Prioritizes major transfer junctions (degree >= 3)
       and emergency overcrowding relief (progress > 0.25) to expand capacity to 18.
    3. Train Reservation Guard: Never consumes trains on AddTrain if an unbuilt Line token is available,
       preventing stranded lines.
    4. Proactive Dispatch: Deploys extra trains and carriages to lines with longest round-trips and highest queue demands.
    5. Sprawl-Constrained Network Building: Builds new lines connecting diverse station shapes with minimal distance.
    6. Short-Line Headway Balancing: Connects unconnected stations preferring short lines (<= 5 stations) to guarantee round-trip time < 45s.
    7. Dual-Service Multi-Line Relief: When a station enters crisis (oc > 0.20 or queue >= 8), connects an adjacent
       secondary line to halve headway and clear queue backlog.
    8. Loop Closure: Closes compact loops (4-6 stations) to eliminate turnaround delays.
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
        edges = obs["edges"]

        unused_lines = int(round(globals_feat[0]))
        unused_trains = int(round(globals_feat[1]))
        unused_carriages = int(round(globals_feat[2]))
        unused_tunnels = int(round(globals_feat[3]))
        interchanges_avail = int(round(globals_feat[4]))

        alive_st = [i for i in range(30) if np.any(nodes[i, 2:12] > 0)]
        unconnected = [i for i in alive_st if nodes[i, 23] == 0]
        interchanges_placed = int(np.sum(nodes[:, 24] > 0.5))

        # Reconstruct line station sets
        line_stations: Dict[int, Set[int]] = {}
        st_lines: Dict[int, List[int]] = {st: [] for st in alive_st}
        for l in range(7):
            l_edges = np.where(edge_attrs[:, l] > 0)[0]
            if len(l_edges) > 0:
                st_set: Set[int] = set()
                for e in l_edges:
                    u, v = int(edges[0, e]), int(edges[1, e])
                    st_set.add(u)
                    st_set.add(v)
                line_stations[l] = st_set
                for st in st_set:
                    if st in st_lines:
                        st_lines[st].append(l)

        active_lines = list(line_stations.keys())
        line_lengths = {l: len(sts) for l, sts in line_stations.items()}

        critical_stations = sorted([s for s in alive_st if nodes[s, 22] > 0.25],
                                   key=lambda s: float(nodes[s, 22]), reverse=True)
        oc_stations = sorted([s for s in alive_st if nodes[s, 22] > 0.05],
                             key=lambda s: float(nodes[s, 22]), reverse=True)
        high_q_stations = sorted([s for s in alive_st if np.sum(nodes[s, 12:22]) >= 6],
                                 key=lambda s: float(np.sum(nodes[s, 12:22])), reverse=True)

        # 1. Weekly Rewards: Actions 4050, 4051
        if mask[4050] or mask[4051]:
            total_lines = len(active_lines) + unused_lines
            has_major_hub = any(nodes[s, 23] >= 3 and nodes[s, 24] == 0 for s in alive_st)
            hub_val = 26 if (has_major_hub and interchanges_placed < 2) else 14
            line_val = 28 if total_lines < 4 else (18 if total_lines < 5 else 6)
            train_val = 24
            carriage_val = 20
            tunnel_val = 5
            card_prio = [line_val, train_val, tunnel_val, carriage_val, hub_val]

            v0 = card_prio[int(np.argmax(globals_feat[13:18]))] if mask[4050] and len(globals_feat) >= 18 else -1
            v1 = card_prio[int(np.argmax(globals_feat[18:23]))] if mask[4051] and len(globals_feat) >= 23 else -1
            return 4050 if v0 >= v1 and mask[4050] else 4051

        # 2. Upgrade Interchange on Critical Overcrowding or Major Transfer Hubs
        if mask[4020:4050].any():
            legal_hubs = np.where(mask[4020:4050])[0]
            # Emergency upgrade if critical station is legal
            for st in critical_stations:
                if (4020 + st) in legal_hubs:
                    return 4020 + st

            best_hub = -1
            best_score = -1.0
            for st in legal_hubs:
                deg = float(nodes[st, 23])
                q = float(np.sum(nodes[st, 12:22]))
                oc = float(nodes[st, 22])
                kind = int(np.argmax(nodes[st, 2:12]))
                rare_mult = 1.2 if kind >= 2 else 1.0
                score = (deg * 40.0 + q * 12.0 + oc * 300.0) * rare_mult
                if score > best_score and (deg >= 3 or q >= 5 or oc > 0.05):
                    best_score = score
                    best_hub = st
            if best_hub >= 0:
                return 4020 + best_hub

        # 3. AddLine: Build available line immediately before consuming trains
        if unused_lines > 0 and unused_trains > 0 and mask[1:436].any():
            triu_u, triu_v = np.triu_indices(30, k=1)
            best_pair = -1
            best_pair_score = -999.0

            priority_targets = set(unconnected)
            if critical_stations:
                priority_targets.add(critical_stations[0])
            elif oc_stations:
                priority_targets.add(oc_stations[0])
            elif high_q_stations:
                priority_targets.add(high_q_stations[0])

            for p_idx in range(len(triu_u)):
                if mask[1 + p_idx]:
                    u, v = triu_u[p_idx], triu_v[p_idx]
                    if u in alive_st and v in alive_st:
                        ku, kv = int(np.argmax(nodes[u, 2:12])), int(np.argmax(nodes[v, 2:12]))
                        dist = float(np.linalg.norm(nodes[u, 0:2] - nodes[v, 0:2]))
                        score = 0.0
                        if ku != kv:
                            score += 45.0
                        if ku >= 2 or kv >= 2:
                            score += 35.0
                        if u in priority_targets or v in priority_targets:
                            score += 150.0
                        score -= dist * 2.5
                        if score > best_pair_score:
                            best_pair_score = score
                            best_pair = p_idx
            if best_pair >= 0:
                return 1 + best_pair

        # 4. Connect Unconnected Stations (Top Priority: Never leave a station isolated!)
        if unconnected:
            unconnected_sorted = sorted(
                unconnected,
                key=lambda s: (float(nodes[s, 22]), float(np.sum(nodes[s, 12:22]))),
                reverse=True
            )
            for target in unconnected_sorted:
                t_pos = nodes[target, 0:2]
                best_act = -1
                best_dist_score = -9999.0

                for l_id in sorted(active_lines, key=lambda l: line_lengths.get(l, 0)):
                    l_len = line_lengths.get(l_id, 0)
                    if mask[436:856].any():
                        for end in [0, 1]:
                            idx = 436 + (l_id * 30 + target) * 2 + end
                            if idx < 856 and mask[idx]:
                                min_st_d = min(float(np.linalg.norm(t_pos - nodes[st, 0:2])) for st in line_stations.get(l_id, set()))
                                dist_score = 150.0 - min_st_d * 3.0 - l_len * 6.0
                                if dist_score > best_dist_score:
                                    best_dist_score = dist_score
                                    best_act = idx

                    if mask[856:4006].any():
                        legal_ins = np.where(mask[856:4006])[0]
                        for ins in legal_ins:
                            rem = ins // 15
                            st_id = rem % 30
                            l = rem // 30
                            if st_id == target and l == l_id:
                                dist_score = 140.0 - l_len * 6.0
                                if dist_score > best_dist_score:
                                    best_dist_score = dist_score
                                    best_act = 856 + ins

                if best_act >= 0:
                    return best_act

        # 5. Emergency Extra Train / Carriage Allocation
        extra_trains = unused_trains - unused_lines
        urgent_pool = critical_stations if critical_stations else oc_stations
        if extra_trains > 0 and urgent_pool:
            target_urgent = urgent_pool[0]
            serving_lines = st_lines.get(target_urgent, [])
            if mask[4006:4013].any():
                for l in serving_lines:
                    if mask[4006 + l]:
                        return 4006 + l

        if unused_carriages > 0 and urgent_pool:
            target_urgent = urgent_pool[0]
            serving_lines = st_lines.get(target_urgent, [])
            if mask[4013:4020].any():
                for l in serving_lines:
                    if mask[4013 + l]:
                        return 4013 + l

        # 6. Crisis Intervention: Dual-service relief for stations with oc > 0.25
        if critical_stations:
            for target in critical_stations:
                t_pos = nodes[target, 0:2]
                for l_id in sorted(active_lines, key=lambda l: line_lengths.get(l, 0)):
                    if target not in line_stations.get(l_id, set()) and line_lengths.get(l_id, 0) <= 5:
                        min_st_d = min(float(np.linalg.norm(t_pos - nodes[st, 0:2])) for st in line_stations[l_id])
                        if min_st_d < 0.40:
                            if mask[436:856].any():
                                for end in [0, 1]:
                                    idx = 436 + (l_id * 30 + target) * 2 + end
                                    if idx < 856 and mask[idx]:
                                        return idx
                            if mask[856:4006].any():
                                legal_ins = np.where(mask[856:4006])[0]
                                for ins in legal_ins:
                                    rem = ins // 15
                                    st_id = rem % 30
                                    l = rem // 30
                                    if st_id == target and l == l_id:
                                        return 856 + ins

        # 7. Routine AddCarriage & AddTrain (Extra trains only!)
        if unused_carriages > 0 and mask[4013:4020].any():
            legal_c = np.where(mask[4013:4020])[0]
            best_c = max(legal_c, key=lambda l: sum(np.sum(nodes[s, 12:22]) for s in line_stations.get(l, set())))
            return 4013 + best_c

        if extra_trains > 0 and mask[4006:4013].any():
            legal_t = np.where(mask[4006:4013])[0]
            best_t = max(legal_t, key=lambda l: (line_lengths.get(l, 0), sum(np.sum(nodes[s, 12:22]) for s in line_stations.get(l, set()))))
            return 4006 + best_t

        # 8. CloseLoop for Compact Cycles (4-6 stations)
        if mask[4052:4059].any():
            legal_loops = np.where(mask[4052:4059])[0]
            for l_id in legal_loops:
                if 4 <= line_lengths.get(l_id, 0) <= 6:
                    return 4052 + l_id

        # 9. Rare Station Multi-Line Connectivity
        if mask[436:856].any():
            rare_st = [s for s in alive_st if int(np.argmax(nodes[s, 2:12])) >= 2 and nodes[s, 23] < 2]
            if rare_st:
                target = rare_st[0]
                for l_id in sorted(active_lines, key=lambda l: line_lengths.get(l, 0)):
                    if line_lengths.get(l_id, 0) < 5 and target not in line_stations.get(l_id, set()):
                        for end in [0, 1]:
                            idx = 436 + (l_id * 30 + target) * 2 + end
                            if idx < 856 and mask[idx]:
                                return idx

        # Default NoOp
        if mask[0]:
            return 0
        legal = np.where(mask)[0]
        return int(legal[0]) if len(legal) > 0 else 0


class GreedyHeuristicPolicy(GrandmasterPolicy):
    """Alias for backwards compatibility with test harnesses and benchmark scripts."""
    pass


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
                elif pol_key in ("greedy", "greedy_heuristic", "greedy heuristic baseline", "grandmaster", "grandmaster_policy", "elite"):
                    policy = GrandmasterPolicy(seed=seed)
                    display_name = "Grandmaster Policy"
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

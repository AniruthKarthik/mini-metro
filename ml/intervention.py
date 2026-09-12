"""
Strategic Network Intervention & Multi-Horizon Counterfactual Evaluation Module (Enhanced).

Provides principled, symmetric candidate intervention evaluation, multi-horizon counterfactual
simulation rollouts via Go state cloning, continuous topology-vs-capacity diagnosis,
reward-aligned candidate utilities, dominating game-over risk constraints, value-based hysteresis,
structural churn tracking, and principled arbitration across:
  - KEEP (preserve network topology, true counterfactual baseline)
  - DISPATCH (AddTrain, AddCarriage, Interchange — cheap capacity relief)
  - LOCAL_EDIT (ExtendLine, InsertStation, ShortenLine, Loop — low disruption topology)
  - MAJOR_REBUILD (RemoveLine + corridor redesign — high disruption reconstruction)
  - EXPANSION (AddLine — high-impact irreversible resource consumption & new topology)

Ensures network modifications (both demolition AND expansion) occur only when the expected
multi-horizon improvement justifies operational disruption, resource cost, and structural churn.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import math
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


def decode_add_line_stations(action_id: int) -> Tuple[int, int]:
    """Decodes flat AddLine action ID (1..435) into station pair (u, v) with u < v."""
    idx = action_id - ADD_LINE_START
    curr = 0
    for i in range(30):
        for j in range(i + 1, 30):
            if curr == idx:
                return (i, j)
            curr += 1
    return (-1, -1)


def encode_add_line_action(u: int, v: int) -> int:
    """Encodes station pair (u, v) into flat AddLine action ID (1..435)."""
    if u > v:
        u, v = v, u
    curr = 0
    for i in range(30):
        for j in range(i + 1, 30):
            if i == u and j == v:
                return ADD_LINE_START + curr
            curr += 1
    return ACTION_NOOP


class InterventionTier(Enum):
    KEEP = "keep"                         # Tier 0: No change, preserve stability
    DISPATCH = "dispatch"                 # Tier 1: AddTrain, AddCarriage, Interchange (zero disruption)
    LOCAL_EDIT = "local_edit"             # Tier 2: ExtendLine, InsertStation, ShortenLine, Loop (low disruption)
    MAJOR_REBUILD = "major_rebuild"       # Tier 3: RemoveLine (high disruption demolition)
    EXPANSION = "expansion"               # Tier 4: AddLine (high impact resource consumption & topology)
    REWARD_CHOICE = "reward_choice"       # Meta: Weekly reward selection


def classify_action_tier(action_id: int) -> InterventionTier:
    """Classifies any discrete action ID into its strategic intervention tier."""
    if action_id == ACTION_NOOP:
        return InterventionTier.KEEP
    if ADD_TRAIN_START <= action_id < ADD_CARRIAGE_END or INTERCHANGE_START <= action_id < INTERCHANGE_END:
        return InterventionTier.DISPATCH
    if CHOOSE_REWARD_START <= action_id < CHOOSE_REWARD_END:
        return InterventionTier.REWARD_CHOICE
    if SHORTEN_LINE_START <= action_id < SHORTEN_LINE_END:
        return InterventionTier.LOCAL_EDIT
    if EXTEND_LINE_START <= action_id < INSERT_STATION_END or CLOSE_LOOP_START <= action_id < OPEN_LOOP_END:
        return InterventionTier.LOCAL_EDIT
    if REMOVE_LINE_START <= action_id < REMOVE_LINE_END:
        return InterventionTier.MAJOR_REBUILD
    if ADD_LINE_START <= action_id < ADD_LINE_END:
        return InterventionTier.EXPANSION
    return InterventionTier.KEEP


def is_high_impact_structural_action(action_id: int, obs: Optional[Dict[str, np.ndarray]] = None) -> bool:
    """
    Symmetric structural impact classifier:
    Identifies high-impact actions that alter network topology or deplete scarce assets
    (RemoveLine, ShortenLine, AddLine, large ExtendLine) and require counterfactual validation.
    """
    if action_id == ACTION_NOOP:
        return False
    if REMOVE_LINE_START <= action_id < REMOVE_LINE_END:
        return True
    if SHORTEN_LINE_START <= action_id < SHORTEN_LINE_END:
        return True
    if ADD_LINE_START <= action_id < ADD_LINE_END:
        return True
    if EXTEND_LINE_START <= action_id < EXTEND_LINE_END and obs is not None:
        nodes = obs.get("nodes", None)
        if nodes is not None:
            idx = action_id - EXTEND_LINE_START
            st_id = (idx // 2) % 30
            if 0 <= st_id < len(nodes):
                deg = float(nodes[st_id, 23]) if nodes.shape[-1] > 23 else 0.0
                if deg >= 2.0:
                    return True
    return False


class InterventionResult(tuple):
    """
    Result of candidate intervention evaluation.
    Subclasses tuple (best_action, candidate_utilities) for 100% backward compatibility
    with existing code: `best_act, utils = arbiter.evaluate_candidates(...)`.
    Also provides attributes `.best_action`, `.candidate_utilities`, `.candidate_set_regret`, `.regret`, `.details`.
    """
    def __new__(
        cls,
        best_action: int,
        candidate_utilities: Dict[int, float],
        candidate_set_regret: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ):
        return super().__new__(cls, (best_action, candidate_utilities))

    def __init__(
        self,
        best_action: int,
        candidate_utilities: Dict[int, float],
        candidate_set_regret: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.best_action = best_action
        self.candidate_utilities = candidate_utilities
        self.candidate_set_regret = float(candidate_set_regret)
        self.details = details or {}

    @property
    def regret(self) -> float:
        """Backward compatibility alias for candidate_set_regret."""
        return self.candidate_set_regret


@dataclass
class LineHealthDiagnosis:
    """
    Continuous diagnostic profile distinguishing capacity bottlenecks from topological defects.
    Replaces discontinuous boolean thresholds with continuous sigmoidal scores.
    """
    line_id: int
    station_ids: List[int] = field(default_factory=list)
    total_queue: int = 0
    max_station_queue: int = 0
    avg_growth_rate: float = 0.0
    busiest_station_id: int = -1
    is_emergency: bool = False
    emergency_station_ids: List[int] = field(default_factory=list)
    redundant_station_count: int = 0
    avg_train_occupancy: float = 0.0
    track_length: float = 0.0
    shape_diversity: float = 0.0

    # Continuous scores in [0.0, 1.0]
    capacity_score: float = 0.0
    topology_score: float = 0.0

    @property
    def is_capacity_issue(self) -> bool:
        return self.capacity_score >= 0.5

    @property
    def is_topology_issue(self) -> bool:
        return self.topology_score >= 0.5


@dataclass
class ChurnTracker:
    """Tracks network modification history to inform structural churn penalties."""
    last_edit_sim_time: float = -999.0
    recent_edits_sim_times: List[float] = field(default_factory=list)
    line_last_modified_sim_time: Dict[int, float] = field(default_factory=dict)
    recent_deletions_count: int = 0
    recent_additions_count: int = 0

    def record_edit(self, action_id: int, sim_time: float):
        self.last_edit_sim_time = sim_time
        self.recent_edits_sim_times.append(sim_time)
        # Prune older than 60s
        self.recent_edits_sim_times = [t for t in self.recent_edits_sim_times if sim_time - t <= 60.0]

        tier = classify_action_tier(action_id)
        if tier == InterventionTier.MAJOR_REBUILD:
            line_id = action_id - REMOVE_LINE_START
            self.line_last_modified_sim_time[line_id] = sim_time
            self.recent_deletions_count += 1
        elif tier == InterventionTier.EXPANSION:
            self.recent_additions_count += 1
        elif SHORTEN_LINE_START <= action_id < SHORTEN_LINE_END:
            line_id = (action_id - SHORTEN_LINE_START) // 2
            self.line_last_modified_sim_time[line_id] = sim_time

    def get_recent_edits_count(self, current_sim_time: float, window: float = 60.0) -> int:
        return sum(1 for t in self.recent_edits_sim_times if current_sim_time - t <= window)

    def get_line_age(self, line_id: int, current_sim_time: float) -> float:
        if line_id not in self.line_last_modified_sim_time:
            return 999.0
        return max(0.0, current_sim_time - self.line_last_modified_sim_time[line_id])


class StrategicInterventionArbiter:
    """
    Evaluates candidate network interventions using multi-horizon Go state cloning.
    Estimates:
        InterventionValue = MultiHorizonFutureBenefit - TierHurdle - ChurnPenalty
    Compares the best available candidate directly against KEEP, enforcing principled
    tie-breaking in favor of lower disruption and risk-constrained terminal safety.
    """

    def __init__(
        self,
        cf_duration: float = 4.0,
        horizons: Optional[List[float]] = None,
        horizon_weights: Optional[List[float]] = None,
        rebuild_threshold: float = 1.0,
        add_line_threshold: float = 0.60,
        local_edit_bias: float = 0.20,
        tie_epsilon: float = 0.35,
        cooldown_period: float = 45.0,
        cooldown_penalty: float = 2.5,
        churn_penalty_weight: float = 0.5,
        game_over_risk_penalty: float = 250.0,
        emergency_override: bool = True,
    ):
        """
        Args:
            cf_duration: Legacy single-horizon duration (seconds).
            horizons: Multi-horizon evaluation checkpoints in seconds (e.g. [4.0, 16.0, 32.0, 48.0]).
            horizon_weights: Weights for aggregating multi-horizon cumulative utilities.
            rebuild_threshold: Base marginal benefit hurdle required for RemoveLine over KEEP.
            add_line_threshold: Base marginal benefit hurdle required for AddLine over KEEP.
            local_edit_bias: Preference offset for lower-disruption local modifications.
            tie_epsilon: Margin within which lower-disruption actions are strictly preferred.
            cooldown_period: Duration in seconds over which recent-edit penalty decays.
            cooldown_penalty: Maximum value penalty for immediately re-modifying a line.
            churn_penalty_weight: Penalty per excess structural modification in recent window.
            game_over_risk_penalty: Dominating penalty applied if any horizon experiences terminal collapse.
            emergency_override: If True, lowers hurdles during severe overcrowding while preventing panic demolition.
        """
        self.cf_duration = cf_duration
        self.horizons = horizons if horizons is not None else [4.0, 16.0, 32.0, 48.0]
        self.horizon_weights = horizon_weights if horizon_weights is not None else [0.15, 0.25, 0.30, 0.30]
        # Normalize weights to sum to 1.0
        total_w = sum(self.horizon_weights)
        if total_w > 0:
            self.horizon_weights = [w / total_w for w in self.horizon_weights]

        self.rebuild_threshold = rebuild_threshold
        self.add_line_threshold = add_line_threshold
        self.local_edit_bias = local_edit_bias
        self.tie_epsilon = tie_epsilon
        self.cooldown_period = cooldown_period
        self.cooldown_penalty = cooldown_penalty
        self.churn_penalty_weight = churn_penalty_weight
        self.game_over_risk_penalty = game_over_risk_penalty
        self.emergency_override = emergency_override

        self.churn_tracker = ChurnTracker()
        self.last_regret: float = 0.0

    def compute_candidate_utility(
        self,
        total_reward: float,
        breakdown: Dict[str, float],
        done: bool,
        useful_coverage_ratio: float = 1.0,
    ) -> float:
        """
        Computes candidate utility tightly aligned with actual environment return:
        - Uses actual cumulative environment reward R_h(a) as the primary value.
        - Disallows unearned graph connectivity bonuses for lines without useful passenger demand.
        - Terminal GameOver is handled as a dominating risk constraint across all horizons.
        """
        if done:
            return -self.game_over_risk_penalty

        adjusted_return = total_reward

        # Adjust connectivity: if a line creates raw graph reachability without serving useful passenger demand,
        # discount the unearned connectivity bonus
        conn_bonus = breakdown.get("connectivity", 0.0)
        if conn_bonus > 0.0 and useful_coverage_ratio < 0.5:
            unearned_conn = conn_bonus * (0.5 - useful_coverage_ratio) * 2.0
            adjusted_return -= unearned_conn

        return adjusted_return

    def compute_network_utility(self, breakdown: Dict[str, float], done: bool) -> float:
        """
        Backward compatibility helper: computes composite network utility from breakdown.
        """
        if done:
            return -self.game_over_risk_penalty

        delivery = breakdown.get("delivery", 0.0)
        connectivity = breakdown.get("connectivity", 0.0)
        crowd = breakdown.get("crowd_penalty", 0.0)
        game_over = breakdown.get("game_over", 0.0)
        redundancy = breakdown.get("redundancy", 0.0)
        track_eff = breakdown.get("track_efficiency", 0.0)
        disruption = breakdown.get("disruption", 0.0)

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

    def diagnose_line_health(self, obs: Dict[str, np.ndarray], line_id: int) -> LineHealthDiagnosis:
        """
        Analyzes line properties from vectorized observation to distinguish capacity issues
        from topological defects using continuous sigmoidal scores.
        """
        diag = LineHealthDiagnosis(line_id=line_id)
        if obs is None or "nodes" not in obs:
            return diag

        nodes = obs["nodes"]
        num_nodes = int(obs["num_nodes"][0]) if "num_nodes" in obs else len(nodes)
        edge_attrs = obs.get("edge_attrs", None)
        edges = obs.get("edges", None)

        served_stations: Set[int] = set()
        track_len = 0.0
        if edge_attrs is not None and edges is not None:
            num_edges = int(obs["num_edges"][0]) if "num_edges" in obs else len(edge_attrs)
            for e_idx in range(min(num_edges, len(edge_attrs))):
                if line_id < edge_attrs.shape[-1] and edge_attrs[e_idx, line_id] > 0.5:
                    u = int(edges[0, e_idx]) if edges.shape[0] == 2 else int(edges[e_idx, 0])
                    v = int(edges[1, e_idx]) if edges.shape[0] == 2 else int(edges[e_idx, 1])
                    if 0 <= u < num_nodes:
                        served_stations.add(u)
                    if 0 <= v < num_nodes:
                        served_stations.add(v)
                    # Length of edge if available in edge_attrs
                    if edge_attrs.shape[-1] > 7:
                        track_len += float(edge_attrs[e_idx, 7])

        diag.station_ids = sorted(list(served_stations))
        diag.track_length = track_len
        growth_rates = []
        max_q = 0
        busiest_st = -1
        station_shapes = set()

        for st_id in diag.station_ids:
            if st_id >= num_nodes:
                continue
            st_feat = nodes[st_id]
            q_sum = int(np.sum(st_feat[12:22]))
            diag.total_queue += q_sum
            if q_sum > max_q:
                max_q = q_sum
                busiest_st = st_id

            # Station shape (features 0..4)
            shape_idx = int(np.argmax(st_feat[0:5])) if st_feat.shape[0] >= 5 else 0
            station_shapes.add(shape_idx)

            # Overcrowd timer & progress
            timer_norm = float(st_feat[27]) if st_feat.shape[0] > 27 else 0.0
            fill_ratio = float(st_feat[25]) if st_feat.shape[0] > 25 else 0.0
            if timer_norm > 0.0 or fill_ratio >= 0.85:
                diag.is_emergency = True
                diag.emergency_station_ids.append(st_id)

            # Growth rate (feature 28)
            growth = float(st_feat[28]) if st_feat.shape[0] > 28 else 0.0
            growth_rates.append(growth)

            # Degree / redundancy (feature 23)
            degree = float(st_feat[23]) if st_feat.shape[0] > 23 else 1.0
            if degree > 2.0:
                diag.redundant_station_count += 1

        diag.max_station_queue = max_q
        diag.busiest_station_id = busiest_st
        diag.avg_growth_rate = float(np.mean(growth_rates)) if growth_rates else 0.0
        diag.shape_diversity = float(len(station_shapes)) / max(1.0, float(len(diag.station_ids)))

        # Train occupancy from globals or nodes
        globals_feat = obs.get("globals", None)
        diag.avg_train_occupancy = 0.5
        if globals_feat is not None and len(globals_feat) > 10:
            diag.avg_train_occupancy = float(globals_feat[8]) if len(globals_feat) > 8 else 0.5

        # Continuous capacity score via smooth sigmoid
        cap_logit = (
            8.0 * (diag.avg_growth_rate - 0.03)
            + 3.0 * (diag.avg_train_occupancy - 0.50)
            + 0.20 * (diag.total_queue - 5.0)
            - 1.0 * float(diag.redundant_station_count)
        )
        diag.capacity_score = float(1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, cap_logit)))))

        # Continuous topology score via smooth sigmoid
        topo_logit = (
            1.5 * (float(diag.redundant_station_count) - 1.0)
            + 0.01 * (diag.track_length - 80.0)
            - 2.5 * (diag.shape_diversity - 0.50)
        )
        diag.topology_score = float(1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, topo_logit)))))

        return diag

    def _estimate_useful_coverage_ratio(self, action_id: int, obs: Dict[str, np.ndarray]) -> float:
        """
        Estimates whether an AddLine action serves genuine passenger demand between endpoints.
        """
        if not (ADD_LINE_START <= action_id < ADD_LINE_END) or obs is None or "nodes" not in obs:
            return 1.0

        u, v = decode_add_line_stations(action_id)
        nodes = obs["nodes"]
        num_nodes = int(obs["num_nodes"][0]) if "num_nodes" in obs else len(nodes)
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            return 0.5

        # Shape of u and v
        shape_u = int(np.argmax(nodes[u, 0:5])) if nodes.shape[-1] >= 5 else 0
        shape_v = int(np.argmax(nodes[v, 0:5])) if nodes.shape[-1] >= 5 else 0

        # Passenger demand at u for shape_v and vice-versa
        # Destination demands are in features 12..21
        demand_u_for_v = float(nodes[u, 12 + shape_v]) if nodes.shape[-1] >= 12 + shape_v else 0.0
        demand_v_for_u = float(nodes[v, 12 + shape_u]) if nodes.shape[-1] >= 12 + shape_u else 0.0

        total_q_u = float(np.sum(nodes[u, 12:22])) if nodes.shape[-1] >= 22 else 0.0
        total_q_v = float(np.sum(nodes[v, 12:22])) if nodes.shape[-1] >= 22 else 0.0

        if total_q_u == 0.0 and total_q_v == 0.0:
            return 0.1  # Connecting two empty stations produces no immediate useful service

        direct_demand = demand_u_for_v + demand_v_for_u
        if direct_demand > 0.0:
            return 1.0

        # Different shapes offer diversity
        if shape_u != shape_v and (total_q_u > 2 or total_q_v > 2):
            return 0.75

        return 0.35

    def generate_candidate_portfolio(
        self,
        env: Any,
        obs: Dict[str, np.ndarray],
        proposed_action_id: int,
        top_k: int = 6,
    ) -> List[int]:
        """
        Symmetric Candidate Portfolio Generator:
        Handles both demolition (RemoveLine/ShortenLine) AND expansion (AddLine/ExtendLine).
        Always includes KEEP (Action 0), the proposed action, top alternative connections/edits,
        and cheap operational dispatches (AddTrain, AddCarriage, Interchange).
        """
        candidates: List[int] = [ACTION_NOOP]
        mask = obs.get("action_mask", None) if obs is not None else None

        if proposed_action_id != ACTION_NOOP and (mask is None or mask[proposed_action_id]):
            candidates.append(proposed_action_id)

        tier = classify_action_tier(proposed_action_id)
        nodes = obs.get("nodes", None) if obs is not None else None
        num_nodes = int(obs["num_nodes"][0]) if obs is not None and "num_nodes" in obs else (len(nodes) if nodes is not None else 1)

        # Case 1: Proposed action is RemoveLine (Major Rebuild)
        if tier == InterventionTier.MAJOR_REBUILD:
            line_id = proposed_action_id - REMOVE_LINE_START
            diag = self.diagnose_line_health(obs, line_id)

            # Capacity alternatives for this line
            train_act = ADD_TRAIN_START + line_id
            if mask is None or (train_act < len(mask) and mask[train_act]):
                candidates.append(train_act)

            carriage_act = ADD_CARRIAGE_START + line_id
            if mask is None or (carriage_act < len(mask) and mask[carriage_act]):
                candidates.append(carriage_act)

            if diag.busiest_station_id >= 0:
                interchange_act = INTERCHANGE_START + diag.busiest_station_id
                if mask is None or (interchange_act < len(mask) and mask[interchange_act]):
                    candidates.append(interchange_act)

            # Local pruning alternatives
            shorten_front = SHORTEN_LINE_START + line_id * 2 + 0
            if mask is None or (shorten_front < len(mask) and mask[shorten_front]):
                candidates.append(shorten_front)
            shorten_back = SHORTEN_LINE_START + line_id * 2 + 1
            if mask is None or (shorten_back < len(mask) and mask[shorten_back]):
                candidates.append(shorten_back)

        # Case 2: Proposed action is AddLine (Network Expansion)
        elif tier == InterventionTier.EXPANSION:
            # Add cheap capacity alternatives on existing congested lines instead of building a new line
            if mask is not None:
                # Find most congested existing line
                best_train_act = -1
                best_carriage_act = -1
                for l_id in range(7):
                    t_act = ADD_TRAIN_START + l_id
                    c_act = ADD_CARRIAGE_START + l_id
                    if t_act < len(mask) and mask[t_act]:
                        best_train_act = t_act
                    if c_act < len(mask) and mask[c_act]:
                        best_carriage_act = c_act
                if best_train_act >= 0:
                    candidates.append(best_train_act)
                if best_carriage_act >= 0:
                    candidates.append(best_carriage_act)

            # Add alternative AddLine candidates that connect highest unserved queues
            if mask is not None and nodes is not None:
                valid_add_lines = []
                for a in range(ADD_LINE_START, ADD_LINE_END):
                    if mask[a] and a != proposed_action_id:
                        u, v = decode_add_line_stations(a)
                        if 0 <= u < num_nodes and 0 <= v < num_nodes:
                            q_u = float(np.sum(nodes[u, 12:22]))
                            q_v = float(np.sum(nodes[v, 12:22]))
                            dist = float(np.linalg.norm(nodes[u, 0:2] - nodes[v, 0:2]))
                            # Score heuristic: higher queue, shorter distance
                            score_heur = (q_u + q_v + 1.0) / (dist + 0.1)
                            valid_add_lines.append((score_heur, a))

                valid_add_lines.sort(key=lambda x: x[0], reverse=True)
                for _, alt_a in valid_add_lines[:2]:
                    candidates.append(alt_a)

        # Include other legal local/dispatch actions up to top_k
        if mask is not None:
            valid_indices = np.where(mask)[0]
            for a in valid_indices:
                if len(candidates) >= top_k:
                    break
                if a not in candidates:
                    a_tier = classify_action_tier(a)
                    if a_tier in (InterventionTier.DISPATCH, InterventionTier.LOCAL_EDIT):
                        candidates.append(int(a))

        # Deduplicate while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        return unique_candidates[:top_k]

    def evaluate_candidate_multi_horizon(
        self,
        env: Any,
        action_id: int,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Simulates an action forward across multi-horizon checkpoints (e.g. 4s, 16s, 32s, 48s)
        and computes the weighted aggregate future network utility.
        Returns: (aggregate_utility, has_game_over, metrics_summary)
        """
        if hasattr(env, "simulate_candidate_multi_horizon"):
            horizon_results = env.simulate_candidate_multi_horizon(action_id, horizons=self.horizons)
            agg_utility = 0.0
            has_game_over = False
            total_actual_reward = 0.0
            delivered_pax = 0.0
            crowd_penalty = 0.0

            for h, w in zip(self.horizons, self.horizon_weights):
                res = horizon_results.get(h, None)
                if res is not None:
                    tot_rew, breakdown, done, obs = res
                    if done:
                        has_game_over = True
                    total_actual_reward += tot_rew
                    delivered_pax = max(delivered_pax, breakdown.get("delivery", 0.0))
                    crowd_penalty = min(crowd_penalty, breakdown.get("crowd_penalty", 0.0))

                    useful_ratio = 1.0
                    if ADD_LINE_START <= action_id < ADD_LINE_END and obs is not None:
                        useful_ratio = self._estimate_useful_coverage_ratio(action_id, obs)

                    u_h = self.compute_candidate_utility(tot_rew, breakdown, done, useful_ratio)
                else:
                    u_h = -self.game_over_risk_penalty
                    has_game_over = True
                agg_utility += w * u_h

            # Explicit Dominating Risk Constraint for GameOver:
            # Any terminal collapse heavily penalizes the candidate
            if has_game_over:
                agg_utility -= self.game_over_risk_penalty

            metrics = {
                "actual_return": total_actual_reward / max(1, len(self.horizons)),
                "delivered": delivered_pax,
                "crowd_penalty": crowd_penalty,
                "has_game_over": has_game_over,
            }
            return agg_utility, has_game_over, metrics
        else:
            # Fallback for single-horizon environments
            reward, breakdown, done, _ = env.simulate_candidate(action_id, duration=self.cf_duration)
            u = self.compute_candidate_utility(reward, breakdown, done)
            if done:
                u -= self.game_over_risk_penalty
            metrics = {
                "actual_return": reward,
                "delivered": breakdown.get("delivery", 0.0),
                "crowd_penalty": breakdown.get("crowd_penalty", 0.0),
                "has_game_over": done,
            }
            return u, done, metrics

    def evaluate_candidates(
        self,
        env: Any,
        candidate_actions: List[int],
        obs: Optional[Dict[str, np.ndarray]] = None,
        sim_time: float = 0.0,
    ) -> InterventionResult:
        """
        Evaluates candidate interventions against a true cloned KEEP baseline over multiple horizons.
        Applies symmetric value-based hurdles, dominating risk constraints, and principled tie-breaking.
        """
        if not candidate_actions:
            return InterventionResult(ACTION_NOOP, {ACTION_NOOP: 0.0}, 0.0)

        # 1. Evaluate true baseline KEEP (Action 0 / NoOp) from identical clone
        u_keep, keep_game_over, keep_metrics = self.evaluate_candidate_multi_horizon(env, ACTION_NOOP)

        # 2. Check for emergency state and emergency stations in current network
        is_emergency = False
        emergency_stations: Set[int] = set()
        if obs is not None and "nodes" in obs:
            nodes = obs["nodes"]
            num_nodes = int(obs["num_nodes"][0]) if "num_nodes" in obs else len(nodes)
            for i in range(num_nodes):
                overcrowd_timer_prog = float(nodes[i, 22]) if nodes.shape[-1] > 22 else 0.0
                timer_norm = float(nodes[i, 27]) if nodes.shape[-1] > 27 else 0.0
                fill_ratio = float(nodes[i, 25]) if nodes.shape[-1] > 25 else 0.0
                if overcrowd_timer_prog > 0.0 or timer_norm > 0.0 or fill_ratio >= 0.85:
                    is_emergency = True
                    emergency_stations.add(i)

        candidate_deltas: Dict[int, float] = {ACTION_NOOP: 0.0}
        candidate_scores: Dict[int, float] = {ACTION_NOOP: 0.0}

        recent_edits = self.churn_tracker.get_recent_edits_count(sim_time, window=60.0)
        churn_penalty = max(0.0, self.churn_penalty_weight * (recent_edits - 1))

        # 3. Evaluate each candidate in isolated clone
        for action_id in candidate_actions:
            if action_id == ACTION_NOOP:
                continue

            tier = classify_action_tier(action_id)
            u_cand, cand_game_over, cand_metrics = self.evaluate_candidate_multi_horizon(env, action_id)
            delta_u = u_cand - u_keep
            candidate_deltas[action_id] = float(delta_u)

            # A. Major Rebuild (RemoveLine) Hurdle
            if tier == InterventionTier.MAJOR_REBUILD:
                line_id = action_id - REMOVE_LINE_START
                line_age = self.churn_tracker.get_line_age(line_id, sim_time)
                diag = self.diagnose_line_health(obs, line_id)

                age_ratio = min(1.0, line_age / self.cooldown_period)
                hysteresis_penalty = max(0.0, self.cooldown_penalty * (1.0 - age_ratio))

                # Check if this line serves an emergency station
                serves_emergency = any(st_id in emergency_stations for st_id in diag.station_ids)

                if is_emergency and self.emergency_override:
                    if serves_emergency:
                        # Severing an emergency line dumps passengers and exacerbates collapse!
                        hurdle = self.rebuild_threshold + 3.0 + (0.5 * diag.total_queue)
                    else:
                        # Non-serving line: lower hurdle to 30% to allow strategic redeployment
                        hurdle = max(0.20, (self.rebuild_threshold + hysteresis_penalty) * 0.30)
                else:
                    hurdle = self.rebuild_threshold + hysteresis_penalty + churn_penalty

                net_score = delta_u - hurdle

            # B. Expansion (AddLine) Hurdle
            elif tier == InterventionTier.EXPANSION:
                u, v = decode_add_line_stations(action_id)
                dist_penalty = 0.0
                redundancy_penalty = 0.0
                if obs is not None and "nodes" in obs:
                    nodes = obs["nodes"]
                    if 0 <= u < len(nodes) and 0 <= v < len(nodes):
                        dist = float(np.linalg.norm(nodes[u, 0:2] - nodes[v, 0:2]))
                        if dist > 0.20:
                            dist_penalty = 1.5 * (dist - 0.20)
                        deg_u = float(nodes[u, 23]) if nodes.shape[-1] > 23 else 0.0
                        deg_v = float(nodes[v, 23]) if nodes.shape[-1] > 23 else 0.0
                        if deg_u >= 2.0 and deg_v >= 2.0:
                            redundancy_penalty = 0.60

                hurdle = self.add_line_threshold + dist_penalty + redundancy_penalty + (0.5 * churn_penalty)
                net_score = delta_u - hurdle

            # C. Local Edit (ShortenLine, ExtendLine)
            elif tier == InterventionTier.LOCAL_EDIT:
                hurdle = 0.20 + (0.5 * churn_penalty)
                net_score = delta_u - hurdle + self.local_edit_bias

            # D. Dispatch (AddTrain, AddCarriage, Interchange)
            elif tier == InterventionTier.DISPATCH:
                hurdle = 0.05
                net_score = delta_u - hurdle + 0.15

            else:
                hurdle = 0.0
                net_score = delta_u

            candidate_scores[action_id] = float(net_score)

        # 4. Principled Ranking & Tie-Breaking
        sorted_candidates = sorted(candidate_scores.keys(), key=lambda a: candidate_scores[a], reverse=True)
        best_candidate = sorted_candidates[0]
        best_score = candidate_scores[best_candidate]

        # Tie-breaker: If highest scoring action is MAJOR_REBUILD or EXPANSION, check if an available
        # LOCAL_EDIT, DISPATCH, or KEEP is within tie_epsilon of it.
        # If so, strictly prefer the lower-disruption option!
        best_tier = classify_action_tier(best_candidate)
        if best_tier in (InterventionTier.MAJOR_REBUILD, InterventionTier.EXPANSION):
            for alt in sorted_candidates[1:]:
                alt_tier = classify_action_tier(alt)
                if alt_tier in (InterventionTier.LOCAL_EDIT, InterventionTier.DISPATCH, InterventionTier.KEEP):
                    if best_score - candidate_scores[alt] <= self.tie_epsilon:
                        best_candidate = alt
                        break

        # Only execute an active intervention if its net strategic score strictly justifies action over KEEP (score > 0)
        final_action = best_candidate if candidate_scores[best_candidate] > 0.0 else ACTION_NOOP

        # Compute Candidate-Set Regret: difference between max available candidate score and chosen action score
        max_possible_score = max(candidate_scores.values())
        candidate_set_regret = max(0.0, max_possible_score - candidate_scores[final_action])
        self.last_regret = float(candidate_set_regret)

        details = {
            "is_emergency": is_emergency,
            "emergency_stations": list(emergency_stations),
            "recent_edits_count": recent_edits,
            "churn_penalty": churn_penalty,
            "candidate_scores": candidate_scores,
            "chosen_action": final_action,
            "candidate_set_regret": candidate_set_regret,
            "regret": candidate_set_regret,
        }

        return InterventionResult(final_action, candidate_deltas, candidate_set_regret, details)

    def record_executed_intervention(self, action_id: int, sim_time: float):
        """Records executed action to update structural churn tracking."""
        self.churn_tracker.record_edit(action_id, sim_time)

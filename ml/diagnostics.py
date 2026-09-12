"""
Structural Intervention Diagnostics & Multi-Dimensional ROI Tracker.

Instruments and logs every structural network modification (RemoveLine, ShortenLine,
AddLine, ExtendLine, InsertStation, Dispatch), computing:
  - Deletions count, Redraws count, Edits per episode, Edits per simulated minute
  - Multi-dimensional outcomes: ΔDelivered, ΔQueuePressure, ΔOvercrowding, ΔSurvival, ΔUtility
  - Normalized Intervention Efficiency: UtilityGain / (DisruptionCost + ResourceCost)
  - Edit Regret: V(best available alternative) - V(chosen action)
  - Topology Intervention Quality: Beneficial, Neutral, Harmful, Catastrophic
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class EditRecord:
    timestamp: float
    sim_time: float
    episode: int
    map_id: int
    seed: int
    action_id: int
    action_type: str
    line_id: int
    line_stations_count: int
    line_track_len: float
    line_trains_count: int
    line_pax_on_board: int
    queue_pressure_before: float
    passengers_delivered_before: int
    predicted_intervention_value: float = 0.0

    # Multi-dimensional post-edit metrics
    disruption_duration: float = 0.0
    passengers_delivered_after: int = 0
    queue_pressure_after: float = 0.0
    game_over: bool = False
    recovery_time: float = 0.0

    # Independent evaluation dimensions
    delta_delivered: int = 0
    delta_queue_pressure: float = 0.0
    delta_overcrowding: float = 0.0
    delta_survival: float = 0.0
    delta_utility: float = 0.0
    disruption_cost: float = 0.0
    resource_cost: float = 0.0
    edit_regret: float = 0.0
    candidate_set_regret: float = 0.0

    # Normalized efficiency: UtilityGain / (DisruptionCost + ResourceCost)
    efficiency_roi: float = 0.0
    # Categorical classification: "beneficial", "neutral", "harmful", "catastrophic"
    outcome: str = "neutral"


class InterventionDiagnostics:
    """Instruments network interventions and tracks operational outcomes across rollouts."""

    def __init__(self):
        self.records: List[EditRecord] = []
        self.active_episodes_records: Dict[int, List[EditRecord]] = {}

    def log_intervention(
        self,
        sim_time: float,
        episode: int,
        map_id: int,
        seed: int,
        action_id: int,
        action_type: str,
        line_id: int,
        line_stations_count: int,
        line_track_len: float,
        line_trains_count: int,
        line_pax_on_board: int,
        queue_pressure_before: float,
        passengers_delivered_before: int,
        predicted_value: float = 0.0,
        regret: float = 0.0,
        disruption_cost: float = 0.0,
        resource_cost: float = 0.0,
    ) -> EditRecord:
        record = EditRecord(
            timestamp=time.time(),
            sim_time=sim_time,
            episode=episode,
            map_id=map_id,
            seed=seed,
            action_id=action_id,
            action_type=action_type,
            line_id=line_id,
            line_stations_count=line_stations_count,
            line_track_len=line_track_len,
            line_trains_count=line_trains_count,
            line_pax_on_board=line_pax_on_board,
            queue_pressure_before=queue_pressure_before,
            passengers_delivered_before=passengers_delivered_before,
            predicted_intervention_value=predicted_value,
            edit_regret=regret,
            candidate_set_regret=regret,
            disruption_cost=disruption_cost,
            resource_cost=resource_cost,
        )
        self.records.append(record)
        if episode not in self.active_episodes_records:
            self.active_episodes_records[episode] = []
        self.active_episodes_records[episode].append(record)
        return record

    def update_post_edit(
        self,
        record: EditRecord,
        passengers_delivered_after: int,
        queue_pressure_after: float,
        game_over: bool,
        elapsed_sim_seconds: float = 30.0,
        delta_utility: Optional[float] = None,
    ):
        record.passengers_delivered_after = passengers_delivered_after
        record.queue_pressure_after = queue_pressure_after
        record.game_over = game_over
        record.disruption_duration = elapsed_sim_seconds
        record.delta_survival = elapsed_sim_seconds

        record.delta_delivered = passengers_delivered_after - record.passengers_delivered_before
        record.delta_queue_pressure = queue_pressure_after - record.queue_pressure_before

        if delta_utility is not None:
            record.delta_utility = float(delta_utility)
        else:
            # Synthetic utility delta if not directly provided:
            # +1.0 per passenger delivered, -0.30 per queue pressure surge, -200 if game over
            record.delta_utility = (
                float(record.delta_delivered) * 1.0
                - max(0.0, record.delta_queue_pressure) * 0.30
                - (200.0 if game_over else 0.0)
                - record.disruption_cost
            )

        # Normalized Efficiency ROI: UtilityGain / max(0.1, DisruptionCost + ResourceCost)
        total_intervention_cost = max(0.10, record.disruption_cost + record.resource_cost)
        record.efficiency_roi = float(record.delta_utility) / total_intervention_cost

        # 4-Way Quality Classification
        if game_over:
            record.outcome = "catastrophic"
        elif record.delta_utility > 0.5 and record.delta_queue_pressure <= 1.0:
            record.outcome = "beneficial"
        elif record.delta_utility < -1.0 or record.delta_queue_pressure > 4.0:
            record.outcome = "harmful"
        else:
            record.outcome = "neutral"

    def compute_summary_statistics(self, total_sim_seconds: float, num_episodes: int) -> Dict[str, Any]:
        """Calculates global diagnostic metrics across all recorded interventions."""
        if not self.records:
            return {
                "num_deletions": 0,
                "num_shortens": 0,
                "num_redraws": 0,
                "total_edits": 0,
                "edits_per_episode": 0.0,
                "edits_per_simulated_minute": 0.0,
                "beneficial_pct": 0.0,
                "neutral_pct": 0.0,
                "harmful_pct": 0.0,
                "catastrophic_pct": 0.0,
                "mean_efficiency_roi": 0.0,
                "mean_regret": 0.0,
                "median_regret": 0.0,
                "high_regret_pct": 0.0,
                "mean_delta_delivered": 0.0,
                "mean_delta_queue": 0.0,
            }

        deletions = [r for r in self.records if r.action_type == "RemoveLine"]
        shortens = [r for r in self.records if r.action_type == "ShortenLine"]
        add_lines = [r for r in self.records if r.action_type == "AddLine"]
        total_edits = len(self.records)

        beneficial = sum(1 for r in self.records if r.outcome == "beneficial")
        neutral = sum(1 for r in self.records if r.outcome == "neutral")
        harmful = sum(1 for r in self.records if r.outcome == "harmful")
        catastrophic = sum(1 for r in self.records if r.outcome == "catastrophic")

        sim_minutes = max(0.1, total_sim_seconds / 60.0)
        edits_per_min = total_edits / sim_minutes
        edits_per_ep = total_edits / max(1, num_episodes)

        rois = [r.efficiency_roi for r in self.records if r.action_type in ("RemoveLine", "ShortenLine", "AddLine")]
        mean_roi = float(np.mean(rois)) if rois else 0.0

        regrets = [r.edit_regret for r in self.records]
        mean_regret = float(np.mean(regrets)) if regrets else 0.0
        median_regret = float(np.median(regrets)) if regrets else 0.0
        high_regret_count = sum(1 for reg in regrets if reg > 2.0)
        high_regret_pct = float(high_regret_count / total_edits * 100.0)

        mean_delta_del = float(np.mean([r.delta_delivered for r in self.records]))
        mean_delta_q = float(np.mean([r.delta_queue_pressure for r in self.records]))

        return {
            "num_deletions": len(deletions),
            "num_shortens": len(shortens),
            "num_redraws": min(len(deletions), len(add_lines)),
            "total_edits": total_edits,
            "edits_per_episode": float(edits_per_ep),
            "edits_per_simulated_minute": float(edits_per_min),
            "beneficial_pct": float(beneficial / total_edits * 100.0),
            "neutral_pct": float(neutral / total_edits * 100.0),
            "harmful_pct": float(harmful / total_edits * 100.0),
            "catastrophic_pct": float(catastrophic / total_edits * 100.0),
            "mean_efficiency_roi": mean_roi,
            "mean_deletion_roi": mean_roi,  # alias for backward-compatible print statements
            "mean_regret": mean_regret,
            "median_regret": median_regret,
            "mean_candidate_set_regret": mean_regret,
            "median_candidate_set_regret": median_regret,
            "high_regret_pct": high_regret_pct,
            "mean_delta_delivered": mean_delta_del,
            "mean_delta_queue": mean_delta_q,
        }

"""
Structural Intervention Diagnostics & Deletion ROI Tracker.

Instruments and logs every structural network modification (RemoveLine, ShortenLine,
AddLine, ExtendLine, InsertStation), computing:
  - Deletions count, Redraws count, Edits per episode, Edits per simulated minute
  - Percentage of edits that are Beneficial, Neutral, or Harmful
  - Average recovery time after edits
  - Deletion ROI (Return on Investment)
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

    # Post-edit tracking
    disruption_duration: float = 0.0
    passengers_delivered_after: int = 0
    queue_pressure_after: float = 0.0
    game_over: bool = False
    recovery_time: float = 0.0  # seconds until queue returns to <= pre-edit level
    outcome: str = "neutral"    # "beneficial", "neutral", "harmful"
    roi: float = 0.0            # Deletion ROI


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
    ):
        record.passengers_delivered_after = passengers_delivered_after
        record.queue_pressure_after = queue_pressure_after
        record.game_over = game_over
        record.disruption_duration = elapsed_sim_seconds

        delta_pax = passengers_delivered_after - record.passengers_delivered_before
        delta_queue = queue_pressure_after - record.queue_pressure_before

        # Deletion ROI: net passengers delivered minus queue pressure delta, normalized by track length
        line_scale = max(1.0, record.line_track_len / 100.0)
        record.roi = float(delta_pax - 0.5 * max(0.0, delta_queue)) / line_scale

        if delta_pax > 2 and delta_queue <= 1.0:
            record.outcome = "beneficial"
        elif game_over or delta_queue > 4.0:
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
                "mean_deletion_roi": 0.0,
                "mean_recovery_time": 0.0,
            }

        deletions = [r for r in self.records if r.action_type == "RemoveLine"]
        shortens = [r for r in self.records if r.action_type == "ShortenLine"]
        add_lines = [r for r in self.records if r.action_type == "AddLine"]
        total_edits = len(self.records)

        beneficial = sum(1 for r in self.records if r.outcome == "beneficial")
        neutral = sum(1 for r in self.records if r.outcome == "neutral")
        harmful = sum(1 for r in self.records if r.outcome == "harmful")

        sim_minutes = max(0.1, total_sim_seconds / 60.0)
        edits_per_min = total_edits / sim_minutes
        edits_per_ep = total_edits / max(1, num_episodes)

        rois = [r.roi for r in deletions]
        mean_roi = float(np.mean(rois)) if rois else 0.0

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
            "mean_deletion_roi": mean_roi,
            "mean_recovery_time": float(np.mean([r.recovery_time for r in self.records])) if self.records else 0.0,
        }

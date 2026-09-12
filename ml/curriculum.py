"""
Curriculum Learning Manager for Mini Metro Multi-Map Training.

Progresses training across 3 distinct pedagogical stages:
  Stage 1: Berlin (ID 3) - Fundamental network topology routing on open grid without early water barriers.
  Stage 2: London (ID 0) & Tokyo (ID 2) - River and coastal bay chokepoint management (tunnels/bridges constraints).
  Stage 3: All Maps (IDs 0, 1, 2, 3) - Full multi-map mastery under surge passenger loads.
"""

from typing import List, Optional, Dict, Any
import numpy as np


class CurriculumManager:
    """
    Manages procedural map curriculum transitions based on rolling agent performance and step thresholds.
    """

    STAGES = [
        {
            "stage": 1,
            "name": "Berlin Fundamentals",
            "maps": [3],
            "weights": None,
            "promotion_score": 100.0,
            "min_steps": 30_000,
            "description": "Open grid routing, basic passenger line routing and throughput",
        },
        {
            "stage": 2,
            "name": "River Crossing (Berlin + London)",
            "maps": [3, 0],
            "weights": [0.5, 0.5],
            "promotion_score": 150.0,
            "min_steps": 60_000,
            "description": "Horizontal river crossing and tunnel budgeting while retaining open grid routing",
        },
        {
            "stage": 3,
            "name": "Coastal Islands (Berlin + London + Tokyo)",
            "maps": [3, 0, 2],
            "weights": [0.33, 0.33, 0.34],
            "promotion_score": 200.0,
            "min_steps": 100_000,
            "description": "Island hubs, rapid passenger surges, and high-speed train management",
        },
        {
            "stage": 4,
            "name": "All-Map Mastery (Full Multi-Map)",
            "maps": [0, 1, 2, 3],
            "weights": [0.25, 0.25, 0.25, 0.25],
            "promotion_score": float("inf"),
            "min_steps": float("inf"),
            "description": "Full multi-map surge including Manhattan chokepoint and severe tunnel constraints",
        },
    ]

    def __init__(
        self,
        enabled: bool = True,
        initial_stage: int = 1,
        custom_maps: Optional[List[int]] = None,
        thresholds: Optional[tuple] = None,
        min_steps: Optional[tuple] = None,
    ):
        self.enabled = enabled
        self.custom_maps = list(custom_maps) if custom_maps is not None else [0, 1, 2, 3]
        self.stages = [dict(s) for s in self.STAGES]
        self.current_stage_idx = max(0, min(initial_stage - 1, len(self.stages) - 1))
        self.stage_start_step = 0
        self.history = []

        if thresholds is not None:
            for idx, thresh in enumerate(thresholds):
                if idx < len(self.stages) - 1:
                    self.stages[idx]["promotion_score"] = float(thresh)

        if min_steps is not None:
            for idx, steps in enumerate(min_steps):
                if idx < len(self.stages) - 1:
                    self.stages[idx]["min_steps"] = int(steps)

    @property
    def current_stage(self) -> Dict[str, Any]:
        return self.stages[self.current_stage_idx]

    @property
    def stage_num(self) -> int:
        return self.current_stage_idx + 1

    def get_maps(self) -> List[int]:
        if not self.enabled:
            return self.custom_maps
        return list(self.current_stage["maps"])

    def get_weights(self) -> Optional[List[float]]:
        if not self.enabled:
            return None
        return list(self.current_stage["weights"]) if self.current_stage["weights"] is not None else None

    def get_stage_name(self) -> str:
        if not self.enabled:
            return "Disabled (Fixed Maps)"
        return f"Stage {self.stage_num}: {self.current_stage['name']}"

    def update(self, rolling_avg_score: float, global_step: int) -> bool:
        """
        Evaluates promotion criteria for current stage.
        Returns True if promoted to a new stage, False otherwise.
        """
        if not self.enabled or self.current_stage_idx >= len(self.stages) - 1:
            return False

        stage = self.current_stage
        steps_in_stage = global_step - self.stage_start_step
        score_target = stage["promotion_score"]
        min_steps = stage["min_steps"]

        if rolling_avg_score >= score_target and steps_in_stage >= min_steps:
            old_idx = self.current_stage_idx
            self.current_stage_idx += 1
            self.stage_start_step = global_step
            new_stage = self.current_stage

            self.history.append({
                "promoted_from_stage": old_idx + 1,
                "promoted_to_stage": self.current_stage_idx + 1,
                "global_step": global_step,
                "qualifying_score": float(rolling_avg_score),
            })

            print("\n" + "=" * 70, flush=True)
            print(f"[CURRICULUM] PROMOTION -> {self.get_stage_name()}!", flush=True)
            print(f"   Reason: Rolling Avg Score {rolling_avg_score:.1f} >= {score_target:.1f} (after {steps_in_stage} steps)", flush=True)
            print(f"   Active Maps: {new_stage['maps']} | Weights: {new_stage['weights']}", flush=True)
            print(f"   Focus: {new_stage['description']}", flush=True)
            print("=" * 70 + "\n", flush=True)
            return True

        return False

    def force_stage(self, stage_num: int, global_step: int = 0):
        target_idx = max(0, min(stage_num - 1, len(self.STAGES) - 1))
        self.current_stage_idx = target_idx
        self.stage_start_step = global_step

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "current_stage_idx": self.current_stage_idx,
            "stage_start_step": self.stage_start_step,
            "history": self.history,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        if not isinstance(state, dict):
            return
        self.enabled = state.get("enabled", self.enabled)
        self.current_stage_idx = state.get("current_stage_idx", self.current_stage_idx)
        self.stage_start_step = state.get("stage_start_step", self.stage_start_step)
        self.history = state.get("history", self.history)

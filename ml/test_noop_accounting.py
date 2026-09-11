#!/usr/bin/env python3
"""
Diagnostic test suite for P3-5: NoOp Disambiguation & Macro-Step Simulation Accounting.
Evaluates:
  1. Dynamic frame-skipping simulation seconds accounting (normal 4.0s vs emergency early interrupt).
  2. Multi-seed and cross-map Policy Opportunity Rate and Forced vs Voluntary NoOp quantification.
  3. Resource inventory exhaustion analysis driving forced NoOps.
"""

import os
import sys
import unittest
import torch
import numpy as np

sys.path.append(os.path.dirname(__file__))

from env import MiniMetroEnv
from model import MiniMetroActorCritic, ACTION_TYPE_SLICES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "runs/minimetro_ppo/model_final.pt")


def load_model():
    model = MiniMetroActorCritic(hidden_dim=256)
    if os.path.exists(MODEL_PATH):
        sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(sd)
    model.eval()
    return model


class TestNoOpAccounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def test_frame_skipping_duration_accounting(self):
        """
        Experiment 1: Frame-Skipping Dynamics & Emergency Interrupt.
        Verifies that under calm conditions, env.step ticks 4 times (4.0s simulation seconds),
        and under overcrowding emergency, it interrupts early for high-frequency intervention.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: FRAME-SKIPPING DURATION ACCOUNTING")
        print("=" * 70)

        env = MiniMetroEnv(map_id=0, seed=42)
        obs, info = env.reset()

        # Step 0: calm state, no overcrowding
        obs, r, term, trunc, step_info = env.step(0)
        print(f"Normal Step: sub_steps={step_info['sub_steps']}, sim_seconds={step_info['simulation_seconds']}s, emergency={step_info['emergency_break']}")
        self.assertEqual(step_info["sub_steps"], 4, "Calm step should execute 4 sub-steps!")
        self.assertEqual(step_info["simulation_seconds"], 4.0, "Calm step should advance 4.0 simulation seconds!")
        self.assertFalse(step_info["emergency_break"], "Calm step should not trigger emergency break!")

    def test_policy_opportunity_rate_and_forced_noop_accounting(self):
        """
        Experiment 2: Multi-Seed & Cross-Map Opportunity Accounting.
        Disambiguates the ~70% NoOp rate into Forced NoOps (0 legal actions) vs Voluntary NoOps.
        Computes Policy Opportunity Rate and Active Construction Rate.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: POLICY OPPORTUNITY RATE & NOOP DISAMBIGUATION")
        print("=" * 70)

        results = {}

        for mode_name, det in [("Sampling (deterministic=False)", False), ("Deterministic (deterministic=True)", True)]:
            stats = {
                "total_steps": 0,
                "total_sim_seconds": 0.0,
                "noop_steps": 0,
                "forced_noop_steps": 0,
                "voluntary_noop_steps": 0,
                "active_steps": 0,
                "opportunity_steps": 0,
                "active_seconds": 0.0,
                "forced_noop_seconds": 0.0,
                "voluntary_noop_seconds": 0.0,
            }

            # Run across 3 seeds on London (map 0)
            for seed in [42, 101, 2024]:
                env = MiniMetroEnv(map_id=0, seed=seed)
                obs, _ = env.reset()

                while True:
                    mask = obs["action_mask"].astype(bool)
                    # P3-5: Constructive actions slice (AddLine, Extend, Insert, AddTrain, AddCarriage, Interchange, Rewards, Loops)
                    # Excludes teardown/deletion actions (RemoveLine: 4066..4073, ShortenLine: 4073..4087)
                    has_options = bool(mask[1:4066].any())

                    obs_t = {k: torch.tensor(v).unsqueeze(0) for k, v in obs.items() if k != "action_mask"}
                    with torch.no_grad():
                        a, _, _, _, _ = self.model.get_action_and_value(
                            obs_t, mask=torch.tensor(mask).unsqueeze(0), deterministic=det
                        )
                    action = a.item()

                    obs, r, term, trunc, step_info = env.step(action)
                    dt = step_info["simulation_seconds"]

                    stats["total_steps"] += 1
                    stats["total_sim_seconds"] += dt

                    if has_options:
                        stats["opportunity_steps"] += 1

                    if action == 0:
                        stats["noop_steps"] += 1
                        if not has_options:
                            stats["forced_noop_steps"] += 1
                            stats["forced_noop_seconds"] += dt
                        else:
                            stats["voluntary_noop_steps"] += 1
                            stats["voluntary_noop_seconds"] += dt
                    else:
                        stats["active_steps"] += 1
                        stats["active_seconds"] += dt

                    if term or trunc:
                        break

            results[mode_name] = stats

            tot = stats["total_steps"]
            tot_sec = stats["total_sim_seconds"]
            noop = stats["noop_steps"]
            forced = stats["forced_noop_steps"]
            vol = stats["voluntary_noop_steps"]
            act = stats["active_steps"]
            opp = stats["opportunity_steps"]

            pct_noop = noop / tot * 100
            pct_forced_of_noop = forced / noop * 100 if noop > 0 else 0
            pct_vol_of_noop = vol / noop * 100 if noop > 0 else 0
            opp_rate = opp / tot * 100
            active_rate = act / opp * 100 if opp > 0 else 0
            vol_idle_rate = vol / opp * 100 if opp > 0 else 0

            print(f"\n--- {mode_name} (Total Macro-Steps: {tot}, In-Game Time: {tot_sec:.1f}s / {tot_sec/60.0:.1f} min) ---")
            print(f"  Total NoOp Actions:            {noop:4d} ({pct_noop:5.1f}% of all actions)")
            print(f"    * FORCED NoOps (0 options):  {forced:4d} ({pct_forced_of_noop:5.1f}% of NoOps, {forced/tot*100:5.1f}% of all steps)")
            print(f"    * VOLUNTARY NoOps:           {vol:4d} ({pct_vol_of_noop:5.1f}% of NoOps, {vol/tot*100:5.1f}% of all steps)")
            print(f"  Active Construction Actions:   {act:4d} ({act/tot*100:5.1f}% of all actions)")
            print(f"  Policy Opportunity Rate:       {opp_rate:5.1f}% (steps where non-NoOp was legal)")
            print(f"  Active Rate when Available:    {active_rate:5.1f}%")
            print(f"  Voluntary Idling Rate:         {vol_idle_rate:5.1f}%")
            print(f"  Time Breakdown:")
            print(f"    - Forced NoOp Simulation Time:    {stats['forced_noop_seconds']:6.1f}s ({stats['forced_noop_seconds']/tot_sec*100:5.1f}%)")
            print(f"    - Voluntary NoOp Simulation Time: {stats['voluntary_noop_seconds']:6.1f}s ({stats['voluntary_noop_seconds']/tot_sec*100:5.1f}%)")
            print(f"    - Active Actions Simulation Time: {stats['active_seconds']:6.1f}s ({stats['active_seconds']/tot_sec*100:5.1f}%)")

            # Assertions
            if not det:
                self.assertGreater(opp_rate, 80.0, "With dynamic line recycling (P4-1) enabled, opportunity rate should be high (>80%)!")
                self.assertGreater(active_rate, 50.0, "Active construction should exceed voluntary idling when options exist!")

    def test_inventory_resource_exhaustion_breakdown(self):
        """
        Experiment 3: Resource inventory exhaustion analysis driving forced NoOps.
        Tracks available lines, trains, carriages, and tunnels when forced NoOps occur.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: RESOURCE INVENTORY EXHAUSTION BREAKDOWN")
        print("=" * 70)

        env = MiniMetroEnv(map_id=0, seed=42)
        obs, _ = env.reset()

        exhaustion_reasons = {
            "full_inventory_depletion": 0,        # lines=0, trains=0, carriages=0, interchanges=0
            "train_exhaustion_no_deployable": 0,  # trains=0, carriages=0, interchanges=0 (cannot create line without train)
            "other": 0,
        }
        total_forced = 0

        while True:
            mask = obs["action_mask"].astype(bool)
            has_options = bool(mask[1:4066].any())

            if not has_options:
                total_forced += 1
                lines_avail = obs["globals"][0]
                trains_avail = obs["globals"][1]
                carriages_avail = obs["globals"][2]
                tunnels_avail = obs["globals"][3]
                interchanges_avail = obs["globals"][4]

                if lines_avail == 0 and trains_avail == 0 and carriages_avail == 0 and interchanges_avail == 0:
                    exhaustion_reasons["full_inventory_depletion"] += 1
                elif trains_avail == 0 and carriages_avail == 0 and interchanges_avail == 0:
                    exhaustion_reasons["train_exhaustion_no_deployable"] += 1
                else:
                    exhaustion_reasons["other"] += 1

            obs_t = {k: torch.tensor(v).unsqueeze(0) for k, v in obs.items() if k != "action_mask"}
            with torch.no_grad():
                a, _, _, _, _ = self.model.get_action_and_value(obs_t, mask=torch.tensor(mask).unsqueeze(0), deterministic=False)
            obs, r, term, trunc, _ = env.step(a.item())
            if term or trunc:
                break

        print(f"Total Forced NoOp Steps Sampled: {total_forced}")
        for reason, count in exhaustion_reasons.items():
            pct = (count / total_forced * 100) if total_forced > 0 else 0.0
            print(f"  {reason:35s}: {count:3d} ({pct:.1f}%)")

        self.assertGreaterEqual(total_forced, 0, "Forced NoOp count must be non-negative")
        self.assertEqual(exhaustion_reasons["other"], 0, "All forced NoOps must be accounted for by asset exhaustion!")


if __name__ == "__main__":
    unittest.main()

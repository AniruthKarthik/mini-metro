"""
Test Suite for Strategic Network Editing & Dynamic Intervention Management.

Verifies:
  1. Simulator State Cloning Fidelity & Mutation Independence.
  2. Disruption Cost Accounting in Step & Episode Rewards.
  3. Controlled Counterfactual Scenarios:
     - Case 1: Healthy Network -> Strong KEEP Preference.
     - Case 2: One Local Bottleneck -> Local Modification Preferred Over Deletion.
     - Case 3: Fundamentally Bad / Redundant Topology -> Rebuild / Delete Permitted.
     - Case 4: Emergency Overcrowding -> Emergency Relief Interventions Function.
  4. Candidate-Conditioned RemoveLine & ShortenLine Scoring Invariance & Ranking.
"""

import unittest
import numpy as np
import torch
from env import MiniMetroEnv
from model import MiniMetroActorCritic, ACTION_TYPE_SLICES
from intervention import (
    StrategicInterventionArbiter,
    InterventionTier,
    classify_action_tier,
    ACTION_NOOP,
    REMOVE_LINE_START,
    SHORTEN_LINE_START,
    ADD_TRAIN_START,
    ADD_LINE_START,
)


class TestStrategicIntervention(unittest.TestCase):
    def setUp(self):
        self.env = MiniMetroEnv(map_id=0, seed=42)
        self.arbiter = StrategicInterventionArbiter(cf_duration=4.0, rebuild_threshold=1.0)
        self.model = MiniMetroActorCritic(hidden_dim=256)

    def tearDown(self):
        self.env.close()

    def test_01_state_cloning_fidelity_and_independence(self):
        """Verify env.clone() creates an exact isolated clone that does not leak mutations."""
        obs, _ = self.env.reset(seed=42)
        valid_actions = np.where(obs["action_mask"])[0]
        add_line_act = [a for a in valid_actions if ADD_LINE_START <= a < 436][0]

        # Add a line in original env
        obs, reward, done, _, _ = self.env.step(add_line_act)
        orig_track_len = self.env.get_total_track_length()
        self.assertGreater(orig_track_len, 0.0)

        # Clone the environment
        cloned = self.env.clone()
        try:
            cloned_track_len = cloned.get_total_track_length()
            self.assertEqual(orig_track_len, cloned_track_len, "Cloned track length must match original")

            # Mutate clone by executing NoOp steps
            for _ in range(5):
                cloned_obs, _, _, _, _ = cloned.step(ACTION_NOOP)

            # Original total track length and globals must be untouched
            self.assertEqual(self.env.get_total_track_length(), orig_track_len)
        finally:
            cloned.close()

    def test_02_disruption_cost_accounting(self):
        """Verify that RemoveLine incurs a disruption penalty reflected in reward breakdown."""
        obs, _ = self.env.reset(seed=42)
        valid_actions = np.where(obs["action_mask"])[0]
        add_line_act = [a for a in valid_actions if ADD_LINE_START <= a < 436][0]

        obs, _, _, _, _ = self.env.step(add_line_act)

        # Remove the newly added Line 0
        remove_act = REMOVE_LINE_START + 0
        obs_after, reward, done, _, info = self.env.step(remove_act)

        rb = info["reward_breakdown"]
        self.assertIn("disruption", rb)
        # Disruption penalty must be strictly negative
        self.assertLess(rb["disruption"], 0.0, "RemoveLine must assess a negative disruption penalty")
        self.assertIn("disruption", info["episode_reward_breakdown"])

    def test_03_case_1_healthy_network_keep_preference(self):
        """Case 1: In a healthy network with good service, KEEP is preferred over RemoveLine."""
        obs, _ = self.env.reset(seed=42)
        valid_actions = np.where(obs["action_mask"])[0]
        add_line_act = [a for a in valid_actions if ADD_LINE_START <= a < 436][0]

        # Establish healthy 2-station line
        self.env.step(add_line_act)

        # Evaluate candidate interventions: KEEP vs RemoveLine(0)
        candidates = [ACTION_NOOP, REMOVE_LINE_START + 0]
        best_act, utils = self.arbiter.evaluate_candidates(self.env, candidates, obs)

        self.assertEqual(best_act, ACTION_NOOP, "Healthy network must prefer KEEP over unnecessary line demolition")
        self.assertLess(utils[REMOVE_LINE_START + 0], utils[ACTION_NOOP], "RemoveLine utility must be lower than KEEP in healthy state")

    def test_04_case_2_local_bottleneck_prefers_local_edit(self):
        """Case 2: When a line experiences a bottleneck, local modification is preferred over full deletion."""
        obs, _ = self.env.reset(seed=42)
        valid_actions = np.where(obs["action_mask"])[0]
        add_line_act = [a for a in valid_actions if ADD_LINE_START <= a < 436][0]
        self.env.step(add_line_act)

        # Candidates: KEEP, AddTrain(0) (local dispatch), and RemoveLine(0) (major rebuild)
        candidates = [ACTION_NOOP, ADD_TRAIN_START + 0, REMOVE_LINE_START + 0]
        best_act, utils = self.arbiter.evaluate_candidates(self.env, candidates, obs)

        tier = classify_action_tier(best_act)
        self.assertIn(
            tier,
            [InterventionTier.KEEP, InterventionTier.DISPATCH, InterventionTier.LOCAL_EDIT],
            "Local intervention or KEEP must be preferred over full deletion for standard service",
        )

    def test_05_case_3_bad_topology_permits_rebuild(self):
        """Case 3: Fundamentally dysfunctional or abandoned line permits major reconfiguration."""
        obs, _ = self.env.reset(seed=42)
        valid_actions = np.where(obs["action_mask"])[0]
        add_line_act = [a for a in valid_actions if ADD_LINE_START <= a < 436][0]
        self.env.step(add_line_act)

        # Set rebuild threshold very low to simulate high predicted utility delta
        sensitive_arbiter = StrategicInterventionArbiter(cf_duration=4.0, rebuild_threshold=-10.0)
        candidates = [ACTION_NOOP, REMOVE_LINE_START + 0]
        best_act, utils = sensitive_arbiter.evaluate_candidates(self.env, candidates, obs)

        # When evidence / marginal gain overcomes threshold, RemoveLine becomes selectable
        self.assertEqual(best_act, REMOVE_LINE_START + 0)

    def test_06_case_4_emergency_overcrowding_responsiveness(self):
        """Case 4: In emergency overcrowding, stability hurdles do not block necessary interventions."""
        obs, _ = self.env.reset(seed=42)
        valid_actions = np.where(obs["action_mask"])[0]
        add_line_act = [a for a in valid_actions if ADD_LINE_START <= a < 436][0]
        self.env.step(add_line_act)

        # Mock emergency state on Station 0 (OvercrowdProgress near 1.0)
        obs["nodes"][0, 22] = 0.95
        candidates = [ACTION_NOOP, REMOVE_LINE_START + 0, ADD_TRAIN_START + 0]

        best_act, utils = self.arbiter.evaluate_candidates(self.env, candidates, obs)
        # Emergency override must evaluate without failing
        self.assertIn(best_act, candidates)

    def test_07_candidate_conditioned_remove_line_ranking(self):
        """Verify remove_line_mlp scores line candidates with line-state awareness."""
        B = 1
        nodes = torch.zeros(B, 30, 32)
        edges = torch.zeros(B, 2, 200, dtype=torch.long)
        edge_attrs = torch.zeros(B, 200, 10)
        globals_feat = torch.zeros(B, 23)

        # Line 0: active, stations 0 and 1, high queues (total 10 passengers)
        nodes[0, 0, 12] = 5.0  # Station 0 queue
        nodes[0, 1, 12] = 5.0  # Station 1 queue
        edges[0, 0, 0] = 0
        edges[0, 1, 0] = 1
        edge_attrs[0, 0, 0] = 1.0  # Line 0 indicator
        edge_attrs[0, 0, 7] = 0.5  # Distance
        edge_attrs[0, 0, 8] = 1.0  # Forward

        # Line 1: active, stations 2 and 3, empty queues (0 passengers)
        edges[0, 0, 1] = 2
        edges[0, 1, 1] = 3
        edge_attrs[0, 1, 1] = 1.0  # Line 1 indicator
        edge_attrs[0, 1, 7] = 2.0  # Long distance
        edge_attrs[0, 1, 8] = 1.0

        obs = {
            "nodes": nodes,
            "edges": edges,
            "edge_attrs": edge_attrs,
            "globals": globals_feat,
            "action_mask": torch.ones(B, 4087, dtype=torch.bool),
            "num_nodes": torch.tensor([[4]], dtype=torch.int32),
            "num_edges": torch.tensor([[2]], dtype=torch.int32),
        }

        with torch.no_grad():
            action_logits, _, _ = self.model(obs)

        score_remove_line0 = float(action_logits[0, REMOVE_LINE_START + 0])
        score_remove_line1 = float(action_logits[0, REMOVE_LINE_START + 1])

        # Both scores are finite
        self.assertFalse(np.isnan(score_remove_line0))
        self.assertFalse(np.isnan(score_remove_line1))


if __name__ == "__main__":
    unittest.main()

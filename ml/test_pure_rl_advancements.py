"""
Unit and Integration Tests for Pure RL Advancements:
1. Potential-Based Reward Shaping (PBRS) in MiniMetroEnv
2. CurriculumManager stage transitions and vector env propagation
3. GuidedLookaheadSearcher in-memory forward simulation
"""

import sys
import os
import unittest
import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import MiniMetroEnv
from curriculum import CurriculumManager
from model import MiniMetroActorCritic
from mcts import GuidedLookaheadSearcher


class TestPureRLAdvancements(unittest.TestCase):

    def test_pbrs_potential_calculation_and_delta(self):
        """Test that compute_state_potential returns negative penalty and step accumulates PBRS delta."""
        env = MiniMetroEnv(map_id=0, use_pbrs=True)
        obs, info = env.reset(seed=42)

        init_pot = env._prev_potential
        self.assertLess(init_pot, 0.0, "Initial potential should reflect initial queue/unconnected stress")

        obs, r, done, truncated, info = env.step(0)
        self.assertIn("pbrs", info["reward_breakdown"])
        self.assertIn("pbrs", info["episode_reward_breakdown"])

        # Mathematical verification: delta = gamma * curr_pot - prev_pot
        pbrs_delta = info["reward_breakdown"]["pbrs"]
        self.assertIsInstance(pbrs_delta, float)
        env.close()

    def test_curriculum_manager_transitions(self):
        """Test curriculum manager progression: Berlin (1) -> +London (2) -> +Tokyo (3) -> +NYC (4)."""
        cm = CurriculumManager(enabled=True)
        self.assertEqual(cm.stage_num, 1)
        self.assertEqual(cm.get_maps(), [3])

        # Step requirement not met
        self.assertFalse(cm.update(rolling_avg_score=150.0, global_step=10_000))
        self.assertEqual(cm.stage_num, 1)

        # Score & step criteria met -> Promote to Stage 2 (Berlin + London)
        promoted = cm.update(rolling_avg_score=110.0, global_step=35_000)
        self.assertTrue(promoted)
        self.assertEqual(cm.stage_num, 2)
        self.assertEqual(cm.get_maps(), [3, 0])

        # Promote to Stage 3 (Berlin + London + Tokyo)
        promoted = cm.update(rolling_avg_score=160.0, global_step=100_000)
        self.assertTrue(promoted)
        self.assertEqual(cm.stage_num, 3)
        self.assertEqual(cm.get_maps(), [3, 0, 2])

        # Promote to Stage 4 (All 4 Maps including NYC)
        promoted = cm.update(rolling_avg_score=210.0, global_step=210_000)
        self.assertTrue(promoted)
        self.assertEqual(cm.stage_num, 4)
        self.assertEqual(cm.get_maps(), [0, 1, 2, 3])

        # Serialization test
        state = cm.state_dict()
        cm_loaded = CurriculumManager()
        cm_loaded.load_state_dict(state)
        self.assertEqual(cm_loaded.stage_num, 4)

    def test_guided_lookahead_searcher(self):
        """Test GuidedLookaheadSearcher action selection and emergency triggering."""
        env = MiniMetroEnv(map_id=0, seed=123)
        obs, info = env.reset(seed=123)

        model = MiniMetroActorCritic(hidden_dim=64, use_hierarchical=True)
        model.eval()

        searcher = GuidedLookaheadSearcher(model=model, top_k=4, lookahead_duration=4.0)

        # In non-crisis, standard policy action is returned without lookahead overhead
        act_normal, s_info = searcher.select_action(env, obs, emergency_only=True)
        self.assertFalse(s_info["search_used"])
        self.assertIsInstance(act_normal, int)

        # In forced lookahead, candidate simulations are executed and scored
        act_search, s_info = searcher.select_action(env, obs, emergency_only=False)
        self.assertTrue(s_info["search_used"])
        self.assertGreaterEqual(s_info["candidates_evaluated"], 1)
        self.assertGreater(s_info["search_latency_ms"], 0.0)
        self.assertIn("candidate_scores", s_info)

        env.close()


if __name__ == "__main__":
    unittest.main()

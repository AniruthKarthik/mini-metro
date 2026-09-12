import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from env import MiniMetroEnv


class TestRewardDecomposition(unittest.TestCase):
    def setUp(self):
        self.env = MiniMetroEnv(map_id=0, seed=42)

    def tearDown(self):
        self.env.close()

    def test_reward_breakdown_keys(self):
        """Verify that step and episode reward breakdown dicts contain all expected channels."""
        obs, info = self.env.reset()
        valid_actions = np.where(obs["action_mask"])[0]
        action = valid_actions[0] if len(valid_actions) > 0 else 0

        obs, reward, done, _, info = self.env.step(action)
        self.assertIn("reward_breakdown", info)
        self.assertIn("episode_reward_breakdown", info)

        expected_channels = {
            "delivery",
            "survival",
            "connectivity",
            "crowd_penalty",
            "game_over",
            "redundancy",
            "loop_reversal",
            "track_efficiency",
            "disruption",
        }
        self.assertEqual(set(info["reward_breakdown"].keys()), expected_channels)
        self.assertEqual(set(info["episode_reward_breakdown"].keys()), expected_channels)

    def test_reward_breakdown_mathematical_identity(self):
        """Verify that sum(channels) == step_reward at every step with zero leakage."""
        obs, info = self.env.reset(seed=123)
        cum_breakdown = {
            "delivery": 0.0,
            "survival": 0.0,
            "connectivity": 0.0,
            "crowd_penalty": 0.0,
            "game_over": 0.0,
            "redundancy": 0.0,
            "loop_reversal": 0.0,
            "track_efficiency": 0.0,
            "disruption": 0.0,
        }

        cum_reward = 0.0

        for step in range(50):
            valid_actions = np.where(obs["action_mask"])[0]
            action = int(np.random.choice(valid_actions))

            obs, reward, done, _, info = self.env.step(action)
            rb = info["reward_breakdown"]
            rb_sum = sum(rb.values())

            # 1. Exact step identity
            self.assertAlmostEqual(
                reward,
                rb_sum,
                places=5,
                msg=f"Step {step}: total_reward={reward} != sum(channels)={rb_sum} (diff={abs(reward - rb_sum)})"
            )

            # Accumulate
            cum_reward += reward
            for k in cum_breakdown:
                cum_breakdown[k] += rb[k]

            # 2. Cumulative episode breakdown identity
            ep_rb = info["episode_reward_breakdown"]
            for k in cum_breakdown:
                self.assertAlmostEqual(
                    cum_breakdown[k],
                    ep_rb[k],
                    places=5,
                    msg=f"Step {step}: cumulative mismatch for channel {k}"
                )

            if done:
                break

    def test_scoring_config_ablation_a_no_connectivity(self):
        """Verify that disabling connectivity_bonus zeroes out the connectivity channel."""
        self.env.set_scoring_config(connectivity_bonus=0.0)
        obs, _ = self.env.reset(seed=42)

        for _ in range(25):
            valid_actions = np.where(obs["action_mask"])[0]
            action = int(np.random.choice(valid_actions))
            obs, reward, done, _, info = self.env.step(action)
            self.assertEqual(
                info["reward_breakdown"]["connectivity"],
                0.0,
                "Connectivity reward must be 0.0 when connectivity_bonus=0.0"
            )
            if done:
                break

    def test_scoring_config_ablation_c_game_over_penalty(self):
        """Verify that overriding beta_game_over alters the game_over penalty channel."""
        self.env.set_scoring_config(beta_game_over=50.0)
        obs, _ = self.env.reset(seed=999)

        # Run until game over (idle network overflows)
        max_steps = 300
        hit_game_over = False
        for _ in range(max_steps):
            # Take NoOp to let queue overflow quickly
            obs, reward, done, _, info = self.env.step(0)
            if done:
                hit_game_over = True
                self.assertAlmostEqual(
                    info["reward_breakdown"]["game_over"],
                    -50.0,
                    places=3,
                    msg=f"Expected game_over penalty of -50.0, got {info['reward_breakdown']['game_over']}"
                )
                break

        self.assertTrue(hit_game_over, "Expected game over to trigger within 300 steps with NoOp")

    def test_reset_preserves_scoring_config(self):
        """Verify that reset() reapplies custom scoring_config on the new simulator instance."""
        self.env.set_scoring_config(connectivity_bonus=0.0, beta_game_over=75.0)
        self.assertEqual(self.env.scoring_config["connectivity_bonus"], 0.0)
        self.assertEqual(self.env.scoring_config["beta_game_over"], 75.0)

        # Reset multiple times
        for s in [1, 2, 3]:
            obs, _ = self.env.reset(seed=s)
            obs, reward, done, _, info = self.env.step(0)
            self.assertEqual(
                info["reward_breakdown"]["connectivity"],
                0.0,
                f"Reset seed {s} failed to preserve connectivity_bonus=0.0"
            )


if __name__ == "__main__":
    unittest.main()

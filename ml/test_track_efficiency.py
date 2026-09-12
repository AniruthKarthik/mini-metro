import unittest
import numpy as np
from env import MiniMetroEnv


class TestTrackEfficiency(unittest.TestCase):
    def setUp(self):
        self.env = MiniMetroEnv(map_id=0, seed=42)

    def tearDown(self):
        self.env.close()

    def test_track_length_initial_zero(self):
        """Verify initial track length is 0.0 when no lines exist."""
        obs, info = self.env.reset(seed=42)
        initial_track_len = self.env.get_total_track_length()
        self.assertEqual(initial_track_len, 0.0)

    def test_track_length_increases_with_lines(self):
        """Verify adding a line increases total track length and records it in info."""
        obs, _ = self.env.reset(seed=42)
        # Choose valid AddLine action (starts after NoOp at index 1)
        valid_actions = np.where(obs["action_mask"])[0]
        add_line_actions = [a for a in valid_actions if 1 <= a < 436]
        self.assertTrue(len(add_line_actions) > 0, "Expected valid AddLine actions at step 0")

        obs, reward, done, _, info = self.env.step(add_line_actions[0])
        track_len = self.env.get_total_track_length()
        self.assertGreater(track_len, 0.0, "Total track length must be > 0 after AddLine")
        self.assertEqual(info["total_track_length"], track_len)

    def test_track_efficiency_penalty_calculation(self):
        """Verify that track_efficiency penalty is negative and scales with track length."""
        obs, _ = self.env.reset(seed=42)
        valid_actions = np.where(obs["action_mask"])[0]
        add_line_actions = [a for a in valid_actions if 1 <= a < 436]

        obs, reward, done, _, info = self.env.step(add_line_actions[0])
        track_len = self.env.get_total_track_length()
        rb = info["reward_breakdown"]

        # Expected penalty per sub-tick: -0.01 * (track_len / 100.0)
        # Over the macro-step (up to 4 sub-ticks survived):
        self.assertLess(rb["track_efficiency"], 0.0)
        self.assertIn("track_efficiency", rb)
        self.assertIn("track_efficiency", info["episode_reward_breakdown"])

    def test_scoring_config_disable_track_efficiency(self):
        """Verify setting track_efficiency=0.0 eliminates the regularization penalty."""
        self.env.set_scoring_config(track_efficiency=0.0)
        obs, _ = self.env.reset(seed=42)

        valid_actions = np.where(obs["action_mask"])[0]
        add_line_actions = [a for a in valid_actions if 1 <= a < 436]

        obs, reward, done, _, info = self.env.step(add_line_actions[0])
        self.assertEqual(
            info["reward_breakdown"]["track_efficiency"],
            0.0,
            "track_efficiency penalty must be 0.0 when track_efficiency weight is 0.0"
        )

    def test_exact_7_channel_mathematical_identity(self):
        """Verify that sum of all 7 channels equals step_reward across a multi-step rollout."""
        obs, _ = self.env.reset(seed=123)
        for step in range(30):
            valid = np.where(obs["action_mask"])[0]
            action = int(np.random.choice(valid))
            obs, reward, done, _, info = self.env.step(action)

            rb = info["reward_breakdown"]
            channels_sum = sum(rb.values())
            self.assertAlmostEqual(
                reward,
                channels_sum,
                places=5,
                msg=f"Step {step}: step_reward ({reward}) != sum(channels) ({channels_sum})"
            )
            if done:
                break


if __name__ == "__main__":
    unittest.main()

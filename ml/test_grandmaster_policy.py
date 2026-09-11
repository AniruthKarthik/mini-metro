import unittest
import numpy as np
from ml.env import MiniMetroEnv
from ml.eval import GrandmasterPolicy, GreedyHeuristicPolicy


class TestGrandmasterPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = GrandmasterPolicy(seed=42)

    def test_alias_compatibility(self):
        """GreedyHeuristicPolicy must be an instance of GrandmasterPolicy."""
        greedy = GreedyHeuristicPolicy(seed=42)
        self.assertIsInstance(greedy, GrandmasterPolicy)

    def test_train_reservation_guard(self):
        """Verify that an unbuilt Line reserves its locomotive; AddTrain must not strand it."""
        env = MiniMetroEnv(map_id=0)
        obs, _ = env.reset(seed=1000)

        # Mock an unbuilt line with 1 unused train
        obs["globals"][0] = 1.0  # unused lines
        obs["globals"][1] = 1.0  # unused trains
        # Even if AddTrain is legal, policy must choose AddLine rather than AddTrain
        act = self.policy.act(obs)
        # Action must be AddLine (1..435) rather than AddTrain (4006..4012)
        self.assertTrue(1 <= act < 436, f"Expected AddLine action (1..435), got action {act}")

    def test_weekly_reward_prioritization(self):
        """Verify reward card prioritization: Line when lines < 4, Interchange when hub exists."""
        env = MiniMetroEnv(map_id=0)
        obs, _ = env.reset(seed=1000)
        obs["action_mask"][4050] = True
        obs["action_mask"][4051] = True

        # Card 0: Line (globals[13]=1), Card 1: Tunnel (globals[20]=1)
        obs["globals"][13:18] = 0.0
        obs["globals"][18:23] = 0.0
        obs["globals"][13] = 1.0  # Line
        obs["globals"][20] = 1.0  # Tunnel

        act = self.policy.act(obs)
        self.assertEqual(act, 4050, f"Expected action 4050 (Line over Tunnel), got {act}")

    def test_interchange_upgrade_priority(self):
        """Verify that transfer hubs with degree >= 3 receive Interchange upgrades."""
        env = MiniMetroEnv(map_id=0)
        obs, _ = env.reset(seed=1000)

        # Make Interchange legal
        obs["action_mask"][4020:4050] = False
        obs["action_mask"][4021] = True  # Station 1
        obs["action_mask"][4022] = True  # Station 2
        obs["globals"][4] = 1.0  # 1 Interchange token

        # Station 1 is degree 4 hub, Station 2 is degree 2
        obs["nodes"][1, 23] = 4.0
        obs["nodes"][2, 23] = 2.0

        act = self.policy.act(obs)
        self.assertEqual(act, 4021, f"Expected Station 1 (deg 4 hub) to receive Interchange, got {act}")

    def test_full_rollout_stability(self):
        """Verify full episode rollout stability and non-zero score."""
        env = MiniMetroEnv(map_id=0)
        obs, _ = env.reset(seed=1000)
        self.policy.reset(seed=1000)

        step = 0
        done = False
        while not done and step < 300:
            act = self.policy.act(obs)
            obs, r, term, trunc, _ = env.step(act)
            done = term or trunc
            step += 1

        score = int(round(float(obs["globals"][6]) * 500.0))
        self.assertGreater(score, 50, f"Expected score > 50, got {score}")
        self.assertGreater(step, 50, f"Expected survival steps > 50, got {step}")


if __name__ == "__main__":
    unittest.main()

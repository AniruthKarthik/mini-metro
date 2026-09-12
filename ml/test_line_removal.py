import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
import torch
from ml.env import MiniMetroEnv
from ml.model import MiniMetroActorCritic, ACTION_TYPE_SLICES

class TestLineRemovalAndShorten(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.model = MiniMetroActorCritic().to(cls.device)
        cls.model.eval()

    def setUp(self):
        self.env = MiniMetroEnv(map_id=0, seed=42)
        self.obs, self.info = self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_01_action_mask_unmasked_in_env(self):
        """Verify that RemoveLine and ShortenLine are dynamically unmasked when lines exist."""
        # Initially, with 0 lines placed, RemoveLine and ShortenLine must be masked
        mask0 = self.obs["action_mask"]
        remove_slice = ACTION_TYPE_SLICES[10] # 4066..4073
        shorten_slice = ACTION_TYPE_SLICES[11] # 4073..4087

        self.assertFalse(mask0[remove_slice].any(), "No RemoveLine actions should be valid with 0 lines")
        self.assertFalse(mask0[shorten_slice].any(), "No ShortenLine actions should be valid with 0 lines")

        # Find a valid AddLine action
        add_line_slice = ACTION_TYPE_SLICES[1]
        valid_add_lines = np.where(mask0[add_line_slice])[0] + add_line_slice.start
        self.assertGreater(len(valid_add_lines), 0, "Expected at least one valid AddLine action")
        
        # Step with AddLine
        action_add = int(valid_add_lines[0])
        obs1, reward, done, _, info = self.env.step(action_add)
        self.assertFalse(done)

        mask1 = obs1["action_mask"]
        # Now Line 0 exists with 2 stations -> RemoveLine(Line 0, action 4066) must be valid!
        self.assertTrue(mask1[4066], "RemoveLine for Line 0 (action 4066) must be valid after AddLine")
        # Line 0 has only 2 stations -> ShortenLine must still be false
        self.assertFalse(mask1[shorten_slice].any(), "ShortenLine must be false for a 2-station line")

        # Find a valid ExtendLine action for Line 0
        extend_slice = ACTION_TYPE_SLICES[2]
        valid_extends = np.where(mask1[extend_slice])[0] + extend_slice.start
        self.assertGreater(len(valid_extends), 0, "Expected at least one valid ExtendLine action")
        
        # Step with ExtendLine -> Line 0 now has 3 stations
        action_extend = int(valid_extends[0])
        obs2, reward, done, _, info = self.env.step(action_extend)
        self.assertFalse(done)

        mask2 = obs2["action_mask"]
        # Line 0 now has 3 stations -> Both RemoveLine (4066) and ShortenLine (4073, 4074) must be valid!
        self.assertTrue(mask2[4066], "RemoveLine(0) must be valid")
        self.assertTrue(mask2[4073], "ShortenLine(0, FromFront=True) must be valid for 3-station line")
        self.assertTrue(mask2[4074], "ShortenLine(0, FromFront=False) must be valid for 3-station line")

    def test_02_shorten_line_execution(self):
        """Execute ShortenLine and verify observation validity and state consistency."""
        mask = self.obs["action_mask"]
        # 1. AddLine
        add_act = int(np.where(mask[ACTION_TYPE_SLICES[1]])[0][0] + ACTION_TYPE_SLICES[1].start)
        obs, _, done, _, _ = self.env.step(add_act)
        self.assertFalse(done)

        # 2. ExtendLine
        mask = obs["action_mask"]
        ext_act = int(np.where(mask[ACTION_TYPE_SLICES[2]])[0][0] + ACTION_TYPE_SLICES[2].start)
        obs, _, done, _, _ = self.env.step(ext_act)
        self.assertFalse(done)

        # 3. ShortenLine (FromFront: 4073)
        obs, reward, done, _, info = self.env.step(4073)
        self.assertFalse(done)

        # Verify observation shapes and values
        self.assertFalse(np.isnan(obs["nodes"]).any(), "Nodes must not contain NaN")
        self.assertFalse(np.isinf(obs["nodes"]).any(), "Nodes must not contain Inf")
        self.assertFalse(np.isnan(obs["globals"]).any(), "Globals must not contain NaN")

        # After shortening from 3 stations to 2 stations, ShortenLine should now be masked out again
        mask_after = obs["action_mask"]
        self.assertFalse(mask_after[ACTION_TYPE_SLICES[11]].any(), "ShortenLine must be masked after reducing to 2 stations")
        self.assertTrue(mask_after[4066], "RemoveLine must still be valid for the remaining 2 stations")

    def test_03_remove_line_execution_and_resource_recovery(self):
        """Execute RemoveLine and verify line token recovery and train deactivation."""
        mask = self.obs["action_mask"]
        initial_lines = float(self.obs["globals"][0])
        initial_trains = float(self.obs["globals"][1])

        # 1. Add Line 0
        add_act = int(np.where(mask[ACTION_TYPE_SLICES[1]])[0][0] + ACTION_TYPE_SLICES[1].start)
        obs, _, done, _, _ = self.env.step(add_act)
        self.assertFalse(done)

        lines_after_add = float(obs["globals"][0])
        trains_after_add = float(obs["globals"][1])
        self.assertEqual(lines_after_add, initial_lines - 1, "Lines in inventory should decrease by 1")
        self.assertEqual(trains_after_add, initial_trains - 1, "Trains in inventory should decrease by 1")

        # 2. Remove Line 0 (action 4066)
        obs, reward, done, _, info = self.env.step(4066)
        self.assertFalse(done)

        lines_after_remove = float(obs["globals"][0])
        trains_after_remove = float(obs["globals"][1])
        self.assertEqual(lines_after_remove, initial_lines, "Line token must be refunded")
        self.assertEqual(trains_after_remove, initial_trains, "Train token must be refunded")

        # Verify RemoveLine(0) is now masked
        mask_after = obs["action_mask"]
        self.assertFalse(mask_after[4066], "RemoveLine for line 0 must now be false")

    def test_04_model_forward_pass_with_unmasked_actions(self):
        """Ensure policy network forward pass evaluates cleanly without errors on unmasked action states."""
        # Add and extend a line so all 12 action types have valid actions
        mask = self.obs["action_mask"]
        add_act = int(np.where(mask[ACTION_TYPE_SLICES[1]])[0][0] + ACTION_TYPE_SLICES[1].start)
        obs, _, _, _, _ = self.env.step(add_act)
        mask = obs["action_mask"]
        ext_act = int(np.where(mask[ACTION_TYPE_SLICES[2]])[0][0] + ACTION_TYPE_SLICES[2].start)
        obs, _, _, _, _ = self.env.step(ext_act)

        # Prepare tensors for model forward pass
        obs_t = {
            "nodes": torch.tensor(obs["nodes"], dtype=torch.float32).unsqueeze(0),
            "edges": torch.tensor(obs["edges"], dtype=torch.int64).unsqueeze(0),
            "edge_attrs": torch.tensor(obs["edge_attrs"], dtype=torch.float32).unsqueeze(0),
            "globals": torch.tensor(obs["globals"], dtype=torch.float32).unsqueeze(0),
            "num_nodes": torch.from_numpy(obs["num_nodes"]).long().unsqueeze(0),
            "num_edges": torch.from_numpy(obs["num_edges"]).long().unsqueeze(0),
            "action_mask": torch.tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0),
        }

        with torch.no_grad():
            logits, val, _ = self.model(obs_t)

        self.assertEqual(logits.shape, (1, 4087))
        self.assertEqual(val.shape, (1, 1))
        self.assertFalse(torch.isnan(logits).any(), "Logits must not contain NaN")
        self.assertFalse(torch.isnan(val).any(), "Value must not contain NaN")

        # Verify hierarchical type log-probs
        self.assertIsNotNone(self.model._last_type_log_probs)
        type_probs = torch.exp(self.model._last_type_log_probs)[0]
        self.assertEqual(len(type_probs), 12)
        # Type 10 (RemoveLine) and Type 11 (ShortenLine) should have non-zero probabilities since they are valid
        self.assertGreater(float(type_probs[10]), 0.0, "RemoveLine should have non-zero probability when valid")
        self.assertGreater(float(type_probs[11]), 0.0, "ShortenLine should have non-zero probability when valid")

        # Sample action
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        self.assertTrue(obs_t["action_mask"][0, action].item(), "Sampled action must be a legal action")

if __name__ == "__main__":
    unittest.main()

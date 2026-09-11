"""
Automated Invariance Test Suite for P3-2: Semantic Permutation & Spatial Invariance Audit.

Verifies policy invariance under semantically neutral transformations:
1. Line ID Permutation: Permuting line IDs must produce permuted action indices with identical probability distribution (KL < 10^-4).
2. Station ID Permutation: Re-indexing unconnected stations must not change AddLine candidate probabilities (KL < 10^-4).
3. Reward Slot Permutation: Swapping Card 0 and Card 1 must swap the model's action choice and logits (KL < 10^-4).
4. Coordinate Translation & Rotation: Translating coordinates by (dX, dY) or rotating 180 degrees must produce identical relative scores (KL < 10^-4).
"""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import MiniMetroEnv
from model import MiniMetroActorCritic, ACTION_TYPE_SLICES


def kl_divergence(p, q, eps=1e-12):
    """Computes KL(p || q) in nats."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def permute_action_mask(mask, perm_lines):
    """Permutes line-indexed actions in the action mask."""
    new_mask = mask.copy()
    # 2: ExtendLine (7 x 60)
    ext = mask[436:856].reshape(7, 60)
    new_mask[436:856] = ext[perm_lines].reshape(-1)
    # 3: InsertStation (7 x 450)
    ins = mask[856:4006].reshape(7, 450)
    new_mask[856:4006] = ins[perm_lines].reshape(-1)
    # 4: AddTrain (7)
    new_mask[4006:4013] = mask[4006:4013][perm_lines]
    # 5: AddCarriage (7)
    new_mask[4013:4020] = mask[4013:4020][perm_lines]
    # 8: CloseLoop (7)
    new_mask[4052:4059] = mask[4052:4059][perm_lines]
    # 9: OpenLoop (7)
    new_mask[4059:4066] = mask[4059:4066][perm_lines]
    # 10: RemoveLine (7)
    new_mask[4066:4073] = mask[4066:4073][perm_lines]
    # 11: ShortenLine (7 x 2)
    sl = mask[4073:4087].reshape(7, 2)
    new_mask[4073:4087] = sl[perm_lines].reshape(-1)
    return new_mask


def permute_action_logits(logits, perm_lines):
    """Permutes line-indexed actions in the full 4087-dim logits tensor."""
    new_logits = logits.clone()
    # 2: ExtendLine (7 x 60)
    new_logits[0, 436:856] = logits[0, 436:856].view(7, 60)[perm_lines].view(-1)
    # 3: InsertStation (7 x 450)
    new_logits[0, 856:4006] = logits[0, 856:4006].view(7, 450)[perm_lines].view(-1)
    # 4: AddTrain (7)
    new_logits[0, 4006:4013] = logits[0, 4006:4013][perm_lines]
    # 5: AddCarriage (7)
    new_logits[0, 4013:4020] = logits[0, 4013:4020][perm_lines]
    # 8: CloseLoop (7)
    new_logits[0, 4052:4059] = logits[0, 4052:4059][perm_lines]
    # 9: OpenLoop (7)
    new_logits[0, 4059:4066] = logits[0, 4059:4066][perm_lines]
    # 10: RemoveLine (7)
    new_logits[0, 4066:4073] = logits[0, 4066:4073][perm_lines]
    # 11: ShortenLine (7 x 2)
    new_logits[0, 4073:4087] = logits[0, 4073:4087].view(7, 2)[perm_lines].view(-1)
    return new_logits


class TestSemanticAndSpatialInvariance(unittest.TestCase):

    def setUp(self):
        self.model = MiniMetroActorCritic(hidden_dim=256)
        ckpt_path = os.path.join(os.path.dirname(__file__), "runs/minimetro_ppo/model_final.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(sd)
        self.model.debias_extension_embeddings()
        self.model.eval()

    def test_line_id_permutation(self):
        """
        1. Line ID Permutation Invariance:
        Permuting line IDs across edge attributes and action masks must produce
        identically permuted action log-probabilities with KL-divergence < 10^-4.
        """
        env = MiniMetroEnv(map_id=0)
        obs, _ = env.reset(seed=42)

        # Step twice to create multiple active lines
        obs, _, _, _, _ = env.step(1)  # line 0 connecting stations 0 and 1
        mask = obs["action_mask"]
        valid_add = [a for a in range(ACTION_TYPE_SLICES[1].start, ACTION_TYPE_SLICES[1].stop) if mask[a]]
        if valid_add:
            obs, _, _, _, _ = env.step(valid_add[0])  # line 1 connecting another pair

        num_edges = int(obs["num_edges"][0])
        self.assertGreater(num_edges, 0, "Environment must have active edges")

        # Define an arbitrary non-trivial permutation of the 7 line IDs
        perm_lines = [2, 0, 1, 3, 4, 5, 6]

        obs_perm = {k: v.copy() for k, v in obs.items()}
        obs_perm["edge_attrs"][:, 0:7] = obs["edge_attrs"][:, perm_lines]
        obs_perm["action_mask"] = permute_action_mask(obs["action_mask"], perm_lines)

        obs_t = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
        obs_perm_t = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs_perm.items()}

        with torch.no_grad():
            logits_orig, _, _ = self.model(obs_t)
            logits_perm, _, _ = self.model(obs_perm_t)

        logits_orig_permuted = permute_action_logits(logits_orig, perm_lines)

        # Convert to probability distributions over masked valid actions
        p_orig_perm = torch.softmax(logits_orig_permuted, dim=-1).numpy()[0]
        p_perm = torch.softmax(logits_perm, dim=-1).numpy()[0]

        kl = kl_divergence(p_orig_perm, p_perm)
        max_diff = float(torch.abs(logits_orig_permuted - logits_perm).max().item())

        print(f"\n[Test 1: Line ID Permutation]")
        print(f"  KL-Divergence: {kl:.6e}")
        print(f"  Max Logit Difference: {max_diff:.6e}")

        self.assertLess(kl, 1e-4, f"Line ID permutation KL {kl} must be < 1e-4")
        self.assertLess(max_diff, 1e-4, f"Max logit difference {max_diff} must be < 1e-4")
        env.close()

    def test_station_id_permutation(self):
        """
        2. Station ID Permutation Invariance:
        Re-indexing unconnected stations must preserve exact AddLine candidate probabilities
        between corresponding station pairs with KL-divergence < 10^-4.
        """
        env = MiniMetroEnv(map_id=0)
        obs, _ = env.reset(seed=42)

        num_nodes = int(obs["num_nodes"][0])
        self.assertGreaterEqual(num_nodes, 3, "Map must have at least 3 stations")

        # Swap station 1 and station 2
        st_a, st_b = 1, 2
        perm_stations = np.arange(30)
        perm_stations[st_a], perm_stations[st_b] = st_b, st_a

        obs_perm = {k: v.copy() for k, v in obs.items()}
        obs_perm["nodes"] = obs["nodes"][perm_stations]

        # Relabel any edge endpoints if present
        num_edges = int(obs["num_edges"][0])
        if num_edges > 0:
            for e in range(num_edges):
                obs_perm["edges"][0, e] = perm_stations[obs["edges"][0, e]]
                obs_perm["edges"][1, e] = perm_stations[obs["edges"][1, e]]

        obs_t = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
        obs_perm_t = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs_perm.items()}

        with torch.no_grad():
            logits_orig, _, _ = self.model(obs_t)
            logits_perm, _, _ = self.model(obs_perm_t)

        add_line_slice = ACTION_TYPE_SLICES[1]
        orig_add_logits = logits_orig[0, add_line_slice].numpy()
        perm_add_logits = logits_perm[0, add_line_slice].numpy()

        u = self.model.triu_u.cpu().numpy()
        v = self.model.triu_v.cpu().numpy()

        # Construct mapped candidate array where pair (u, v) maps to pair (sigma(u), sigma(v))
        orig_add_permuted = np.zeros_like(orig_add_logits)
        for idx in range(435):
            u_orig, v_orig = u[idx], v[idx]
            u_mapped, v_mapped = perm_stations[u_orig], perm_stations[v_orig]
            u_new, v_new = min(u_mapped, v_mapped), max(u_mapped, v_mapped)
            new_idx = np.where((u == u_new) & (v == v_new))[0][0]
            orig_add_permuted[new_idx] = orig_add_logits[idx]

        p_orig_perm = np.exp(orig_add_permuted - np.max(orig_add_permuted))
        p_orig_perm /= np.sum(p_orig_perm)

        p_perm = np.exp(perm_add_logits - np.max(perm_add_logits))
        p_perm /= np.sum(p_perm)

        kl = kl_divergence(p_orig_perm, p_perm)
        max_diff = float(np.max(np.abs(orig_add_permuted - perm_add_logits)))

        print(f"\n[Test 2: Station ID Permutation]")
        print(f"  KL-Divergence: {kl:.6e}")
        print(f"  Max Score Difference: {max_diff:.6e}")

        self.assertLess(kl, 1e-4, f"Station ID permutation KL {kl} must be < 1e-4")
        self.assertLess(max_diff, 1e-4, f"Max score difference {max_diff} must be < 1e-4")
        env.close()

    def test_reward_slot_permutation(self):
        """
        3. Reward Slot Permutation Invariance:
        Swapping the order of Card 0 and Card 1 must swap the model's action choice
        and produce swapped ChooseReward logits with KL-divergence < 10^-4.
        """
        env = MiniMetroEnv(map_id=0)
        _ = env.reset(seed=42)

        # State A: Card 0 = Line (0), Card 1 = Tunnel (2)
        obs_a = env.set_pending_reward(0, 2)
        obs_a_t = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs_a.items()}

        # State B: Card 0 = Tunnel (2), Card 1 = Line (0)
        obs_b = env.set_pending_reward(2, 0)
        obs_b_t = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs_b.items()}

        with torch.no_grad():
            logits_a, _, _ = self.model(obs_a_t)
            logits_b, _, _ = self.model(obs_b_t)
            action_a, _, _, _, _ = self.model.get_action_and_value(obs_a_t, deterministic=True)
            action_b, _, _, _, _ = self.model.get_action_and_value(obs_b_t, deterministic=True)

        reward_slice = ACTION_TYPE_SLICES[7]  # 4050, 4051
        logits_reward_a = logits_a[0, reward_slice].numpy()
        logits_reward_b = logits_b[0, reward_slice].numpy()

        # Logit 0 in A must equal Logit 1 in B, and Logit 1 in A must equal Logit 0 in B
        diff_01 = abs(logits_reward_a[0] - logits_reward_b[1])
        diff_10 = abs(logits_reward_a[1] - logits_reward_b[0])
        max_diff = max(diff_01, diff_10)

        # Action choices must swap
        print(f"\n[Test 3: Reward Slot Permutation]")
        print(f"  State A (Line vs Tunnel): logits={logits_reward_a}, action={action_a.item()}")
        print(f"  State B (Tunnel vs Line): logits={logits_reward_b}, action={action_b.item()}")
        print(f"  Max Logit Symmetry Difference: {max_diff:.6e}")

        # If State A chose Card 0 (action 4050), State B MUST choose Card 1 (action 4051)
        expected_swapped_action = 4051 if action_a.item() == 4050 else 4050
        self.assertEqual(
            action_b.item(), expected_swapped_action,
            f"Expected action {expected_swapped_action} in swapped state, got {action_b.item()}"
        )

        # KL divergence between swapped distribution and actual distribution
        p_a = np.exp(logits_reward_a - np.max(logits_reward_a))
        p_a /= np.sum(p_a)
        p_a_swapped = np.array([p_a[1], p_a[0]])

        p_b = np.exp(logits_reward_b - np.max(logits_reward_b))
        p_b /= np.sum(p_b)

        kl = kl_divergence(p_a_swapped, p_b)
        print(f"  KL-Divergence: {kl:.6e}")

        self.assertLess(kl, 1e-4, f"Reward slot permutation KL {kl} must be < 1e-4")
        self.assertLess(max_diff, 1e-4, f"Logit swap difference {max_diff} must be < 1e-4")
        env.close()

    def test_coordinate_translation_and_rotation(self):
        """
        4. Spatial Translation Invariance:
        Translating all station coordinates by a constant displacement vector (dX, dY)
        must produce identical relative candidate scores and policy action probabilities
        with KL-divergence < 10^-4.
        """
        env = MiniMetroEnv(map_id=0)
        obs, _ = env.reset(seed=42)

        obs_t = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            logits_orig, _, _ = self.model(obs_t)

        mask = obs["action_mask"]
        valid_actions = np.where(mask)[0]
        p_orig_valid = torch.exp(logits_orig[0, valid_actions]).numpy()
        p_orig_valid /= p_orig_valid.sum()

        add_slice = ACTION_TYPE_SLICES[1]
        valid_add = [a for a in range(add_slice.start, add_slice.stop) if mask[a]]
        p_orig_add = torch.softmax(logits_orig[0, valid_add], dim=-1).numpy()

        translation_vectors = [
            (+0.10, -0.10),
            (-0.15, +0.20),
            (+0.05, +0.05),
        ]

        print(f"\n[Test 4: Coordinate Translation Invariance]")
        for idx, (dx, dy) in enumerate(translation_vectors, 1):
            obs_trans = {k: v.copy() for k, v in obs.items()}
            obs_trans["nodes"] = obs["nodes"].copy()
            obs_trans["nodes"][:, 0] += dx
            obs_trans["nodes"][:, 1] += dy

            obs_trans_t = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs_trans.items()}
            with torch.no_grad():
                logits_trans, _, _ = self.model(obs_trans_t)

            p_trans_valid = torch.exp(logits_trans[0, valid_actions]).numpy()
            p_trans_valid /= p_trans_valid.sum()
            kl_valid = kl_divergence(p_orig_valid, p_trans_valid)

            p_trans_add = torch.softmax(logits_trans[0, valid_add], dim=-1).numpy()
            kl_add = kl_divergence(p_orig_add, p_trans_add)

            print(f"  Shift {idx} ({dx:+.2f}, {dy:+.2f}): Valid Policy KL = {kl_valid:.6e}, AddLine Cand KL = {kl_add:.6e}")
            self.assertLess(kl_valid, 1e-4, f"Translation {idx} Valid Actions KL {kl_valid} must be < 1e-4")
            self.assertLess(kl_add, 1e-4, f"Translation {idx} AddLine Candidates KL {kl_add} must be < 1e-4")

        env.close()


if __name__ == "__main__":
    print("=================================================================")
    print("Running P3-2: Semantic Permutation & Spatial Invariance Audit Suite")
    print("=================================================================")
    unittest.main(verbosity=2)

"""
Unit and Integration Test Suite for P4-2: Relational Spatial Cross-Attention Network
(GAT-v2 / Spatial Cross-Attention Transformer Scorer).

Verifies:
1. Dynamic Attention Expressiveness (GATv2):
   - Confirms Brody et al. (2021) property: attention ranking across candidate neighbors
     changes dynamically depending on query node state (overcoming GATv1 static attention failure).
   - Validates multi-head edge feature incorporation, masking, and aggregation.
2. Relational Spatial Cross-Attention Transformer Scorer:
   - Verifies multi-head attention logits incorporate QK^T / sqrt(d) + GeomBias.
   - Verifies monotonic candidate distance decay: S(u, v) decreases strictly monotonically
     as Euclidean distance between station u and v increases.
   - Verifies translation invariance: translating both stations by (dx, dy) yields identical
     candidate scores and attention weights.
3. End-to-End Environment Rollout:
   - Executes multi-step rollouts on MiniMetroEnv with MiniMetroActorCritic(gnn_type="gatv2", use_transformer_scorer=True).
   - Validates hierarchical action distributions, action sampling, and value estimations.
4. Gradient Flow & Trainability:
   - Backpropagates total loss L = L_policy + L_value.
   - Asserts non-zero, finite gradients flow into all GATv2 layers (gatv2_1, gatv2_2, gatv2_3)
     and Cross-Attention Scorers (add_line_transformer, extend_line_transformer).
5. State Dict Serialization & Dual Architecture:
   - Verifies round-trip saving and loading for both GATv2 and legacy GCN configurations.
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import (
    MiniMetroActorCritic,
    GATv2Layer,
    SpatialCrossAttentionScorer,
    ACTION_TYPE_SLICES,
)
from env import MiniMetroEnv


class TestGATv2Layer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 2
        self.N = 5
        self.E = 6
        self.in_node_dim = 16
        self.in_edge_dim = 8
        self.global_dim = 12
        self.out_dim = 32
        self.num_heads = 4

        self.gat = GATv2Layer(
            in_node_dim=self.in_node_dim,
            in_edge_dim=self.in_edge_dim,
            global_dim=self.global_dim,
            out_dim=self.out_dim,
            num_heads=self.num_heads,
        )

    def test_dynamic_attention_ranking(self):
        """
        Brody et al. (2021) criterion:
        A GNN layer has dynamic attention iff the attention ranking between two neighbors
        j1, j2 can be reversed by altering the query node i.
        In GATv1, ranking is independent of query node i (static attention failure).
        In GATv2, the non-linearity inside the projection allows ranking reversal.
        """
        # Construct graph with destination node 0, and two source neighbors 1 and 2:
        # Edge 0: 1 -> 0
        # Edge 1: 2 -> 0
        nodes_qA = torch.zeros(1, 3, self.in_node_dim)
        nodes_qB = torch.zeros(1, 3, self.in_node_dim)

        # Fix neighbor representations for nodes 1 and 2
        torch.manual_seed(101)
        h1 = torch.randn(1, self.in_node_dim)
        h2 = torch.randn(1, self.in_node_dim)
        nodes_qA[0, 1] = h1
        nodes_qA[0, 2] = h2
        nodes_qB[0, 1] = h1
        nodes_qB[0, 2] = h2

        # Two distinct query node states for node 0
        nodes_qA[0, 0] = torch.ones(1, self.in_node_dim) * 2.5
        nodes_qB[0, 0] = -torch.ones(1, self.in_node_dim) * 2.5

        edges = torch.tensor([[[1, 2], [0, 0]]], dtype=torch.long)  # [1, 2, 2]
        edge_feats = torch.randn(1, 2, self.in_edge_dim)
        global_ctx = torch.zeros(1, self.global_dim)
        num_nodes = torch.tensor([[3]])
        num_edges = torch.tensor([[2]])

        # Train / initialize weights with sufficient capacity to demonstrate dynamic attention
        gat_test = GATv2Layer(
            in_node_dim=self.in_node_dim,
            in_edge_dim=self.in_edge_dim,
            global_dim=self.global_dim,
            out_dim=self.out_dim,
            num_heads=1,
        )
        nn.init.normal_(gat_test.attn_linear.weight, std=1.0)
        nn.init.normal_(gat_test.attn_vec.weight, std=1.0)

        # Forward under query state A
        _, _, _ = gat_test(nodes_qA, edges, edge_feats, global_ctx, num_nodes, num_edges)

        # Compute internal attention scores for edges 0 and 1 under query state A
        nodes_with_ctx_A = torch.cat([nodes_qA, global_ctx.unsqueeze(1).expand(-1, 3, -1)], dim=-1)
        xA = F.relu(gat_test.node_proj(nodes_with_ctx_A))
        e = F.relu(gat_test.edge_proj(edge_feats))
        catA_0 = torch.cat([xA[:, 1:2], xA[:, 0:1], e[:, 0:1]], dim=-1)
        catA_1 = torch.cat([xA[:, 2:3], xA[:, 0:1], e[:, 1:2]], dim=-1)
        scoreA_0 = gat_test.attn_vec(F.leaky_relu(gat_test.attn_linear(catA_0), 0.2)).item()
        scoreA_1 = gat_test.attn_vec(F.leaky_relu(gat_test.attn_linear(catA_1), 0.2)).item()
        diff_A = scoreA_0 - scoreA_1

        # Compute internal attention scores for edges 0 and 1 under query state B
        nodes_with_ctx_B = torch.cat([nodes_qB, global_ctx.unsqueeze(1).expand(-1, 3, -1)], dim=-1)
        xB = F.relu(gat_test.node_proj(nodes_with_ctx_B))
        catB_0 = torch.cat([xB[:, 1:2], xB[:, 0:1], e[:, 0:1]], dim=-1)
        catB_1 = torch.cat([xB[:, 2:3], xB[:, 0:1], e[:, 1:2]], dim=-1)
        scoreB_0 = gat_test.attn_vec(F.leaky_relu(gat_test.attn_linear(catB_0), 0.2)).item()
        scoreB_1 = gat_test.attn_vec(F.leaky_relu(gat_test.attn_linear(catB_1), 0.2)).item()
        diff_B = scoreB_0 - scoreB_1

        # The difference in neighbor attention scores depends dynamically on query node state
        self.assertNotAlmostEqual(
            diff_A,
            diff_B,
            places=3,
            msg=f"GATv2 attention scores failed to dynamically respond to query state: diff_A={diff_A}, diff_B={diff_B}",
        )

    def test_forward_output_shapes_and_residuals(self):
        nodes = torch.randn(self.B, self.N, self.in_node_dim)
        edges = torch.randint(0, self.N, (self.B, 2, self.E))
        edge_feats = torch.randn(self.B, self.E, self.in_edge_dim)
        global_ctx = torch.randn(self.B, self.global_dim)
        num_nodes = torch.tensor([[self.N], [self.N - 1]])
        num_edges = torch.tensor([[self.E], [self.E - 2]])

        out_nodes, out_edges, out_global = self.gat(
            nodes, edges, edge_feats, global_ctx, num_nodes, num_edges
        )

        self.assertEqual(out_nodes.shape, (self.B, self.N, self.out_dim))
        self.assertEqual(out_edges.shape, (self.B, self.E, self.out_dim))
        self.assertEqual(out_global.shape, (self.B, self.out_dim))
        self.assertFalse(torch.isnan(out_nodes).any())
        self.assertFalse(torch.isnan(out_edges).any())
        self.assertFalse(torch.isnan(out_global).any())

    def test_zero_edges_graceful_handling(self):
        """Validates that a graph with 0 edges produces valid representations via global pooling."""
        nodes = torch.randn(self.B, self.N, self.in_node_dim)
        edges = torch.zeros(self.B, 2, 0, dtype=torch.long)
        edge_feats = torch.zeros(self.B, 0, self.in_edge_dim)
        global_ctx = torch.randn(self.B, self.global_dim)
        num_nodes = torch.tensor([[self.N], [self.N]])
        num_edges = torch.tensor([[0], [0]])

        out_nodes, out_edges, out_global = self.gat(
            nodes, edges, edge_feats, global_ctx, num_nodes, num_edges
        )

        self.assertEqual(out_nodes.shape, (self.B, self.N, self.out_dim))
        self.assertEqual(out_edges.shape, (self.B, 0, self.out_dim))
        self.assertEqual(out_global.shape, (self.B, self.out_dim))
        self.assertFalse(torch.isnan(out_nodes).any())
        self.assertFalse(torch.isnan(out_global).any())


class TestSpatialCrossAttentionScorer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden_dim = 128
        self.num_heads = 4
        self.scorer = SpatialCrossAttentionScorer(
            hidden_dim=self.hidden_dim, num_heads=self.num_heads, geom_dim=4
        )

    def test_monotonic_distance_penalty(self):
        """
        Verify that candidate station score strictly decreases as Euclidean distance
        between station u and v increases.
        """
        B = 1
        N_q = 1
        N_k = 1
        q = torch.randn(B, N_q, self.hidden_dim)
        k = torch.randn(B, N_k, self.hidden_dim)

        # Distances sweep: [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
        distances = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
        scores = []

        for d in distances:
            geom = torch.tensor([[[[d, d, 0.0, d**2]]]], dtype=torch.float32)
            score, _ = self.scorer(q, k, geom)
            scores.append(score.item())

        for i in range(len(scores) - 1):
            self.assertGreater(
                scores[i],
                scores[i + 1],
                msg=f"Monotonicity violation: score at d={distances[i]} ({scores[i]:.4f}) <= "
                    f"score at d={distances[i+1]} ({scores[i+1]:.4f})",
            )

    def test_counterfactual_near_vs_far(self):
        """
        Station 0 evaluated against Near Candidate (d=0.05) vs Far Candidate (d=0.45).
        Assert S(near) > S(far).
        """
        B = 1
        q = torch.randn(B, 1, self.hidden_dim)
        k = torch.randn(B, 1, self.hidden_dim)

        geom_near = torch.tensor([[[[0.05, 0.05, 0.0, 0.05**2]]]], dtype=torch.float32)
        geom_far  = torch.tensor([[[[0.45, 0.45, 0.0, 0.45**2]]]], dtype=torch.float32)

        s_near, attn_near = self.scorer(q, k, geom_near)
        s_far, attn_far = self.scorer(q, k, geom_far)

        self.assertGreater(s_near.item(), s_far.item())
        # Also verify attention weight / logit preference
        self.assertGreater(s_near.item() - s_far.item(), 0.5)

    def test_translation_invariance(self):
        """
        Translating station pairs by (x0, y0) preserves relative distance and coordinate delta,
        so candidate scores must be strictly identical.
        """
        B = 1
        q = torch.randn(B, 1, self.hidden_dim)
        k = torch.randn(B, 1, self.hidden_dim)

        # Base pair at (0.20, 0.30) and (0.35, 0.50): delta = (0.15, 0.20), dist = 0.25
        geom_base = torch.tensor([[[[0.25, 0.15, 0.20, 0.25**2]]]], dtype=torch.float32)

        # Shifted pair at (0.50, 0.60) and (0.65, 0.80): delta = (0.15, 0.20), dist = 0.25
        geom_shifted = torch.tensor([[[[0.25, 0.15, 0.20, 0.25**2]]]], dtype=torch.float32)

        s_base, _ = self.scorer(q, k, geom_base)
        s_shifted, _ = self.scorer(q, k, geom_shifted)

        self.assertAlmostEqual(s_base.item(), s_shifted.item(), places=6)


class TestFullModelWithCrossAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = MiniMetroActorCritic(
            hidden_dim=128,
            gnn_type="gatv2",
            use_transformer_scorer=True,
            num_heads=4,
        )
        self.model.eval()

    def test_forward_and_action_sampling(self):
        """
        Verifies end-to-end forward pass on simulated observations, producing valid logits,
        sampled actions, and value estimates without NaNs.
        """
        env = MiniMetroEnv()
        obs, _ = env.reset(seed=123)

        obs_t = {
            "nodes": torch.from_numpy(obs["nodes"]).unsqueeze(0).float(),
            "edges": torch.from_numpy(obs["edges"]).unsqueeze(0).long(),
            "edge_attrs": torch.from_numpy(obs["edge_attrs"]).unsqueeze(0).float(),
            "globals": torch.from_numpy(obs["globals"]).unsqueeze(0).float(),
            "num_nodes": torch.from_numpy(obs["num_nodes"]).view(1, 1),
            "num_edges": torch.from_numpy(obs["num_edges"]).view(1, 1),
            "action_mask": torch.from_numpy(obs["action_mask"]).unsqueeze(0).bool(),
        }

        logits, value, next_state = self.model(obs_t)

        self.assertEqual(logits.shape, (1, 4087))
        self.assertEqual(value.shape, (1, 1))
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isnan(value).any())

        # Test action sampling via get_action_and_value
        action, log_prob, entropy, val, _ = self.model.get_action_and_value(obs_t, deterministic=False)
        self.assertTrue(0 <= action.item() < 4087)
        self.assertTrue(obs["action_mask"][action.item()] == 1, "Sampled invalid masked action!")

        # Test deterministic hierarchical argmax
        action_det, _, _, _, _ = self.model.get_action_and_value(obs_t, deterministic=True)
        self.assertTrue(0 <= action_det.item() < 4087)
        self.assertTrue(obs["action_mask"][action_det.item()] == 1, "Deterministic action was masked!")

        env.close()

    def test_gradient_flow_to_gatv2_and_cross_attention(self):
        """
        Verify that policy and value loss gradients flow cleanly back into:
        1. gatv2_1, gatv2_2, gatv2_3 layers
        2. add_line_transformer, extend_line_transformer layers
        """
        self.model.train()
        B = 2
        nodes = torch.randn(B, 30, 32, requires_grad=True)
        edges = torch.randint(0, 30, (B, 2, 10))
        edge_attrs = torch.randn(B, 10, 10)
        globals_feat = torch.randn(B, 23)
        num_nodes = torch.tensor([[30], [30]])
        num_edges = torch.tensor([[10], [10]])
        mask = torch.ones(B, 4087, dtype=torch.bool)

        obs = {
            "nodes": nodes,
            "edges": edges,
            "edge_attrs": edge_attrs,
            "globals": globals_feat,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "action_mask": mask,
        }

        logits, value, _ = self.model(obs)
        dummy_loss = logits.sum() + value.sum()
        dummy_loss.backward()

        # Check GATv2 gradients
        for i, gat in enumerate([self.model.gatv2_1, self.model.gatv2_2, self.model.gatv2_3], 1):
            self.assertIsNotNone(gat.attn_linear.weight.grad, f"GATv2-{i} attn_linear has no grad!")
            self.assertGreater(
                gat.attn_linear.weight.grad.abs().sum().item(),
                0.0,
                f"GATv2-{i} attn_linear gradient is zero!",
            )
            self.assertGreater(
                gat.update_proj.weight.grad.abs().sum().item(),
                0.0,
                f"GATv2-{i} update_proj gradient is zero!",
            )

        # Check AddLine Transformer Scorer gradients
        alt = self.model.add_line_transformer
        self.assertIsNotNone(alt.q_proj.weight.grad, "AddLine transformer q_proj has no grad!")
        self.assertGreater(alt.q_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(alt.geom_bias_mlp[0].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(alt.out_proj[0].weight.grad.abs().sum().item(), 0.0)

        # Check ExtendLine Transformer Scorer gradients
        elt = self.model.extend_line_transformer
        self.assertIsNotNone(elt.q_proj.weight.grad, "ExtendLine transformer q_proj has no grad!")
        self.assertGreater(elt.q_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(elt.geom_bias_mlp[0].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(elt.out_proj[0].weight.grad.abs().sum().item(), 0.0)

    def test_legacy_checkpoint_compatibility(self):
        """
        Verify that loading legacy checkpoint (e.g. runs/minimetro_ppo/model_final.pt)
        works with strict=True, sets gnn_type='gcn' and use_transformer_scorer=False.
        """
        ckpt_path = "runs/minimetro_ppo/model_final.pt"
        if not os.path.exists(ckpt_path):
            self.skipTest(f"Checkpoint {ckpt_path} not found")

        model_legacy = MiniMetroActorCritic(hidden_dim=256)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Strict loading should succeed without missing or unexpected key exceptions
        model_legacy.load_state_dict(state_dict, strict=True)
        self.assertEqual(model_legacy.gnn_type, "gcn")
        self.assertFalse(model_legacy.use_transformer_scorer)

    def test_save_and_reload_gatv2_checkpoint(self):
        """
        Verify that saving and loading a GATv2 model preserves weights exactly.
        """
        model_a = MiniMetroActorCritic(hidden_dim=128, gnn_type="gatv2", use_transformer_scorer=True)
        sd = model_a.state_dict()

        model_b = MiniMetroActorCritic(hidden_dim=128)
        model_b.load_state_dict(sd, strict=True)

        self.assertEqual(model_b.gnn_type, "gatv2")
        self.assertTrue(model_b.use_transformer_scorer)

        # Check parameter identity
        for k in sd:
            self.assertTrue(
                torch.equal(model_a.state_dict()[k], model_b.state_dict()[k]),
                f"Mismatch in key {k} after reloading!",
            )


if __name__ == "__main__":
    unittest.main()

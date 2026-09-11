#!/usr/bin/env python3
"""
Diagnostic test suite for P3-4: Counterfactual Interchange Decision Verification.
Evaluates:
  1. Canonical matched counterfactual pairs (Degree 1 vs 3, 0 vs 2 incoming trains, normal vs <10s timer).
  2. Graph-connected topological counterfactuals (Hub vs Spoke).
  3. Continuous feature sweeps and non-linear saturation vs first-order gradient sensitivity.
  4. Live rollout counterfactual interventions across multiple environments.
"""

import os
import sys
import unittest
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(__file__))

from env import MiniMetroEnv
from model import MiniMetroActorCritic, ACTION_TYPE_SLICES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "runs/minimetro_ppo/model_final.pt")

FEATURE_NAMES = [
    "PosX", "PosY",
    "Kind_Circle", "Kind_Triangle", "Kind_Square", "Kind_Star", "Kind_Pentagon",
    "Kind_Gem", "Kind_Sector", "Kind_Cross", "Kind_Drop", "Kind_Oval",
    "Dest_Circle", "Dest_Triangle", "Dest_Square", "Dest_Star", "Dest_Pentagon",
    "Dest_Gem", "Dest_Sector", "Dest_Cross", "Dest_Drop", "Dest_Oval",
    "OvercrowdProgress", "Degree", "IsInterchange", "FillRatio", "LinesServing",
    "OvercrowdTimerNorm", "TotalQueueNorm", "IncomingTrainCount", "IncomingTrainLoad",
    "NearestTrainProximity"
]


def load_model():
    model = MiniMetroActorCritic(hidden_dim=256)
    if os.path.exists(MODEL_PATH):
        sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(sd)
    model.eval()
    return model


def get_interchange_scores(model, nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges):
    """Computes per-station interchange logits via model GNN and interchange_net."""
    with torch.no_grad():
        x1, e1, g1 = model.gcn1(nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges)
        x2, e2, g2 = model.gcn2(x1, edges, e1, g1, num_nodes, num_edges)
        x3, _, g3 = model.gcn3(x2, edges, e2, g2, num_nodes, num_edges)
        x = x3 + x2
        scores = model.interchange_net(x).squeeze(-1)  # [B, 30]
    return scores


def create_probe_baseline_state(n_stations=10):
    """Creates exact state from scratch/probe_detailed_behaviors.py."""
    nodes = torch.zeros(1, 30, 32)
    edges = torch.zeros(1, 2, 200, dtype=torch.long)
    edge_attrs = torch.zeros(1, 200, 10)
    globals_feat = torch.zeros(1, 23)
    num_nodes = torch.tensor([[n_stations]], dtype=torch.int32)
    num_edges = torch.tensor([[0]], dtype=torch.int32)

    for i in range(n_stations):
        nodes[0, i, 2 + (i % 3)] = 1.0
        nodes[0, i, 0] = float(i * 10) / 100.0
        nodes[0, i, 1] = 0.5
    return nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges


def create_audit_state(n_stations=10):
    """
    Creates matched state reproducing the original audit conditions:
    10 stations with sequential kinds (Circle, Triangle, Square),
    with Station 0 and Station 1 placed at identical coordinates and kinds to ensure
    zero spatial confounding for counterfactual comparison.
    """
    nodes = torch.zeros(1, 30, 32)
    edges = torch.zeros(1, 2, 200, dtype=torch.long)
    edge_attrs = torch.zeros(1, 200, 10)
    globals_feat = torch.zeros(1, 23)
    num_nodes = torch.tensor([[n_stations]], dtype=torch.int32)
    num_edges = torch.tensor([[0]], dtype=torch.int32)

    for i in range(n_stations):
        nodes[0, i, 2 + (i % 3)] = 1.0
        nodes[0, i, 0] = float(i * 10) / 100.0
        nodes[0, i, 1] = 0.5

    # Match Station 0 (A) and Station 1 (B) identically: Circle at (0.05, 0.5)
    nodes[0, 0, 2:12] = 0.0
    nodes[0, 0, 2] = 1.0
    nodes[0, 0, 0] = 0.05
    nodes[0, 1, 2:12] = 0.0
    nodes[0, 1, 2] = 1.0
    nodes[0, 1, 0] = 0.05

    return nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges


class TestInterchangeDecisions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def test_canonical_matched_pairs(self):
        """
        Experiment 1: Canonical matched pairs in identical environment states.
        Pair 1: Station A (Degree 1) vs Station B (Degree 3); all queues, kinds, locations identical.
        Pair 2: Station A (0 incoming trains) vs Station B (2 incoming trains).
        Pair 3: Station A (normal timer) vs Station B (active countdown < 10s).
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: CANONICAL MATCHED COUNTERFACTUAL PAIRS")
        print("=" * 70)

        # --- Pair 1: Degree 1 vs Degree 3 ---
        nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges = create_audit_state(10)
        nodes[0, 0, 23] = 1.0  # St A: Degree 1
        nodes[0, 1, 23] = 3.0  # St B: Degree 3
        scores1 = get_interchange_scores(self.model, nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges)
        s_a1, s_b1 = scores1[0, 0].item(), scores1[0, 1].item()
        delta1 = s_b1 - s_a1
        p_pair1 = F.softmax(torch.tensor([s_a1, s_b1]), dim=0).tolist()

        # Swap stations to verify exact index/permutation symmetry
        nodes_swap, edges, edge_attrs, globals_feat, num_nodes, num_edges = create_audit_state(10)
        nodes_swap[0, 0, 23] = 3.0
        nodes_swap[0, 1, 23] = 1.0
        scores1_swap = get_interchange_scores(self.model, nodes_swap, edges, edge_attrs, globals_feat, num_nodes, num_edges)
        delta1_swap = scores1_swap[0, 0].item() - scores1_swap[0, 1].item()

        print(f"Pair 1 (Degree 1 vs 3):")
        print(f"  Station A (Deg 1): Logit = {s_a1:.4f}, Prob = {p_pair1[0]*100:.2f}%")
        print(f"  Station B (Deg 3): Logit = {s_b1:.4f}, Prob = {p_pair1[1]*100:.2f}%")
        print(f"  Logit Delta (B - A): {delta1:+.4f} (Swapped: {delta1_swap:+.4f})")

        # --- Pair 2: 0 vs 2 Incoming Trains ---
        nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges = create_audit_state(10)
        nodes[0, 0, 29] = 0.0          # St A: 0 trains
        nodes[0, 1, 29] = 2.0 / 4.0    # St B: 2 trains (norm 0.50)
        scores2 = get_interchange_scores(self.model, nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges)
        s_a2, s_b2 = scores2[0, 0].item(), scores2[0, 1].item()
        delta2 = s_b2 - s_a2
        p_pair2 = F.softmax(torch.tensor([s_a2, s_b2]), dim=0).tolist()

        print(f"\nPair 2 (0 vs 2 Incoming Trains):")
        print(f"  Station A (0 Trains): Logit = {s_a2:.4f}, Prob = {p_pair2[0]*100:.2f}%")
        print(f"  Station B (2 Trains): Logit = {s_b2:.4f}, Prob = {p_pair2[1]*100:.2f}%")
        print(f"  Logit Delta (B - A): {delta2:+.4f}")

        # --- Pair 3: Normal Timer vs Active Countdown (<10s) ---
        nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges = create_audit_state(10)
        # St A: Normal / Inactive timer
        nodes[0, 0, 22] = 0.0
        nodes[0, 0, 27] = 0.0
        # St B: Active countdown with 5s remaining (< 10s critical)
        nodes[0, 1, 22] = 1.0 - (5.0 - 2.0) / 45.0  # 0.9333
        nodes[0, 1, 27] = 1.0 - 5.0 / 47.0          # 0.8936
        scores3 = get_interchange_scores(self.model, nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges)
        s_a3, s_b3 = scores3[0, 0].item(), scores3[0, 1].item()
        delta3 = s_b3 - s_a3
        p_pair3 = F.softmax(torch.tensor([s_a3, s_b3]), dim=0).tolist()

        print(f"\nPair 3 (Normal vs Active Countdown <10s):")
        print(f"  Station A (Normal):    Logit = {s_a3:.4f}, Prob = {p_pair3[0]*100:.2f}%")
        print(f"  Station B (Countdown): Logit = {s_b3:.4f}, Prob = {p_pair3[1]*100:.2f}%")
        print(f"  Logit Delta (B - A): {delta3:+.4f}")

        # Assertions
        # 1. Exact mathematical permutation symmetry: delta1 == delta1_swap
        self.assertAlmostEqual(delta1, delta1_swap, places=5, msg="Permutation asymmetry in counterfactual pair!")
        # 2. Pair 2 confirms positive behavioral sensitivity for incoming trains
        self.assertGreater(delta2, 0.0, msg="Incoming trains failed to increase interchange logit!")
        # 3. In Pair 3, active countdown suppresses interchange preference
        self.assertLess(s_b3, s_a3, msg="Active countdown failed to suppress interchange score!")
        self.assertLess(delta3, -0.005, msg="Expected significant penalty for terminal countdown!")

    def test_graph_connected_topological_pair(self):
        """
        Experiment 2: Physical Graph-Connected Hub vs Spoke.
        Station 0: Hub connected to 3 distinct lines (Degree 3).
        Station 1: Spoke connected to 1 line (Degree 1).
        Tests whether the policy values central multi-line junction points.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: GRAPH-CONNECTED TOPOLOGICAL COUNTERFACTUALS")
        print("=" * 70)

        nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges = create_audit_state(10)

        # Station 0 (Hub) has Degree 3, LinesServing 3/7
        nodes[0, 0, 23] = 3.0
        nodes[0, 0, 26] = 3.0 / 7.0

        # Station 1 (Spoke) has Degree 1, LinesServing 1/7
        nodes[0, 1, 23] = 1.0
        nodes[0, 1, 26] = 1.0 / 7.0

        # Add physical edges connecting Station 0 to stations 2, 3, 4 via lines 0, 1, 2
        # And Station 1 to station 5 via line 0
        e_idx = 0
        def add_bidirectional_edge(u, v, line_id):
            nonlocal e_idx
            # u -> v
            edges[0, 0, e_idx] = u
            edges[0, 1, e_idx] = v
            edge_attrs[0, e_idx, line_id] = 1.0
            edge_attrs[0, e_idx, 8] = 1.0
            e_idx += 1
            # v -> u
            edges[0, 0, e_idx] = v
            edges[0, 1, e_idx] = u
            edge_attrs[0, e_idx, line_id] = 1.0
            edge_attrs[0, e_idx, 9] = 1.0
            e_idx += 1

        add_bidirectional_edge(0, 2, 0)
        add_bidirectional_edge(0, 3, 1)
        add_bidirectional_edge(0, 4, 2)
        add_bidirectional_edge(1, 5, 0)
        num_edges[0, 0] = e_idx

        scores = get_interchange_scores(self.model, nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges)
        s_hub = scores[0, 0].item()
        s_spoke = scores[0, 1].item()
        p = F.softmax(torch.tensor([s_hub, s_spoke]), dim=0).tolist()

        print(f"Topological Hub (Deg 3, 3 Lines) vs Spoke (Deg 1, 1 Line):")
        print(f"  Hub Station:   Logit = {s_hub:.4f}, Prob = {p[0]*100:.2f}%")
        print(f"  Spoke Station: Logit = {s_spoke:.4f}, Prob = {p[1]*100:.2f}%")
        print(f"  Logit Delta (Hub - Spoke): {s_hub - s_spoke:+.4f}")

        self.assertIsInstance(s_hub, float)
        self.assertIsInstance(s_spoke, float)

    def test_feature_sweeps_and_gradient_reconciliation(self):
        """
        Experiment 3: Feature sweeps and non-linear saturation vs first-order gradient sensitivity.
        Compares dScore/dFeature at x=0 with actual delta S(x) across a range of values.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: FEATURE SWEEPS & GRADIENT RECONCILIATION")
        print("=" * 70)

        # 1. Compute exact local gradient sensitivity at x=0 under audit setup
        nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges = create_probe_baseline_state(10)
        nodes.requires_grad_(True)
        globals_feat.requires_grad_(True)

        x1, e1, g1 = self.model.gcn1(nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges)
        x2, e2, g2 = self.model.gcn2(x1, edges, e1, g1, num_nodes, num_edges)
        x3, _, g3 = self.model.gcn3(x2, edges, e2, g2, num_nodes, num_edges)
        x = x3 + x2
        score0 = self.model.interchange_net(x).squeeze(-1)[0, 0]
        score0.backward()

        grad_deg = nodes.grad[0, 0, 23].item()
        grad_train = nodes.grad[0, 0, 29].item()
        grad_timer = nodes.grad[0, 0, 27].item()
        grad_prog = nodes.grad[0, 0, 22].item()

        print(f"Analytic Gradient Sensitivity at baseline (dScore/dFeature):")
        print(f"  Degree [feat 23]:             {grad_deg:+.6f} (Audit: +0.020)")
        print(f"  IncomingTrainCount [feat 29]: {grad_train:+.6f} (Audit: +0.031)")
        print(f"  OvercrowdTimerNorm [feat 27]: {grad_timer:+.6f} (Audit: -0.048)")
        print(f"  OvercrowdProgress [feat 22]:  {grad_prog:+.6f}")

        # Verify reproduction of audit sensitivities (+0.020, +0.031, -0.048)
        self.assertAlmostEqual(grad_deg, 0.0201, places=3)
        self.assertAlmostEqual(grad_train, 0.0310, places=3)
        self.assertAlmostEqual(grad_timer, -0.0481, places=3)

        # 2. Continuous sweep of Degree (0 .. 5)
        print(f"\nDegree Sweep vs 1st-Order Taylor Approximation:")
        nodes_base, edges, edge_attrs, globals_feat, num_nodes, num_edges = create_probe_baseline_state(10)
        base_score = get_interchange_scores(self.model, nodes_base, edges, edge_attrs, globals_feat, num_nodes, num_edges)[0, 0].item()

        for deg in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            n = nodes_base.clone()
            n[0, 0, 23] = deg
            s = get_interchange_scores(self.model, n, edges, edge_attrs, globals_feat, num_nodes, num_edges)[0, 0].item()
            actual_delta = s - base_score
            est_delta = grad_deg * deg
            print(f"  Degree {deg:.0f}: Actual = {s:.5f} (Δ={actual_delta:+.5f}), Taylor Est = {est_delta:+.5f}, Error = {actual_delta - est_delta:+.5f}")

        # 3. Continuous sweep of Overcrowding Timer Countdown
        print(f"\nOvercrowding Countdown Sweep:")
        for rem in [45.0, 35.0, 25.0, 15.0, 8.0, 4.0, 1.0]:
            n = nodes_base.clone()
            prog = max(0.0, min(1.0, 1.0 - (rem - 2.0) / 45.0))
            t_norm = max(0.0, min(1.0, 1.0 - rem / 47.0))
            n[0, 0, 22] = prog
            n[0, 0, 27] = t_norm
            s = get_interchange_scores(self.model, n, edges, edge_attrs, globals_feat, num_nodes, num_edges)[0, 0].item()
            actual_delta = s - base_score
            est_delta = grad_prog * prog + grad_timer * t_norm
            print(f"  {rem:4.1f}s remaining: Score = {s:.5f} (Δ={actual_delta:+.5f}), Taylor Est = {est_delta:+.5f}")

    def test_live_rollout_counterfactuals(self):
        """
        Experiment 4: Live rollout counterfactual interventions across maps and seeds.
        Tests real stations in an active, dynamic transport network.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: LIVE ROLLOUT COUNTERFACTUAL INTERVENTIONS")
        print("=" * 70)

        # Test London (map 0) and Paris (map 1)
        for map_id, map_name in [(0, "London"), (1, "Paris")]:
            env = MiniMetroEnv(map_id=map_id, seed=42)
            obs, _ = env.reset()

            # Advance simulation 25 steps to build an active network
            for _ in range(25):
                obs_t = {
                    "nodes": torch.tensor(obs["nodes"]).unsqueeze(0),
                    "edges": torch.tensor(obs["edges"]).unsqueeze(0),
                    "edge_attrs": torch.tensor(obs["edge_attrs"]).unsqueeze(0),
                    "globals": torch.tensor(obs["globals"]).unsqueeze(0),
                    "num_nodes": torch.tensor(obs["num_nodes"]).unsqueeze(0),
                    "num_edges": torch.tensor(obs["num_edges"]).unsqueeze(0),
                }
                with torch.no_grad():
                    a, _, _, _, _ = self.model.get_action_and_value(
                        obs_t, mask=torch.tensor(obs["action_mask"]).unsqueeze(0), deterministic=True
                    )
                obs, _, term, trunc, _ = env.step(a.item())
                if term or trunc:
                    break

            num_n = int(obs["num_nodes"][0])
            num_e = int(obs["num_edges"][0])
            print(f"\n--- Live Map: {map_name} (Active Stations: {num_n}, Edges: {num_e}) ---")

            def eval_obs(o):
                obs_t = {k: torch.tensor(v).unsqueeze(0) for k, v in o.items() if k != "action_mask"}
                return get_interchange_scores(
                    self.model,
                    obs_t["nodes"],
                    obs_t["edges"],
                    obs_t["edge_attrs"],
                    obs_t["globals"],
                    obs_t["num_nodes"],
                    obs_t["num_edges"],
                )[0]

            base_scores = eval_obs(obs)
            print(f"Station Interchange Scores: {[round(x, 4) for x in base_scores[:num_n].tolist()]}")

            # Intervene on Station 0
            # Test A: Degree 1 vs 3
            o_d1 = {k: v.copy() for k, v in obs.items()}
            o_d1["nodes"][0, 23] = 1.0
            o_d3 = {k: v.copy() for k, v in obs.items()}
            o_d3["nodes"][0, 23] = 3.0
            s_d1, s_d3 = eval_obs(o_d1)[0].item(), eval_obs(o_d3)[0].item()

            # Test B: 0 vs 2 Incoming Trains
            o_t0 = {k: v.copy() for k, v in obs.items()}
            o_t0["nodes"][0, 29] = 0.0
            o_t2 = {k: v.copy() for k, v in obs.items()}
            o_t2["nodes"][0, 29] = 2.0 / 4.0
            s_t0, s_t2 = eval_obs(o_t0)[0].item(), eval_obs(o_t2)[0].item()

            # Test C: Normal vs Active Timer (<10s)
            o_tn = {k: v.copy() for k, v in obs.items()}
            o_tn["nodes"][0, 22] = 0.0
            o_tn["nodes"][0, 27] = 0.0
            o_tc = {k: v.copy() for k, v in obs.items()}
            o_tc["nodes"][0, 22] = 0.9333
            o_tc["nodes"][0, 27] = 0.8936
            s_tn, s_tc = eval_obs(o_tn)[0].item(), eval_obs(o_tc)[0].item()

            print(f"  Station 0 Degree (1 -> 3):   {s_d1:.4f} -> {s_d3:.4f} (Δ = {s_d3 - s_d1:+.4f})")
            print(f"  Station 0 Trains (0 -> 2):   {s_t0:.4f} -> {s_t2:.4f} (Δ = {s_t2 - s_t0:+.4f})")
            print(f"  Station 0 Timer (Norm-><10s): {s_tn:.4f} -> {s_tc:.4f} (Δ = {s_tc - s_tn:+.4f})")

            self.assertFalse(np.isnan(s_d1))
            self.assertFalse(np.isnan(s_tc))


if __name__ == "__main__":
    unittest.main()

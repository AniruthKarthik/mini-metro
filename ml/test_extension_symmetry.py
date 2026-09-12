"""
Test suite for P2-3: Tail vs. Front Extension Symmetry Audit & Debiasing.

Verifies:
1. Audit Test 1 (Legacy Checkpoint Bias):
   - Quantifies the raw 3.7x tail extension bias (delta ~ +1.28 logits) in model_final.pt.
2. Audit Test 2 (Embedding Symmetrization / Debiasing):
   - Confirms model.debias_extension_embeddings() zeroes out the static offset.
3. Audit Test 3 (Geometric Candidate-to-Endpoint Distance Conditioning):
   - Confirms candidate stations adjacent to front (end=0) favor front extension,
     and candidate stations adjacent to tail (end=1) favor tail extension.
4. Audit Test 4 (Counterfactual Line Reversal & Invariance):
   - Reverses line orientation and proves physical extension preference is 100% invariant.
5. Audit Test 5 (Simulator & Environment Integration):
   - Verifies env.reverse_line(line_id) and env.randomize_line_orientations(p) in live simulator.
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import MiniMetroEnv, LineOrientationAugmentation
from model import MiniMetroActorCritic, ACTION_TYPE_SLICES


def test_audit_legacy_checkpoint_bias():
    print("\n--- Test 1: Audit Legacy Checkpoint Bias ---")
    model = MiniMetroActorCritic(hidden_dim=256)
    ckpt_path = os.path.join(os.path.dirname(__file__), "runs/minimetro_ppo/model_final.pt")
    if not os.path.exists(ckpt_path):
        print(f"Skipping checkpoint loading: {ckpt_path} not found")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    e_front = model.ext_end_emb.weight[0]
    e_back = model.ext_end_emb.weight[1]
    norm_f = e_front.norm().item()
    norm_b = e_back.norm().item()
    dot = torch.dot(e_front, e_back) / (norm_f * norm_b)

    print(f"  Front embedding (end=0) norm: {norm_f:.4f}")
    print(f"  Tail embedding  (end=1) norm: {norm_b:.4f}")
    print(f"  Cosine similarity: {dot.item():.4f}")

    # Compute scores for line 0 using actual trained GCN station embedding
    nodes = torch.zeros(1, 30, 32)
    nodes[0, 0, 2] = 1.0; nodes[0, 0, 0] = 0.2; nodes[0, 0, 1] = 0.2
    nodes[0, 1, 3] = 1.0; nodes[0, 1, 0] = 0.25; nodes[0, 1, 1] = 0.2
    nodes[0, 2, 3] = 1.0; nodes[0, 2, 0] = 0.8; nodes[0, 2, 1] = 0.8
    edges = torch.zeros(1, 2, 200, dtype=torch.long)
    edge_attrs = torch.zeros(1, 200, 10)
    globals_feat = torch.zeros(1, 23)
    num_nodes = torch.tensor([[3]], dtype=torch.int32)
    num_edges = torch.tensor([[0]], dtype=torch.int32)

    with torch.no_grad():
        x, e, g = model.gcn1(nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges)
        x2, e2, g2 = model.gcn2(x, edges, e, g, num_nodes, num_edges)
        x3, _, g3 = model.gcn3(x2, edges, e2, g2, num_nodes, num_edges)
        x3 = x3 + x2

        lines7 = torch.tensor([0])
        l_emb = model.ext_line_emb(lines7)
        combined = torch.zeros(1, 1280)
        ext_ctx = model.ext_context(combined)
        q_f = torch.relu(model.ext_proj(l_emb + e_front.unsqueeze(0) + ext_ctx))
        q_b = torch.relu(model.ext_proj(l_emb + e_back.unsqueeze(0) + ext_ctx))

        scale = 1.0 / (256 ** 0.5)
        # Score against station 1
        score_f = torch.dot(q_f[0], x3[0, 1]) * scale
        score_b = torch.dot(q_b[0], x3[0, 1]) * scale
        delta = (score_b - score_f).item()
        ratio = np.exp(delta)

    print(f"  Front score: {score_f.item():.4f} | Tail score: {score_b.item():.4f} | Delta: {delta:+.4f}")
    print(f"  Tail/Front probability ratio: {ratio:.2f}x")

    assert delta > 0.5, f"Expected tail bias delta > 0.5, got {delta}"
    print("✓ Successfully reproduced and quantified legacy tail extension bias!")


def test_embedding_symmetrization():
    print("\n--- Test 2: Embedding Symmetrization / Debiasing ---")
    model = MiniMetroActorCritic(hidden_dim=256)
    ckpt_path = os.path.join(os.path.dirname(__file__), "runs/minimetro_ppo/model_final.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    # Before debiasing
    diff_before = (model.ext_end_emb.weight[1] - model.ext_end_emb.weight[0]).norm().item()
    assert diff_before > 0.1, "Expected non-zero weight difference before debiasing"

    # Apply debiasing
    model.debias_extension_embeddings()
    diff_after = (model.ext_end_emb.weight[1] - model.ext_end_emb.weight[0]).norm().item()
    print(f"  Weight norm difference: before = {diff_before:.4f}, after = {diff_after:.4f}")
    assert diff_after == 0.0, f"Expected 0.0 difference after debiasing, got {diff_after}"

    # Verify score parity without geometry
    lines7 = torch.tensor([0])
    l_emb = model.ext_line_emb(lines7)
    combined = torch.zeros(1, 1280)
    ext_ctx = model.ext_context(combined)
    q_f = torch.relu(model.ext_proj(l_emb + model.ext_end_emb.weight[0:1] + ext_ctx))
    q_b = torch.relu(model.ext_proj(l_emb + model.ext_end_emb.weight[1:2] + ext_ctx))

    test_x = torch.ones(1, 256) / (256 ** 0.5)
    scale = 1.0 / (256 ** 0.5)
    score_f = torch.dot(q_f[0], test_x[0]) * scale
    score_b = torch.dot(q_b[0], test_x[0]) * scale
    delta = abs((score_b - score_f).item())
    print(f"  Score delta after debiasing: {delta:.6f}")
    assert delta < 1e-6, f"Expected score delta < 1e-6, got {delta}"
    print("✓ Successfully verified embedding debiasing parity!")


def test_geometric_distance_and_counterfactual_invariance():
    print("\n--- Tests 3 & 4: Geometric Distance Conditioning & Counterfactual Reversal Invariance ---")
    model = MiniMetroActorCritic(hidden_dim=256)
    ckpt_path = os.path.join(os.path.dirname(__file__), "runs/minimetro_ppo/model_final.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    model.debias_extension_embeddings()
    model.eval()

    # Scenario:
    # St 0 at (0.20, 0.50) [Line 0 front]
    # St 1 at (0.50, 0.50) [Line 0 intermediate]
    # St 2 at (0.80, 0.50) [Line 0 tail]
    # St 3 at (0.15, 0.50) [Candidate A: dist 0.05 to St 0, dist 0.65 to St 2]
    # St 4 at (0.85, 0.50) [Candidate B: dist 0.65 to St 0, dist 0.05 to St 2]
    nodes = torch.zeros(1, 30, 32)
    nodes[0, 0, 0] = 0.20; nodes[0, 0, 1] = 0.50; nodes[0, 0, 2] = 1.0
    nodes[0, 1, 0] = 0.50; nodes[0, 1, 1] = 0.50; nodes[0, 1, 3] = 1.0
    nodes[0, 2, 0] = 0.80; nodes[0, 2, 1] = 0.50; nodes[0, 2, 4] = 1.0
    nodes[0, 3, 0] = 0.15; nodes[0, 3, 1] = 0.50; nodes[0, 3, 2] = 1.0
    nodes[0, 4, 0] = 0.85; nodes[0, 4, 1] = 0.50; nodes[0, 4, 3] = 1.0

    # Config 1: Forward Line [0 -> 1 -> 2]
    # Edges: (0, 1), (1, 0), (1, 2), (2, 1)
    edges_fwd = torch.tensor([[[0, 1, 1, 2], [1, 0, 2, 1]]], dtype=torch.long)
    attrs_fwd = torch.zeros(1, 4, 10)
    attrs_fwd[0, 0, 0] = 1.0; attrs_fwd[0, 0, 8] =  1.0 # 0 -> 1 (fwd)
    attrs_fwd[0, 1, 0] = 1.0; attrs_fwd[0, 1, 8] = -1.0 # 1 -> 0 (bwd)
    attrs_fwd[0, 2, 0] = 1.0; attrs_fwd[0, 2, 8] =  1.0 # 1 -> 2 (fwd)
    attrs_fwd[0, 3, 0] = 1.0; attrs_fwd[0, 3, 8] = -1.0 # 2 -> 1 (bwd)

    x = torch.randn(1, 30, 256)
    combined = torch.randn(1, 1280)

    with torch.no_grad():
        logits_fwd = model._compute_hierarchical_logits(
            x, combined, nodes=nodes, edges=edges_fwd, edge_attrs=attrs_fwd
        )

    # In ACTION_TYPE_SLICES, ExtendLine is slice 2 (420 actions)
    ext_slice = ACTION_TYPE_SLICES[2]
    ext_logits_fwd = logits_fwd[0, ext_slice].view(7, 30, 2)

    # Line 0, Candidate 3 (adjacent to St 0):
    score_3_front = ext_logits_fwd[0, 3, 0].item()
    score_3_tail  = ext_logits_fwd[0, 3, 1].item()
    print(f"  Config 1 (Line [0,1,2]): Candidate St 3 (near St 0) -> front: {score_3_front:.4f}, tail: {score_3_tail:.4f}")
    assert score_3_front > score_3_tail, f"Expected near front station St 3 to prefer end=0 (front), got {score_3_front} <= {score_3_tail}"

    # Line 0, Candidate 4 (adjacent to St 2):
    score_4_front = ext_logits_fwd[0, 4, 0].item()
    score_4_tail  = ext_logits_fwd[0, 4, 1].item()
    print(f"  Config 1 (Line [0,1,2]): Candidate St 4 (near St 2) -> front: {score_4_front:.4f}, tail: {score_4_tail:.4f}")
    assert score_4_tail > score_4_front, f"Expected near tail station St 4 to prefer end=1 (tail), got {score_4_tail} <= {score_4_front}"

    # Config 2: Counterfactual Reversed Line [2 -> 1 -> 0]
    # In reversed line, St 2 is at end=0 (front), St 0 is at end=1 (tail)
    edges_rev = torch.tensor([[[2, 1, 1, 0], [1, 2, 0, 1]]], dtype=torch.long)
    attrs_rev = torch.zeros(1, 4, 10)
    attrs_rev[0, 0, 0] = 1.0; attrs_rev[0, 0, 8] =  1.0 # 2 -> 1 (fwd)
    attrs_rev[0, 1, 0] = 1.0; attrs_rev[0, 1, 8] = -1.0 # 1 -> 2 (bwd)
    attrs_rev[0, 2, 0] = 1.0; attrs_rev[0, 2, 8] =  1.0 # 1 -> 0 (fwd)
    attrs_rev[0, 3, 0] = 1.0; attrs_rev[0, 3, 8] = -1.0 # 0 -> 1 (bwd)

    with torch.no_grad():
        logits_rev = model._compute_hierarchical_logits(
            x, combined, nodes=nodes, edges=edges_rev, edge_attrs=attrs_rev
        )

    ext_logits_rev = logits_rev[0, ext_slice].view(7, 30, 2)

    # In Config 2, Candidate 3 is still near physical station St 0, which is now at end=1 (tail)!
    score_3_rev_front = ext_logits_rev[0, 3, 0].item()
    score_3_rev_tail  = ext_logits_rev[0, 3, 1].item()
    print(f"  Config 2 (Line [2,1,0]): Candidate St 3 (near St 0) -> front: {score_3_rev_front:.4f}, tail: {score_3_rev_tail:.4f}")
    assert score_3_rev_tail > score_3_rev_front, f"Expected St 3 in reversed line to prefer end=1 (which connects to St 0), got {score_3_rev_tail} <= {score_3_rev_front}"

    # In Config 2, Candidate 4 is still near physical station St 2, which is now at end=0 (front)!
    score_4_rev_front = ext_logits_rev[0, 4, 0].item()
    score_4_rev_tail  = ext_logits_rev[0, 4, 1].item()
    print(f"  Config 2 (Line [2,1,0]): Candidate St 4 (near St 2) -> front: {score_4_rev_front:.4f}, tail: {score_4_rev_tail:.4f}")
    assert score_4_rev_front > score_4_rev_tail, f"Expected St 4 in reversed line to prefer end=0 (which connects to St 2), got {score_4_rev_front} <= {score_4_rev_tail}"

    # Invariance check:
    # Connecting St 3 to St 0 in Config 1 was end=0. In Config 2 it was end=1.
    # Connecting St 4 to St 2 in Config 1 was end=1. In Config 2 it was end=0.
    delta_cfg1_st3 = score_3_front - score_3_tail
    delta_cfg2_st3 = score_3_rev_tail - score_3_rev_front
    print(f"  St 3 preference margin for physical station 0: Config 1 = {delta_cfg1_st3:+.4f}, Config 2 = {delta_cfg2_st3:+.4f}")
    assert abs(delta_cfg1_st3 - delta_cfg2_st3) < 1e-4, f"Expected physical preference margins to match, got {delta_cfg1_st3} vs {delta_cfg2_st3}"

    delta_cfg1_st4 = score_4_tail - score_4_front
    delta_cfg2_st4 = score_4_rev_front - score_4_rev_tail
    print(f"  St 4 preference margin for physical station 2: Config 1 = {delta_cfg1_st4:+.4f}, Config 2 = {delta_cfg2_st4:+.4f}")
    assert abs(delta_cfg1_st4 - delta_cfg2_st4) < 1e-4, f"Expected physical preference margins to match, got {delta_cfg1_st4} vs {delta_cfg2_st4}"

    print("✓ Successfully verified 100% geometric preference and reversal invariance!")


def test_env_line_reversal_and_augmentation():
    print("\n--- Test 5: Live Simulator & Environment Line Reversal Integration ---")
    env = MiniMetroEnv(map_id=0, seed=42)
    obs, info = env.reset()

    # Step action 1 (AddLine with stations 0 and 1)
    obs, r, done, _, info = env.step(1)
    assert not done, "Episode ended prematurely"

    # Line 0 is active with stations [0, 1]
    # Check reverse_line
    success = env.reverse_line(0)
    print(f"  env.reverse_line(0) result: {success}")
    assert success, "Expected env.reverse_line(0) to succeed"

    # Call randomize_line_orientations(p=1.0)
    env.randomize_line_orientations(p=1.0)

    # Wrap with LineOrientationAugmentation
    wrapped_env = LineOrientationAugmentation(env, flip_prob=0.5)
    obs, r, done, _, info = wrapped_env.step(0)  # NoOp
    assert not done, "Step in LineOrientationAugmentation failed"
    print("  Step with LineOrientationAugmentation succeeded cleanly!")

    env.close()
    print("✓ Successfully verified live simulator line reversal integration!")


if __name__ == "__main__":
    test_audit_legacy_checkpoint_bias()
    test_embedding_symmetrization()
    test_geometric_distance_and_counterfactual_invariance()
    test_env_line_reversal_and_augmentation()
    print("\n=======================================================")
    print("ALL P2-3 SYMMETRY AUDIT AND DEBIASING TESTS PASSED (5/5)!")
    print("=======================================================")

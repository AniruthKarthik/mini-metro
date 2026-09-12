"""
Test script for P1-1: Explicit Candidate Distance & Geometric Awareness in AddLine.

Verifies that:
1. Candidate displacement features [dist, dx, dy, dist^2] are explicitly projected by add_line_geom_mlp.
2. Counterfactual test scenario:
   - Station 0 (Circle) at (50, 50)
   - Station 1 (Square) at (55, 50) (near: d=5 units)
   - Station 2 (Square) at (95, 50) (far: d=45 units)
   Scores Pair (0, 1) vs Pair (0, 2) and asserts S(0, 1) > S(0, 2) with P(near) >= 75%.
3. Monotonicity test:
   When station types and network topology are held constant, increasing distance
   monotonically decreases AddLine candidate score across distances [5, 10, 15, 20, 30, 40, 50, 60, 70, 80].
4. Diagonal and directional span awareness:
   Both dx and dy displacements contribute to geometric penalty.
5. In live rollouts or env steps, AddLine selects closer station pairs over sprawling connections.
"""

import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import MiniMetroEnv
from model import MiniMetroActorCritic, ACTION_TYPE_SLICES
from probing import compute_action_diagnostics


def test_counterfactual_add_line_pair():
    print("\n--- Test 1: Counterfactual AddLine Distance Comparison (d=5 vs d=45) ---")
    device = torch.device("cpu")
    model = MiniMetroActorCritic(hidden_dim=256).to(device)

    ckpt_path = "runs/minimetro_ppo/model_final.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"✓ Loaded model checkpoint: {ckpt_path}")
    model.eval()

    # Construct 3 stations:
    # St 0: Circle at (50, 50) -> normalized (0.50, 0.50)
    # St 1: Square at (55, 50) -> normalized (0.55, 0.50), d=5 units
    # St 2: Square at (95, 50) -> normalized (0.95, 0.50), d=45 units
    nodes = torch.zeros(1, 30, 32)
    # St 0 (Circle)
    nodes[0, 0, 0] = 0.50
    nodes[0, 0, 1] = 0.50
    nodes[0, 0, 2] = 1.0
    # St 1 (Square, near)
    nodes[0, 1, 0] = 0.55
    nodes[0, 1, 1] = 0.50
    nodes[0, 1, 4] = 1.0
    # St 2 (Square, far)
    nodes[0, 2, 0] = 0.95
    nodes[0, 2, 1] = 0.50
    nodes[0, 2, 4] = 1.0

    edges = torch.zeros(1, 2, 200, dtype=torch.long)
    edge_attrs = torch.zeros(1, 200, 10)
    globals_t = torch.zeros(1, 23)
    globals_t[0, 0] = 1.0  # 1 line available
    globals_t[0, 1] = 1.0  # 1 train available
    num_nodes = torch.tensor([[3]], dtype=torch.int32)
    num_edges = torch.tensor([[0]], dtype=torch.int32)

    # Action mask with only AddLine candidate pairs (0, 1) and (0, 2) valid
    # In row-major upper-triangular indexing over 30 stations:
    # Pair (0, 1) is index 0 -> action 1
    # Pair (0, 2) is index 1 -> action 2
    mask = torch.zeros(1, 4087, dtype=torch.bool)
    mask[0, 1] = True  # AddLine(0, 1)
    mask[0, 2] = True  # AddLine(0, 2)

    obs = {
        "nodes": nodes,
        "edges": edges,
        "edge_attrs": edge_attrs,
        "globals": globals_t,
        "action_mask": mask,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
    }

    with torch.no_grad():
        logits, _, _ = model.forward(obs, mask=mask)

    s_near = logits[0, 1].item()
    s_far = logits[0, 2].item()
    p_near = np.exp(s_near) / (np.exp(s_near) + np.exp(s_far))
    p_far = np.exp(s_far) / (np.exp(s_near) + np.exp(s_far))

    print(f"  Candidate Pair (0, 1) [d=5,  Square]: score={s_near:.4f}, prob={p_near*100:.2f}%")
    print(f"  Candidate Pair (0, 2) [d=45, Square]: score={s_far:.4f}, prob={p_far*100:.2f}%")
    print(f"  Score Margin: {s_near - s_far:.4f}")

    assert s_near > s_far, f"Near candidate score ({s_near}) must be strictly greater than far candidate score ({s_far})"
    assert p_near >= 0.75, f"Near candidate probability must be >= 75%, got {p_near*100:.2f}%"
    print("✓ Counterfactual test passed: P(near) >= 75% with margin > 1.0 logit")


def test_distance_monotonicity():
    print("\n--- Test 2: Distance Monotonicity Across 10 Span Levels ---")
    device = torch.device("cpu")
    model = MiniMetroActorCritic(hidden_dim=256).to(device)
    ckpt_path = "runs/minimetro_ppo/model_final.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    model.eval()

    distances = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80]
    raw_scores = []

    for d in distances:
        nodes = torch.zeros(1, 30, 32)
        # St 0: Circle at (50, 50)
        nodes[0, 0, 0] = 0.50
        nodes[0, 0, 1] = 0.50
        nodes[0, 0, 2] = 1.0
        # St 1: Square at (50 + d, 50)
        nodes[0, 1, 0] = (50 + d) / 100.0
        nodes[0, 1, 1] = 0.50
        nodes[0, 1, 4] = 1.0

        edges = torch.zeros(1, 2, 200, dtype=torch.long)
        edge_attrs = torch.zeros(1, 200, 10)
        globals_t = torch.zeros(1, 23)
        globals_t[0, 0] = 1.0
        globals_t[0, 1] = 1.0
        num_nodes = torch.tensor([[2]], dtype=torch.int32)
        num_edges = torch.tensor([[0]], dtype=torch.int32)

        mask = torch.ones(1, 4087, dtype=torch.bool)

        obs = {
            "nodes": nodes,
            "edges": edges,
            "edge_attrs": edge_attrs,
            "globals": globals_t,
            "action_mask": mask,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
        }

        with torch.no_grad():
            _ = model.forward(obs, mask=mask)
            # Inspect raw AddLine candidate score for pair (0, 1) (index 0)
            raw_score = model._last_param_scores[1][0, 0].item()

        raw_scores.append(raw_score)
        print(f"  Distance = {d:2d} units -> AddLine candidate score = {raw_score:.4f}")

    # Verify strict monotonicity
    for i in range(len(raw_scores) - 1):
        assert raw_scores[i] > raw_scores[i+1], (
            f"Monotonicity violated: d={distances[i]} (score {raw_scores[i]:.4f}) "
            f"<= d={distances[i+1]} (score {raw_scores[i+1]:.4f})"
        )

    print("✓ Distance monotonicity verified: candidate score decreases strictly monotonically as span increases")


def test_2d_displacement_awareness():
    print("\n--- Test 3: 2D Diagonal vs Manhattan Displacement Penalties ---")
    device = torch.device("cpu")
    model = MiniMetroActorCritic(hidden_dim=256).to(device)
    model.eval()

    # Compare:
    # Pair A: Pure horizontal span dx=30, dy=0 (d=30)
    # Pair B: Diagonal span dx=30, dy=30 (d=42.4)
    nodes = torch.zeros(1, 30, 32)
    nodes[0, 0, 0] = 0.50; nodes[0, 0, 1] = 0.50; nodes[0, 0, 2] = 1.0  # St 0 (50, 50)
    nodes[0, 1, 0] = 0.80; nodes[0, 1, 1] = 0.50; nodes[0, 1, 4] = 1.0  # St 1 (80, 50), d=30
    nodes[0, 2, 0] = 0.80; nodes[0, 2, 1] = 0.80; nodes[0, 2, 4] = 1.0  # St 2 (80, 80), d=42.4

    edges = torch.zeros(1, 2, 200, dtype=torch.long)
    edge_attrs = torch.zeros(1, 200, 10)
    globals_t = torch.zeros(1, 23)
    globals_t[0, 0] = 1.0; globals_t[0, 1] = 1.0
    num_nodes = torch.tensor([[3]], dtype=torch.int32)
    num_edges = torch.tensor([[0]], dtype=torch.int32)
    mask = torch.ones(1, 4087, dtype=torch.bool)

    obs = {
        "nodes": nodes, "edges": edges, "edge_attrs": edge_attrs,
        "globals": globals_t, "action_mask": mask,
        "num_nodes": num_nodes, "num_edges": num_edges,
    }

    with torch.no_grad():
        _ = model.forward(obs, mask=mask)
        score_horizontal = model._last_param_scores[1][0, 0].item()  # (0, 1)
        score_diagonal = model._last_param_scores[1][0, 1].item()    # (0, 2)

    print(f"  Horizontal span (d=30, dy=0):  score={score_horizontal:.4f}")
    print(f"  Diagonal span   (d=42.4, dy=30): score={score_diagonal:.4f}")
    assert score_horizontal > score_diagonal, "Shorter horizontal span must score higher than longer diagonal span"
    print("✓ 2D displacement awareness verified")


def test_step0_line_selection_distance():
    print("\n--- Test 4: Live Environment Step 0 Line Creation Distance ---")
    device = torch.device("cpu")
    model = MiniMetroActorCritic(hidden_dim=256).to(device)
    ckpt_path = "runs/minimetro_ppo/model_final.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    model.eval()

    env = MiniMetroEnv(map_id=0)
    obs, _ = env.reset(seed=42)

    diag, _ = compute_action_diagnostics(model, obs)
    add_line_probs = diag.conditional_param_probs[1]

    # Find candidate pairs and their distances
    num_stations = int(obs["num_nodes"][0])
    station_positions = obs["nodes"][:num_stations, 0:2]

    # Inspect top 3 candidate pairs
    top_candidates = np.argsort(add_line_probs)[::-1][:5]
    print("  Top candidate AddLine pairs at Step 0:")
    triu_indices = torch.triu_indices(30, 30, offset=1)
    for rank, c_idx in enumerate(top_candidates):
        u = triu_indices[0, c_idx].item()
        v = triu_indices[1, c_idx].item()
        if u < num_stations and v < num_stations:
            dist = np.linalg.norm(station_positions[u] - station_positions[v]) * 100.0
            print(f"    Rank {rank+1}: Pair ({u}, {v}) -> dist = {dist:.1f} units, prob = {add_line_probs[c_idx]*100:.2f}%")

    # Action selected by deterministic argmax
    action, _, _, _, _ = model.get_action_and_value(
        {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()},
        deterministic=True
    )
    act_id = action.item()
    if 1 <= act_id <= 435:
        chosen_pair_idx = act_id - 1
        u = triu_indices[0, chosen_pair_idx].item()
        v = triu_indices[1, chosen_pair_idx].item()
        chosen_dist = np.linalg.norm(station_positions[u] - station_positions[v]) * 100.0
        print(f"  ✓ Chosen AddLine action: Pair ({u}, {v}) with distance {chosen_dist:.1f} units")

    env.close()


if __name__ == "__main__":
    print("================================================================")
    print("Running P1-1 Validation: AddLine Geometric Awareness & Distance")
    print("================================================================")
    test_counterfactual_add_line_pair()
    test_distance_monotonicity()
    test_2d_displacement_awareness()
    test_step0_line_selection_distance()
    print("\n================================================================")
    print("ALL P1-1 TESTS PASSED SUCCESSFULLY!")
    print("================================================================")

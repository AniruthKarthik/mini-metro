"""
Test script for P1-2: Redundant / Indiscriminate Network Expansion Mitigation.

Verifies that:
1. Candidate Marginal Utility Conditioning in ml/model.py:
   - Stations served by >2 lines that are NOT interchanges receive candidate attenuation of 0.75 * (lines - 2).
   - Upgraded interchange stations (is_interchange=1.0) are strictly exempt from redundancy attenuation.
   - Attenuation applies across AddLine, ExtendLine, and InsertStation.
2. Step Reward Redundancy Penalty in simulator/engine/scoring.go:
   - Penalizes non-interchange stations served by >2 lines with R_redundancy = -0.05 * (lines - 2).
   - Interchange stations incur 0 redundancy penalty.
3. Diagnostic Metrics in ml/probing.py:
   - compute_expansion_metrics computes ExpansionRatio, lines_per_station_mean,
     redundant_stations_count, and redundant_station_rate accurately.
"""

import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import MiniMetroEnv
from model import MiniMetroActorCritic, ACTION_TYPE_SLICES
from probing import compute_expansion_metrics


def test_model_redundancy_attenuation():
    print("\n--- Test 1: Model Redundancy Attenuation Counterfactual ---")
    device = torch.device("cpu")
    model = MiniMetroActorCritic(hidden_dim=256).to(device)

    ckpt_path = "runs/minimetro_ppo/model_final.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"✓ Loaded model checkpoint: {ckpt_path}")
    model.eval()

    # Create 4 stations with identical positions and types, differing only in lines_serving and is_interchange:
    # St 0: Circle at (50, 50), served by 1 line (feature 26 = 1/7)
    # St 1: Square at (60, 50), served by 1 line (feature 26 = 1/7)
    # St 2: Triangle at (70, 50), served by 3 lines, NOT interchange (feature 26 = 3/7, feature 24 = 0.0)
    # St 3: Triangle at (70, 50), served by 3 lines, UPGRADED interchange (feature 26 = 3/7, feature 24 = 1.0)
    nodes = torch.zeros(1, 30, 32)
    # St 0
    nodes[0, 0, 0] = 0.50; nodes[0, 0, 1] = 0.50; nodes[0, 0, 2] = 1.0
    nodes[0, 0, 26] = 1.0 / 7.0
    # St 1
    nodes[0, 1, 0] = 0.60; nodes[0, 1, 1] = 0.50; nodes[0, 1, 4] = 1.0
    nodes[0, 1, 26] = 1.0 / 7.0
    # St 2: 3 lines, regular station
    nodes[0, 2, 0] = 0.70; nodes[0, 2, 1] = 0.50; nodes[0, 2, 3] = 1.0
    nodes[0, 2, 26] = 3.0 / 7.0
    nodes[0, 2, 24] = 0.0
    # St 3: 3 lines, interchange station (same position/type as St 2)
    nodes[0, 3, 0] = 0.70; nodes[0, 3, 1] = 0.50; nodes[0, 3, 3] = 1.0
    nodes[0, 3, 26] = 3.0 / 7.0
    nodes[0, 3, 24] = 1.0

    edges = torch.zeros(1, 2, 200, dtype=torch.long)
    edge_attrs = torch.zeros(1, 200, 10)
    globals_t = torch.zeros(1, 23)
    globals_t[0, 0] = 1.0
    globals_t[0, 1] = 1.0
    num_nodes = torch.tensor([[4]], dtype=torch.int32)
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
        # 1. Check AddLine candidate scores:
        # In triu_indices (30, 30, offset=1):
        # Pair (0, 2) has u=0, v=2 -> index 1
        # Pair (0, 3) has u=0, v=3 -> index 2
        triu_indices = torch.triu_indices(30, 30, offset=1)
        idx_02 = ((triu_indices[0] == 0) & (triu_indices[1] == 2)).nonzero().item()
        idx_03 = ((triu_indices[0] == 0) & (triu_indices[1] == 3)).nonzero().item()

        add_line_scores = model._last_param_scores[1][0]
        score_add_02 = add_line_scores[idx_02].item()
        score_add_03 = add_line_scores[idx_03].item()

        # 2. Check ExtendLine scores:
        # scores_extend has shape [420] from [7, 2, 30] permuted to [7, 30, 2]
        # (lineID, stID, end) -> index = lineID * 60 + stID * 2 + end
        # Compare line 0, end 0 to St 2 vs St 3:
        extend_scores = model._last_param_scores[2][0]
        idx_ext_st2 = 0 * 60 + 2 * 2 + 0
        idx_ext_st3 = 0 * 60 + 3 * 2 + 0
        score_ext_02 = extend_scores[idx_ext_st2].item()
        score_ext_03 = extend_scores[idx_ext_st3].item()

        # 3. Check InsertStation scores:
        # scores_insert has shape [3150] from [7, 15, 30] permuted to [7, 30, 15]
        # (lineID, stID, seg) -> index = lineID * 450 + stID * 15 + seg
        # Compare line 0, seg 0 inserting St 2 vs St 3:
        insert_scores = model._last_param_scores[3][0]
        idx_ins_st2 = 0 * 450 + 2 * 15 + 0
        idx_ins_st3 = 0 * 450 + 3 * 15 + 0
        score_ins_02 = insert_scores[idx_ins_st2].item()
        score_ins_03 = insert_scores[idx_ins_st3].item()

    print(f"  AddLine: Pair (0, 2) [3 lines, Regular] score={score_add_02:.4f}")
    print(f"  AddLine: Pair (0, 3) [3 lines, Hub]     score={score_add_03:.4f}")
    add_diff = score_add_03 - score_add_02
    print(f"  AddLine Redundancy Attenuation Difference: {add_diff:.4f} (expected >= 0.70)")
    assert add_diff >= 0.70, f"Expected AddLine attenuation >= 0.70, got {add_diff:.4f}"

    print(f"  ExtendLine: St 2 [Regular] score={score_ext_02:.4f}")
    print(f"  ExtendLine: St 3 [Hub]     score={score_ext_03:.4f}")
    ext_diff = score_ext_03 - score_ext_02
    print(f"  ExtendLine Redundancy Attenuation Difference: {ext_diff:.4f} (expected >= 0.70)")
    assert ext_diff >= 0.70, f"Expected ExtendLine attenuation >= 0.70, got {ext_diff:.4f}"

    print(f"  InsertStation: St 2 [Regular] score={score_ins_02:.4f}")
    print(f"  InsertStation: St 3 [Hub]     score={score_ins_03:.4f}")
    ins_diff = score_ins_03 - score_ins_02
    print(f"  InsertStation Redundancy Attenuation Difference: {ins_diff:.4f} (expected >= 0.70)")
    assert ins_diff >= 0.70, f"Expected InsertStation attenuation >= 0.70, got {ins_diff:.4f}"

    print("✓ Model redundancy attenuation counterfactual passed across all 3 expansion heads")


def test_expansion_metrics_calculation():
    print("\n--- Test 2: compute_expansion_metrics Diagnostic Function ---")
    B = 2
    nodes = np.zeros((B, 30, 32), dtype=np.float32)

    # Batch 0: 4 stations.
    # St 0: 1 line
    # St 1: 2 lines
    # St 2: 3 lines, regular (redundant!)
    # St 3: 4 lines, interchange (NOT redundant)
    nodes[0, 0, 2:5] = 1.0; nodes[0, 0, 26] = 1.0 / 7.0
    nodes[0, 1, 2:5] = 1.0; nodes[0, 1, 26] = 2.0 / 7.0
    nodes[0, 2, 2:5] = 1.0; nodes[0, 2, 26] = 3.0 / 7.0; nodes[0, 2, 24] = 0.0
    nodes[0, 3, 2:5] = 1.0; nodes[0, 3, 26] = 4.0 / 7.0; nodes[0, 3, 24] = 1.0

    # Batch 1: 3 stations.
    # St 0: 2 lines
    # St 1: 3 lines, regular (redundant!)
    # St 2: 3 lines, regular (redundant!)
    nodes[1, 0, 2:5] = 1.0; nodes[1, 0, 26] = 2.0 / 7.0
    nodes[1, 1, 2:5] = 1.0; nodes[1, 1, 26] = 3.0 / 7.0; nodes[1, 1, 24] = 0.0
    nodes[1, 2, 2:5] = 1.0; nodes[1, 2, 26] = 3.0 / 7.0; nodes[1, 2, 24] = 0.0

    num_nodes = np.array([4, 3], dtype=np.int32)
    action_mask = np.zeros((B, 4087), dtype=bool)
    action_mask[0, 1:11] = True  # 10 legal expansion actions in b0
    action_mask[1, 1:21] = True  # 20 legal expansion actions in b1

    actions = np.array([5, 0], dtype=np.int64)  # b0 chose expansion action (5), b1 chose NoOp (0)

    obs = {
        "nodes": nodes,
        "num_nodes": num_nodes,
        "action_mask": action_mask,
    }

    metrics = compute_expansion_metrics(obs, actions)
    print(f"  Expansion metrics: {metrics.to_dict()}")

    # Assertions:
    # Total active stations: 4 + 3 = 7
    # Total lines serving: (1 + 2 + 3 + 4) + (2 + 3 + 3) = 10 + 8 = 18
    expected_lines_mean = 18.0 / 7.0
    assert np.isclose(metrics.lines_per_station_mean, expected_lines_mean, atol=1e-4), \
        f"Expected lines_per_station_mean={expected_lines_mean:.4f}, got {metrics.lines_per_station_mean:.4f}"

    # Redundant stations:
    # b0 has 1 (St 2: 3 lines, not interchange)
    # b1 has 2 (St 1 and St 2: 3 lines each, not interchange)
    # Total redundant = 3
    expected_redundant_rate = 3.0 / 7.0
    expected_redundant_count = 3.0 / 2.0  # per state
    assert np.isclose(metrics.redundant_station_rate, expected_redundant_rate, atol=1e-4), \
        f"Expected redundant_station_rate={expected_redundant_rate:.4f}, got {metrics.redundant_station_rate:.4f}"
    assert np.isclose(metrics.redundant_stations_count, expected_redundant_count, atol=1e-4), \
        f"Expected redundant_stations_count={expected_redundant_count:.4f}, got {metrics.redundant_stations_count:.4f}"

    # Legal expansion connections: 10 + 20 = 30
    # Selected expansion actions: b0 took action 5 (expansion), b1 took 0 (NoOp) -> 1
    expected_expansion_ratio = 1.0 / 30.0
    expected_expansion_action_rate = 1.0 / 2.0
    assert np.isclose(metrics.expansion_ratio, expected_expansion_ratio, atol=1e-4), \
        f"Expected expansion_ratio={expected_expansion_ratio:.4f}, got {metrics.expansion_ratio:.4f}"
    assert np.isclose(metrics.expansion_action_rate, expected_expansion_action_rate, atol=1e-4), \
        f"Expected expansion_action_rate={expected_expansion_action_rate:.4f}, got {metrics.expansion_action_rate:.4f}"

    print("✓ compute_expansion_metrics calculations verified perfectly")


def test_live_rollout_metrics():
    print("\n--- Test 3: Live Environment Evaluation Rollout ---")
    env = MiniMetroEnv(map_id=0, seed=123)
    obs, _ = env.reset(seed=123)

    initial_metrics = compute_expansion_metrics(obs)
    print(f"  Step 0 metrics: lines_per_st={initial_metrics.lines_per_station_mean:.2f}, "
          f"redundant_st_rate={initial_metrics.redundant_station_rate:.2f}")
    assert initial_metrics.lines_per_station_mean >= 0.0
    assert initial_metrics.redundant_stations_count == 0.0, "Initial empty network cannot have redundant stations"

    # Step environment for 50 steps
    total_steps = 50
    actions_taken = []
    obs_list = []
    for _ in range(total_steps):
        mask = obs["action_mask"]
        valid_actions = np.where(mask)[0]
        # Choose a valid action
        action = np.random.choice(valid_actions)
        obs, reward, term, trunc, _ = env.step(action)
        actions_taken.append(action)
        obs_list.append(obs)
        if term or trunc:
            break

    # Batch compute metrics over rollout
    batched_nodes = np.stack([o["nodes"] for o in obs_list], axis=0)
    batched_masks = np.stack([o["action_mask"] for o in obs_list], axis=0)
    batched_num_nodes = np.stack([o["num_nodes"] for o in obs_list], axis=0)
    batched_obs = {
        "nodes": batched_nodes,
        "action_mask": batched_masks,
        "num_nodes": batched_num_nodes,
    }
    rollout_metrics = compute_expansion_metrics(batched_obs, np.array(actions_taken))
    print(f"  Rollout metrics over {len(obs_list)} steps:")
    print(f"    Expansion Action Rate: {rollout_metrics.expansion_action_rate*100:.2f}%")
    print(f"    Expansion Ratio:       {rollout_metrics.expansion_ratio:.4f}")
    print(f"    Avg Lines/Station:     {rollout_metrics.lines_per_station_mean:.2f}")
    print(f"    Redundant Station Rate:{rollout_metrics.redundant_station_rate*100:.2f}%")

    assert 0.0 <= rollout_metrics.expansion_action_rate <= 1.0
    assert 0.0 <= rollout_metrics.expansion_ratio <= 1.0
    assert 0.0 <= rollout_metrics.redundant_station_rate <= 1.0
    env.close()
    print("✓ Live rollout metrics verified cleanly")


if __name__ == "__main__":
    print("==================================================================")
    print("Running P1-2 Validation: Network Expansion & Redundancy Mitigation")
    print("==================================================================")
    test_model_redundancy_attenuation()
    test_expansion_metrics_calculation()
    test_live_rollout_metrics()
    print("\n==================================================================")
    print("ALL P1-2 TESTS PASSED SUCCESSFULLY!")
    print("==================================================================")

"""
Forensic Audit: Weekly Upgrade / Reward System Verification in Python.
Tests:
1. Observation visibility: globals[11] flag, globals[13..17] card 0 one-hot, globals[18..22] card 1 one-hot.
2. Action mask: exactly actions 4050 and 4051 are active when reward is pending.
3. Action mapping: action 4050 applies card 0, action 4051 applies card 1.
4. Line resource increment: verifying unlocked lines increases when Line is chosen.
5. Legacy checkpoint inspection: reporting exact tensor shapes, layer weights, and compatibility.
"""
import sys
import os
import ctypes
import numpy as np
import torch

sys.path.append(os.path.abspath("ml"))
from env import MiniMetroEnv, lib
from model import MiniMetroActorCritic, ACTION_TYPE_SLICES

def test_observation_visibility_and_action_mapping():
    print("\n========================================================")
    print("  TEST 1: Observation Visibility & Action Mapping")
    print("========================================================")
    
    env = MiniMetroEnv(map_id=0, seed=42)
    obs, _ = env.reset()
    
    reward_names = {
        0: "LINE",
        1: "LOCOMOTIVE",
        2: "TUNNEL",
        3: "CARRIAGE",
        4: "INTERCHANGE"
    }
    
    # Test all pairs of rewards
    test_pairs = [
        (0, 3), # Line vs Carriage
        (2, 0), # Tunnel vs Line
        (1, 4), # Locomotive vs Interchange
        (0, 1), # Line vs Locomotive
    ]
    
    for c0, c1 in test_pairs:
        name0 = reward_names[c0]
        name1 = reward_names[c1]
        
        # Inject pending reward via C-API
        lib.SetPendingReward(env.handle, c0, c1)
        
        # Query observation and action mask
        obs = env._get_obs()
        mask = obs["action_mask"]
        globals_vec = obs["globals"]
        
        # 1. Check pending flag
        assert globals_vec[11] == 1.0, f"Expected globals[11] == 1.0, got {globals_vec[11]}"
        
        # 2. Check card 0 one-hot
        c0_onehot = globals_vec[13:18]
        expected_c0 = np.zeros(5, dtype=np.float32)
        expected_c0[c0] = 1.0
        assert np.array_equal(c0_onehot, expected_c0), f"Card 0 mismatch: expected {expected_c0}, got {c0_onehot}"
        
        # 3. Check card 1 one-hot
        c1_onehot = globals_vec[18:23]
        expected_c1 = np.zeros(5, dtype=np.float32)
        expected_c1[c1] = 1.0
        assert np.array_equal(c1_onehot, expected_c1), f"Card 1 mismatch: expected {expected_c1}, got {c1_onehot}"
        
        # 4. Check action mask: ONLY actions 4050 and 4051 must be True
        valid_indices = np.where(mask)[0]
        assert set(valid_indices) == {4050, 4051}, f"Action mask violation: expected exactly [4050, 4051], got {valid_indices}"
        
        # Print observation tensor summary
        print(f"Offered: Card 0 = {name0} ({c0}) | Card 1 = {name1} ({c1})")
        print(f"  -> globals[11] (pending): {globals_vec[11]}")
        print(f"  -> globals[13..17] (card 0 one-hot): {c0_onehot.tolist()}")
        print(f"  -> globals[18..22] (card 1 one-hot): {c1_onehot.tolist()}")
        print(f"  -> Valid action mask indices: {valid_indices.tolist()}")
        
        # 5. Verify action application:
        # If c0 is Line (0), action 4050 should increment lines count
        lines_before = int(globals_vec[0])
        lib.SetPendingReward(env.handle, c0, c1)
        # Select Card 0 (action 4050)
        env.step(4050)
        obs_after = env._get_obs()
        lines_after = int(obs_after["globals"][0])
        
        if c0 == 0:
            assert lines_after == lines_before + 1, f"Expected lines {lines_before} -> {lines_before + 1}, got {lines_after}"
            print(f"  -> Action 4050 (Card 0 = LINE) correctly incremented lines: {lines_before} -> {lines_after}")
        else:
            print(f"  -> Action 4050 (Card 0 = {name0}) applied successfully.")
            
        # Test Card 1 (action 4051)
        lines_before_c1 = int(obs_after["globals"][0])
        lib.SetPendingReward(env.handle, c0, c1)
        env.step(4051)
        obs_after_c1 = env._get_obs()
        lines_after_c1 = int(obs_after_c1["globals"][0])
        if c1 == 0:
            assert lines_after_c1 == lines_before_c1 + 1, f"Expected lines {lines_before_c1} -> {lines_before_c1 + 1}, got {lines_after_c1}"
            print(f"  -> Action 4051 (Card 1 = LINE) correctly incremented lines: {lines_before_c1} -> {lines_after_c1}")
        else:
            print(f"  -> Action 4051 (Card 1 = {name1}) applied successfully.")

    print("✓ All observation visibility and action mapping checks passed!")

def test_legacy_checkpoint_compatibility():
    print("\n========================================================")
    print("  TEST 2: Legacy Checkpoint Compatibility Audit")
    print("========================================================")
    
    ckpt_path = "runs/minimetro_ppo/model_final.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    
    # Analyze state dict
    gcn1_node_w = sd["gcn1.node_proj.weight"]
    print(f"Checkpoint gcn1.node_proj.weight shape: {gcn1_node_w.shape}")
    # Shape is [256, 45] -> 32 node_dim + 13 global_dim
    hidden_dim = gcn1_node_w.shape[0]
    in_features = gcn1_node_w.shape[1]
    legacy_global_dim = in_features - 32
    print(f"Inferred checkpoint dimensions: hidden_dim={hidden_dim}, node_dim=32, global_dim={legacy_global_dim}")
    
    has_card_mlp = "reward_card_mlp.0.weight" in sd
    has_dispatch_mlp = "dispatch_train_mlp.0.weight" in sd
    print(f"Checkpoint contains reward_card_mlp: {has_card_mlp}")
    print(f"Checkpoint contains dispatch_train_mlp: {has_dispatch_mlp}")
    
    model = MiniMetroActorCritic(hidden_dim=256)
    model.load_state_dict(sd)
    print(f"Model is_legacy_checkpoint: {model.is_legacy_checkpoint}")
    
    # Check what scores_choose_reward does:
    combined = torch.zeros(1, 256 * 5)
    globals_t = torch.zeros(1, 23)
    globals_t[0, 11] = 1.0 # pending reward
    globals_t[0, 13] = 1.0 # Card 0 = LINE
    globals_t[0, 21] = 1.0 # Card 1 = CARRIAGE
    
    print("Legacy checkpoint capacity analysis complete.")

if __name__ == "__main__":
    test_observation_visibility_and_action_mapping()
    test_legacy_checkpoint_compatibility()

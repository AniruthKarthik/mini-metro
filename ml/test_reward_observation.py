"""
Test script for P0-2: Weekly Reward-Card Observation Blindness Fix.

Verifies that:
1. Vectorized observation global_dim is 23 (was 13, +10 for two 5-class reward card one-hots).
2. Card 0 one-hot encoding occupies indices 13..17 (Line=0, Train=1, Tunnel=2, Carriage=3, Interchange=4).
3. Card 1 one-hot encoding occupies indices 18..22.
4. When no reward is offered, indices 13..22 are strictly 0.0.
5. In-game weekly reward event triggers valid one-hot representations for both offered cards.
6. Permutation test: Mocking Card 0 = Line, Card 1 = Tunnel vs Card 0 = Tunnel, Card 1 = Line
   correctly swaps the observation features.
7. Choosing a reward clears the pending flag and resets card features to 0.0.
8. PyTorch model (MiniMetroActorCritic) forward pass and get_action_and_value run cleanly
   without dimension mismatch, including backward-compatible loading of legacy checkpoints.
"""

import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import MiniMetroEnv
from model import MiniMetroActorCritic, ACTION_TYPE_SLICES

REWARD_NAMES = ["Line", "Train", "Tunnel", "Carriage", "Interchange"]


def test_dimensions():
    print("\n--- Test 1: Dimension Verification ---")
    env = MiniMetroEnv(map_id=0)
    assert env.global_dim == 23, f"Expected env.global_dim == 23, got {env.global_dim}"
    assert env.observation_space["globals"].shape == (23,), f"Expected shape (23,), got {env.observation_space['globals'].shape}"

    obs, _ = env.reset(seed=42)
    assert obs["globals"].shape == (23,), f"Expected obs globals shape (23,), got {obs['globals'].shape}"
    assert obs["globals"][11] == 0.0, f"Expected pending flag 0.0, got {obs['globals'][11]}"
    assert np.all(obs["globals"][13:23] == 0.0), f"Expected reward card features all 0.0, got {obs['globals'][13:23]}"
    print("✓ Environment and observation space dimensions verified: global_dim=23")
    env.close()


def test_in_game_reward_event():
    print("\n--- Test 2: In-Game Weekly Reward Trigger ---")
    env = MiniMetroEnv(map_id=0)
    obs, _ = env.reset(seed=42)

    triggered = False
    for step in range(80):
        action = 0
        if obs["action_mask"][4050]:
            triggered = True
            break
        obs, reward, term, trunc, info = env.step(action)
        if obs["globals"][11] == 1.0:
            triggered = True
            break
        if term:
            break

    assert triggered, "Reward event did not trigger within rollout steps"
    print(f"✓ In-game weekly reward event triggered at step {step}")

    assert obs["globals"][11] == 1.0, f"Expected pending flag globals[11] == 1.0, got {obs['globals'][11]}"

    card0_feats = obs["globals"][13:18]
    assert np.sum(card0_feats) == 1.0, f"Expected exactly one 1.0 in Card 0 [13..17], got {card0_feats}"
    card0_type = int(np.argmax(card0_feats))
    print(f"  Card 0: {REWARD_NAMES[card0_type]} (one-hot index {13 + card0_type})")

    card1_feats = obs["globals"][18:23]
    assert np.sum(card1_feats) == 1.0, f"Expected exactly one 1.0 in Card 1 [18..22], got {card1_feats}"
    card1_type = int(np.argmax(card1_feats))
    print(f"  Card 1: {REWARD_NAMES[card1_type]} (one-hot index {18 + card1_type})")

    assert obs["action_mask"][4050], "Action 4050 (Choose Card 0) should be valid"
    assert obs["action_mask"][4051], "Action 4051 (Choose Card 1) should be valid"

    obs_after, _, _, _, _ = env.step(4050)
    assert obs_after["globals"][11] == 0.0, f"Expected pending flag cleared to 0.0, got {obs_after['globals'][11]}"
    assert np.all(obs_after["globals"][13:23] == 0.0), f"Expected card features cleared to 0.0, got {obs_after['globals'][13:23]}"
    print("✓ Successfully selected reward card and verified state returned to normal (cards cleared)")
    env.close()


def test_controlled_permutation():
    print("\n--- Test 3: Controlled Card Permutation Test ---")
    env = MiniMetroEnv(map_id=0)
    obs, _ = env.reset(seed=100)

    # Permutation A: Card 0 = Line (0), Card 1 = Tunnel (2)
    obs_a = env.set_pending_reward(0, 2)
    assert obs_a["globals"][11] == 1.0
    assert obs_a["globals"][13] == 1.0 and np.all(obs_a["globals"][14:18] == 0.0), "Card 0 must be Line"
    assert obs_a["globals"][20] == 1.0 and np.all(obs_a["globals"][18:20] == 0.0) and np.all(obs_a["globals"][21:23] == 0.0), "Card 1 must be Tunnel"
    print("  State A: Card 0 = Line, Card 1 = Tunnel -> globals[13]=1.0, globals[20]=1.0")

    # Permutation B: Swapped: Card 0 = Tunnel (2), Card 1 = Line (0)
    obs_b = env.set_pending_reward(2, 0)
    assert obs_b["globals"][11] == 1.0
    assert obs_b["globals"][15] == 1.0 and np.all(obs_b["globals"][13:15] == 0.0) and np.all(obs_b["globals"][16:18] == 0.0), "Card 0 must be Tunnel"
    assert obs_b["globals"][18] == 1.0 and np.all(obs_b["globals"][19:23] == 0.0), "Card 1 must be Line"
    print("  State B: Card 0 = Tunnel, Card 1 = Line -> globals[15]=1.0, globals[18]=1.0")

    assert not np.array_equal(obs_a["globals"][13:23], obs_b["globals"][13:23]), "Card encodings must differ under permutation"

    obs_c = env.set_pending_reward(3, 4)
    assert obs_c["globals"][16] == 1.0 and obs_c["globals"][22] == 1.0
    print("  State C: Card 0 = Carriage, Card 1 = Interchange -> globals[16]=1.0, globals[22]=1.0")

    obs_cleared = env.set_pending_reward(-1, -1)
    assert obs_cleared["globals"][11] == 0.0
    assert np.all(obs_cleared["globals"][13:23] == 0.0)
    print("  Cleared: globals[11]=0.0, globals[13:23]=0.0")

    print("✓ Controlled permutation assertions all passed")
    env.close()


def test_model_forward_and_compatibility():
    print("\n--- Test 4: PyTorch Model Forward Pass & Checkpoint Compatibility ---")
    device = torch.device("cpu")
    model = MiniMetroActorCritic(hidden_dim=256).to(device)

    assert model.gcn1.node_proj.in_features == 32 + 23, f"Expected 55 input features, got {model.gcn1.node_proj.in_features}"
    assert model.gcn1.global_update[0].in_features == 256 + 23, f"Expected 279 input features, got {model.gcn1.global_update[0].in_features}"

    ckpt_path = "runs/minimetro_ppo/model_final.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"✓ Successfully loaded legacy checkpoint '{ckpt_path}' with automatic weight adaptation")

    model.eval()

    env = MiniMetroEnv(map_id=0)
    obs, _ = env.reset(seed=42)
    obs_t = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}

    with torch.no_grad():
        logits, value, _ = model(obs_t)
        action, logprob, entropy, val, _ = model.get_action_and_value(obs_t, deterministic=True)
    assert logits.shape == (1, 4087), f"Expected logits shape (1, 4087), got {logits.shape}"
    assert value.shape == (1, 1), f"Expected value shape (1, 1), got {value.shape}"
    print(f"✓ Normal forward pass: selected action {action.item()}, value={value.item():.3f}")

    obs_reward_a = env.set_pending_reward(0, 2)
    obs_reward_a_t = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs_reward_a.items()}
    with torch.no_grad():
        logits_a, _, _ = model(obs_reward_a_t)
        action_a, _, _, _, _ = model.get_action_and_value(obs_reward_a_t, deterministic=True)
    print(f"  Reward State A (Line vs Tunnel): ChooseReward logits = [{logits_a[0, 4050]:.3f}, {logits_a[0, 4051]:.3f}], action={action_a.item()}")

    obs_reward_b = env.set_pending_reward(2, 0)
    obs_reward_b_t = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs_reward_b.items()}
    with torch.no_grad():
        logits_b, _, _ = model(obs_reward_b_t)
        action_b, _, _, _, _ = model.get_action_and_value(obs_reward_b_t, deterministic=True)
    print(f"  Reward State B (Tunnel vs Line): ChooseReward logits = [{logits_b[0, 4050]:.3f}, {logits_b[0, 4051]:.3f}], action={action_b.item()}")

    print("✓ Model forward pass and action decoding executed cleanly with no shape errors")
    env.close()


if __name__ == "__main__":
    print("================================================================")
    print("Running P0-2 Validation: Weekly Reward-Card Observation Blindness")
    print("================================================================")
    test_dimensions()
    test_in_game_reward_event()
    test_controlled_permutation()
    test_model_forward_and_compatibility()
    print("\n================================================================")
    print("ALL P0-2 TESTS PASSED SUCCESSFULLY!")
    print("================================================================")

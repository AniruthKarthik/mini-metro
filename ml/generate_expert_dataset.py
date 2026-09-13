"""
Expert Dataset Generator for Behavioral Cloning (BC) in Mini Metro.

Executes high-performing heuristic and hybrid policies across all maps (London, NYC,
Tokyo, Berlin) and random seeds, recording (obs, mask, expert_action) tuples into
a compressed .npz dataset for supervised pre-training.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from env import MiniMetroEnv
from eval import GreedyHeuristicPolicy, ModelLiveAgentPolicy
from model import MiniMetroActorCritic
from agent import load_model

DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "runs", "expert_dataset.npz")


def generate_dataset(
    target_transitions: int = 20000,
    maps: list = None,
    output_path: str = DEFAULT_OUTPUT,
    model_path: str = None,
    device_name: str = "cpu",
    seed_start: int = 5000,
):
    if maps is None:
        maps = [0, 1, 2, 3]  # London, NYC, Tokyo, Berlin

    device = torch.device(device_name)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Initialize policies
    policies = {}
    policies["greedy"] = GreedyHeuristicPolicy()

    if model_path and os.path.exists(model_path):
        try:
            model, _ = load_model(device, model_override=model_path)
            model.eval()
            policies["hybrid"] = ModelLiveAgentPolicy(model, device=device, deterministic=True)
            print(f"[Dataset] Loaded hybrid policy with model: {model_path}")
        except Exception as e:
            print(f"[Dataset] Could not load model for hybrid policy: {e}")

    # Storage buffers
    all_nodes = []
    all_edges = []
    all_edge_attrs = []
    all_globals = []
    all_masks = []
    all_num_nodes = []
    all_num_edges = []
    all_actions = []

    total_transitions = 0
    episodes = 0
    start_time = time.time()

    print("=" * 70)
    print("MINI METRO: EXPERT DEMONSTRATION DATASET GENERATOR")
    print(f"Target Transitions : {target_transitions}")
    print(f"Maps               : {maps}")
    print(f"Output File        : {output_path}")
    print("=" * 70)

    policy_keys = list(policies.keys())
    seed = seed_start

    while total_transitions < target_transitions:
        map_id = maps[episodes % len(maps)]
        pol_key = policy_keys[episodes % len(policy_keys)]
        policy = policies[pol_key]

        env = MiniMetroEnv(map_id=map_id, seed=seed)
        obs, _ = env.reset(seed=seed)
        policy.reset(seed=seed)

        ep_transitions = 0
        ep_score = 0
        done = False

        while not done:
            try:
                action = policy.act(obs, env=env)
            except TypeError:
                action = policy.act(obs)

            # Record transition
            all_nodes.append(np.array(obs["nodes"], dtype=np.float32).reshape(30, 32))
            all_edges.append(np.array(obs["edges"], dtype=np.int64).reshape(200, 2))
            all_edge_attrs.append(np.array(obs["edge_attrs"], dtype=np.float32).reshape(200, 10))
            all_globals.append(np.array(obs["globals"], dtype=np.float32).reshape(23))
            all_masks.append(np.array(obs["action_mask"], dtype=bool))
            
            num_nodes = int(obs["num_nodes"][0]) if "num_nodes" in obs else len(obs["nodes"])
            num_edges = int(obs["num_edges"][0]) if "num_edges" in obs else len(obs["edges"])
            all_num_nodes.append(num_nodes)
            all_num_edges.append(num_edges)
            all_actions.append(int(action))

            total_transitions += 1
            ep_transitions += 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs
            ep_score = info.get("score", ep_score)

            if total_transitions >= target_transitions:
                break

        env.close()
        episodes += 1
        seed += 1

        elapsed = time.time() - start_time
        fps = total_transitions / max(0.1, elapsed)
        print(f"  Ep {episodes:03d} (Map {map_id}, {pol_key}): Score={ep_score:3d}, Steps={ep_transitions:3d} | Total: {total_transitions}/{target_transitions} ({fps:.1f} steps/s)", flush=True)

    print("\n[Dataset] Compressing and writing arrays to disk...")
    np.savez_compressed(
        output_path,
        nodes=np.stack(all_nodes, axis=0),
        edges=np.stack(all_edges, axis=0),
        edge_attrs=np.stack(all_edge_attrs, axis=0),
        globals=np.stack(all_globals, axis=0),
        action_masks=np.stack(all_masks, axis=0),
        num_nodes=np.array(all_num_nodes, dtype=np.int32).reshape(-1, 1),
        num_edges=np.array(all_num_edges, dtype=np.int32).reshape(-1, 1),
        actions=np.array(all_actions, dtype=np.int64),
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Dataset] Saved {total_transitions} expert transitions to {output_path} ({size_mb:.2f} MB)")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Mini Metro Expert Dataset Generator")
    parser.add_argument("--transitions", type=int, default=10000, help="Target transition count")
    parser.add_argument("--maps", type=int, nargs="+", default=[0, 1, 2, 3], help="Map IDs: 0=London, 1=NYC, 2=Tokyo, 3=Berlin")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output .npz file path")
    parser.add_argument("--model", type=str, default=os.path.join(SCRIPT_DIR, "runs", "minimetro_ppo", "model_best.pt"), help="Model checkpoint for hybrid expert")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(
        target_transitions=args.transitions,
        maps=args.maps,
        output_path=args.output,
        model_path=args.model,
        device_name=args.device,
    )

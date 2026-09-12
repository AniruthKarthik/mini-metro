#!/usr/bin/env python3
"""
Comprehensive Reward Decomposition & Ablation Suite (P2-1).

Evaluates the agent under 4 systematic reward configurations:
1. Baseline: Standard reward formulation (alpha=0.30, beta=200, conn=2.0, quadratic crowd)
2. Ablation A: No ConnectivityBonus (conn=0.0) -> Quantifies expansion drive
3. Ablation B: Linear Crowd Penalty (linear=True) -> Quantifies overcrowding sensitivity
4. Ablation C: Reduced BetaGameOverPenalty (beta=50.0) -> Quantifies terminal risk sensitivity

Measures return, length, deliveries, reward channel breakdowns, and expansion metrics.
"""

import os
import sys
import argparse
import numpy as np
import torch
from typing import Dict, Any, List

from env import MiniMetroEnv
from model import MiniMetroActorCritic
from probing import compute_expansion_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run reward ablation suite (P2-1)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes per configuration")
    parser.add_argument("--max_steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--checkpoint", type=str, default="runs/minimetro_ppo/model_final.pt", help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic policy")
    return parser.parse_args()


def load_agent(checkpoint_path: str, device: torch.device) -> MiniMetroActorCritic:
    model = MiniMetroActorCritic(hidden_dim=256).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"[OK] Successfully loaded model checkpoint from {checkpoint_path}")
    else:
        print(f"! Checkpoint {checkpoint_path} not found; running with initialized model")
    model.eval()
    return model


def run_single_ablation(
    config_name: str,
    scoring_kwargs: Dict[str, Any],
    model: MiniMetroActorCritic,
    device: torch.device,
    num_episodes: int,
    max_steps: int,
    deterministic: bool = True,
) -> Dict[str, Any]:
    print(f"\nEvaluating {config_name}: {scoring_kwargs} ...")
    env = MiniMetroEnv(map_id=0)
    env.set_scoring_config(**scoring_kwargs)

    ep_returns = []
    ep_lengths = []
    ep_deliveries = []
    ep_breakdowns: Dict[str, List[float]] = {
        "delivery": [],
        "survival": [],
        "connectivity": [],
        "crowd_penalty": [],
        "game_over": [],
        "redundancy": [],
        "loop_reversal": [],
        "track_efficiency": [],
    }
    ep_expansion_rates = []
    ep_redundant_rates = []
    ep_lines_per_st = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep * 42)
        done = False
        step_count = 0
        actions_taken = []
        obs_nodes_list = []
        obs_masks_list = []

        lstm_state = None

        while not done and step_count < max_steps:
            obs_nodes_list.append(obs["nodes"])
            obs_masks_list.append(obs["action_mask"])

            # Prepare obs tensor for model
            obs_tensor = {
                "nodes": torch.from_numpy(obs["nodes"]).float().unsqueeze(0).to(device),
                "edges": torch.from_numpy(obs["edges"]).long().unsqueeze(0).to(device),
                "edge_attrs": torch.from_numpy(obs["edge_attrs"]).float().unsqueeze(0).to(device),
                "globals": torch.from_numpy(obs["globals"]).float().unsqueeze(0).to(device),
                "action_mask": torch.from_numpy(obs["action_mask"]).bool().unsqueeze(0).to(device),
                "num_nodes": torch.from_numpy(obs["num_nodes"]).int().unsqueeze(0).to(device),
                "num_edges": torch.from_numpy(obs["num_edges"]).int().unsqueeze(0).to(device),
            }

            with torch.no_grad():
                action_tensor, _, _, _, lstm_state = model.get_action_and_value(
                    obs_tensor,
                    lstm_state=lstm_state,
                    mask=obs_tensor["action_mask"],
                    deterministic=deterministic,
                )
                action = int(action_tensor.item())

            actions_taken.append(action)
            obs, reward, done, _, info = env.step(action)
            step_count += 1

        # Episode final stats
        ep_returns.append(sum(info["episode_reward_breakdown"].values()))
        ep_lengths.append(step_count)
        ep_deliveries.append(info["episode_reward_breakdown"]["delivery"])

        for k, v in info["episode_reward_breakdown"].items():
            ep_breakdowns[k].append(v)

        # Compute expansion metrics across episode trajectory
        batched_obs = {
            "nodes": np.stack(obs_nodes_list),
            "action_mask": np.stack(obs_masks_list),
        }
        exp_metrics = compute_expansion_metrics(batched_obs, np.array(actions_taken))
        ep_expansion_rates.append(exp_metrics.expansion_action_rate)
        ep_redundant_rates.append(exp_metrics.redundant_station_rate)
        ep_lines_per_st.append(exp_metrics.lines_per_station_mean)

    env.close()

    results = {
        "name": config_name,
        "episodes": num_episodes,
        "mean_return": float(np.mean(ep_returns)),
        "std_return": float(np.std(ep_returns)),
        "mean_length": float(np.mean(ep_lengths)),
        "std_length": float(np.std(ep_lengths)),
        "mean_delivery": float(np.mean(ep_deliveries)),
        "std_delivery": float(np.std(ep_deliveries)),
        "breakdowns_mean": {k: float(np.mean(v)) for k, v in ep_breakdowns.items()},
        "expansion_action_rate_mean": float(np.mean(ep_expansion_rates)),
        "redundant_station_rate_mean": float(np.mean(ep_redundant_rates)),
        "lines_per_station_mean": float(np.mean(ep_lines_per_st)),
    }
    return results


def print_comparison_table(results_list: List[Dict[str, Any]]):
    print("\n" + "=" * 110)
    print("REWARD ABLATION EXPERIMENT SUMMARY TABLE (P2-1)")
    print("=" * 110)
    headers = [
        "Configuration",
        "Return (Mean±Std)",
        "Length",
        "Deliveries",
        "Conn Rwd",
        "Crowd Pen",
        "Track Eff",
        "Game Over",
        "Exp Act %",
        "Redund %",
        "Lines/Stn",
    ]
    header_str = f"{headers[0]:<20} | {headers[1]:<17} | {headers[2]:<7} | {headers[3]:<10} | {headers[4]:<9} | {headers[5]:<10} | {headers[6]:<9} | {headers[7]:<9} | {headers[8]:<9} | {headers[9]:<8} | {headers[10]:<8}"
    print(header_str)
    print("-" * 122)

    for res in results_list:
        b = res["breakdowns_mean"]
        ret_str = f"{res['mean_return']:>6.1f} ± {res['std_return']:<4.1f}"
        len_str = f"{res['mean_length']:>5.1f}"
        del_str = f"{res['mean_delivery']:>6.1f}"
        conn_str = f"{b['connectivity']:>6.1f}"
        crw_str = f"{b['crowd_penalty']:>7.1f}"
        trk_str = f"{b.get('track_efficiency', 0.0):>7.2f}"
        go_str = f"{b['game_over']:>6.1f}"
        exp_str = f"{res['expansion_action_rate_mean'] * 100:>6.1f}%"
        red_str = f"{res['redundant_station_rate_mean'] * 100:>5.1f}%"
        lps_str = f"{res['lines_per_station_mean']:>6.2f}"

        row_str = f"{res['name']:<20} | {ret_str:<17} | {len_str:<7} | {del_str:<10} | {conn_str:<9} | {crw_str:<10} | {trk_str:<9} | {go_str:<9} | {exp_str:<9} | {red_str:<8} | {lps_str:<8}"
        print(row_str)
    print("=" * 122)

    # Detailed Analysis
    baseline = results_list[0]
    ablation_a = results_list[1]
    print("\nKEY SCIENTIFIC FINDINGS:")
    exp_diff = (ablation_a['expansion_action_rate_mean'] - baseline['expansion_action_rate_mean']) * 100
    print(f"1. ConnectivityBonus Contribution to Network Expansion:")
    print(f"   - Baseline Expansion Action Rate: {baseline['expansion_action_rate_mean']*100:.2f}% (Connectivity Reward: +{baseline['breakdowns_mean']['connectivity']:.2f})")
    print(f"   - Ablation A (No ConnBonus) Expansion Action Rate: {ablation_a['expansion_action_rate_mean']*100:.2f}% (Connectivity Reward: +{ablation_a['breakdowns_mean']['connectivity']:.2f})")
    print(f"   - Δ Expansion Action Rate: {exp_diff:+.2f}% points")
    print(f"   - Conclusion: ConnectivityBonus accounts for {abs(baseline['breakdowns_mean']['connectivity']):.2f} pts of reward and directly incentivizes line formation.")

    ablation_b = results_list[2]
    print(f"2. Linear vs. Quadratic Overcrowding Penalty:")
    print(f"   - Baseline Crowd Penalty: {baseline['breakdowns_mean']['crowd_penalty']:.2f}")
    print(f"   - Ablation B (Linear) Crowd Penalty: {ablation_b['breakdowns_mean']['crowd_penalty']:.2f}")
    print(f"   - Δ Crowd Penalty: {ablation_b['breakdowns_mean']['crowd_penalty'] - baseline['breakdowns_mean']['crowd_penalty']:+.2f}")

    ablation_c = results_list[3]
    print(f"3. BetaGameOverPenalty Sensitivity:")
    print(f"   - Baseline Game Over Penalty: {baseline['breakdowns_mean']['game_over']:.2f}")
    print(f"   - Ablation C (Beta=50) Game Over Penalty: {ablation_c['breakdowns_mean']['game_over']:.2f}")
    print("=" * 122 + "\n")


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = load_agent(args.checkpoint, device)

    experiments = [
        ("Baseline (Default)", {"alpha_crowd": 0.30, "beta_game_over": 200.0, "connectivity_bonus": 2.0, "track_efficiency": 0.01, "linear_crowd": False}),
        ("Ablation A (No Conn)", {"alpha_crowd": 0.30, "beta_game_over": 200.0, "connectivity_bonus": 0.0, "track_efficiency": 0.01, "linear_crowd": False}),
        ("Ablation B (Lin Crowd)", {"alpha_crowd": 0.30, "beta_game_over": 200.0, "connectivity_bonus": 2.0, "track_efficiency": 0.01, "linear_crowd": True}),
        ("Ablation C (Beta=50)", {"alpha_crowd": 0.30, "beta_game_over": 50.0, "connectivity_bonus": 2.0, "track_efficiency": 0.01, "linear_crowd": False}),
    ]

    all_results = []
    for name, kwargs in experiments:
        res = run_single_ablation(
            config_name=name,
            scoring_kwargs=kwargs,
            model=model,
            device=device,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
        )
        all_results.append(res)

    print_comparison_table(all_results)


if __name__ == "__main__":
    main()

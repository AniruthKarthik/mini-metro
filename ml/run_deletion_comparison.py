"""
Three-Way Empirical Deletion & Intervention Experiment:
  Mode A: Dynamic editing enabled with unguided policy
  Mode B: Dynamic editing disabled (RemoveLine and ShortenLine masked)
  Mode C: Dynamic editing enabled with strategic intervention-value & stability mechanism

Evaluates across identical seeds, measuring:
  - Score (passengers delivered)
  - Survival Time (seconds / steps)
  - Game-Over Rate (%)
  - Number of Deletions & Redraws
  - Network Edits per Simulated Minute
  - Deletion ROI
"""

import sys
import os
import time
import argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import MiniMetroEnv
from model import MiniMetroActorCritic, ACTION_TYPE_SLICES
from intervention import (
    StrategicInterventionArbiter,
    InterventionTier,
    classify_action_tier,
    ACTION_NOOP,
    REMOVE_LINE_START,
    SHORTEN_LINE_START,
)
from diagnostics import InterventionDiagnostics

MODEL_PATH = os.path.join(os.path.dirname(__file__), "runs/minimetro_ppo/model_final.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs/minimetro_ppo/model_final.pt")


def evaluate_episode(
    env: MiniMetroEnv,
    model: MiniMetroActorCritic,
    mode: str,
    seed: int,
    map_id: int,
    arbiter: StrategicInterventionArbiter,
    diagnostics: InterventionDiagnostics,
    episode_idx: int,
    max_steps: int = 250,
) -> dict:
    obs, info = env.reset(seed=seed)
    lstm_state = None
    total_reward = 0.0
    steps = 0
    done = False
    sim_seconds_total = 0.0

    episode_deletions = 0
    episode_shortens = 0
    episode_add_lines = 0

    while not done and steps < max_steps:
        steps += 1
        mask = obs["action_mask"].copy()

        # Mode B: disable dynamic deletion / shorten completely
        if mode == "B":
            mask[ACTION_TYPE_SLICES[10]] = False  # RemoveLine
            mask[ACTION_TYPE_SLICES[11]] = False  # ShortenLine
            if not mask.any():
                mask[0] = True

        obs_tensor = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
        mask_t = torch.as_tensor(mask).bool().unsqueeze(0)
        obs_tensor["action_mask"] = mask_t

        with torch.no_grad():
            action_t, _, _, _, next_lstm = model.get_action_and_value(
                obs_tensor,
                lstm_state=lstm_state,
                mask=mask_t,
                deterministic=False,
            )
            lstm_state = next_lstm
            action_id = int(action_t.item())

        tier = classify_action_tier(action_id)

        # Mode C: Strategic Intervention Arbiter
        if mode == "C":
            if tier == InterventionTier.MAJOR_REBUILD:
                # Propose candidates: KEEP, valid local edits, and the proposed RemoveLine
                valid_indices = np.where(mask)[0]
                local_candidates = [
                    a for a in valid_indices
                    if classify_action_tier(a) in (InterventionTier.LOCAL_EDIT, InterventionTier.DISPATCH)
                ][:5]
                candidates = [ACTION_NOOP] + local_candidates + [action_id]

                best_act, utils = arbiter.evaluate_candidates(env, candidates, obs)
                action_id = best_act
                tier = classify_action_tier(action_id)

        # Track structural actions
        is_structural = False
        action_name = "Other"
        line_target = -1
        if REMOVE_LINE_START <= action_id < REMOVE_LINE_START + 7:
            is_structural = True
            action_name = "RemoveLine"
            line_target = action_id - REMOVE_LINE_START
            episode_deletions += 1
        elif SHORTEN_LINE_START <= action_id < SHORTEN_LINE_START + 14:
            is_structural = True
            action_name = "ShortenLine"
            line_target = (action_id - SHORTEN_LINE_START) // 2
            episode_shortens += 1
        elif 1 <= action_id < 436:
            is_structural = True
            action_name = "AddLine"
            episode_add_lines += 1

        rec = None
        if is_structural:
            rec = diagnostics.log_intervention(
                sim_time=sim_seconds_total,
                episode=episode_idx,
                map_id=map_id,
                seed=seed,
                action_id=action_id,
                action_type=action_name,
                line_id=line_target,
                line_stations_count=0,
                line_track_len=env.get_total_track_length(),
                line_trains_count=0,
                line_pax_on_board=0,
                queue_pressure_before=float(obs["nodes"][:, 12:22].sum()),
                passengers_delivered_before=int(obs["globals"][6] * 500.0),
            )

        obs, reward, done, truncated, step_info = env.step(action_id)
        total_reward += reward
        sim_seconds = step_info.get("simulation_seconds", 4.0)
        sim_seconds_total += sim_seconds

        if rec is not None:
            diagnostics.update_post_edit(
                rec,
                passengers_delivered_after=int(obs["globals"][6] * 500.0),
                queue_pressure_after=float(obs["nodes"][:, 12:22].sum()),
                game_over=done,
                elapsed_sim_seconds=sim_seconds,
            )

    score = int(obs["globals"][6] * 500.0)
    return {
        "score": score,
        "survival_seconds": sim_seconds_total,
        "steps": steps,
        "game_over": done,
        "deletions": episode_deletions,
        "shortens": episode_shortens,
        "add_lines": episode_add_lines,
        "redraws": min(episode_deletions, episode_add_lines),
        "total_reward": total_reward,
    }


def run_experiment(seeds: list, map_id: int = 0):
    print("=" * 80)
    print("STRATEGIC NETWORK EDITING EVALUATION: MODES A vs B vs C")
    print("=" * 80)
    print(f"Seeds: {seeds} | Map ID: {map_id} (London)")

    model = MiniMetroActorCritic(hidden_dim=256)
    if os.path.exists(MODEL_PATH):
        sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(sd)
        print(f"Loaded trained policy checkpoint: {MODEL_PATH}")
    model.eval()

    arbiter = StrategicInterventionArbiter(cf_duration=4.0, rebuild_threshold=1.0)

    modes = ["A", "B", "C"]
    mode_names = {
        "A": "Mode A: Dynamic Editing Enabled (Unguided Policy)",
        "B": "Mode B: Dynamic Editing Disabled (Masked Baseline)",
        "C": "Mode C: Strategic Intervention & Stability (Proposed)",
    }

    results = {m: [] for m in modes}
    diagnostics_by_mode = {m: InterventionDiagnostics() for m in modes}

    for m in modes:
        print(f"\nEvaluating {mode_names[m]}...")
        env = MiniMetroEnv(map_id=map_id, seed=seeds[0])
        for idx, seed in enumerate(seeds):
            res = evaluate_episode(
                env=env,
                model=model,
                mode=m,
                seed=seed,
                map_id=map_id,
                arbiter=arbiter,
                diagnostics=diagnostics_by_mode[m],
                episode_idx=idx,
            )
            results[m].append(res)
            print(f"  [Seed {seed:3d}] Score: {res['score']:3d} | Survival: {res['survival_seconds']:5.1f}s ({res['steps']:3d} steps) | Del: {res['deletions']} | Redraw: {res['redraws']}")
        env.close()

    print("\n" + "=" * 80)
    print("COMPARATIVE EVALUATION SUMMARY ACROSS IDENTICAL SEEDS")
    print("=" * 80)
    headers = ["Metric", "Mode A (Unguided)", "Mode B (Disabled)", "Mode C (Strategic)"]
    print(f"{headers[0]:<32} | {headers[1]:<18} | {headers[2]:<18} | {headers[3]:<18}")
    print("-" * 92)

    def stats(key, m):
        vals = [r[key] for r in results[m]]
        return float(np.mean(vals)), float(np.std(vals))

    m_score_a, s_score_a = stats("score", "A")
    m_score_b, s_score_b = stats("score", "B")
    m_score_c, s_score_c = stats("score", "C")
    print(f"{'Mean Score (Pax Delivered)':<32} | {m_score_a:6.2f} ± {s_score_a:4.1f}    | {m_score_b:6.2f} ± {s_score_b:4.1f}    | {m_score_c:6.2f} ± {s_score_c:4.1f}")

    med_a = float(np.median([r["score"] for r in results["A"]]))
    med_b = float(np.median([r["score"] for r in results["B"]]))
    med_c = float(np.median([r["score"] for r in results["C"]]))
    print(f"{'Median Score':<32} | {med_a:6.1f}             | {med_b:6.1f}             | {med_c:6.1f}")

    m_surv_a, _ = stats("survival_seconds", "A")
    m_surv_b, _ = stats("survival_seconds", "B")
    m_surv_c, _ = stats("survival_seconds", "C")
    print(f"{'Mean Survival Time (s)':<32} | {m_surv_a:6.1f}s            | {m_surv_b:6.1f}s            | {m_surv_c:6.1f}s")

    go_a = float(np.mean([1.0 if r["game_over"] else 0.0 for r in results["A"]]) * 100.0)
    go_b = float(np.mean([1.0 if r["game_over"] else 0.0 for r in results["B"]]) * 100.0)
    go_c = float(np.mean([1.0 if r["game_over"] else 0.0 for r in results["C"]]) * 100.0)
    print(f"{'Game-Over Rate (%)':<32} | {go_a:5.1f}%             | {go_b:5.1f}%             | {go_c:5.1f}%")

    m_del_a, _ = stats("deletions", "A")
    m_del_b, _ = stats("deletions", "B")
    m_del_c, _ = stats("deletions", "C")
    print(f"{'Mean Deletions / Episode':<32} | {m_del_a:5.2f}              | {m_del_b:5.2f}              | {m_del_c:5.2f}")

    m_red_a, _ = stats("redraws", "A")
    m_red_b, _ = stats("redraws", "B")
    m_red_c, _ = stats("redraws", "C")
    print(f"{'Mean Redraws / Episode':<32} | {m_red_a:5.2f}              | {m_red_b:5.2f}              | {m_red_c:5.2f}")

    diag_a = diagnostics_by_mode["A"].compute_summary_statistics(sum(r["survival_seconds"] for r in results["A"]), len(seeds))
    diag_b = diagnostics_by_mode["B"].compute_summary_statistics(sum(r["survival_seconds"] for r in results["B"]), len(seeds))
    diag_c = diagnostics_by_mode["C"].compute_summary_statistics(sum(r["survival_seconds"] for r in results["C"]), len(seeds))

    print(f"{'Edits per Simulated Minute':<32} | {diag_a['edits_per_simulated_minute']:5.2f}              | {diag_b['edits_per_simulated_minute']:5.2f}              | {diag_c['edits_per_simulated_minute']:5.2f}")
    print(f"{'Beneficial Edits (%)':<32} | {diag_a['beneficial_pct']:5.1f}%             | {diag_b['beneficial_pct']:5.1f}%             | {diag_c['beneficial_pct']:5.1f}%")
    print(f"{'Harmful Edits (%)':<32} | {diag_a['harmful_pct']:5.1f}%             | {diag_b['harmful_pct']:5.1f}%             | {diag_c['harmful_pct']:5.1f}%")
    print(f"{'Mean Deletion ROI':<32} | {diag_a['mean_deletion_roi']:+5.2f}              | {diag_b['mean_deletion_roi']:+5.2f}              | {diag_c['mean_deletion_roi']:+5.2f}")

    print("=" * 80)
    print("HYPOTHESIS VALIDATION:")
    print(f"  Mode A Score ({m_score_a:.1f}) vs Mode B Score ({m_score_b:.1f}): A < B? {'YES' if m_score_a < m_score_b else 'NO'}")
    print(f"  Mode C Score ({m_score_c:.1f}) vs Mode B Score ({m_score_b:.1f}): C > B? {'YES' if m_score_c >= m_score_b else 'NO'}")
    print(f"  Mode C Score ({m_score_c:.1f}) vs Mode A Score ({m_score_a:.1f}): C > A? {'YES' if m_score_c > m_score_a else 'NO'}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105, 106, 107, 108])
    parser.add_argument("--map", type=int, default=0)
    args = parser.parse_args()

    run_experiment(seeds=args.seeds, map_id=args.map)

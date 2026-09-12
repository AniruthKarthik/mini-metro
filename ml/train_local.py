import sys
import os
try:
    import gymnasium as gym
except ImportError:
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
    if os.path.exists(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
        os.execv(venv_python, [venv_python] + sys.argv)

import argparse
import time
import glob
import collections

import numpy as np
import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass
from torch.utils.tensorboard import SummaryWriter

from env import MiniMetroEnv, LineOrientationAugmentation
from model import MiniMetroActorCritic
from ppo import PPO
from probing import compute_expansion_metrics
from curriculum import CurriculumManager


# ============================================================
# ENVIRONMENT
# ============================================================

MAP_NAMES = {
    0: "London",
    1: "New York City",
    2: "Tokyo",
    3: "Berlin",
}

def make_env(seed, map_id=0, map_pool=None, map_weights=None, flip_prob=0.5, use_pbrs=True):
    def thunk():
        env = MiniMetroEnv(
            map_id=map_id,
            seed=seed,
            map_pool=map_pool,
            map_weights=map_weights,
            use_pbrs=use_pbrs,
        )
        if flip_prob > 0:
            env = LineOrientationAugmentation(env, flip_prob=flip_prob)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

CHECKPOINT_DIR = "runs/minimetro_ppo_local"


def save_checkpoint(
    model,
    agent,
    update,
    global_step,
    checkpoint_dir=CHECKPOINT_DIR,
    curriculum_state_dict=None,
    best_avg_score=-1.0,
):
    """
    Save everything needed to resume training.

    Saves:
        - model parameters
        - optimizer state (if available)
        - current update
        - global step
        - RNG states
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"checkpoint_{update:05d}.pt"
    )

    checkpoint = {
        "update": update,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "best_avg_score": best_avg_score,
    }
    if curriculum_state_dict is not None:
        checkpoint["curriculum_state_dict"] = curriculum_state_dict

    # PPO implementation may expose optimizer as agent.optimizer.
    if hasattr(agent, "optimizer"):
        checkpoint["optimizer_state_dict"] = (
            agent.optimizer.state_dict()
        )

    # Save RNG states so resumed training is more reproducible.
    checkpoint["torch_rng_state"] = torch.get_rng_state()

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    checkpoint["numpy_rng_state"] = np.random.get_state()

    torch.save(
        checkpoint,
        checkpoint_path
    )

    print(
        f"Checkpoint saved: {checkpoint_path}",
        flush=True
    )

    return checkpoint_path


def find_latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    """
    Find the checkpoint with the highest update number.
    """
    if not os.path.exists(checkpoint_dir):
        return None

    valid_ckpts = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_") and f.endswith(".pt"):
            if f in ("checkpoint_error.pt", "checkpoint_emergency.pt"):
                continue
            parts = f.replace("checkpoint_", "").replace(".pt", "")
            if parts.isdigit():
                valid_ckpts.append((int(parts), os.path.join(checkpoint_dir, f)))

    if not valid_ckpts:
        return None

    valid_ckpts.sort(key=lambda x: x[0])
    return valid_ckpts[-1][1]


def load_checkpoint(
    checkpoint_path,
    model,
    agent,
    device,
    curriculum=None,
):
    """
    Load model/optimizer/RNG state.

    Returns:
        update,
        global_step
    """

    print(
        f"Loading checkpoint: {checkpoint_path}",
        flush=True
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model_sd = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if model_sd is not None:
        model.load_state_dict(model_sd)

    if hasattr(agent, "optimizer"):
        optim_sd = checkpoint.get("optimizer_state_dict", checkpoint.get("optimizer"))
        if optim_sd is not None:
            try:
                agent.optimizer.load_state_dict(optim_sd)
            except Exception as opt_err:
                print(
                    f"[WARNING] Could not restore optimizer state ({opt_err}). "
                    f"Architecture parameters changed — using reinitialized optimizer."
                )

    if curriculum is not None and "curriculum_state_dict" in checkpoint:
        curriculum.load_state_dict(checkpoint["curriculum_state_dict"])
        print(f"[CURRICULUM] Restored Curriculum State -> {curriculum.get_stage_name()}", flush=True)

    # Restore RNG state when available.
    if "torch_rng_state" in checkpoint:
        try:
            torch.set_rng_state(
                checkpoint["torch_rng_state"]
            )
        except Exception:
            pass

    if (
        torch.cuda.is_available()
        and "cuda_rng_state" in checkpoint
    ):
        try:
            torch.cuda.set_rng_state_all(
                checkpoint["cuda_rng_state"]
            )
        except Exception:
            pass

    if "numpy_rng_state" in checkpoint:
        try:
            np.random.set_state(
                checkpoint["numpy_rng_state"]
            )
        except Exception:
            pass

    update = int(
        checkpoint.get("update", 0)
    )

    global_step = int(
        checkpoint.get("global_step", 0)
    )

    print(
        f"[OK] Resumed from update {update} "
        f"| global_step={global_step}",
        flush=True
    )

    return update, global_step


# ============================================================
# MAIN TRAINING
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Mini Metro Local PPO Multi-Map Training")
    parser.add_argument("--maps", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="List of map IDs to train on: 0=London, 1=NYC, 2=Tokyo, 3=Berlin")
    parser.add_argument("--map-mode", type=str, choices=["stratified", "mixed"], default="stratified",
                        help="Worker map allocation: 'stratified' or 'mixed'")
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable procedural multi-map curriculum learning (Stage 1 Berlin -> Stage 2 London/Tokyo -> Stage 3 All Maps)")
    parser.add_argument("--pbrs", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Potential-Based Reward Shaping (Ng et al., 1999) for dense temporal credit assignment")
    parser.add_argument("--hierarchical", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable two-stage Hierarchical Action Factorization")
    parser.add_argument("--hidden-dim", type=int, default=32,
                        help="Model hidden dimension (default: 32)")
    parser.add_argument("--total-timesteps", type=int, default=40_000,
                        help="Total environment steps to train (default: 40,000)")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="Number of parallel environments (default: 16)")
    parser.add_argument("--curriculum-thresholds", type=float, nargs="+", default=[35.0, 55.0, 75.0],
                        help="Rolling score promotion thresholds for 4 stages: Berlin -> +London -> +Tokyo -> +NYC (default: 35 55 75)")
    parser.add_argument("--curriculum-min-steps", type=int, nargs="+", default=[8000, 16000, 26000],
                        help="Minimum steps required before promoting each stage (default: 8000 16000 26000)")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Fine-tune from latest checkpoint")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Explicit pretrained model path")
    return parser.parse_args()

def run_training(args=None):
    if args is None:
        args = parse_args()

    # --------------------------------------------------------
    # TRAINING CONFIGURATION
    # --------------------------------------------------------

    num_envs = args.num_envs
    num_steps = 128
    total_timesteps = args.total_timesteps
    maps = args.maps
    map_mode = args.map_mode

    curriculum = CurriculumManager(
        enabled=args.curriculum,
        custom_maps=args.maps,
        thresholds=tuple(args.curriculum_thresholds),
        min_steps=tuple(args.curriculum_min_steps),
    )
    maps = curriculum.get_maps()
    map_mode = "mixed" if args.curriculum else args.map_mode
    map_weights = curriculum.get_weights() if args.curriculum else None

    # Number of environment transitions per update.
    rollout_size = num_envs * num_steps

    # Phase 3 Config
    num_minibatches = 8
    minibatch_size = rollout_size // num_minibatches
    update_epochs = 8

    # Ceiling instead of silently stopping below total_timesteps.
    num_updates = int(
        np.ceil(total_timesteps / rollout_size)
    )

    # CPU configuration.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(2)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon integrated graphics
        device = torch.device("mps")
        torch.set_num_threads(8)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        # Intel integrated graphics
        device = torch.device("xpu")
        torch.set_num_threads(8)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(8)

    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    print("=" * 70)
    print("MiniMetro Local Multi-Map PPO Training")
    print("=" * 70)
    print(f"Device           : {device}")
    print(f"Curriculum       : {curriculum.get_stage_name() if args.curriculum else 'Disabled'}")
    print(f"PBRS Shaping     : {'Enabled (Ng et al. 1999)' if args.pbrs else 'Disabled'}")
    print(f"Hierarchical     : {'Enabled' if args.hierarchical else 'Disabled'}")
    print(f"Maps             : {[MAP_NAMES.get(m, f'Map_{m}') for m in maps]} (IDs: {maps})")
    print(f"Map Strategy     : {map_mode.upper()}")
    print(f"Num environments : {num_envs}")
    print(f"Steps/update     : {num_steps}")
    print(f"Rollout size     : {rollout_size}")
    print(f"Target steps     : {total_timesteps}")
    print(f"Total updates    : {num_updates}")
    print("=" * 70)

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # TensorBoard logs.
    writer = SummaryWriter(
        log_dir=CHECKPOINT_DIR
    )

    # --------------------------------------------------------
    # ENVIRONMENTS
    # --------------------------------------------------------

    print("Creating vectorized environments...")
    env_fns = []
    if map_mode == "stratified":
        for i in range(num_envs):
            assigned_map = maps[i % len(maps)]
            env_fns.append(make_env(seed=i, map_id=assigned_map, flip_prob=0.5, use_pbrs=args.pbrs))
            print(f"  Worker {i:02d} -> {MAP_NAMES.get(assigned_map, f'Map_{assigned_map}')} (ID {assigned_map})")
    else:
        for i in range(num_envs):
            env_fns.append(make_env(seed=i, map_id=-1, map_pool=maps, map_weights=map_weights, flip_prob=0.5, use_pbrs=args.pbrs))
            print(f"  Worker {i:02d} -> Dynamic Curriculum over maps {maps}")

    envs = gym.vector.AsyncVectorEnv(
        env_fns,
        context='spawn'
    )
    
    print("Environments created.")

    # --------------------------------------------------------
    # DEVICE & MODEL
    # --------------------------------------------------------

    hidden_dim = args.hidden_dim
    model = MiniMetroActorCritic(hidden_dim=hidden_dim, use_hierarchical=args.hierarchical).to(device)
    agent = PPO(
        model,
        lr=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5
    )

    # --------------------------------------------------------
    # CHECKPOINT LOADING
    # --------------------------------------------------------

    start_update = 1
    global_step = 0

    latest_ckpt = find_latest_checkpoint(CHECKPOINT_DIR)

    if latest_ckpt:
        try:
            loaded_update, global_step = load_checkpoint(
                latest_ckpt,
                model,
                agent,
                device,
                curriculum=curriculum if args.curriculum else None,
            )
            start_update = loaded_update + 1
            print(f"[OK] Resumed from update {start_update - 1} | global_step={global_step}")
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint:\n{e}\nStarting a new training run.")
            start_update = 1
            global_step = 0
    else:
        print("No checkpoint found. Starting new training run.")

    # --------------------------------------------------------
    # TENSORS
    # --------------------------------------------------------

    obs = {
        k: torch.zeros(
            (num_steps, num_envs) + v.shape,
            dtype=torch.float32,
            device=device,
        )
        for k, v in envs.single_observation_space.items()
    }

    actions = torch.zeros(
        (num_steps, num_envs),
        dtype=torch.float32,
        device=device,
    )

    logprobs = torch.zeros(
        (num_steps, num_envs),
        dtype=torch.float32,
        device=device,
    )

    rewards = torch.zeros(
        (num_steps, num_envs),
        dtype=torch.float32,
        device=device,
    )

    dones = torch.zeros(
        (num_steps, num_envs),
        dtype=torch.float32,
        device=device,
    )

    values = torch.zeros(
        (num_steps, num_envs),
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # LSTM STATE STORAGE
    # --------------------------------------------------------

    lstm_hx = torch.zeros(
        (num_steps, num_envs, hidden_dim * 5),
        dtype=torch.float32,
        device=device,
    )

    lstm_cx = torch.zeros(
        (num_steps, num_envs, hidden_dim * 5),
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # INITIAL ENV RESET
    # --------------------------------------------------------

    next_obs, _ = envs.reset()

    next_obs_tensor = {
        k: torch.as_tensor(
            v,
            device=device
        )
        for k, v in next_obs.items()
    }

    # Make sure action mask is boolean.
    next_obs_tensor["action_mask"] = (
        next_obs_tensor["action_mask"].bool()
    )

    next_done = torch.zeros(
        num_envs,
        dtype=torch.float32,
        device=device,
    )

    next_lstm_state = (
        torch.zeros(1, num_envs, hidden_dim * 5, device=device),
        torch.zeros(1, num_envs, hidden_dim * 5, device=device)
    )

    start_time = time.time()
    recent_scores = collections.deque(maxlen=20)
    best_avg_score = -1.0
    best_model_path = os.path.join(CHECKPOINT_DIR, "model_best.pt")

    # --------------------------------------------------------
    # TRAINING EPOCHS
    # --------------------------------------------------------

    try:
        for update in range(start_update, num_updates + 1):
            update_start_time = time.time()
            print(f"\n⏳ Performing {update}/{num_updates}...")

            for step in range(num_steps):
                global_step += num_envs

                for k in obs.keys():
                    obs[k][step] = next_obs_tensor[k]

                dones[step] = next_done

                # --------------------------------------------
                # ACTION
                # --------------------------------------------
                
                lstm_hx[step].copy_(next_lstm_state[0].squeeze(0))
                lstm_cx[step].copy_(next_lstm_state[1].squeeze(0))

                with torch.no_grad():

                    mask = (
                        next_obs_tensor[
                            "action_mask"
                        ].bool()
                    )

                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        action, logprob, _, value, next_lstm_state = (
                            model.get_action_and_value(
                                next_obs_tensor,
                                lstm_state=next_lstm_state,
                                mask=mask,
                            )
                        )

                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob

                # --------------------------------------------
                # ENV STEP
                # --------------------------------------------

                next_obs, reward, terminated, truncated, infos = envs.step(
                    action.cpu().numpy()
                )

                done = np.logical_or(
                    terminated,
                    truncated
                )

                # --------------------------------------------
                # STORE REWARD & RESET LSTM
                # --------------------------------------------

                rewards[step].copy_(
                    torch.as_tensor(
                        reward,
                        dtype=torch.float32,
                        device=device,
                    ).view(-1)
                )

                done_mask = torch.tensor(
                    done,
                    dtype=torch.float32,
                    device=device,
                ).view(1, num_envs, 1)

                next_lstm_state = (
                    next_lstm_state[0] * (1.0 - done_mask),
                    next_lstm_state[1] * (1.0 - done_mask)
                )

                # --------------------------------------------
                # NEXT OBSERVATION
                # --------------------------------------------

                next_obs_tensor = {
                    k: torch.as_tensor(
                        v,
                        device=device,
                    )
                    for k, v in next_obs.items()
                }

                next_obs_tensor["action_mask"] = (
                    next_obs_tensor[
                        "action_mask"
                    ].bool()
                )

                next_done = torch.tensor(
                    done,
                    dtype=torch.float32,
                    device=device,
                )

                # --------------------------------------------
                # LOGGING EPISODE STATS
                # --------------------------------------------

                # Extract completed episode statistics (supports Gymnasium 1.0+ vectorized format and legacy final_info)
                completed_episodes = []
                if "_episode" in infos:
                    for idx, has_ep in enumerate(infos["_episode"]):
                        if has_ep:
                            completed_episodes.append({
                                "idx": idx,
                                "r": float(infos["episode"]["r"][idx]),
                                "l": int(infos["episode"]["l"][idx]),
                                "score": int(infos["score"][idx]) if "score" in infos else 0,
                                "map_name": str(infos["map_name"][idx]) if "map_name" in infos else MAP_NAMES.get(maps[idx % len(maps)], f"Map_{idx}"),
                                "breakdown": {k: float(v[idx]) for k, v in infos.get("episode_reward_breakdown", {}).items()} if "episode_reward_breakdown" in infos else {},
                                "total_track_length": float(infos["total_track_length"][idx]) if "total_track_length" in infos else 0.0,
                            })
                elif "final_info" in infos:
                    for idx, info in enumerate(infos["final_info"]):
                        if info and "episode" in info:
                            completed_episodes.append({
                                "idx": idx,
                                "r": float(np.asarray(info["episode"]["r"]).reshape(-1)[0]),
                                "l": int(np.asarray(info["episode"]["l"]).reshape(-1)[0]),
                                "score": int(info.get("score", 0)),
                                "map_name": str(info.get("map_name", MAP_NAMES.get(maps[idx % len(maps)], f"Map_{idx}"))),
                                "breakdown": info.get("episode_reward_breakdown", {}),
                                "total_track_length": float(info.get("total_track_length", 0.0)),
                            })

                for ep_data in completed_episodes:
                    idx = ep_data["idx"]
                    episode_return = ep_data["r"]
                    episode_length = ep_data["l"]
                    score = ep_data["score"]
                    map_name = ep_data["map_name"]
                    map_key = map_name.lower().replace(" ", "_")

                    print(f"[{map_name.upper()} | Env {idx:02d}] step={global_step} | Return={episode_return:.2f} | Score={score} | Length={episode_length}", flush=True)

                    writer.add_scalar("charts/episodic_return", episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)
                    writer.add_scalar(f"charts/episodic_return_{map_key}", episode_return, global_step)
                    writer.add_scalar(f"charts/episodic_length_{map_key}", episode_length, global_step)
                    writer.add_scalar(f"charts/score_{map_key}", score, global_step)
                    for channel, val in ep_data["breakdown"].items():
                        writer.add_scalar(f"rewards/{channel}", val, global_step)
                    if ep_data["total_track_length"] > 0:
                        writer.add_scalar("metrics/total_track_length", ep_data["total_track_length"], global_step)

                    # Track all-time best model based on rolling average score
                    recent_scores.append(score)
                    if len(recent_scores) >= 5:
                        current_avg = float(np.mean(recent_scores))
                        if current_avg > best_avg_score:
                            best_avg_score = current_avg
                            torch.save(model.state_dict(), best_model_path)
                            print(f"[BEST] New all-time best model! Rolling Avg Score: {best_avg_score:.1f} (Latest: {score}) -> Saved {best_model_path}", flush=True)
                            writer.add_scalar("charts/best_rolling_score", best_avg_score, global_step)

                        if args.curriculum and curriculum.update(best_avg_score, global_step):
                            envs.call("set_map_pool", curriculum.get_maps(), curriculum.get_weights())
                            writer.add_scalar("charts/curriculum_stage", curriculum.stage_num, global_step)

            # ------------------------------------------------
            # GAE
            # ------------------------------------------------

            with torch.no_grad():

                next_value = (
                    model.get_value(
                        next_obs_tensor,
                        lstm_state=next_lstm_state
                    )
                    .reshape(1, -1)
                )

                advantages, returns = (
                    agent.compute_gae(
                        rewards,
                        values,
                        next_value,
                        dones,
                        next_done,
                    )
                )

            # ------------------------------------------------
            # PREPARE BATCH
            # ------------------------------------------------

            b_obs = obs
            b_actions = actions
            b_logprobs = logprobs
            b_advantages = advantages
            b_returns = returns
            b_masks = b_obs["action_mask"].bool()
            b_values = values

            # ------------------------------------------------
            # NORMALIZATION & LR DECAY
            # ------------------------------------------------

            b_advantages = (
                b_advantages - b_advantages.mean()
            ) / (b_advantages.std() + 1e-8)

            frac = 1.0 - (update - 1.0) / num_updates
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = 3e-4 * frac
                
            # PHASE-3: Entropy Annealing
            agent.ent_coef = 0.05 * frac

            # ------------------------------------------------
            # PPO UPDATE
            # ------------------------------------------------

            pg_loss, v_loss, ent_loss, clipfrac, approx_kl = (
                agent.update(
                    b_obs,
                    b_actions,
                    b_logprobs,
                    b_advantages,
                    b_returns,
                    b_masks,
                    values=b_values,
                    init_lstm_hx=lstm_hx,
                    init_lstm_cx=lstm_cx,
                    update_epochs=update_epochs,
                    num_minibatches=num_minibatches,
                )
            )

            # Convert tensors/numpy scalars safely.
            pg_loss = float(
                np.asarray(
                    pg_loss
                )
            )

            v_loss = float(
                np.asarray(
                    v_loss
                )
            )

            ent_loss = float(
                np.asarray(
                    ent_loss
                )
            )

            clipfrac = float(
                np.asarray(
                    clipfrac
                )
            )

            approx_kl = float(
                np.asarray(
                    approx_kl
                )
            )

            # ------------------------------------------------
            # NaN DETECTION
            # ------------------------------------------------

            if not all(
                np.isfinite(x)
                for x in [
                    pg_loss,
                    v_loss,
                    ent_loss,
                    clipfrac,
                    approx_kl,
                ]
            ):

                print(
                    "[WARNING] "
                    "Non-finite PPO metric detected!",
                    flush=True
                )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            elapsed = time.time() - start_time

            sps = int(
                global_step / max(
                    elapsed,
                    1e-6
                )
            )

            update_time = (
                time.time()
                - update_start_time
            )

            writer.add_scalar(
                "losses/value_loss",
                v_loss,
                global_step,
            )

            writer.add_scalar(
                "losses/policy_loss",
                pg_loss,
                global_step,
            )

            writer.add_scalar(
                "losses/entropy",
                ent_loss,
                global_step,
            )

            writer.add_scalar(
                "losses/approx_kl",
                approx_kl,
                global_step,
            )

            writer.add_scalar(
                "losses/clipfrac",
                clipfrac,
                global_step,
            )

            writer.add_scalar(
                "charts/SPS",
                sps,
                global_step,
            )

            writer.add_scalar(
                "charts/update_time",
                update_time,
                global_step,
            )

            # P1-2: Track network expansion and station redundancy metrics
            exp_metrics = compute_expansion_metrics(b_obs, b_actions)
            writer.add_scalar(
                "charts/expansion_action_rate",
                exp_metrics.expansion_action_rate,
                global_step,
            )
            writer.add_scalar(
                "charts/expansion_ratio",
                exp_metrics.expansion_ratio,
                global_step,
            )
            writer.add_scalar(
                "charts/avg_lines_per_station",
                exp_metrics.lines_per_station_mean,
                global_step,
            )
            writer.add_scalar(
                "charts/redundant_station_rate",
                exp_metrics.redundant_station_rate,
                global_step,
            )

            writer.flush()

            # ------------------------------------------------
            # PRINT UPDATE
            # ------------------------------------------------

            print(
                f"Completed "
                f"{update}/{num_updates} | "
                f"steps={global_step} | "
                f"SPS={sps} | "
                f"v_loss={v_loss:.4f} | "
                f"pg_loss={pg_loss:.4f} | "
                f"entropy={ent_loss:.4f} | "
                f"KL={approx_kl:.6f} | "
                f"time={update_time:.2f}s",
                flush=True
            )

            # ------------------------------------------------
            # SAVE CHECKPOINT
            # ------------------------------------------------
            #
            # IMPORTANT:
            # Save EVERY update.
            #
            # This means if Kaggle kills the session after
            # 6 hours, at worst you lose the current update
            # rather than several hours of work.
            #

            save_checkpoint(
                model,
                agent,
                update,
                global_step,
                curriculum_state_dict=curriculum.state_dict() if args.curriculum else None,
                best_avg_score=best_avg_score,
            )

    except KeyboardInterrupt:

        print(
            "\nTraining interrupted by user.",
            flush=True
        )

        # Try to save the current model even if
        # interruption happens between checkpoints.
        try:

            emergency_path = os.path.join(
                CHECKPOINT_DIR,
                "checkpoint_interrupted.pt"
            )

            emergency_checkpoint = {
                "update": update
                if "update" in locals()
                else 0,

                "global_step": global_step,

                "model_state_dict":
                    model.state_dict(),

                "torch_rng_state":
                    torch.get_rng_state(),

                "numpy_rng_state":
                    np.random.get_state(),
            }

            if hasattr(
                agent,
                "optimizer"
            ):

                emergency_checkpoint[
                    "optimizer_state_dict"
                ] = (
                    agent.optimizer.state_dict()
                )

            torch.save(
                emergency_checkpoint,
                emergency_path
            )

            print(
                f"Emergency checkpoint saved: "
                f"{emergency_path}",
                flush=True
            )

        except Exception as e:

            print(
                f"[WARNING] Could not save emergency "
                f"checkpoint: {e}",
                flush=True
            )

    except Exception as e:

        print(
            "\nTRAINING FAILED",
            flush=True
        )

        print(
            f"Error: {type(e).__name__}: {e}",
            flush=True
        )

        # Save emergency checkpoint.
        try:

            emergency_path = os.path.join(
                CHECKPOINT_DIR,
                "checkpoint_error.pt"
            )

            emergency_checkpoint = {
                "update": update
                if "update" in locals()
                else 0,

                "global_step": global_step,

                "model_state_dict":
                    model.state_dict(),

                "torch_rng_state":
                    torch.get_rng_state(),

                "numpy_rng_state":
                    np.random.get_state(),
            }

            if hasattr(
                agent,
                "optimizer"
            ):

                emergency_checkpoint[
                    "optimizer_state_dict"
                ] = (
                    agent.optimizer.state_dict()
                )

            torch.save(
                emergency_checkpoint,
                emergency_path
            )

            print(
                f"Emergency checkpoint saved: "
                f"{emergency_path}",
                flush=True
            )

        except Exception as save_error:

            print(
                f"[WARNING] Could not save emergency "
                f"checkpoint: {save_error}",
                flush=True
            )

        # Re-raise so the actual traceback is visible.
        raise

    finally:

        # ----------------------------------------------------
        # FINAL MODEL
        # ----------------------------------------------------

        try:

            final_model_path = os.path.join(
                CHECKPOINT_DIR,
                "model_final.pt"
            )

            torch.save(
                model.state_dict(),
                final_model_path
            )

            print(
                f"Final model saved: "
                f"{final_model_path}",
                flush=True
            )

        except Exception as e:

            print(
                f"[WARNING] Could not save final model: {e}",
                flush=True
            )

        # ----------------------------------------------------
        # CLOSE ENVIRONMENTS
        # ----------------------------------------------------

        try:
            envs.close()
        except Exception:
            pass

        # ----------------------------------------------------
        # CLOSE TENSORBOARD
        # ----------------------------------------------------

        try:
            writer.close()
        except Exception:
            pass

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(
        f"Total environment steps: {global_step}"
    )
    print(
        f"Final model: "
        f"{os.path.join(CHECKPOINT_DIR, 'model_final.pt')}"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_training()
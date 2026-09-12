import sys
import os
try:
    import gymnasium as gym
except ImportError:
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
    if os.path.exists(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
        os.execv(venv_python, [venv_python] + sys.argv)

import time
import glob

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


# ============================================================
# ENVIRONMENT
# ============================================================

def make_env(seed, map_id=0, flip_prob=0.5):
    def thunk():
        env = MiniMetroEnv(
            map_id=map_id,
            seed=seed
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
    }

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
        f"💾 Checkpoint saved: {checkpoint_path}",
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
):
    """
    Load model/optimizer/RNG state.

    Returns:
        update,
        global_step
    """

    print(
        f"🔄 Loading checkpoint: {checkpoint_path}",
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
                    f"⚠️ Could not restore optimizer state ({opt_err}). "
                    f"Architecture parameters changed — using reinitialized optimizer."
                )

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
        f"✅ Resumed from update {update} "
        f"| global_step={global_step}",
        flush=True
    )

    return update, global_step


# ============================================================
# MAIN TRAINING
# ============================================================

def run_training():

    # --------------------------------------------------------
    # TRAINING CONFIGURATION
    # --------------------------------------------------------

    num_envs = 16
    num_steps = 128

    total_timesteps = 40_000

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
    print("MiniMetro PPO Training")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Num environments : {num_envs}")
    print(f"Steps/update    : {num_steps}")
    print(f"Rollout size    : {rollout_size}")
    print(f"Target steps    : {total_timesteps}")
    print(f"Total updates   : {num_updates}")
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
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(i)
            for i in range(num_envs)
        ],
        context='spawn'
    )
    
    print("✅ Environments created.")

    # --------------------------------------------------------
    # DEVICE & MODEL
    # --------------------------------------------------------

    model = MiniMetroActorCritic(hidden_dim=32).to(device)
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
            )
            start_update = loaded_update + 1
            print(f"✅ Resumed from update {start_update - 1} | global_step={global_step}")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint:\n{e}\nStarting a new training run.")
            start_update = 1
            global_step = 0
    else:
        print("🆕 No checkpoint found. Starting new training run.")

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

    hidden_dim = 32
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

                if "final_info" in infos:
                    for idx, info in enumerate(infos["final_info"]):
                        if info and "episode" in info:
                            episode_return = info["episode"]["r"]
                            episode_length = info["episode"]["l"]

                            # Convert possible numpy scalars to Python values.
                            try:
                                episode_return = float(np.asarray(episode_return).reshape(-1)[0])
                            except Exception:
                                pass

                            try:
                                episode_length = int(np.asarray(episode_length).reshape(-1)[0])
                            except Exception:
                                pass

                            print(f"global_step={global_step}, env={idx}, episodic_return={episode_return:.3f}, length={episode_length}", flush=True)

                            writer.add_scalar("charts/episodic_return", episode_return, global_step)
                            writer.add_scalar("charts/episodic_length", episode_length, global_step)
                            if "episode_reward_breakdown" in info:
                                for channel, val in info["episode_reward_breakdown"].items():
                                    writer.add_scalar(f"rewards/{channel}", val, global_step)
                            if "total_track_length" in info:
                                writer.add_scalar("metrics/total_track_length", info["total_track_length"], global_step)

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
                    "⚠️ WARNING: "
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
                f"✅ Completed "
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
            )

    except KeyboardInterrupt:

        print(
            "\n🛑 Training interrupted by user.",
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
                f"💾 Emergency checkpoint saved: "
                f"{emergency_path}",
                flush=True
            )

        except Exception as e:

            print(
                f"⚠️ Could not save emergency "
                f"checkpoint: {e}",
                flush=True
            )

    except Exception as e:

        print(
            "\n❌ TRAINING FAILED",
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
                f"💾 Emergency checkpoint saved: "
                f"{emergency_path}",
                flush=True
            )

        except Exception as save_error:

            print(
                f"⚠️ Could not save emergency "
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
                f"💾 Final model saved: "
                f"{final_model_path}",
                flush=True
            )

        except Exception as e:

            print(
                f"⚠️ Could not save final model: {e}",
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
    print("✅ TRAINING FINISHED")
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
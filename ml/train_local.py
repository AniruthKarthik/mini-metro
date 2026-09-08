import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import glob
import numpy as np
import torch
import gymnasium as gym
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass
from torch.utils.tensorboard import SummaryWriter

from env import MiniMetroEnv
from model import MiniMetroActorCritic
from ppo import PPO

CHECKPOINT_DIR = "runs/minimetro_ppo_local"


def make_env(seed, map_id=0):
    def thunk():
        env = MiniMetroEnv(map_id=map_id, seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def find_latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    """Find the latest checkpoint file in checkpoint_dir."""
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    checkpoints.sort(key=os.path.getmtime)
    return checkpoints[-1]


def run_training():
    num_envs = int(os.environ.get("NUM_ENVS", "16"))
    num_steps = 512          # Longer rollout for credit assignment
    total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", "10000000"))
    batch_size = num_envs * num_steps
    num_minibatches = 4
    minibatch_size = batch_size // num_minibatches
    update_epochs = 4        # 4 gradient steps per rollout
    num_updates = total_timesteps // batch_size

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(CHECKPOINT_DIR)

    # Use AsyncVectorEnv with 'spawn' to prevent Go CGO runtime crashes on fork
    envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(num_envs)], context='spawn')
    torch.set_num_threads(8)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            if major >= 7:
                device = torch.device("cuda")
                torch.set_float32_matmul_precision('high')
            else:
                print(f"⚠️ Warning: GPU capability sm_{major}{minor} unsupported. Defaulting to CPU.", flush=True)
        except Exception as e:
            print(f"⚠️ GPU check error: {e}. Defaulting to CPU.", flush=True)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")

    print("=" * 70)
    print("MiniMetro PPO Local Training (Parity with train.py)")
    print("=" * 70)
    print(f"Device           : {device}")
    print(f"Hidden dim       : 256 (matches train.py)")
    print(f"Num environments : {num_envs}")
    print(f"Steps/update     : {num_steps}")
    print(f"Rollout size     : {batch_size}")
    print(f"Minibatch size   : {minibatch_size} ({num_minibatches} minibatches)")
    print(f"Update epochs    : {update_epochs}")
    print(f"Target steps     : {total_timesteps}")
    print(f"Total updates    : {num_updates}")
    print("=" * 70, flush=True)

    base_model = MiniMetroActorCritic(hidden_dim=256).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"🔥 Enabling DataParallel across {torch.cuda.device_count()} GPUs!", flush=True)
        model = torch.nn.DataParallel(base_model)
    else:
        model = base_model

    raw_model = base_model
    agent = PPO(model)

    start_update = 1
    global_step = 0

    # Resume from latest checkpoint if available
    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt is not None:
        try:
            ckpt_data = torch.load(latest_ckpt, map_location=device, weights_only=False)
            # Verify hidden_dim matches 256
            sd = ckpt_data.get("model_state_dict", ckpt_data)
            if sd.get("gcn1.node_proj.weight", torch.empty(0)).shape[0] == 256:
                raw_model.load_state_dict(sd)
                if "optimizer_state_dict" in ckpt_data and hasattr(agent, "optimizer"):
                    agent.optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
                start_update = ckpt_data.get("update", 0) + 1
                global_step = ckpt_data.get("global_step", (start_update - 1) * batch_size)
                print(f"🔄 Resumed from checkpoint: {latest_ckpt} (update={start_update}, step={global_step})", flush=True)
            else:
                print(f"ℹ️ Found previous checkpoint with different dims. Initializing fresh 256-dim model.", flush=True)
        except Exception as e:
            print(f"⚠️ Could not resume from checkpoint {latest_ckpt}: {e}. Starting fresh.", flush=True)

    obs = {k: torch.zeros((num_steps, num_envs) + v.shape).to(device) for k, v in envs.single_observation_space.items()}
    actions = torch.zeros((num_steps, num_envs)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    start_time = time.time()

    next_obs, _ = envs.reset()
    next_obs_tensor = {k: torch.as_tensor(v, device=device) for k, v in next_obs.items()}
    next_done = torch.zeros(num_envs).to(device)

    try:
        for update in range(start_update, num_updates + 1):
            update_start_time = time.time()
            print(f"\n⏳ Performing {update}/{num_updates}...", flush=True)

            if update % 10 == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{update:05d}.pt")
                model_path = os.path.join(CHECKPOINT_DIR, f"model_{update}.pt")
                torch.save(raw_model.state_dict(), model_path)
                torch.save({
                    "update": update,
                    "global_step": global_step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                }, ckpt_path)
                print(f"💾 Checkpoint saved: {ckpt_path}", flush=True)

            for step in range(num_steps):
                global_step += num_envs

                for k in obs.keys():
                    obs[k][step] = next_obs_tensor[k]
                dones[step] = next_done

                with torch.no_grad():
                    mask = next_obs_tensor["action_mask"].bool()
                    action, logprob, _, value = raw_model.get_action_and_value(next_obs_tensor, mask=mask)
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)

                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs_tensor = {k: torch.as_tensor(v, device=device) for k, v in next_obs.items()}
                next_done = torch.tensor(done, dtype=torch.float32).to(device)

                if "final_info" in infos:
                    for idx, info in enumerate(infos["final_info"]):
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}, length={info['episode']['l']}", flush=True)
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            with torch.no_grad():
                next_value = raw_model.get_value(next_obs_tensor).reshape(1, -1)
                advantages, returns = agent.compute_gae(rewards, values, next_value, dones, next_done)

            b_obs = {k: v.reshape((-1,) + envs.single_observation_space[k].shape) for k, v in obs.items()}
            b_actions = actions.reshape(-1)
            b_logprobs = logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_masks = b_obs["action_mask"].bool()

            # PHASE-1 fix PPO-1: normalize over FULL batch, not per-minibatch (matches train.py)
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # PHASE-1 fix PPO-4: linear LR decay toward 0 over training (matches train.py)
            frac = 1.0 - (update - 1.0) / num_updates
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = 3e-4 * frac

            pg_loss, v_loss, ent_loss, clipfrac, approx_kl = agent.update(
                b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_masks,
                b_values=b_values, update_epochs=update_epochs, num_minibatches=num_minibatches
            )

            writer.add_scalar("losses/value_loss", v_loss, global_step)
            writer.add_scalar("losses/policy_loss", pg_loss, global_step)
            writer.add_scalar("losses/entropy", ent_loss, global_step)
            writer.add_scalar("losses/approx_kl", approx_kl, global_step)
            writer.add_scalar("losses/clipfrac", clipfrac, global_step)
            writer.add_scalar("charts/SPS", int(global_step / max(time.time() - start_time, 1e-6)), global_step)
            writer.add_scalar("charts/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step)

            # Diagnostic: monitor NoOp rate (matches train.py)
            noop_rate = (b_actions == 0).float().mean().item()
            writer.add_scalar("charts/noop_rate", noop_rate, global_step)

            update_time = time.time() - update_start_time
            print(f"✅ Completed {update}/{num_updates} | steps={global_step} | SPS={int(global_step / max(time.time() - start_time, 1e-6))} | v_loss={v_loss:.4f} | pg_loss={pg_loss:.4f} | entropy={ent_loss:.4f} | KL={approx_kl:.6f} | time={update_time:.2f}s", flush=True)

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    finally:
        final_model_path = os.path.join(CHECKPOINT_DIR, "model_final.pt")
        torch.save(raw_model.state_dict(), final_model_path)
        print(f"💾 Final model saved: {final_model_path}", flush=True)
        envs.close()
        writer.close()

    print("=" * 70)
    print("✅ TRAINING FINISHED")
    print(f"Total environment steps: {global_step}")
    print("=" * 70)


if __name__ == "__main__":
    run_training()
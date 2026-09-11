import sys
import os
try:
    import gymnasium as gym
except ImportError:
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
    if os.path.exists(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
        os.execv(venv_python, [venv_python] + sys.argv)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import time

from env import MiniMetroEnv
from model import MiniMetroActorCritic
from ppo import PPO
from probing import compute_expansion_metrics

def make_env(seed, map_id=-1):
    def thunk():
        env = MiniMetroEnv(map_id=map_id, seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def find_latest_checkpoint(checkpoint_dir="runs/minimetro_ppo"):
    if not os.path.exists(checkpoint_dir):
        return None, 0
    files = [f for f in os.listdir(checkpoint_dir) if (f.startswith("checkpoint_") or f.startswith("model_")) and f.endswith(".pt")]
    
    def get_update_num(f):
        base = f.replace("checkpoint_", "").replace("model_", "").replace(".pt", "")
        try:
            return int(base)
        except ValueError:
            return -1

    files = sorted([f for f in files if get_update_num(f) >= 0], key=get_update_num)
    if not files:
        return None, 0
    
    latest_file = files[-1]
    return os.path.join(checkpoint_dir, latest_file), get_update_num(latest_file)

def cleanup_old_checkpoints(checkpoint_dir="runs/minimetro_ppo", keep_last=5):
    if not os.path.exists(checkpoint_dir):
        return
    files = [f for f in os.listdir(checkpoint_dir) if (f.startswith("checkpoint_") or f.startswith("model_")) and f.endswith(".pt")]
    
    def get_update_num(f):
        base = f.replace("checkpoint_", "").replace("model_", "").replace(".pt", "")
        try:
            return int(base)
        except ValueError:
            return -1

    files = sorted([f for f in files if get_update_num(f) >= 0], key=get_update_num)
    if len(files) > keep_last:
        for old_file in files[:-keep_last]:
            old_path = os.path.join(checkpoint_dir, old_file)
            try:
                os.remove(old_path)
                print(f"🗑️ Cleaned up old checkpoint: {old_file}", flush=True)
            except Exception:
                pass

def run_training():
    cpu_cores = os.cpu_count() or 2
    num_envs = int(os.environ.get("NUM_ENVS", min(16, max(4, cpu_cores))))
    total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", 5000000))
    target_rollout_size = int(os.environ.get("TARGET_ROLLOUT_SIZE", 2048))
    num_steps = int(os.environ.get("NUM_STEPS", target_rollout_size // num_envs))
    batch_size = num_envs * num_steps
    num_minibatches = int(os.environ.get("NUM_MINIBATCHES", 4))
    minibatch_size = batch_size // num_minibatches
    update_epochs = int(os.environ.get("UPDATE_EPOCHS", 2))
    target_kl = float(os.environ.get("TARGET_KL", 0.015))
    num_updates = total_timesteps // batch_size
    
    os.makedirs("runs/minimetro_ppo", exist_ok=True)
    writer = SummaryWriter("runs/minimetro_ppo")
    
    # Use AsyncVectorEnv with 'spawn' and shared_memory=False to prevent Python 3.13 BufferError and POSIX semaphore leaks
    envs = gym.vector.AsyncVectorEnv(
        [make_env(i) for i in range(num_envs)],
        context='spawn',
        shared_memory=False
    )
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            if major >= 7:
                device = torch.device("cuda")
                # Enable TensorFloat-32 (TF32) for massive speedups on Ampere/Ada/Blackwell GPUs
                torch.set_float32_matmul_precision('high')
                torch.backends.cudnn.benchmark = True
                # Set 2 CPU threads for PyTorch when using GPU to avoid CPU contention with env worker processes
                torch.set_num_threads(2)
            else:
                torch.set_num_threads(min(4, max(1, cpu_cores)))
                print(f"⚠️ Warning: GPU {torch.cuda.get_device_name(0)} has CUDA capability sm_{major}{minor}, which is not supported by PyTorch 2.x wheels. Falling back to CPU.", flush=True)
        except Exception as e:
            torch.set_num_threads(min(4, max(1, cpu_cores)))
            print(f"⚠️ GPU check error: {e}. Defaulting to CPU.", flush=True)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_num_threads(min(4, max(1, cpu_cores)))
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        torch.set_num_threads(min(4, max(1, cpu_cores)))
    else:
        torch.set_num_threads(min(4, max(1, cpu_cores)))
            
    print("=" * 70)
    print("MiniMetro PPO Training")
    print("=" * 70)
    print(f"Device          : {device}")
    if device.type == "cuda":
        print(f"GPU Model       : {torch.cuda.get_device_name(0)}")
        print(f"GPU VRAM        : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        use_bf16 = torch.cuda.is_bf16_supported()
        print(f"AMP Precision   : {'bfloat16' if use_bf16 else 'float16'}")
    print(f"Num environments: {num_envs}")
    print(f"Steps/update    : {num_steps}")
    print(f"Rollout size    : {batch_size}")
    print(f"Minibatch size  : {minibatch_size} ({num_minibatches} minibatches)")
    print(f"Update epochs   : {update_epochs}")
    print(f"Target steps    : {total_timesteps}")
    print(f"Total updates   : {num_updates}")
    print("=" * 70, flush=True)
    
    hidden_dim = int(os.environ.get("HIDDEN_DIM", 256))
    base_model = MiniMetroActorCritic(hidden_dim=hidden_dim).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"🔥 Enabling DataParallel across {torch.cuda.device_count()} GPUs!", flush=True)
        model = torch.nn.DataParallel(base_model)
    else:
        model = base_model
        
    raw_model = base_model
    agent = PPO(model)
    
    start_update = 1
    global_step = 0
    latest_ckpt_path, latest_update = find_latest_checkpoint("runs/minimetro_ppo")
    if latest_ckpt_path and latest_update > 0:
        try:
            print(f"🔄 Loading existing checkpoint: {latest_ckpt_path}", flush=True)
            checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                raw_model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint and hasattr(agent, "optimizer"):
                    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_update = checkpoint.get("update", latest_update) + 1
                global_step = checkpoint.get("global_step", (start_update - 1) * batch_size)
            elif isinstance(checkpoint, dict):
                raw_model.load_state_dict(checkpoint)
                start_update = latest_update + 1
                global_step = (start_update - 1) * batch_size
            print(f"✅ Resuming training from update {start_update}/{num_updates} | global_step={global_step}", flush=True)
        except Exception as e:
            print(f"⚠️ Could not load checkpoint {latest_ckpt_path}: {e}. Starting fresh.", flush=True)
            start_update = 1
            global_step = 0

    # Store rollout buffer on CPU to prevent static VRAM allocation overhead
    cpu_device = torch.device("cpu")
    obs = {k: torch.zeros((num_steps, num_envs) + v.shape, device=cpu_device) for k, v in envs.single_observation_space.items()}
    actions = torch.zeros((num_steps, num_envs), device=cpu_device)
    logprobs = torch.zeros((num_steps, num_envs), device=cpu_device)
    rewards = torch.zeros((num_steps, num_envs), device=cpu_device)
    dones = torch.zeros((num_steps, num_envs), device=cpu_device)
    values = torch.zeros((num_steps, num_envs), device=cpu_device)
    
    lstm_hx = torch.zeros((num_steps, num_envs, hidden_dim * 5), device=cpu_device)
    lstm_cx = torch.zeros((num_steps, num_envs, hidden_dim * 5), device=cpu_device)
    
    start_time = time.time()
    
    next_obs, _ = envs.reset()
    next_obs_tensor = {k: torch.as_tensor(v).to(device, non_blocking=True) for k, v in next_obs.items()}
    next_done = torch.zeros(num_envs, device=cpu_device)
    next_lstm_state = (torch.zeros(1, num_envs, hidden_dim * 5, device=device),
                       torch.zeros(1, num_envs, hidden_dim * 5, device=device))
    
    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    for update in range(start_update, num_updates + 1):
        update_start_time = time.time()
        print(f"\n⏳ Performing {update}/{num_updates}...", flush=True)
        
        if update % 10 == 0:
            checkpoint_path = f"runs/minimetro_ppo/checkpoint_{update:05d}.pt"
            checkpoint_data = {
                "update": update,
                "global_step": global_step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
            }
            torch.save(checkpoint_data, checkpoint_path)
            cleanup_old_checkpoints("runs/minimetro_ppo", keep_last=5)

        for step in range(num_steps):
            global_step += num_envs
            
            for k in obs.keys():
                obs[k][step] = next_obs_tensor[k].to(cpu_device)
            dones[step] = next_done
            
            lstm_hx[step] = next_lstm_state[0].squeeze(0).to(cpu_device)
            lstm_cx[step] = next_lstm_state[1].squeeze(0).to(cpu_device)
            
            with torch.no_grad():
                mask = next_obs_tensor["action_mask"].bool()
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    action, logprob, _, value, next_lstm_state = raw_model.get_action_and_value(
                        next_obs_tensor, lstm_state=next_lstm_state, mask=mask
                    )
                values[step] = value.flatten().to(cpu_device)
            
            actions[step] = action.to(cpu_device)
            logprobs[step] = logprob.to(cpu_device)
            
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            # Reset LSTM state for done envs
            done_mask = torch.tensor(done, dtype=torch.float32, device=device).view(1, num_envs, 1)
            next_lstm_state = (
                next_lstm_state[0] * (1.0 - done_mask),
                next_lstm_state[1] * (1.0 - done_mask)
            )
            
            rewards[step] = torch.tensor(reward, dtype=torch.float32).view(-1)
            next_obs_tensor = {k: torch.as_tensor(v).to(device, non_blocking=True) for k, v in next_obs.items()}
            next_done = torch.tensor(done, dtype=torch.float32, device=cpu_device)
            
            if "final_info" in infos:
                for idx, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}, length={info['episode']['l']}", flush=True)
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                next_value = raw_model.get_value(next_obs_tensor, lstm_state=next_lstm_state).reshape(1, -1).to(cpu_device)
            advantages, returns = agent.compute_gae(rewards, values, next_value, dones, next_done)
            advantages, returns = agent.compute_gae(rewards, values, next_value, dones, next_done)
            
        b_obs = obs
        b_actions = actions
        b_logprobs = logprobs
        b_advantages = advantages
        b_returns = returns
        b_values = values
        b_masks = b_obs["action_mask"].bool()

        # PHASE-1 fix PPO-1: normalize over FULL batch, not per-minibatch.
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PHASE-1 fix PPO-4: linear LR decay toward 0 over training.
        frac = 1.0 - (update - 1.0) / num_updates
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = 3e-4 * frac
            
        # PHASE-3: Entropy Annealing
        agent.ent_coef = 0.05 * frac
        
        pg_loss, v_loss, ent_loss, clipfrac, approx_kl = agent.update(
            b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_masks,
            values=b_values, init_lstm_hx=lstm_hx, init_lstm_cx=lstm_cx, update_epochs=update_epochs, num_minibatches=num_minibatches, target_kl=target_kl
        )
        
        writer.add_scalar("losses/value_loss", v_loss, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss, global_step)
        writer.add_scalar("losses/entropy", ent_loss, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", clipfrac, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("charts/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step)

        # PHASE-1 diagnostic: monitor NoOp rate (high = agent learned passivity).
        noop_rate = (b_actions == 0).float().mean().item()
        writer.add_scalar("charts/noop_rate", noop_rate, global_step)

        # P1-2: Track network expansion and station redundancy metrics
        exp_metrics = compute_expansion_metrics(b_obs, b_actions)
        writer.add_scalar("charts/expansion_action_rate", exp_metrics.expansion_action_rate, global_step)
        writer.add_scalar("charts/expansion_ratio", exp_metrics.expansion_ratio, global_step)
        writer.add_scalar("charts/avg_lines_per_station", exp_metrics.lines_per_station_mean, global_step)
        writer.add_scalar("charts/redundant_station_rate", exp_metrics.redundant_station_rate, global_step)
        
        update_time = time.time() - update_start_time
        print(f"✅ Completed {update}/{num_updates} | steps={global_step} | SPS={int(global_step / max(time.time() - start_time, 1e-6))} | v_loss={v_loss:.4f} | pg_loss={pg_loss:.4f} | entropy={ent_loss:.4f} | KL={approx_kl:.6f} | time={update_time:.2f}s", flush=True)
        
    envs.close()
    writer.close()
    torch.save(raw_model.state_dict(), "runs/minimetro_ppo/model_final.pt")
    print("\n🎉 Training finished! Saved final model to runs/minimetro_ppo/model_final.pt", flush=True)

if __name__ == "__main__":
    run_training()

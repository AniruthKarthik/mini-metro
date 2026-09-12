import sys
import os
try:
    import gymnasium as gym
except ImportError:
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
    if os.path.exists(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
        os.execv(venv_python, [venv_python] + sys.argv)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time
import numpy as np
import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from env import MiniMetroEnv, LineOrientationAugmentation
from model import MiniMetroActorCritic
from ppo import PPO
from probing import compute_expansion_metrics

MAP_NAMES = {
    0: "London",
    1: "New York City",
    2: "Tokyo",
    3: "Berlin",
}

def make_env(seed, map_id=-1, map_pool=None, map_weights=None, flip_prob=0.5):
    def thunk():
        env = MiniMetroEnv(
            map_id=map_id,
            seed=seed,
            map_pool=map_pool,
            map_weights=map_weights
        )
        if flip_prob > 0:
            env = LineOrientationAugmentation(env, flip_prob=flip_prob)
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
        if "model_final.pt" in os.listdir(checkpoint_dir):
            return os.path.join(checkpoint_dir, "model_final.pt"), 0
        return None, 0
    
    latest_file = files[-1]
    return os.path.join(checkpoint_dir, latest_file), get_update_num(latest_file)

def cleanup_old_checkpoints(checkpoint_dir="runs/minimetro_ppo", keep_last=5):
    if not os.path.exists(checkpoint_dir):
        return
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    
    def get_update_num(f):
        base = f.replace("checkpoint_", "").replace(".pt", "")
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

def parse_args():
    parser = argparse.ArgumentParser(description="Mini Metro Multi-Map PPO Training & Fine-Tuning")
    parser.add_argument("--maps", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="List of map IDs to train on: 0=London, 1=NYC, 2=Tokyo, 3=Berlin (default: [0, 1, 2, 3])")
    parser.add_argument("--map-mode", type=str, choices=["stratified", "mixed"], default="stratified",
                        help="Worker map allocation: 'stratified' (round-robin per worker) or 'mixed' (sampled on reset)")
    parser.add_argument("--map-weights", type=float, nargs="+", default=None,
                        help="Relative sampling weights for maps in 'mixed' mode (e.g., 1 1 1 2 to emphasize Berlin)")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Fine-tune starting from the latest existing model checkpoint")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Explicit path to pretrained checkpoint .pt file to fine-tune from")
    parser.add_argument("--lr", type=float, default=None,
                        help="Base learning rate (default: 3e-4 for scratch, 1e-4 for fine-tuning)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total timesteps to train (default: 5,000,000 scratch, 500,000 fine-tune)")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments (default: auto based on CPU cores)")
    parser.add_argument("--target-rollout-size", type=int, default=None,
                        help="Target rollout size across all envs (default: 2048)")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Rollout steps per environment")
    parser.add_argument("--num-minibatches", type=int, default=None,
                        help="Number of minibatches per PPO update (default: 4)")
    parser.add_argument("--update-epochs", type=int, default=None,
                        help="PPO update epochs per rollout (default: 2)")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="PPO target KL divergence threshold (default: 0.015)")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Model hidden dimension (default: 256, or introspected from checkpoint)")
    parser.add_argument("--checkpoint-dir", type=str, default="runs/minimetro_ppo",
                        help="Directory to save checkpoints (default: runs/minimetro_ppo)")
    return parser.parse_args()

def run_training(args=None):
    if args is None:
        args = parse_args()

    cpu_cores = os.cpu_count() or 2
    num_envs = args.num_envs or int(os.environ.get("NUM_ENVS", min(16, max(4, cpu_cores))))
    
    # Determine timesteps: default 500,000 for fine-tuning, 5,000,000 for scratch
    if args.total_timesteps is not None:
        total_timesteps = args.total_timesteps
    elif "TOTAL_TIMESTEPS" in os.environ:
        total_timesteps = int(os.environ["TOTAL_TIMESTEPS"])
    elif args.fine_tune or args.pretrained:
        total_timesteps = 500000
    else:
        total_timesteps = 5000000

    target_rollout_size = args.target_rollout_size or int(os.environ.get("TARGET_ROLLOUT_SIZE", 2048))
    num_steps = args.num_steps or int(os.environ.get("NUM_STEPS", max(16, target_rollout_size // num_envs)))
    batch_size = num_envs * num_steps
    num_minibatches = args.num_minibatches or int(os.environ.get("NUM_MINIBATCHES", 4))
    minibatch_size = batch_size // num_minibatches
    update_epochs = args.update_epochs or int(os.environ.get("UPDATE_EPOCHS", 2))
    target_kl = args.target_kl or float(os.environ.get("TARGET_KL", 0.015))
    num_updates = max(1, total_timesteps // batch_size)

    # Base learning rate
    if args.lr is not None:
        base_lr = args.lr
    elif args.fine_tune or args.pretrained:
        base_lr = 1e-4
    else:
        base_lr = 3e-4

    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(checkpoint_dir)

    # Construct multi-map environments
    maps = args.maps
    map_mode = args.map_mode
    map_weights = args.map_weights

    print("=" * 70)
    print("🚇 MiniMetro Multi-Map PPO Training & Fine-Tuning")
    print("=" * 70)
    print(f"Mode            : {'Fine-Tuning' if (args.fine_tune or args.pretrained) else 'Standard Training'}")
    print(f"Maps            : {[MAP_NAMES.get(m, f'Map_{m}') for m in maps]} (IDs: {maps})")
    print(f"Map Strategy    : {map_mode.upper()}")
    if map_mode == "mixed" and map_weights:
        print(f"Map Weights     : {map_weights}")

    env_fns = []
    if map_mode == "stratified":
        for i in range(num_envs):
            assigned_map = maps[i % len(maps)]
            env_fns.append(make_env(seed=i, map_id=assigned_map, flip_prob=0.5))
            print(f"  Worker {i:02d} -> {MAP_NAMES.get(assigned_map, f'Map_{assigned_map}')} (ID {assigned_map})")
    else:
        for i in range(num_envs):
            env_fns.append(make_env(seed=i, map_id=-1, map_pool=maps, map_weights=map_weights, flip_prob=0.5))
            print(f"  Worker {i:02d} -> Dynamic Curriculum over maps {maps}")

    # Use AsyncVectorEnv with 'spawn' and shared_memory=False to prevent Python buffer and POSIX semaphore leaks
    envs = gym.vector.AsyncVectorEnv(
        env_fns,
        context='spawn',
        shared_memory=False
    )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            if major >= 7:
                device = torch.device("cuda")
                torch.set_float32_matmul_precision('high')
                torch.backends.cudnn.benchmark = True
                torch.set_num_threads(2)
            else:
                torch.set_num_threads(min(4, max(1, cpu_cores)))
                print(f"⚠️ Warning: GPU {torch.cuda.get_device_name(0)} capability sm_{major}{minor} not supported. Using CPU.", flush=True)
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
    print(f"Base LR         : {base_lr}")
    print(f"Update epochs   : {update_epochs}")
    print(f"Target steps    : {total_timesteps}")
    print(f"Total updates   : {num_updates}")
    print("=" * 70, flush=True)

    # Checkpoint resolution
    ckpt_to_load = None
    if args.pretrained:
        ckpt_to_load = args.pretrained
    elif args.fine_tune:
        search_dirs = [checkpoint_dir, "runs/minimetro_ppo", "runs/minimetro_ppo_local"]
        for d in search_dirs:
            f, _ = find_latest_checkpoint(d)
            if f and os.path.exists(f):
                ckpt_to_load = f
                break
    else:
        latest_ckpt_path, latest_update = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt_path:
            ckpt_to_load = latest_ckpt_path

    # Introspect architecture dimension if checkpoint exists
    hidden_dim = args.hidden_dim or int(os.environ.get("HIDDEN_DIM", 0))
    checkpoint_data = None
    if ckpt_to_load and os.path.exists(ckpt_to_load):
        try:
            print(f"🔄 Inspecting checkpoint: {ckpt_to_load}", flush=True)
            checkpoint_data = torch.load(ckpt_to_load, map_location=device, weights_only=False)
            state_dict = checkpoint_data.get("model_state_dict", checkpoint_data) if isinstance(checkpoint_data, dict) else checkpoint_data
            
            node_w = state_dict.get("gcn1.node_proj.weight", state_dict.get("gatv2_1.node_proj.weight", None))
            if node_w is not None and hidden_dim <= 0:
                hidden_dim = int(node_w.shape[0])
                print(f"🔍 Introspected hidden_dim={hidden_dim} from checkpoint weights.")
        except Exception as e:
            print(f"⚠️ Error reading checkpoint {ckpt_to_load}: {e}")

    if hidden_dim <= 0:
        hidden_dim = 256
    print(f"Model hidden_dim: {hidden_dim}", flush=True)

    base_model = MiniMetroActorCritic(hidden_dim=hidden_dim).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"🔥 Enabling DataParallel across {torch.cuda.device_count()} GPUs!", flush=True)
        model = torch.nn.DataParallel(base_model)
    else:
        model = base_model
        
    raw_model = base_model
    agent = PPO(model, lr=base_lr)

    start_update = 1
    global_step = 0

    if checkpoint_data is not None:
        try:
            state_dict = checkpoint_data.get("model_state_dict", checkpoint_data) if isinstance(checkpoint_data, dict) else checkpoint_data
            raw_model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded model weights from {ckpt_to_load}", flush=True)

            if args.fine_tune or args.pretrained:
                # Fine-tuning: fresh start on updates/optimizer for clean adaptation
                start_update = 1
                global_step = 0
                print(f"🎯 Fine-Tuning Active: Reinitialized optimizer with lr={base_lr} for multi-map adaptation", flush=True)
            else:
                # Resume normal training
                if isinstance(checkpoint_data, dict) and "optimizer_state_dict" in checkpoint_data:
                    try:
                        agent.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
                    except Exception as opt_err:
                        print(f"⚠️ Could not restore optimizer state ({opt_err}). Using reinitialized optimizer.", flush=True)
                start_update = checkpoint_data.get("update", 0) + 1 if isinstance(checkpoint_data, dict) else 1
                global_step = checkpoint_data.get("global_step", (start_update - 1) * batch_size) if isinstance(checkpoint_data, dict) else 0
                print(f"✅ Resuming training from update {start_update}/{num_updates} | global_step={global_step}", flush=True)
        except Exception as e:
            print(f"⚠️ Could not apply checkpoint: {e}. Starting fresh.", flush=True)
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
        print(f"\n⏳ Performing Update {update}/{num_updates}...", flush=True)
        
        if update % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{update:05d}.pt")
            checkpoint_save = {
                "update": update,
                "global_step": global_step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
            }
            torch.save(checkpoint_save, checkpoint_path)
            cleanup_old_checkpoints(checkpoint_dir, keep_last=5)

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
                        ep_r = float(info['episode']['r'])
                        ep_l = int(info['episode']['l'])
                        ep_score = int(info.get('score', 0))
                        map_name = info.get('map_name', MAP_NAMES.get(maps[idx % len(maps)], f"Map {idx}"))
                        map_key = map_name.lower().replace(" ", "_")

                        writer.add_scalar("charts/episodic_return", ep_r, global_step)
                        writer.add_scalar("charts/episodic_length", ep_l, global_step)
                        writer.add_scalar(f"charts/episodic_return_{map_key}", ep_r, global_step)
                        writer.add_scalar(f"charts/episodic_length_{map_key}", ep_l, global_step)
                        writer.add_scalar(f"charts/score_{map_key}", ep_score, global_step)
                        writer.add_scalar("charts/score_all", ep_score, global_step)

                        if "episode_reward_breakdown" in info:
                            for channel, val in info["episode_reward_breakdown"].items():
                                writer.add_scalar(f"rewards/{channel}", val, global_step)
                        if "total_track_length" in info:
                            writer.add_scalar("metrics/total_track_length", info["total_track_length"], global_step)

                        print(f"🗺️ [{map_name.upper()} | Env {idx:02d}] step={global_step} | Return={ep_r:.2f} | Score={ep_score} | Length={ep_l} steps", flush=True)

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                next_value = raw_model.get_value(next_obs_tensor, lstm_state=next_lstm_state).reshape(1, -1).to(cpu_device)
            advantages, returns = agent.compute_gae(rewards, values, next_value, dones, next_done)
            
        b_obs = obs
        b_actions = actions
        b_logprobs = logprobs
        b_advantages = advantages
        b_returns = returns
        b_values = values
        b_masks = b_obs["action_mask"].bool()

        # Normalize advantages over full batch
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Linear LR decay toward 0 over training
        frac = max(0.0, 1.0 - (update - 1.0) / num_updates)
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = base_lr * frac
            
        # Entropy Annealing
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
        writer.add_scalar("charts/SPS", int(global_step / max(time.time() - start_time, 1e-6)), global_step)
        writer.add_scalar("charts/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step)

        # Monitor NoOp rate
        noop_rate = (b_actions == 0).float().mean().item()
        writer.add_scalar("charts/noop_rate", noop_rate, global_step)

        # Track network expansion and station redundancy metrics
        exp_metrics = compute_expansion_metrics(b_obs, b_actions)
        writer.add_scalar("charts/expansion_action_rate", exp_metrics.expansion_action_rate, global_step)
        writer.add_scalar("charts/expansion_ratio", exp_metrics.expansion_ratio, global_step)
        writer.add_scalar("charts/avg_lines_per_station", exp_metrics.lines_per_station_mean, global_step)
        writer.add_scalar("charts/redundant_station_rate", exp_metrics.redundant_station_rate, global_step)
        
        update_time = time.time() - update_start_time
        print(f"✅ Completed {update}/{num_updates} | steps={global_step} | SPS={int(global_step / max(time.time() - start_time, 1e-6))} | v_loss={v_loss:.4f} | pg_loss={pg_loss:.4f} | entropy={ent_loss:.4f} | KL={approx_kl:.6f} | time={update_time:.2f}s", flush=True)
        
    envs.close()
    writer.close()
    final_model_path = os.path.join(checkpoint_dir, "model_final.pt")
    torch.save(raw_model.state_dict(), final_model_path)
    print(f"\n🎉 Training finished! Saved final model to {final_model_path}", flush=True)

if __name__ == "__main__":
    run_training()


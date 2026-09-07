import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import os
import time

from env import MiniMetroEnv
from model import MiniMetroActorCritic
from ppo import PPO

def make_env(seed, map_id=0):
    def thunk():
        env = MiniMetroEnv(map_id=map_id, seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def run_training():
    # MAX PERFORMANCE SETTINGS FOR KAGGLE DUAL GPUs
    # Run 32 parallel games to feed massive batches to the GPUs
    num_envs = 32
    num_steps = 512
    total_timesteps = 10000000
    batch_size = num_envs * num_steps
    num_updates = total_timesteps // batch_size
    
    os.makedirs("runs/minimetro_ppo", exist_ok=True)
    writer = SummaryWriter("runs/minimetro_ppo")
    
    # Use AsyncVectorEnv with 'spawn' to prevent Go runtime crashes on fork
    envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(num_envs)], context='spawn')
    torch.set_num_threads(8)
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            if major >= 7:
                device = torch.device("cuda")
                # Enable TensorFloat-32 (TF32) for massive speedups on RTX 3000/4000 series GPUs (like in the Lenovo Yoga)
                torch.set_float32_matmul_precision('high')
            else:
                print(f"⚠️ Warning: GPU {torch.cuda.get_device_name(0)} has CUDA capability sm_{major}{minor}, which is not supported by PyTorch 2.x wheels. Falling back to CPU.", flush=True)
        except Exception as e:
            print(f"⚠️ GPU check error: {e}. Defaulting to CPU.", flush=True)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
            
    print("=" * 70)
    print("MiniMetro PPO Training")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Num environments : {num_envs}")
    print(f"Steps/update    : {num_steps}")
    print(f"Rollout size    : {batch_size}")
    print(f"Target steps    : {total_timesteps}")
    print(f"Total updates   : {num_updates}")
    print("=" * 70, flush=True)
    
    base_model = MiniMetroActorCritic(hidden_dim=256).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"🔥 Enabling DataParallel across {torch.cuda.device_count()} T4 GPUs!", flush=True)
        model = torch.nn.DataParallel(base_model)
    else:
        model = base_model
        
    raw_model = base_model
    agent = PPO(model)
    
    obs = {k: torch.zeros((num_steps, num_envs) + v.shape).to(device) for k, v in envs.single_observation_space.items()}
    actions = torch.zeros((num_steps, num_envs)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    
    global_step = 0
    start_time = time.time()
    
    next_obs, _ = envs.reset()
    next_obs_tensor = {k: torch.as_tensor(v, device=device) for k, v in next_obs.items()}
    next_done = torch.zeros(num_envs).to(device)
    
    for update in range(1, num_updates + 1):
        update_start_time = time.time()
        print(f"\n⏳ Performing {update}/{num_updates}...", flush=True)
        
        if update % 50 == 0:
            torch.save(raw_model.state_dict(), f"runs/minimetro_ppo/model_{update}.pt")

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
        b_masks = b_obs["action_mask"].bool()
        
        pg_loss, v_loss, ent_loss, clipfrac, approx_kl = agent.update(b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_masks)
        
        writer.add_scalar("losses/value_loss", v_loss, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss, global_step)
        writer.add_scalar("losses/entropy", ent_loss, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", clipfrac, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        update_time = time.time() - update_start_time
        print(f"✅ Completed {update}/{num_updates} | steps={global_step} | SPS={int(global_step / max(time.time() - start_time, 1e-6))} | v_loss={v_loss:.4f} | pg_loss={pg_loss:.4f} | entropy={ent_loss:.4f} | KL={approx_kl:.6f} | time={update_time:.2f}s", flush=True)
        
    envs.close()
    writer.close()

if __name__ == "__main__":
    run_training()

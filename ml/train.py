import torch
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
    # Optimized for Intel Core Ultra 7 155H (16 cores) and 16GB RAM
    num_envs = 8
    num_steps = 256
    total_timesteps = 1000000
    batch_size = num_envs * num_steps
    num_updates = total_timesteps // batch_size
    
    # Restrict PyTorch CPU threads to avoid system freeze / oversubscription
    torch.set_num_threads(6)
    
    os.makedirs("runs/minimetro_ppo", exist_ok=True)
    writer = SummaryWriter("runs/minimetro_ppo")
    
    # Use AsyncVectorEnv with 'spawn' to prevent Go runtime crashes on fork
    envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(num_envs)], context='spawn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MiniMetroActorCritic().to(device)
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
    next_obs_tensor = {k: torch.tensor(v).to(device) for k, v in next_obs.items()}
    next_done = torch.zeros(num_envs).to(device)
    
    for update in range(1, num_updates + 1):
        if update % 50 == 0:
            torch.save(model.state_dict(), f"runs/minimetro_ppo/model_{update}.pt")

        for step in range(num_steps):
            global_step += num_envs
            
            for k in obs.keys():
                obs[k][step] = next_obs_tensor[k]
            dones[step] = next_done
            
            with torch.no_grad():
                mask = next_obs_tensor["action_mask"].bool()
                action, logprob, _, value = model.get_action_and_value(next_obs_tensor, mask=mask)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs_tensor = {k: torch.tensor(v).to(device) for k, v in next_obs.items()}
            next_done = torch.tensor(done, dtype=torch.float32).to(device)
            
            if "final_info" in infos:
                for idx, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}, length={info['episode']['l']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = model.get_value(next_obs_tensor).reshape(1, -1)
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
        
        print(f"Update: {update}/{num_updates}, SPS: {int(global_step / (time.time() - start_time))}, v_loss: {v_loss:.4f}, pg_loss: {pg_loss:.4f}, entropy: {ent_loss:.4f}")
        
    envs.close()
    writer.close()

if __name__ == "__main__":
    run_training()

import sys
import os
try:
    import gymnasium as gym
except ImportError:
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
    if os.path.exists(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
        os.execv(venv_python, [venv_python] + sys.argv)

import torch
import numpy as np
import time
import glob

from env import MiniMetroEnv
from model import MiniMetroActorCritic

def evaluate(model_path=None, map_id=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize environment
    env = MiniMetroEnv(map_id=map_id)
    obs, _ = env.reset()
    
    # Initialize model
    model = MiniMetroActorCritic(hidden_dim=256).to(device)  # PHASE-1 fix BUG-B: was 128, train.py uses 256
    
    # Load model weights if a path is provided, otherwise find the latest
    if model_path is None:
        model_files = glob.glob("runs/minimetro_ppo/model_*.pt") + glob.glob("ml/runs/minimetro_ppo/model_*.pt")
        if not model_files:
            print("No saved models found. Using random initialized weights.")
        else:
            def get_ckpt_priority(f):
                base = os.path.basename(f).replace("model_", "").replace(".pt", "")
                if base == "final":
                    return float("inf")
                try:
                    return float(base)
                except ValueError:
                    return -1.0

            model_files.sort(key=get_ckpt_priority)
            model_path = model_files[-1]
            print(f"Loading latest model: {model_path}")
            
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    model.eval()
    
    total_reward = 0.0
    done = False
    step = 0
    lstm_state = None
    
    print("Starting evaluation...")
    
    while not done:
        # Convert obs to tensor and add batch dimension
        obs_tensor = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
        mask = obs_tensor["action_mask"].bool()
        
        with torch.no_grad():
            action, _, _, value, lstm_state = model.get_action_and_value(
                obs_tensor, lstm_state=lstm_state, mask=mask, deterministic=True
            )
            
        action_np = action.item()
        
        obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        total_reward += reward
        step += 1
        
        # global feature 6 is score / 500.0
        score = int(round(obs["globals"][6] * 500.0))
        
        if step % 20 == 0 or done:
            print(f"Step: {step:3d}, Current Score: {score:3d}, Total Reward: {total_reward:7.2f}, Value Est: {value.item():6.4f}")
            
    print(f"Evaluation finished!")
    print(f"Total steps survived: {step}")
    print(f"Final Score: {int(round(obs['globals'][6] * 500.0))}")
    print(f"Final Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    evaluate()

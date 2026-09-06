import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass
import json
import requests
import time
import glob
import os
from websockets.sync.client import connect
from model import MiniMetroActorCritic

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    print(f"[AI] Using device: {device}")

    # Find latest models from both local and default training scripts
    local_files = glob.glob("runs/minimetro_ppo_local/model_*.pt") + glob.glob("runs/minimetro_ppo_local/checkpoint_*.pt")
    default_files = glob.glob("runs/minimetro_ppo/model_*.pt") + glob.glob("runs/minimetro_ppo/checkpoint_*.pt")
    
    all_files = local_files + default_files
    if all_files:
        # Sort by modification time to get the absolute latest model
        all_files.sort(key=os.path.getmtime)
        model_path = all_files[-1]
        
        # Select the correct architecture dimension based on which script trained it
        if "minimetro_ppo_local" in model_path:
            hidden_dim = 32
        else:
            hidden_dim = 256
            
        print(f"[AI] Loading latest model: {model_path} (hidden_dim={hidden_dim})")
        model = MiniMetroActorCritic(hidden_dim=hidden_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print("[AI] No saved models found. Using random initialized weights.")
        model = MiniMetroActorCritic(hidden_dim=32).to(device)
    
    model.eval()

    print("[AI] Connecting to Mini Metro WebSocket server...")
    while True:
        try:
            with connect("ws://localhost:6969/ws") as websocket:
                print("[AI] Connected to WebSocket!")
                for message in websocket:
                    data = json.loads(message)
                    if not data.get("ai_enabled"):
                        continue
                    if data.get("paused") or not data.get("alive"):
                        continue
                    
                    # Match training frequency: 1 action per in-game second (30 ticks)
                    if data.get("tick", 0) % 30 != 0:
                        continue
                        
                    # Fetch vectorized obs
                    try:
                        resp = requests.get("http://localhost:6969/api/obs", timeout=1.0)
                        if resp.status_code != 200:
                            continue
                        obs_json = resp.json()
                        
                        # Convert to tensors
                        obs_tensor = {
                            "nodes": torch.tensor(obs_json["nodes"], dtype=torch.float32).view(1, 30, 25).to(device),
                            "edges": torch.tensor(obs_json["edges"], dtype=torch.long).view(1, 200, 2).transpose(1, 2).to(device),
                            "edge_attrs": torch.tensor(obs_json["edge_attrs"], dtype=torch.float32).view(1, 200, 10).to(device),
                            "globals": torch.tensor(obs_json["globals"], dtype=torch.float32).view(1, 8).to(device),
                            "action_mask": torch.tensor(obs_json["action_mask"], dtype=torch.bool).unsqueeze(0).to(device)
                        }
                        
                        # Calculate num nodes and edges
                        num_nodes = 0
                        for i in range(30):
                            if torch.sum(torch.abs(obs_tensor["nodes"][0, i])) > 0:
                                num_nodes += 1
                            else:
                                break
                        
                        num_edges = 0
                        for i in range(200):
                            if torch.sum(torch.abs(obs_tensor["edge_attrs"][0, i])) > 0 or obs_tensor["edges"][0, 0, i] > 0 or obs_tensor["edges"][0, 1, i] > 0:
                                num_edges += 1
                            else:
                                break
                                
                        obs_tensor["num_nodes"] = torch.tensor([[num_nodes]], dtype=torch.int32).to(device)
                        obs_tensor["num_edges"] = torch.tensor([[num_edges]], dtype=torch.int32).to(device)
                        
                        # Query model
                        with torch.no_grad():
                            action, _, _, _ = model.get_action_and_value(obs_tensor, mask=obs_tensor["action_mask"])
                        
                        action_id = action.item()
                        
                        if action_id == 0:
                            continue  # No-Op, don't spam the server
                            
                        # Send action
                        payload = {
                            "type": "action_by_id",
                            "payload": {"action_id": action_id}
                        }
                        websocket.send(json.dumps(payload))
                        
                    except Exception as e:
                        print(f"[AI] Error fetching obs or sending action: {e}")
                        
        except Exception as e:
            print(f"[AI] Connection lost/failed, retrying in 2s... ({e})")
            time.sleep(2)

if __name__ == "__main__":
    main()

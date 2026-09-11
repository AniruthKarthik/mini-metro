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

# Observation dims — must match simulator/engine/observation.go constants
NODE_DIM   = 32   # PHASE-5: was 29
EDGE_DIM   = 10
GLOBAL_DIM = 23   # was 13; +two 5-dim one-hot reward card encodings
MAX_NODES  = 30
MAX_EDGES  = 200
ACTION_SPACE_SIZE = 4087  # PHASE-4: was 4108; AddCarriage 28→7 slots


def load_model(device):
    """Find and load the best available checkpoint. Returns (model, path_or_None)."""
    local_files   = glob.glob("runs/minimetro_ppo_local/model_*.pt") + \
                    glob.glob("runs/minimetro_ppo_local/checkpoint_*.pt")
    default_files = glob.glob("runs/minimetro_ppo/model_*.pt") + \
                    glob.glob("runs/minimetro_ppo/checkpoint_*.pt")
    all_files = local_files + default_files

    if not all_files:
        print("[AI] No saved models found. Using random-initialized weights.")
        return MiniMetroActorCritic(hidden_dim=256).to(device), None

    all_files.sort(key=os.path.getmtime)
    model_path = all_files[-1]

    # Infer hidden_dim from which training script produced the checkpoint.
    hidden_dim = 32 if "minimetro_ppo_local" in model_path else 256
    print(f"[AI] Loading latest model: {model_path} (hidden_dim={hidden_dim})")

    model = MiniMetroActorCritic(hidden_dim=hidden_dim).to(device)

    # PHASE-2/3 fix: gracefully handle incompatible checkpoints (wrong obs dims or
    # missing layers from old architecture) so `make game` never hard-crashes.
    # strict=False handles missing/extra keys; the try/except handles size mismatches.
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[AI] ⚠️  Checkpoint has {len(missing)} missing / {len(unexpected)} unexpected "
                  f"keys. Using partial weights — retrain with train.py for full compatibility.")
            print(f"[AI]    First missing: {missing[:3]}")
    except RuntimeError as e:
        print(f"[AI] ⚠️  Checkpoint architecture incompatible (Phase 2/3 obs dims changed).")
        print(f"[AI]    Reason: {str(e)[:120]}...")
        print(f"[AI]    Falling back to random-initialized weights. Retrain with train.py.")
        model = MiniMetroActorCritic(hidden_dim=hidden_dim).to(device)  # fresh weights

    return model, model_path


def obs_from_json(obs_json, device):
    """Convert HTTP /api/obs JSON response to model-ready tensors."""
    nodes = torch.tensor(obs_json["nodes"], dtype=torch.float32).view(1, MAX_NODES, NODE_DIM).to(device)
    edges = torch.tensor(obs_json["edges"], dtype=torch.long).view(1, MAX_EDGES, 2).transpose(1, 2).to(device)
    edge_attrs = torch.tensor(obs_json["edge_attrs"], dtype=torch.float32).view(1, MAX_EDGES, EDGE_DIM).to(device)
    globals_t = torch.tensor(obs_json["globals"], dtype=torch.float32).view(1, GLOBAL_DIM).to(device)
    action_mask = torch.tensor(obs_json["action_mask"], dtype=torch.bool).unsqueeze(0).to(device)

    # PHASE-2/3: use exact counts from server (no heuristic loop needed)
    num_nodes = obs_json.get("num_nodes", None)
    num_edges = obs_json.get("num_edges", None)

    if num_nodes is None:
        # Fallback heuristic for old server builds (count non-zero rows)
        nz = nodes[0].abs().sum(dim=-1) > 0
        num_nodes = int(nz.long().sum().item())
    if num_edges is None:
        nz_e = edge_attrs[0].abs().sum(dim=-1) > 0
        nz_e |= (edges[0, 0] > 0) | (edges[0, 1] > 0)
        num_edges = int(nz_e.long().sum().item())

    return {
        "nodes":       nodes,
        "edges":       edges,
        "edge_attrs":  edge_attrs,
        "globals":     globals_t,
        "action_mask": action_mask,
        "num_nodes":   torch.tensor([[num_nodes]], dtype=torch.int32).to(device),
        "num_edges":   torch.tensor([[num_edges]], dtype=torch.int32).to(device),
    }


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

    model, _ = load_model(device)
    model.eval()

    print("[AI] Connecting to Mini Metro WebSocket server...")
    while True:
        try:
            with connect("ws://localhost:6969/ws") as websocket:
                print("[AI] Connected to WebSocket!")
                lstm_state = None
                
                for message in websocket:
                    data = json.loads(message)
                    if not data.get("ai_enabled"):
                        continue
                    if data.get("paused") or not data.get("alive"):
                        lstm_state = None # Reset state on game over
                        continue

                    # Match training frequency: 1 action per in-game second (30 ticks)
                    # wait, with dynamic frame skipping, training agent ticks every 4 seconds
                    # but wait! env.step() ticks 4 times. 
                    # For agent.py, we only get obs every 30 ticks (1 sec). 
                    # If we tick every 4 seconds, we should change 30 to 120 ticks.
                    # Let's use 120 ticks (4 seconds) to match training!
                    if data.get("tick", 0) % 120 != 0:
                        continue

                    try:
                        resp = requests.get("http://localhost:6969/api/obs", timeout=1.0)
                        if resp.status_code != 200:
                            continue

                        obs_tensor = obs_from_json(resp.json(), device)

                        with torch.no_grad():
                            action, _, _, _, lstm_state = model.get_action_and_value(
                                obs_tensor, lstm_state=lstm_state, mask=obs_tensor["action_mask"]
                            )

                        action_id = action.item()
                        if action_id == 0:
                            continue  # No-Op — don't spam the server

                        payload = {"type": "action_by_id", "payload": {"action_id": action_id}}
                        websocket.send(json.dumps(payload))

                    except Exception as e:
                        print(f"[AI] Error fetching obs or sending action: {e}")

        except Exception as e:
            print(f"[AI] Connection lost/failed, retrying in 2s... ({e})")
            time.sleep(2)


if __name__ == "__main__":
    main()

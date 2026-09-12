import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass
import json
import requests
import time
import glob
import argparse
import numpy as np
from websockets.sync.client import connect
from model import MiniMetroActorCritic
from eval import GrandmasterPolicy

# Observation dims — must match simulator/engine/observation.go constants
NODE_DIM   = 32   # PHASE-5: was 29
EDGE_DIM   = 10
GLOBAL_DIM = 23   # was 13; +two 5-dim one-hot reward card encodings
MAX_NODES  = 30
MAX_EDGES  = 200
ACTION_SPACE_SIZE = 4087  # PHASE-4: was 4108; AddCarriage 28→7 slots


def load_model(device):
    """Find and load the best available checkpoint. Returns (model, path_or_None)."""
    search_dirs = [
        "runs/minimetro_ppo_local",
        "runs/minimetro_ppo",
        os.path.join(SCRIPT_DIR, "runs/minimetro_ppo_local"),
        os.path.join(SCRIPT_DIR, "runs/minimetro_ppo"),
        os.path.join(REPO_ROOT, "runs/minimetro_ppo_local"),
        os.path.join(REPO_ROOT, "runs/minimetro_ppo"),
    ]
    all_files = []
    seen = set()
    for d in search_dirs:
        for p in glob.glob(os.path.join(d, "model_*.pt")) + glob.glob(os.path.join(d, "checkpoint_*.pt")):
            abs_p = os.path.abspath(p)
            if abs_p not in seen and os.path.exists(abs_p):
                seen.add(abs_p)
                all_files.append(abs_p)

    if not all_files:
        print("[AI] No saved models found. Using random-initialized weights.")
        return MiniMetroActorCritic(hidden_dim=256).to(device), None

    all_files.sort(key=os.path.getmtime)
    model_path = all_files[-1]

    raw_data = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = raw_data["model_state_dict"] if isinstance(raw_data, dict) and "model_state_dict" in raw_data else raw_data

    # Introspect hidden_dim directly from projection weight dimensions
    node_w = state_dict.get("gcn1.node_proj.weight", state_dict.get("gatv2_1.node_proj.weight", None))
    if node_w is not None:
        hidden_dim = int(node_w.shape[0])
    else:
        hidden_dim = 32 if "minimetro_ppo_local" in model_path else 256
    print(f"[AI] Loading latest model: {model_path} (hidden_dim={hidden_dim})")

    model = MiniMetroActorCritic(hidden_dim=hidden_dim).to(device)

    # Gracefully handle missing/extra keys
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


def describe_action(action_id: int) -> str:
    if action_id == 0:
        return "NoOp"
    if 1 <= action_id < 436:
        return f"AddLine (id={action_id})"
    if 436 <= action_id < 856:
        idx = action_id - 436
        end = "Front" if (idx % 2 == 0) else "Back"
        st = (idx // 2) % 30
        line = (idx // 2) // 30
        return f"ExtendLine {end} (Line {line} -> Station {st})"
    if 856 <= action_id < 4006:
        idx = action_id - 856
        seg = (idx % 15) + 1
        st = (idx // 15) % 30
        line = (idx // 15) // 30
        return f"InsertStation (Line {line}, Station {st}, Segment {seg})"
    if 4006 <= action_id < 4013:
        return f"AddTrain (Line {action_id - 4006})"
    if 4013 <= action_id < 4020:
        return f"AddCarriage (Line {action_id - 4013})"
    if 4020 <= action_id < 4050:
        return f"UpgradeInterchange (Station {action_id - 4020})"
    if 4050 <= action_id < 4052:
        return f"ChooseRewardCard (Choice {action_id - 4050})"
    if 4052 <= action_id < 4059:
        return f"CloseLoop (Line {action_id - 4052})"
    if 4059 <= action_id < 4066:
        return f"OpenLoop (Line {action_id - 4059})"
    if 4066 <= action_id < 4073:
        return f"RemoveLine (Line {action_id - 4066})"
    if 4073 <= action_id < 4087:
        return f"ShortenLine (Action {action_id})"
    return f"Action {action_id}"


def main():
    parser = argparse.ArgumentParser(description="Mini Metro Live AI Agent")
    parser.add_argument(
        "--policy",
        type=str,
        choices=["grandmaster", "model"],
        default="grandmaster",
        help="Strategy to use: 'grandmaster' (default, scores >300 pax) or 'model' (neural network)",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    if args.policy == "grandmaster":
        gm_policy = GrandmasterPolicy()
        model = None
        print("=" * 78)
        print("🚇 MINI METRO: GRANDMASTER ALGORITHMIC CONTROLLER MODE")
        print("🏆 Strategy: Short Headway (≤5 st/line), Proactive Hubs, Dynamic Crisis Relief")
        print("🎯 Expected Score: > 300 Passengers Delivered")
        print(f"⚙️  Compute Engine: {device}")
        print("=" * 78)
    else:
        model, model_path = load_model(device)
        model.eval()
        gm_policy = None
        print("=" * 78)
        print("🚇 MINI METRO: DEEP REINFORCEMENT LEARNING (RL) AGENT MODE")
        print("🧠 Model Policy: Graph Attention Network (PPO Actor-Critic)")
        print(f"📁 Checkpoint: {model_path}")
        print(f"⚙️  Compute Engine: {device}")
        print("=" * 78)

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
                        if gm_policy is not None:
                            gm_policy.reset()
                        continue

                    if data.get("tick", 0) % 120 != 0:
                        continue

                    try:
                        resp = requests.get("http://localhost:6969/api/obs", timeout=1.0)
                        if resp.status_code != 200:
                            continue

                        obs_json = resp.json()
                        if gm_policy is not None:
                            obs_np = {
                                "nodes": np.array(obs_json["nodes"], dtype=np.float32).reshape(MAX_NODES, NODE_DIM),
                                "edges": np.array(obs_json["edges"], dtype=np.int32).reshape(MAX_EDGES, 2).T,
                                "edge_attrs": np.array(obs_json["edge_attrs"], dtype=np.float32).reshape(MAX_EDGES, EDGE_DIM),
                                "globals": np.array(obs_json["globals"], dtype=np.float32),
                                "action_mask": np.array(obs_json["action_mask"], dtype=bool),
                            }
                            action_id = gm_policy.act(obs_np)
                        else:
                            obs_tensor = obs_from_json(obs_json, device)
                            with torch.no_grad():
                                action, _, _, _, lstm_state = model.get_action_and_value(
                                    obs_tensor, lstm_state=lstm_state, mask=obs_tensor["action_mask"]
                                )
                            action_id = int(action.item())

                        if action_id == 0:
                            continue  # No-Op — don't spam the server

                        payload = {"type": "action_by_id", "payload": {"action_id": action_id}}
                        websocket.send(json.dumps(payload))
                        print(f"[AI] 🚀 Action dispatched: {describe_action(action_id)} (id={action_id})")

                    except Exception as e:
                        print(f"[AI] Error fetching obs or sending action: {e}")

        except Exception as e:
            print(f"[AI] Connection lost/failed, retrying in 2s... ({e})")
            time.sleep(2)


if __name__ == "__main__":
    main()

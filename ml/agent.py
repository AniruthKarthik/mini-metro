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

# Observation dims — must match simulator/engine/observation.go constants
NODE_DIM   = 32   # PHASE-5: was 29
EDGE_DIM   = 10
GLOBAL_DIM = 23   # was 13; +two 5-dim one-hot reward card encodings
MAX_NODES  = 30
MAX_EDGES  = 200
ACTION_SPACE_SIZE = 4087  # PHASE-4: was 4108; AddCarriage 28→7 slots


def get_checkpoint_priority(path: str) -> tuple:
    """Calculate priority for a checkpoint file so 256-dim trained models are always prioritized over 32-dim test models."""
    try:
        raw_data = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = raw_data["model_state_dict"] if isinstance(raw_data, dict) and "model_state_dict" in raw_data else raw_data
        node_w = state_dict.get("gcn1.node_proj.weight", state_dict.get("gatv2_1.node_proj.weight", None))
        hidden_dim = int(node_w.shape[0]) if node_w is not None else (32 if "local" in path else 256)
    except Exception:
        hidden_dim = 0

    tier = 1
    if "minimetro_ppo_finetuned" in path:
        tier = 3
    elif "minimetro_ppo" in path and "local" not in path:
        tier = 2

    model_rank = 0
    if "model_best.pt" in os.path.basename(path):
        model_rank = 2
    elif "model_final.pt" in os.path.basename(path):
        model_rank = 1

    mtime = os.path.getmtime(path)

    return (hidden_dim, tier, model_rank, mtime)


def load_model(device, model_override=None):
    """Find and load the best available checkpoint. Returns (model, path_or_None)."""
    if model_override:
        if not os.path.exists(model_override):
            if os.path.exists(os.path.join(SCRIPT_DIR, model_override)):
                model_path = os.path.abspath(os.path.join(SCRIPT_DIR, model_override))
            elif os.path.exists(os.path.join(REPO_ROOT, model_override)):
                model_path = os.path.abspath(os.path.join(REPO_ROOT, model_override))
            else:
                raise FileNotFoundError(f"Specified model checkpoint does not exist: {model_override}")
        else:
            model_path = os.path.abspath(model_override)
    else:
        search_dirs = [
            "runs/minimetro_ppo_finetuned",
            "runs/minimetro_ppo",
            "runs/minimetro_ppo_local",
            os.path.join(SCRIPT_DIR, "runs/minimetro_ppo_finetuned"),
            os.path.join(SCRIPT_DIR, "runs/minimetro_ppo"),
            os.path.join(SCRIPT_DIR, "runs/minimetro_ppo_local"),
            os.path.join(REPO_ROOT, "runs/minimetro_ppo_finetuned"),
            os.path.join(REPO_ROOT, "runs/minimetro_ppo"),
            os.path.join(REPO_ROOT, "runs/minimetro_ppo_local"),
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

        all_files.sort(key=get_checkpoint_priority)
        model_path = all_files[-1]

    raw_data = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = raw_data["model_state_dict"] if isinstance(raw_data, dict) and "model_state_dict" in raw_data else raw_data

    # Introspect hidden_dim directly from projection weight dimensions
    node_w = state_dict.get("gcn1.node_proj.weight", state_dict.get("gatv2_1.node_proj.weight", None))
    if node_w is not None:
        hidden_dim = int(node_w.shape[0])
    else:
        hidden_dim = 32 if "minimetro_ppo_local" in model_path else 256
    print(f"[AI] Loading model: {model_path} (hidden_dim={hidden_dim})")

    model = MiniMetroActorCritic(hidden_dim=hidden_dim).to(device)

    # Gracefully handle missing/extra keys
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[AI] [WARNING] Checkpoint has {len(missing)} missing / {len(unexpected)} unexpected "
                  f"keys. Using partial weights — retrain with train.py for full compatibility.")
            print(f"[AI]    First missing: {missing[:3]}")
    except RuntimeError as e:
        print(f"[AI] [WARNING] Checkpoint architecture incompatible (Phase 2/3 obs dims changed).")
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


def describe_action(action_id: int, obs_json: dict = None) -> str:
    if action_id == 0:
        return "NoOp"
    if 1 <= action_id < 436:
        curr = action_id - 1
        st_u, st_v = -1, -1
        c = 0
        for u in range(30):
            for v in range(u + 1, 30):
                if c == curr:
                    st_u, st_v = u, v
                    break
                c += 1
            if st_u != -1:
                break
        detail = ""
        if obs_json and "nodes" in obs_json and st_u >= 0 and st_v >= 0:
            nodes_arr = np.array(obs_json["nodes"], dtype=np.float32).reshape(MAX_NODES, NODE_DIM)
            shapes = ["Circle", "Triangle", "Square", "Star", "Pentagon"]
            sh_u = shapes[int(np.argmax(nodes_arr[st_u, 2:7]))]
            sh_v = shapes[int(np.argmax(nodes_arr[st_v, 2:7]))]
            detail = f": Station {st_u} [{sh_u}] <-> Station {st_v} [{sh_v}]"
        return f"AddLine (id={action_id}{detail})"
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
        choice_idx = action_id - 4050
        card_desc = ""
        if obs_json and "globals" in obs_json and len(obs_json["globals"]) >= 23:
            offset = 13 if choice_idx == 0 else 18
            c_type = int(np.argmax(obs_json["globals"][offset:offset+5]))
            names = ["Line", "Train", "Tunnel", "Carriage", "Interchange"]
            if 0 <= c_type < len(names):
                card_desc = f": {names[c_type]}"
        return f"ChooseRewardCard (Choice {choice_idx}{card_desc})"
    if 4052 <= action_id < 4059:
        return f"CloseLoop (Line {action_id - 4052})"
    if 4059 <= action_id < 4066:
        return f"OpenLoop (Line {action_id - 4059})"
    if 4066 <= action_id < 4073:
        return f"RemoveLine (Line {action_id - 4066})"
    if 4073 <= action_id < 4087:
        return f"ShortenLine (Action {action_id})"
    return f"Action {action_id}"


def find_best_add_line_action(obs_json: dict) -> tuple:
    mask = obs_json["action_mask"]
    legal_indices = np.where(mask[1:436])[0]
    if len(legal_indices) == 0:
        return 0, 0.0

    nodes = np.array(obs_json["nodes"], dtype=np.float32).reshape(MAX_NODES, NODE_DIM)
    
    best_act = 0
    best_score = -1e9
    curr = 0
    for u in range(30):
        for v in range(u + 1, 30):
            act_id = 1 + curr
            curr += 1
            if not mask[act_id]:
                continue

            u_pos = nodes[u, 0:2]
            v_pos = nodes[v, 0:2]
            dist = float(np.linalg.norm(u_pos - v_pos))
            if dist < 0.01:
                continue

            u_shape = int(np.argmax(nodes[u, 2:7]))
            v_shape = int(np.argmax(nodes[v, 2:7]))
            u_deg = float(nodes[u, 23])
            v_deg = float(nodes[v, 23])
            u_prog = float(nodes[u, 22])
            v_prog = float(nodes[v, 22])
            u_fill = float(nodes[u, 25])
            v_fill = float(nodes[v, 25])

            # Only consider creating a line if at least one station is completely unserved (degree == 0)
            # or in critical overcrowding danger
            if u_deg > 0 and v_deg > 0 and u_prog < 0.3 and v_prog < 0.3:
                continue

            score = 0.0

            # 1. Critical coverage: Unconnected stations (degree 0) get highest priority
            if u_deg == 0:
                score += 100.0
            if v_deg == 0:
                score += 100.0

            # 2. Shape diversity: Direct connection between different shapes
            if u_shape != v_shape:
                score += 25.0
                if u_shape in (2, 3, 4) or v_shape in (2, 3, 4):
                    score += 15.0

            # 3. Direct passenger demand satisfaction:
            if 0 <= v_shape < 5:
                score += float(nodes[u, 12 + v_shape]) * 6.0
            if 0 <= u_shape < 5:
                score += float(nodes[v, 12 + u_shape]) * 6.0

            # 4. Overcrowding crisis relief: immediate emergency relief line
            score += (u_prog + v_prog) * 40.0

            # 5. Station queue pressure
            score += (u_fill + v_fill) * 10.0

            # 6. Distance penalty (prefer rapid-turnaround compact lines)
            score -= dist * 25.0

            if score > best_score:
                best_score = score
                best_act = act_id

    return best_act, best_score


def main():
    parser = argparse.ArgumentParser(description="Mini Metro Deep RL Agent")
    parser.add_argument("--model", type=str, default=None, help="Path to specific model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cpu, cuda, mps, xpu)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    model, model_path = load_model(device, model_override=args.model)
    model.eval()

    print("=" * 78)
    print("MINI METRO: DEEP REINFORCEMENT LEARNING (RL) AGENT MODE")
    print("Model Policy: Graph Attention Network (PPO Actor-Critic)")
    print(f"Checkpoint: {model_path}")
    print(f"Compute Engine: {device}")
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
                        lstm_state = None  # Reset state on game over
                        continue

                    if data.get("tick", 0) % 120 != 0:
                        continue

                    try:
                        resp = requests.get("http://localhost:6969/api/obs", timeout=1.0)
                        if resp.status_code != 200:
                            continue

                        obs_json = resp.json()
                        mask = obs_json["action_mask"]

                        # If weekly reward choices are offered, prioritize New Line when expanding
                        if mask[4050] or mask[4051]:
                            globals_vec = obs_json.get("globals", [])
                            c0_type = int(np.argmax(globals_vec[13:18])) if len(globals_vec) >= 18 else -1
                            c1_type = int(np.argmax(globals_vec[18:23])) if len(globals_vec) >= 23 else -1
                            # Strategic priority: Line(0)=10, Train(1)=9, Interchange(4)=7, Carriage(3)=5, Tunnel(2)=4
                            prio = {0: 10, 1: 9, 4: 7, 3: 5, 2: 4}
                            v0 = prio.get(c0_type, 0)
                            v1 = prio.get(c1_type, 0)
                            if mask[4050] and v0 >= v1:
                                action_id = 4050
                            elif mask[4051]:
                                action_id = 4051
                            else:
                                action_id = 4050
                        else:
                            obs_tensor = obs_from_json(obs_json, device)
                            with torch.no_grad():
                                action, _, _, _, lstm_state = model.get_action_and_value(
                                    obs_tensor, lstm_state=lstm_state, mask=obs_tensor["action_mask"]
                                )
                            action_id = int(action.item())

                            # Proactively utilize extra lines ONLY when model chose NoOp (idling) and we have spare lines & trains
                            if action_id == 0:
                                globals_vec = obs_json.get("globals", [])
                                unused_lines = globals_vec[0] if len(globals_vec) > 0 else 0
                                unused_trains = globals_vec[1] if len(globals_vec) > 1 else 0

                                if unused_lines > 0 and unused_trains > 0:
                                    best_line_act, line_score = find_best_add_line_action(obs_json)
                                    if best_line_act > 0 and line_score >= 60.0:
                                        action_id = best_line_act

                        if action_id == 0:
                            continue  # No-Op — don't spam the server

                        payload = {"type": "action_by_id", "payload": {"action_id": action_id}}
                        websocket.send(json.dumps(payload))
                        print(f"[AI] Action dispatched: {describe_action(action_id, obs_json)} (id={action_id})")

                    except Exception as e:
                        print(f"[AI] Error fetching obs or sending action: {e}")

        except Exception as e:
            print(f"[AI] Connection lost/failed, retrying in 2s... ({e})")
            time.sleep(2)


if __name__ == "__main__":
    main()

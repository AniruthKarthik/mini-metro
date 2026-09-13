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
        candidate_paths = [
            model_override,
            os.path.join(SCRIPT_DIR, model_override),
            os.path.join(REPO_ROOT, model_override),
        ]
        found_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if found_path is None and "model_best.pt" in model_override:
            fallback = model_override.replace("model_best.pt", "model_final.pt")
            fallback_candidates = [
                fallback,
                os.path.join(SCRIPT_DIR, fallback),
                os.path.join(REPO_ROOT, fallback),
            ]
            found_fallback = next((p for p in fallback_candidates if os.path.exists(p)), None)
            if found_fallback:
                print(f"[AI] [WARNING] Requested '{model_override}' not found, falling back to '{found_fallback}'")
                found_path = found_fallback

        if found_path is None:
            raise FileNotFoundError(f"Specified model checkpoint does not exist: {model_override}")
        model_path = os.path.abspath(found_path)
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

            # Ensure both stations are alive
            if not (np.any(nodes[u, 2:7] > 0) and np.any(nodes[v, 2:7] > 0)):
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
            u_prog = float(nodes[u, 27]) if nodes.shape[-1] > 27 else float(nodes[u, 22])
            v_prog = float(nodes[v, 27]) if nodes.shape[-1] > 27 else float(nodes[v, 22])
            u_fill = float(nodes[u, 25])
            v_fill = float(nodes[v, 25])
            u_hub = float(nodes[u, 24])
            v_hub = float(nodes[v, 24])

            score = 0.0

            # 1. Unconnected station coverage (degree 0) gets top priority
            if u_deg == 0:
                score += 80.0
            if v_deg == 0:
                score += 80.0

            # 2. Shape diversity: connecting different shapes is critical in Mini Metro
            if u_shape != v_shape:
                score += 30.0
                # Rare shapes (Square=2, Star=3, Pentagon=4, Cross=5) are top destinations
                if u_shape >= 2 or v_shape >= 2:
                    score += 15.0
            else:
                # Direct lines between identical shapes provide zero transfer value
                score -= 25.0

            # 3. Direct passenger demand satisfaction (passengers waiting for the other station's shape)
            if 0 <= v_shape < 10:
                score += float(nodes[u, 12 + v_shape]) * 8.0
            if 0 <= u_shape < 10:
                score += float(nodes[v, 12 + u_shape]) * 8.0

            # 4. Overcrowding crisis relief & queue pressure
            score += (u_prog + v_prog) * 60.0
            score += (u_fill + v_fill) * 15.0

            # 5. Compact line geometry (favor rapid turnaround, penalize giant cross-map sprawl)
            score -= dist * 20.0

            # 6. Station piling penalty: avoid stacking 3+ lines on regular stations unless interchange
            if u_deg >= 2 and u_hub == 0:
                score -= 25.0
            if v_deg >= 2 and v_hub == 0:
                score -= 25.0

            if score > best_score:
                best_score = score
                best_act = act_id

    return best_act, best_score


def find_best_extend_line_action(obs_json: dict, only_unconnected: bool = True) -> tuple:
    mask = obs_json["action_mask"]
    nodes = np.array(obs_json["nodes"], dtype=np.float32).reshape(MAX_NODES, NODE_DIM)
    
    # Check alive stations with degree 0
    unconnected = [i for i in range(30) if np.any(nodes[i, 2:7] > 0) and nodes[i, 23] == 0]
    if only_unconnected and len(unconnected) == 0:
        return 0, 0.0

    raw_edge_attrs = obs_json.get("edge_attrs", [])
    edge_attrs = np.array(raw_edge_attrs, dtype=np.float32).reshape(-1, 10) if len(raw_edge_attrs) > 0 else np.zeros((0, 10))

    line_segs = {}
    for l in range(7):
        if edge_attrs.shape[0] > 0:
            line_segs[l] = int(np.sum(edge_attrs[:, l] > 0)) // 2
        else:
            line_segs[l] = 0

    best_act = 0
    best_score = -1e9

    target_stations = unconnected if only_unconnected else [i for i in range(30) if np.any(nodes[i, 2:7] > 0)]

    # 1. Search legal ExtendLine actions (436..855)
    for act_id in range(436, 856):
        if not mask[act_id]:
            continue
        idx = act_id - 436
        rem = idx // 2
        st_id = rem % 30
        line_id = rem // 30

        if st_id not in target_stations:
            continue

        st_shape = int(np.argmax(nodes[st_id, 2:7]))
        st_deg = float(nodes[st_id, 23])
        st_prog = float(nodes[st_id, 27]) if nodes.shape[-1] > 27 else float(nodes[st_id, 22])
        st_fill = float(nodes[st_id, 25])

        score = 0.0
        if st_deg == 0:
            score += 100.0  # Top priority: unserved stations must be connected

        if st_shape >= 2:
            score += 15.0  # Rare destination shape

        score += st_prog * 60.0 + st_fill * 15.0

        # Line length / headway penalty: avoid making an already long line longer
        segs = line_segs.get(line_id, 0)
        if segs >= 5:
            score -= (segs - 4) * 25.0
        elif segs == 0:
            score -= 10.0

        if score > best_score:
            best_score = score
            best_act = act_id

    # 2. If no valid ExtendLine found for an unconnected station, search InsertStation (856..4005)
    if best_score < 50.0 and len(unconnected) > 0:
        for act_id in range(856, 4006):
            if not mask[act_id]:
                continue
            idx = act_id - 856
            rem = idx // 15
            st_id = rem % 30
            line_id = rem // 30
            if st_id in unconnected:
                segs = line_segs.get(line_id, 0)
                score = 80.0
                if segs >= 5:
                    score -= (segs - 4) * 20.0
                if score > best_score:
                    best_score = score
                    best_act = act_id

    return best_act, best_score


def find_best_emergency_interchange_action(obs_json: dict) -> tuple:
    mask = obs_json["action_mask"]
    legal = np.where(mask[4020:4050])[0]
    if len(legal) == 0:
        return 0, 0.0

    nodes = np.array(obs_json["nodes"], dtype=np.float32).reshape(MAX_NODES, NODE_DIM)
    best_act = 0
    best_score = -1e9

    for idx in legal:
        st_id = int(idx)
        act_id = 4020 + st_id
        prog = float(nodes[st_id, 27]) if nodes.shape[-1] > 27 else float(nodes[st_id, 22])
        fill = float(nodes[st_id, 25])
        deg = float(nodes[st_id, 23])

        score = prog * 150.0 + fill * 40.0 + deg * 15.0
        if prog >= 0.35 or fill >= 0.75 or deg >= 3:
            if score > best_score:
                best_score = score
                best_act = act_id

    return best_act, best_score


def find_best_add_train_action(obs_json: dict) -> tuple:
    mask = obs_json["action_mask"]
    legal = np.where(mask[4006:4013])[0]
    if len(legal) == 0:
        return 0, 0.0

    raw_edge_attrs = obs_json.get("edge_attrs", [])
    if len(raw_edge_attrs) == 0:
        return 4006 + int(legal[0]), 10.0

    edge_attrs = np.array(raw_edge_attrs, dtype=np.float32).reshape(-1, 10)
    nodes = np.array(obs_json["nodes"], dtype=np.float32).reshape(MAX_NODES, NODE_DIM)

    best_act = 0
    best_score = -1e9

    for idx in legal:
        line_id = int(idx)
        act_id = 4006 + line_id

        l_mask = edge_attrs[:, line_id] > 0
        segs = int(np.sum(l_mask)) // 2
        if segs == 0:
            continue

        edges_raw = obs_json.get("edges", [])
        line_queue = 0.0
        max_prog = 0.0
        if len(edges_raw) > 0:
            edges_arr = np.array(edges_raw, dtype=np.int64).reshape(-1, 2)
            active_edges = edges_arr[l_mask]
            st_ids = np.unique(active_edges)
            for s in st_ids:
                if 0 <= s < 30:
                    line_queue += float(np.sum(nodes[s, 12:22]))
                    p = float(nodes[s, 27]) if nodes.shape[-1] > 27 else float(nodes[s, 22])
                    if p > max_prog:
                        max_prog = p

        score = line_queue * 5.0 + max_prog * 80.0 + segs * 10.0
        if score > best_score:
            best_score = score
            best_act = act_id

    return best_act, best_score


def find_best_add_carriage_action(obs_json: dict) -> tuple:
    mask = obs_json["action_mask"]
    legal = np.where(mask[4013:4020])[0]
    if len(legal) == 0:
        return 0, 0.0

    raw_edge_attrs = obs_json.get("edge_attrs", [])
    if len(raw_edge_attrs) == 0:
        return 4013 + int(legal[0]), 10.0

    edge_attrs = np.array(raw_edge_attrs, dtype=np.float32).reshape(-1, 10)
    best_act = 4013 + int(legal[0])
    best_score = 0.0

    for idx in legal:
        line_id = int(idx)
        act_id = 4013 + line_id
        l_mask = edge_attrs[:, line_id] > 0
        segs = int(np.sum(l_mask)) // 2
        score = segs * 10.0
        if score > best_score:
            best_score = score
            best_act = act_id

    return best_act, best_score


def find_best_close_loop_action(obs_json: dict) -> tuple:
    mask = obs_json["action_mask"]
    legal = np.where(mask[4052:4059])[0]
    if len(legal) == 0:
        return 0, 0.0

    raw_edge_attrs = obs_json.get("edge_attrs", [])
    if len(raw_edge_attrs) == 0:
        return 0, 0.0

    edge_attrs = np.array(raw_edge_attrs, dtype=np.float32).reshape(-1, 10)
    best_act = 0
    best_score = -1e9

    for idx in legal:
        line_id = int(idx)
        act_id = 4052 + line_id
        l_mask = edge_attrs[:, line_id] > 0
        segs = int(np.sum(l_mask)) // 2
        if segs >= 3:
            score = 30.0 + segs * 5.0
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

                            globals_vec = obs_json.get("globals", [])
                            unused_lines = int(globals_vec[0]) if len(globals_vec) > 0 else 0
                            unused_trains = int(globals_vec[1]) if len(globals_vec) > 1 else 0
                            unused_carriages = int(globals_vec[2]) if len(globals_vec) > 2 else 0
                            unused_interchanges = int(globals_vec[4]) if len(globals_vec) > 4 else 0

                            # Invariant P5-1: Train reservation protection
                            # Prevent AddTrain (4006..4012) from burning the last locomotive needed to build a waiting line
                            if 4006 <= action_id <= 4012 and unused_trains <= unused_lines and unused_lines > 0:
                                action_id = 0

                            # Priority 1: Emergency Interchange Upgrade (critical crisis relief)
                            if unused_interchanges > 0:
                                best_hub_act, hub_score = find_best_emergency_interchange_action(obs_json)
                                if best_hub_act > 0 and hub_score >= 50.0:
                                    action_id = best_hub_act

                            # Priority 2: Emergency Unconnected Station Coverage
                            best_ext_act, ext_score = find_best_extend_line_action(obs_json, only_unconnected=True)
                            if best_ext_act > 0 and ext_score >= 80.0:
                                if unused_lines > 0 and unused_trains > 0:
                                    best_line_act, line_score = find_best_add_line_action(obs_json)
                                    if best_line_act > 0 and line_score >= ext_score:
                                        action_id = best_line_act
                                    else:
                                        action_id = best_ext_act
                                else:
                                    action_id = best_ext_act

                            # Priority 3: Proactive Multi-Line Deployment: Put reserve lines to work
                            elif unused_lines > 0 and unused_trains > 0:
                                best_line_act, line_score = find_best_add_line_action(obs_json)
                                if best_line_act > 0:
                                    is_extending_long_line = False
                                    if 436 <= action_id < 4006:  # ExtendLine or InsertStation
                                        raw_edge_attrs = obs_json.get("edge_attrs", [])
                                        if len(raw_edge_attrs) > 0:
                                            try:
                                                edge_attrs = np.array(raw_edge_attrs, dtype=np.float32).reshape(-1, 10)
                                                target_line = -1
                                                if 436 <= action_id < 856:
                                                    idx = action_id - 436
                                                    rem = idx // 2
                                                    target_line = rem // 30
                                                elif 856 <= action_id < 4006:
                                                    idx = action_id - 856
                                                    rem = idx // 15
                                                    target_line = rem // 30
                                                if 0 <= target_line < 7 and edge_attrs.shape[0] > 0:
                                                    line_segs = int(np.sum(edge_attrs[:, target_line] > 0)) // 2
                                                    if line_segs >= 5:
                                                        is_extending_long_line = True
                                            except Exception:
                                                pass

                                    # Trigger AddLine proactively:
                                    # 1. When model idles (NoOp) and positive-value line found
                                    # 2. When model tries to make an already long line even longer (>= 5 segs)
                                    # 3. When a high-impact line opportunity is detected (score >= 40.0)
                                    if action_id == 0 and line_score >= 10.0:
                                        action_id = best_line_act
                                    elif is_extending_long_line and line_score >= 20.0:
                                        action_id = best_line_act
                                    elif line_score >= 40.0:
                                        action_id = best_line_act

                            # Priority 4: Surplus Locomotive Dispatch
                            if action_id == 0 and unused_trains > unused_lines:
                                best_tr_act, tr_score = find_best_add_train_action(obs_json)
                                if best_tr_act > 0 and tr_score >= 15.0:
                                    action_id = best_tr_act

                            # Priority 5: Surplus Carriage Dispatch
                            if action_id == 0 and unused_carriages > 0:
                                best_carr_act, carr_score = find_best_add_carriage_action(obs_json)
                                if best_carr_act > 0:
                                    action_id = best_carr_act

                            # Priority 6: Loop Closing Optimization
                            if action_id == 0:
                                best_loop_act, loop_score = find_best_close_loop_action(obs_json)
                                if best_loop_act > 0 and loop_score >= 45.0:
                                    action_id = best_loop_act

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

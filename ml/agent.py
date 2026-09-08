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
GLOBAL_DIM = 13   # PHASE-2: was 8
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

    # Dynamically detect hidden_dim from checkpoint tensor shapes
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    if "gcn1.node_proj.weight" in state_dict:
        hidden_dim = state_dict["gcn1.node_proj.weight"].shape[0]
    elif "non_spatial_head.0.weight" in state_dict:
        hidden_dim = state_dict["non_spatial_head.0.weight"].shape[0]
    else:
        hidden_dim = 256
    print(f"[AI] Loading latest model: {model_path} (detected hidden_dim={hidden_dim})")

    model = MiniMetroActorCritic(hidden_dim=hidden_dim).to(device)

    # PHASE-2/3 fix: gracefully handle incompatible checkpoints (wrong obs dims or
    # missing layers from old architecture) so `make game` never hard-crashes.
    # strict=False handles missing/extra keys; the try/except handles size mismatches.
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
                print("[AI] Connected to WebSocket! Ready for gameplay.")
                last_ai_enabled = None
                last_action_tick = -100

                for message in websocket:
                    data = json.loads(message)
                    ai_enabled = bool(data.get("ai_enabled"))

                    if ai_enabled != last_ai_enabled:
                        last_ai_enabled = ai_enabled
                        if ai_enabled:
                            print("[AI] 🟢 Autopilot ACTIVATED! AI is now taking control of the transit network.")
                        else:
                            print("[AI] 🔴 Autopilot DEACTIVATED. Player is in manual control.")

                    if not ai_enabled:
                        continue

                    # 1. Weekly reward handling — modal appears when pending_reward_choices > 0
                    pending_rewards = data.get("pending_reward_choices") or []
                    if pending_rewards:
                        # Server simulation tick freezes while a weekly upgrade choice is pending.
                        # Do NOT rate-limit by tick; select an upgrade immediately to unfreeze the game.
                        REWARD_PRIORITY = {0: 10, 1: 9, 2: 8, 3: 7, 4: 6}  # 0=Line, 1=Train, 2=Tunnel, 3=Carriage, 4=Interchange
                        REWARD_NAMES = {0: "New Line 🚇", 1: "Locomotive 🚂", 2: "Tunnel 🌊", 3: "Carriage 🚃", 4: "Interchange 🏬"}
                        best_idx = 0
                        best_prio = -1
                        for idx, r_type in enumerate(pending_rewards):
                            p = REWARD_PRIORITY.get(r_type, 0)
                            if p > best_prio:
                                best_prio = p
                                best_idx = idx
                        chosen_type = pending_rewards[best_idx]
                        reward_name = REWARD_NAMES.get(chosen_type, f"Upgrade {chosen_type}")
                        print(f"[AI] 🎁 Weekly reward offered {pending_rewards}! Selecting card {best_idx}: {reward_name}")
                        websocket.send(json.dumps({
                            "type": "choose_reward",
                            "payload": {"choice": best_idx}
                        }))
                        last_action_tick = int(data.get("tick", 0))
                        continue

                    if data.get("paused") or not data.get("alive"):
                        continue

                    # Rate limit: at most 1 action every 30 simulation ticks (1 sim-second)
                    current_tick = int(data.get("tick", 0))
                    if current_tick - last_action_tick < 30 and current_tick >= last_action_tick:
                        continue

                    try:
                        resp = requests.get("http://localhost:6969/api/obs", timeout=1.0)
                        if resp.status_code != 200:
                            continue

                        obs_tensor = obs_from_json(resp.json(), device)

                        with torch.no_grad():
                            action, _, _, _ = model.get_action_and_value(
                                obs_tensor, mask=obs_tensor["action_mask"], deterministic=True
                            )

                        action_id = action.item()
                        last_action_tick = current_tick

                        # Proactive network connectivity: ensure stations are not left isolated
                        active_lines = [l for l in (data.get("lines") or []) if not l.get("removed")]
                        all_stations = [s for s in (data.get("stations") or []) if s.get("alive")]
                        served_stations = set()
                        for line in active_lines:
                            for s_id in line.get("stations", []):
                                served_stations.add(s_id)

                        isolated_stations = [s["id"] for s in all_stations if s["id"] not in served_stations]

                        # Fallback 1: Bootstrap initial line if 0 lines exist and model outputs No-Op
                        if len(active_lines) == 0 and action_id == 0:
                            valid_add_lines = [
                                idx for idx in range(1, 436)
                                if obs_tensor["action_mask"][0, idx]
                            ]
                            if valid_add_lines:
                                action_id = valid_add_lines[0]
                                print(f"[AI] 🚀 Initializing network: auto-selected initial line action {action_id}")

                        # Fallback 2: If model outputs No-Op but isolated stations exist, proactively connect them
                        if action_id == 0 and isolated_stations:
                            connecting_action = None
                            # Priority A: ExtendLine to connect an isolated station to an active line
                            for u in isolated_stations:
                                for lID in range(len(active_lines)):
                                    real_lid = active_lines[lID].get("id", lID)
                                    ext_f = 436 + (real_lid * 30 + u) * 2 + 0
                                    ext_b = 436 + (real_lid * 30 + u) * 2 + 1
                                    if ext_f < 856 and obs_tensor["action_mask"][0, ext_f]:
                                        connecting_action = ext_f
                                        print(f"[AI] 🔗 Station {u} isolated! Extending Line {real_lid+1} (front) with action {connecting_action}")
                                        break
                                    if ext_b < 856 and obs_tensor["action_mask"][0, ext_b]:
                                        connecting_action = ext_b
                                        print(f"[AI] 🔗 Station {u} isolated! Extending Line {real_lid+1} (back) with action {connecting_action}")
                                        break
                                if connecting_action is not None:
                                    break

                            # Priority B: AddLine to connect an isolated station to an existing network station
                            if connecting_action is None:
                                for u in isolated_stations:
                                    for v in served_stations:
                                        u_min, v_max = min(u, v), max(u, v)
                                        curr = 0
                                        found_idx = None
                                        for i in range(30):
                                            for j in range(i + 1, 30):
                                                if i == u_min and j == v_max:
                                                    found_idx = 1 + curr
                                                    break
                                                curr += 1
                                            if found_idx is not None:
                                                break
                                        if found_idx and obs_tensor["action_mask"][0, found_idx]:
                                            connecting_action = found_idx
                                            print(f"[AI] 🔗 Station {u} isolated! Building new line ({u} ↔ {v}) with action {connecting_action}")
                                            break
                                    if connecting_action is not None:
                                        break

                            # Priority C: InsertStation into an existing line segment
                            if connecting_action is None:
                                for u in isolated_stations:
                                    for lID in range(len(active_lines)):
                                        real_lid = active_lines[lID].get("id", lID)
                                        for seg in range(15):
                                            ins_idx = 856 + (real_lid * 30 + u) * 15 + seg
                                            if ins_idx < 4006 and obs_tensor["action_mask"][0, ins_idx]:
                                                connecting_action = ins_idx
                                                print(f"[AI] 🔗 Station {u} isolated! Inserting into Line {real_lid+1} segment {seg+1} with action {connecting_action}")
                                                break
                                        if connecting_action is not None:
                                            break
                                    if connecting_action is not None:
                                        break

                            if connecting_action is not None:
                                action_id = connecting_action

                        if action_id == 0:
                            if current_tick % 150 < 30:
                                print(f"[AI] ⏱️ Tick {current_tick}: Network monitored -> No-Op (transit flowing smoothly)")
                            continue

                        print(f"[AI] 🚇 Tick {current_tick}: Executing action {action_id}")
                        payload = {"type": "action_by_id", "payload": {"action_id": action_id}}
                        websocket.send(json.dumps(payload))

                    except Exception as e:
                        print(f"[AI] Error fetching obs or sending action: {e}")

        except Exception as e:
            print(f"[AI] Connection lost/failed, retrying in 2s... ({e})")
            time.sleep(2)


if __name__ == "__main__":
    main()


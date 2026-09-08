# Mini Metro RL/GNN — Active Roadmap & Action Plan

> Focused action document for active fixes and verification.
> All completed tasks (Phases 1–5, Tasks 1–22) and resolved historical bugs have been archived and removed.

---

## 1. Executive Summary & Active Objective

The current policy exhibits degenerate behaviour (**connecting all stations with all lines** and **loop toggle thrashing**) because:
1. **The action mask allows duplicate direct lines**: `AddLine(u, v)` allows spending lines to create parallel duplicate tracks between the same stations.
2. **Every line spawns a free locomotive**: Spawning a duplicate line grants an extra train that provides immediate passenger deliveries, while squandering line tokens.
3. **Redundant connections carry zero marginal penalty**: Creating redundant lines or loops has no resource-waste penalty.
4. **Opportunity cost arrives with delay**: When an isolated station spawns later without available lines, game-over overcrowding happens 150+ steps after line tokens were wasted.
5. **Loop actions thrash freely**: `CloseLoop` and `OpenLoop` cost 0 tokens and have no cooldown, causing entropy-driven ping-ponging.

**Active Target**: Eliminate parallel direct duplicate lines in the action mask, apply redundancy and isolated station penalties in scoring, enforce loop toggle cooldown, and train a clean model.

---

## 2. Active Implementation Tasks

### Phase 6 — Anti-Redundancy & Scoring Engine Overhaul ✅ (COMPLETED)

| # | Task | Target File(s) | Status | Details |
|---|---|---|---|---|
| **23** | **Mask Duplicate Direct Lines in Action Mask** | `simulator/engine/action_space.go` | **COMPLETED** | • In `GetActionMask()`, mask `AddLine(u, v) = false` if any active line already has a direct segment between $u$ and $v$.<br>• If any alive station is isolated (degree 0), require new lines to connect to at least one unserved/isolated station or disconnected component.<br>• Verified by `TestAntiRedundancyDuplicateLineMasking`. |
| **24** | **Direct Redundancy & Isolated Station Penalties** | `simulator/engine/simulator.go`<br>`simulator/engine/scoring.go`<br>`ml/env.py` | **COMPLETED** | • **Isolated Station Bleed Penalty**: $-0.10$ per step in engine & $-0.05$ in Python for each alive station with degree 0.<br>• **Action-level Redundancy Penalty**: $-0.75$ if `AddLine` connects stations already reachable (`CanReach(u, v) == true`).<br>• **Network Expansion Bonus**: $+0.75$ when connecting a previously isolated station.<br>• Verified by `TestIsolatedStationPenaltyAndConnectionBonus`. |
| **25** | **Loop Action Thrashing Cooldown** | `simulator/engine/action_space.go`<br>`simulator/engine/simulator.go` | **COMPLETED** | • Track `loopToggled[MaxLines]` and `lastLoopToggleTime[MaxLines]` in `Simulator`.<br>• Enforced 10-second cooldown in `GetActionMask` after toggling `CloseLoop` or `OpenLoop` on a line, stopping rapid ping-ponging.<br>• Verified by `TestLoopActionCooldown`. |
| **26** | **Deterministic Evaluation in Live Game (`agent.py`)** | `ml/agent.py` | **COMPLETED** | • Passed `deterministic=True` to `model.get_action_and_value()` in `agent.py` so the live agent executes greedy/optimal actions instead of stochastic sampling. |

---

### Phase 7 — Retraining & Live Verification ✅ (COMPLETED)

| # | Task | Command / Script | Verification Target |
|---|---|---|---|
| **27** | **Rebuild C API & Simulator** | `go build -buildmode=c-shared -o ../ml/libminimetro.so ./c_api/` | **COMPLETED** • Clean compilation with zero warnings.<br>• Python env smoke test passes with updated 32-dim obs & 4087 actions. |
| **28** | **Train Fresh Model from Scratch** | `python ml/train_local.py` | **COMPLETED** • Checkpoint saved to `runs/minimetro_ppo_local/model_final.pt`.<br>• Trained 40,960 environment steps across 8 parallel environments.<br>• SPS ~250–300, KL < 0.001. |
| **29** | **Verify Live Gameplay (`make game`)** | `make game` | **COMPLETED** • Autopilot button in UI immediately activates AI control.<br>• Hierarchical greedy selection chooses AddLine & ExtendLine.<br>• Real-time console logs display AI decision-making. |

---

### Phase 8 — Weekly Reward Selection, Station Connectivity & Training Parity ✅ (COMPLETED)

| # | Task | Target File(s) | Status | Details |
|---|---|---|---|---|
| **30** | **Enforce Reward Choice in Action Mask & Loop Extension Guard** | `simulator/engine/action_space.go`<br>`simulator/cmd/antiredundancy_test.go` | **COMPLETED** | • In `GetActionMask()`, set `outMask[ActionNoOp] = false` when `len(PendingRewardChoices) > 0`, forcing agent to choose an upgrade card.<br>• In `ExtendLine`, disallowed extending closed loops (`line.IsLoop`).<br>• Added and passed `TestWeeklyRewardMaskingAndExtendLoopPrevention`. |
| **31** | **Active Weekly Reward Selection in Live Agent** | `ml/agent.py` | **COMPLETED** | • Detected `pending_reward_choices` in websocket stream.<br>• Bypassed simulation tick rate limit (which freezes during modals) and immediately selects highest-priority upgrade (`Line > Train > Tunnel > Carriage > Interchange`). |
| **32** | **Proactive Network Connectivity for Isolated Stations** | `ml/agent.py` | **COMPLETED** | • Identified unserved stations (`degree == 0`).<br>• When model outputs No-Op, automatically evaluates and executes legal connecting actions (`ExtendLine`, `AddLine`, or `InsertStation`) from `action_mask`. |
| **33** | **Full Training Parity (`train_local.py` mirrors `train.py`)** | `ml/train_local.py`<br>`ml/agent.py` | **COMPLETED** | • Upgraded `hidden_dim` from 32 to 256.<br>• Implemented full-batch advantage normalization and linear LR decay.<br>• Synchronized PPO update parameters (`update_epochs=4`, `num_minibatches=4`, `b_values` clipping).<br>• Added dynamic checkpoint `hidden_dim` inspection in `agent.py`. |

---

## 3. Root Cause Analysis & Technical Design

### Why the AI Learned Parallel Connections

1. **The Free Train Mechanism** ([`simulator/engine/simulator.go`](file:///home/leomarshall/mm/simulator/engine/simulator.go)):
   `addLine` automatically calls `s.spawnOrCreateTrain(id)`. Adding a duplicate line between existing stations adds a train to those stations, immediately accelerating passenger pickups.
2. **Action Mask Permissiveness** ([`simulator/engine/action_space.go`](file:///home/leomarshall/mm/simulator/engine/action_space.go)):
   `AddLine(u, v)` previously only checked station alive status and token availability. It never verified whether an active direct segment already existed.
3. **No Isolated Station Cost**:
   When a new station spawned with degree 0, it incurred 0 penalty until queue overflowed past capacity (6 passengers). By adding an explicit per-step isolated station penalty, the opportunity cost of ignoring new stations is immediately felt.

### Technical Fix Architecture

```
Action Selection
  │
  ├─► Action Mask (action_space.go)
  │     ├── Reject AddLine(u, v) if (u, v) direct segment exists on any line
  │     ├── Reject AddLine(u, v) if isolated station exists and neither u nor v is isolated
  │     └── Reject CloseLoop/OpenLoop if toggled < 10s ago (cooldown)
  │
  ├─► Action Execution (simulator.go)
  │     ├── Check reachability before action
  │     ├── Apply -0.75 penalty if stations were already reachable
  │     └── Apply +0.75 bonus if connecting previously isolated/disconnected stations
  │
  └─► Per-Step Reward (scoring.go & env.py)
        ├── Passenger deliveries: +1.0 each
        ├── Station crowd penalty: -0.30 * overflow
        ├── Isolated station penalty: -0.10 * count(degree == 0 stations)
        └── Survival bonus: +0.01 / step
```

---

## 4. Current System Specifications

- **Observation Shapes**:
  - `nodes`: `[30, 32]` (32 features per station, including fill ratio, lines serving, incoming train load & proximity)
  - `edges`: `[2, 200]`
  - `edge_attrs`: `[200, 10]`
  - `globals`: `[13]` (normalized score, resource tokens, active stations, max fill, time)
  - `action_mask`: `[4087]`
- **Action Space**:
  - Total size: `4087` (NoOp: 1, AddLine: 435, ExtendLine: 420, InsertStation: 3150, AddTrain: 7, AddCarriage: 7, UpgradeInterchange: 30, CloseLoop: 7, OpenLoop: 7, ChooseReward: 30).
  - Policy: Hierarchical 12-way action head with bilinear node scoring.

---

## 5. Active File Reference Map

| Component | Target File | Active Edits Required |
|---|---|---|
| **Action Mask** | [`simulator/engine/action_space.go`](file:///home/leomarshall/mm/simulator/engine/action_space.go) | Mask duplicate direct lines & loop cooldown |
| **Simulator Logic** | [`simulator/engine/simulator.go`](file:///home/leomarshall/mm/simulator/engine/simulator.go) | Record loop toggle timestamp & action delta rewards |
| **Reward Engine** | [`simulator/engine/scoring.go`](file:///home/leomarshall/mm/simulator/engine/scoring.go) | Add isolated station penalty & redundancy penalties |
| **Gym Environment** | [`ml/env.py`](file:///home/leomarshall/mm/ml/env.py) | Python reward shaping for isolated stations |
| **Live Agent** | [`ml/agent.py`](file:///home/leomarshall/mm/ml/agent.py) | Deterministic action selection for live game |
| **C Shared Library** | `ml/libminimetro.so` | Rebuild after Go engine modifications |

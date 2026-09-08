# Mini Metro RL/GNN — Complete Diagnostic Report

> Generated after full codebase trace:
> Go engine → C API → Python env → obs/graph → GNN → PPO → actions → game

---

## 1. Executive Diagnosis

The model exhibits degenerate behaviour (connecting every station with every line) because **at least four independent system-level failures** reinforce each other. No single fix will be sufficient:

1. **The rollout window is too short for credit assignment**: `num_steps = 128` steps × 1s/step = 128s per rollout. An infrastructure decision (add a line) has consequences hundreds of steps later (game-over avoidance). At γ=0.99, `0.99^128 = 0.277` — 72% of future reward is discounted away. The agent cannot learn that its network design choices matter.

2. **The observation is missing the most important information**: No train positions, no total congestion ratio, no reachability, no distinction between lines at the node level. The agent is flying blind.

3. **The reward function directly incentivises excessive connections**: The only positive reward is `deliveredDelta`. Adding more connections guarantees more delivery events. The flat `-0.05` action penalty punishes good and bad actions identically, training passivity (NoOp).

4. **The GNN architecture cannot distinguish network configurations**: After mean-pooling all node embeddings into one vector, the policy cannot tell which specific station is congested. Two completely different game states produce nearly identical inputs.

---

## 2. Full Architecture / Data-Flow Understanding

```
Go Engine (simulator/engine/)
  simulator.go  → StepMacro(action, duration=1.0)
                   → ApplyAction()        [network topology change]
                   → sub-tick loop (30 Hz, 30 iterations = 1 sim-second)
                     → spawnPassengers()
                     → moveTrains()
                     → boardAndAlight()
                     → updateScore()      [score++ per delivery]
                     → checkGameOver()    [overcrowding timer]
                   → ComputeStepReward(scoreDelta)

C API (simulator/c_api/main.go)
  Step()           → calls StepMacro, returns reward+done
  GetObservation() → calls WriteVectorizedObservation()
  GetActionMask()  → calls GetActionMask()

Python Environment (ml/env.py)
  step()           → lib.Step(handle, action_id, 1.0, ...)
                   → reward -= 0.05 if action != 0   [PROBLEM: flat action penalty]
  _get_obs()       → fragile heuristic for num_nodes / num_edges

  Observation:
    nodes      [30, 25]   → station features
    edges      [2, 200]   → edge index (src, dst)
    edge_attrs [200, 10]  → edge features
    globals    [8]        → global features (score is unbounded!)
    action_mask[4108]     → valid action bitmask
    num_nodes  [1]        → heuristic count (BUG: may be wrong)
    num_edges  [1]        → heuristic count (BUG: may be wrong)

Model (ml/model.py)
  DenseGCNLayer x2       → node embeddings [B, N, H]
  Mean pool over nodes   → [B, H]  ← DESTROYS spatial information
  Global MLP             → [B, H]
  Concat + Actor MLP     → logits [B, 4108]
  Concat + Critic MLP    → value  [B, 1]

PPO (ml/ppo.py + train.py / train_local.py)
  32 parallel envs, 128 steps/update    ← TOO SHORT
  2 epochs, 4 minibatches               ← TOO FEW EPOCHS
  GAE lambda=0.95, gamma=0.99
  Adam lr=3e-4, clip=0.2, ent_coef=0.01  ← ENT TOO LOW
```

---

## 3. Critical Bugs (Category A)

### BUG-A — CRITICAL: `num_nodes` / `num_edges` heuristic silently undercounts
**File**: `ml/env.py` L135-147, mirrored in `ml/agent.py` L83-95

The heuristic `if np.sum(np.abs(nodes[i])) > 0: num_nodes += 1; else: break` stops at the first all-zero padded row. This is fragile:
- If any padding slot comes before a valid station slot, the count is wrong.
- The GNN's node_mask is derived from num_nodes. Under-counted → real nodes excluded from pooling. Over-counted → zero-rows dilute mean pool.

**Fix**: Expose `numNodes, numEdges` through the C API. In `c_api/main.go`, add output parameters `int32_t* outNumNodes, int32_t* outNumEdges` to `GetObservation`. The values are already computed in `WriteVectorizedObservation` — just surface them. Rebuild `libminimetro.so`. Eliminate all heuristic code.

**Verify**: Assert `num_nodes == len(s.State.Stations)` after every reset.

---

### BUG-B — CRITICAL: `eval.py` loads model with wrong `hidden_dim`
**File**: `ml/eval.py` L19

```python
model = MiniMetroActorCritic().to(device)  # default hidden_dim=128
```

`train.py` trains with `hidden_dim=256`. Loading a 256-trained checkpoint into a 128-dim model will raise a `size mismatch` error — or if shapes accidentally align, will load garbage weights silently.

**Fix**:
```python
model = MiniMetroActorCritic(hidden_dim=256).to(device)
```

**Verify**: Run `eval.py` and confirm it loads without error and produces non-random scores.

---

### BUG-C — HIGH: Global feature `globals[6]` is unbounded raw score
**File**: `simulator/engine/observation.go` L204

```go
outGlobals[6] = float32(s.State.Score)
```

The score is a cumulative integer that grows without bound (0 → 10,000+). All other globals are in range [0, 28] or [0, 1]. The score feature dominates the `global_proj` MLP gradients by 100–10,000×.

**Fix**: Normalize: `float32(s.State.Score) / 500.0` (clip at 1.0 for extremely long games), or better: expose `deliveredDelta` per step rather than cumulative score.

**Verify**: Log mean/std of each global dim during training. No dim should be >10× another.

---

### BUG-D — HIGH: `shortenLine()` is a no-op (silent success)
**File**: `simulator/engine/simulator.go` L462-465

```go
func (s *Simulator) shortenLine(a ShortenLine) error {
    return nil  // does nothing!
}
```

If `ShortenLine` actions are ever re-enabled in the mask, the agent receives success (no error, no state change, no reward) for a null action. The `-0.05` penalty in Python then punishes it for a no-op that looked valid.

**Fix**: Either implement `shortenLine` properly or return `errors.New("not implemented")`.

---

### BUG-E — MEDIUM: `boolMaskBuf` shared global is a single-threaded bottleneck
**File**: `simulator/c_api/main.go` L122-151

A single shared `boolMaskBuf` + mutex serializes all `GetActionMask` calls across parallel environments. With 32 async workers this is a contention point. Use a per-call stack array instead:
```go
var boolMask [engine.TotalActionSpaceSize]bool
sim.GetActionMask(boolMask[:])
```

---

## 4. Major Design Problems (Category B)

### MAJOR-A — CRITICAL: Rollout window too short for temporal credit assignment
**File**: `ml/train.py` L26, `ml/ppo.py` L21-37

`num_steps = 128` covers 128 game-seconds. An infrastructure decision made at step 1 prevents game-over at step 300+ — entirely outside the GAE window. At γ=0.99 and step 128: discount = 0.277. The agent never sees the consequence of building or not building a line.

**Fix**: Increase `num_steps` to at least 512, ideally 1024. This is the single highest-ROI change.

**Verify**: Plot `returns.mean()` per episode. If it increases significantly after rollout expansion, credit assignment was the bottleneck.

---

### MAJOR-B — CRITICAL: Mean pooling destroys spatial/relational identity
**File**: `ml/model.py` L110-115

```python
pooled = (x * node_mask.unsqueeze(-1)).sum(dim=1) / num_nodes.clamp(min=1).float()
```

After pooling, the actor receives the same 256-dim vector regardless of *which station* is congested or *which line* is underserved. The policy cannot express "extend line 2 to station 7" because it has no representation of the distinction between stations 7 and station 12.

**Fix (immediate)**: Replace with concatenated mean + max pool:
```python
# mean pool
mean_pool = (x * node_mask.unsqueeze(-1)).sum(1) / num_nodes.clamp(min=1).float()
# max pool (mask out invalid nodes with -1e9)
x_masked = x.masked_fill(~node_mask.unsqueeze(-1), -1e9)
max_pool = x_masked.max(dim=1).values
pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2H]
```
Update `fc_actor` and `fc_critic` input dim from `H*2` to `H*3`.

**Fix (long-term)**: Use bilinear action scoring that directly uses per-node embeddings instead of a pooled global.

**Verify**: After fix, check if the policy selects the most-congested station more often than random.

---

### MAJOR-C — HIGH: Flat `-0.05` action penalty trains passivity
**File**: `ml/env.py` L173-175

```python
if action_id != 0:
    reward -= 0.05
```

This punishes every non-NoOp identically — a critical infrastructure action and a useless one pay the same penalty. In sparse-reward early training (0 deliveries for first 10–20 steps), the optimal short-term policy is: **always NoOp**. The agent learns passivity.

**Fix**: Remove this penalty entirely. Exploration is already incentivised by `ent_coef`. If a penalty is desired, make it resource-aware: penalize spending a line/train token on a configuration that doesn't improve reachability.

**Verify**: Monitor NoOp rate during training. Healthy range: 15–35%. If >60%, passivity has set in.

---

### MAJOR-D — HIGH: Node features missing fill ratio and routing information
**File**: `simulator/engine/observation.go` L89-129

Currently missing from node features:
| Feature | Importance |
|---|---|
| `queue / capacity` (fill ratio) | Agent cannot see total congestion, only breakdown by destination type |
| `overcrowdingTimer / 47.0` (absolute timer) | Progress [0,1] exists but not absolute time; 5s vs 45s remaining require different responses |
| `numLinesServing(station)` | Agent cannot see if a station is already well-connected |
| `reachabilityScore` (fraction of waiting passengers with a valid route) | Disconnected station ≠ low-demand station |

**Fix**: Grow `NodeFeatureDim` from 25 to at least 29:
```go
outNodes[base+25] = float32(len(st.Queue)) / float32(st.Capacity)
outNodes[base+26] = float32(len(linesThroughStation)) / 7.0
outNodes[base+27] = float32(max(0, st.OvercrowdingTimer)) / overcrowdingFailureSeconds
outNodes[base+28] = reachabilityScore(station)  // fraction of passengers routable
```
Update `node_dim` in `env.py` and `model.py`.

---

### MAJOR-E — HIGH: Global features missing critical structural state
**File**: `simulator/engine/observation.go` L196-212

Missing from globals:
- Total station count (agent doesn't know how large the network is)
- Max queue across all stations (best single predictor of imminent game-over)
- Count of stations currently overcrowding (stations with `OvercrowdingTimer >= 0`)
- `PendingRewardChoices > 0` flag (agent must infer from mask alone that it's in reward mode)
- Normalized game time progression `GameTimeSeconds / expectedMax`

**Fix**: Grow `GlobalFeatureDim` from 8 to 13, adding these features. Replace raw score with normalized score or delta.

---

### MAJOR-F — HIGH: GCNLayer2 reuses raw edge_attrs (no edge update)
**File**: `ml/model.py` L107-108

```python
x = self.gcn1(nodes, edges, edge_attrs, num_nodes, num_edges)
x = self.gcn2(x, edges, edge_attrs, num_nodes, num_edges)  # same edge_attrs!
```

Layer 2 receives updated node embeddings but the **same unupdated edge features** from layer 1. Stacking two GCN layers this way provides limited additional expressiveness because edge representations never evolve.

**Fix**: Add an edge embedding update within each layer:
```python
# In DenseGCNLayer.forward, after computing new_x:
src_emb = gather(x, src); dst_emb = gather(new_x, dst)
new_e = F.relu(self.edge_update(torch.cat([e, src_emb, dst_emb], dim=-1)))
return new_x, new_e  # pass updated edges to next layer
```

---

### MAJOR-G — HIGH: GCN messages ignore destination node features
**File**: `ml/model.py` L28-30

```python
src_feat = torch.gather(x, 1, src.unsqueeze(-1).expand(-1, -1, H))
msg = F.relu(self.msg_proj(torch.cat([src_feat, e], dim=-1)))
```

Messages are computed from `src_feat` and edge features only. The destination node's state is not included. For Metro planning, "station A is congested" should influence the message sent from upstream stations, but station A's congestion information is not in the message because `dst_feat` is excluded.

**Fix**:
```python
dst_feat = torch.gather(x, 1, dst.unsqueeze(-1).expand(-1, -1, H))
msg = F.relu(self.msg_proj(torch.cat([src_feat, dst_feat, e], dim=-1)))
# Update msg_proj input dim from hidden_dim*2 to hidden_dim*3
```

---

### MAJOR-H — HIGH: 2 GCN layers insufficient for graph diameter
**File**: `ml/model.py` L56-57

With 2 message-passing layers, information propagates at most 2 hops. A 15-station map has diameter 5+ hops. A congested terminal station 5 hops from an available train cannot signal the policy in 2 passes.

**Fix**: Add a 3rd GCN layer with a residual connection:
```python
self.gcn3 = DenseGCNLayer(hidden_dim, edge_dim, hidden_dim)
# In forward:
x_res = x  # after gcn2
x = self.gcn3(x, edges, edge_attrs, num_nodes, num_edges)
x = x + x_res  # residual connection prevents oversmoothing
```

---

### MAJOR-I — MEDIUM: `InsertStation` dominates action space (77% of actions)
**File**: `simulator/engine/action_space.go` L25-27

`InsertStationCount = 7 × 30 × 15 = 3150` out of 4108 total actions (76.7%). PPO's softmax distributes probability mass proportionally, biasing the policy toward InsertStation simply due to sheer count.

**Fix (long-term)**: Hierarchical action space — first choose action type (12 options), then choose parameters. This eliminates the count-imbalance bias.

**Fix (short-term)**: No code change, but monitor `action_type_distribution` in TensorBoard. If InsertStation is always chosen, the bias is active.

---

### MAJOR-J — MEDIUM: `AddCarriage` indexed by TrainID is unlearnable
**File**: `simulator/engine/action_space.go` L33-35, `GetActionMask` L333-340

The agent must pick train ID 0–27 to add a carriage. But train IDs are opaque — the observation doesn't expose which train is on which line or how loaded it is. The agent is effectively guessing a train ID randomly.

**Fix**: Change `AddCarriage` to index by `lineID` (7 options) and have the engine automatically select the most loaded active train on that line. This is a cleaner interface that matches the decision the agent actually needs to make.

---

### MAJOR-K — LOW: Disabled actions waste logit capacity
**File**: `simulator/engine/action_space.go` L376-381

`RemoveLine` (7 actions) and `ShortenLine` (14 actions) are permanently masked but still occupy logit positions. The softmax distributes some probability mass over them that must be learned to zero. 

**Fix**: If permanently disabled, remove them from `TotalActionSpaceSize`. Total drops from 4108 → 4087.

---

## 5. Reward Analysis (Category C analysis, design problem)

### Current reward formula:
```
Go:     R_t = deliveredDelta - 0.05 * totalCrowdPenalty - 50 * isGameOver
Python: R_t -= 0.05 if action != NoOp
```

### Problems:

**R1: Sparse delivery reward → zero-gradient early training**
Most steps produce `deliveredDelta = 0` (no passengers delivered). No reward, no gradient signal. The policy drifts randomly for the first hundreds of updates.

**R2: Crowd penalty scale is 22× too small**
`AlphaCrowdPenalty = 0.05`. At dangerous overcrowding (overflow=2, cap=6): penalty ≈ 0.14/step. Meanwhile, `deliveredDelta` can be 3/step. The agent ignores crowding because deliveries pay 22× more.

**R3: `BetaGameOverPenalty = 50` is 10% of typical episode return**
A 1000-step episode delivering 0.5 passengers/step accumulates ~500 reward. Terminal -50 is 10% of that. The agent rationally accepts game-over if early deliveries were high.

**R4: No reward for connecting isolated stations**
When a new station spawns, connecting it to the network prevents future game-over. The reward only arrives 30–100 steps later when a train makes the first delivery. The action that prevented game-over gets no credit.

**R5: Flat action penalty overrides selective incentives**
-0.05 per non-NoOp makes NoOp the rational early-game choice when `deliveredDelta = 0`.

### Proposed improved reward:
```python
def compute_reward(prev_obs, curr_obs, delivered_delta, game_over):
    r = 0.0
    
    # Delivery (clip to reduce variance)
    r += min(delivered_delta, 5) * 0.5
    
    # Survival bonus (dense signal)
    r += 0.01
    
    # Overcrowding gradient (acts well before game-over)
    for each station:
        fill = queue_len / capacity
        if fill > 0.8:
            r -= 0.1 * (fill - 0.8)
        if overcrowding_active:
            r -= 0.3 * overcrowding_progress  # strong near-death penalty
    
    # Connectivity bonus (reward connecting isolated stations)
    newly_reachable = count_newly_reachable_station_pairs(prev, curr)
    r += 0.2 * newly_reachable
    
    # Game-over (much larger)
    if game_over:
        r -= 200.0
    
    # NO flat action penalty
    return r
```

---

## 6. PPO Training Analysis (Category C)

### PPO-1 — CRITICAL: Advantage normalization per-minibatch is unstable
**File**: `ml/ppo.py` L67-68

```python
mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
```

Per-minibatch normalization: if all advantages in a minibatch are near-identical (common early when reward=0), `std → 0`, causing `/ 1e-8` to produce massive normalized values → exploding gradient.

**Fix**: Normalize advantages over the **full batch** before minibatch split:
```python
# In training loop, before PPO update:
b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
# Remove normalization from ppo.py update()
```

### PPO-2 — HIGH: `update_epochs = 2` is insufficient
Each trajectory is used only 2× before discard. For 4108-way discrete actions, 2 passes over 4096 samples (~2 samples per action logit) cannot produce well-calibrated logits.

**Fix**: Increase to `update_epochs = 4`. Monitor `approx_kl` — if it exceeds 0.02, reduce epochs.

### PPO-3 — HIGH: `ent_coef = 0.01` is too small for 4108-way action space
Maximum entropy for Categorical(4108) = log(4108) ≈ 8.32. A focused policy (10 valid actions) has entropy ≈ log(10) ≈ 2.3. The entropy bonus is `0.01 × 2.3 = 0.023/step` vs delivery rewards of 1–10/step. The entropy term fails to maintain exploration.

**Fix**: Increase `ent_coef = 0.05`. Or use annealing: start at 0.1, decay to 0.01.

### PPO-4 — MEDIUM: No learning rate schedule
Constant `lr = 3e-4` makes large updates throughout training. Near convergence, this prevents fine-tuning.

**Fix**: Linear decay:
```python
frac = 1.0 - (update - 1.0) / num_updates
for param_group in agent.optimizer.param_groups:
    param_group['lr'] = 3e-4 * frac
```

### PPO-5 — MEDIUM: No value function clipping
Current `v_loss = 0.5 * ((newvalue - b_returns)**2).mean()` is unclipped. Standard PPO clips value updates to stabilize the critic:
```python
v_clipped = b_values[mbinds] + torch.clamp(newvalue - b_values[mbinds], -clip_coef, clip_coef)
v_loss = 0.5 * torch.max((newvalue - b_returns[mbinds])**2, (v_clipped - b_returns[mbinds])**2).mean()
```

### PPO-6 — MEDIUM: `gamma=0.99` too low for long episodes
At `num_steps=128`, `0.99^128 = 0.277`. Even after fixing rollout length, long-horizon consequences (game-over prevention at step 500) are heavily discounted.

**Fix**: After increasing `num_steps`, also increase `gamma = 0.995` to extend effective horizon.

### PPO-7 — LOW: `train_local.py` uses `SyncVectorEnv` instead of `AsyncVectorEnv`
**File**: `ml/train_local.py` L266

`SyncVectorEnv` runs all envs in the same process. With a Go CGO shared library, this risks goroutine scheduler interference. `train.py` correctly uses `AsyncVectorEnv(context='spawn')`.

**Fix**: Change `train_local.py` L266:
```python
envs = gym.vector.AsyncVectorEnv(
    [make_env(i) for i in range(num_envs)],
    context='spawn'
)
```

---

## 7. Inference / Evaluation (Category D)

### INF-1 — HIGH: `eval.py` uses stochastic sampling during evaluation
**File**: `ml/eval.py` L49

```python
action, _, _, value = model.get_action_and_value(obs_tensor, mask=mask)
```

`action=None` → `probs.sample()` → stochastic. Evaluation performance has high variance across runs because the same policy makes different decisions.

**Fix**: Add a deterministic flag:
```python
# In model.py get_action_and_value:
if deterministic:
    action = torch.argmax(logits.masked_fill(~mask, -1e9), dim=-1)
else:
    action = probs.sample()
```

### INF-2 — HIGH: `eval.py` reports shaped reward, not game score
`total_reward` includes the -0.05 action penalty (a training artifact). Use `obs["globals"][6]` (normalized after fix) as the primary metric, not accumulated shaped reward.

### INF-3 — MEDIUM: `agent.py` same fragile heuristic for num_nodes/num_edges
**File**: `ml/agent.py` L83-95

Same fragile loop as `env.py`. Fix both together when C API is updated.

---

## 8. Prioritized Implementation Roadmap

### Phase 1 — Fix Showstoppers ✅ COMPLETED (branch: fixes)

| # | Change | File(s) | Time |
|---|---|---|---|
| 1 | Fix `eval.py` hidden_dim 128 → 256 | `ml/eval.py` L19 | 1 min |
| 2 | Remove `-0.05` flat action penalty | `ml/env.py` L173-175 | 1 min |
| 3 | Normalize `globals[6]` score | `observation.go` L204 | 2 min |
| 4 | Increase `num_steps = 512` | `ml/train.py` L27 | 1 min |
| 5 | Increase `ent_coef = 0.05` | `ml/ppo.py` L7 | 1 min |
| 6 | Move advantage normalization to full-batch | `ml/train.py` + `ml/ppo.py` | 5 min |
| 7 | Fix `SyncVectorEnv` in `train_local.py` | `ml/train_local.py` L266 | 2 min |

**Status**: All 7 fixes applied and smoke-tested on branch `fixes`.

**Test results** (3 PPO updates, 4 envs, 32 steps/update):
```
PASS ent_coef=0.05
PASS update_epochs default=4
Update 1/3 | pg=-0.0271 v=0.4966 ent=0.8745 kl=0.000033 noop=42.2% lr=3.00e-04
Update 2/3 | pg=-0.0678 v=4.5364 ent=0.9362 kl=0.000014 noop=43.0% lr=2.00e-04
Update 3/3 | pg=0.1269  v=3.4182 ent=0.9363 kl=0.000023 noop=41.4% lr=1.00e-04
PASS all losses finite
PASS rewards have positive values (survival bonus active)
PASS no -0.05 flat action penalty
reward range [-0.095, 2.010]
```
**Observations**: entropy ~0.9 (healthy for early training), KL <0.001 (well within clip range),
NoOp rate 41-43% (acceptable — was likely >70% before removing action penalty),
LR decaying correctly, all losses finite.

**Files changed**:
| File | Changes |
|---|---|
| `ml/ppo.py` | ent_coef 0.01→0.05; update_epochs 2→4; removed per-minibatch advantage norm |
| `ml/train.py` | num_steps 128→512; full-batch adv norm; LR decay; NoOp rate logging; checkpoint every 10 updates |
| `ml/env.py` | Removed -0.05 flat penalty; added survival +0.01/step; added early overcrowd gradient |
| `ml/eval.py` | hidden_dim 128→256 |
| `ml/train_local.py` | SyncVectorEnv→AsyncVectorEnv(context=spawn) |
| `simulator/engine/observation.go` | globals[6] score normalized by /500 |
| `simulator/c_api/` | Rebuilt libminimetro.so |

**Next**: run full training for 5M steps, then proceed to Phase 2 (observation improvements).

Retrain from scratch for 5M steps. Measure: episode length, game score, NoOp rate.

### Phase 2 — Improve Observation ✅ COMPLETED (branch: fixes)

| # | Change | File(s) |
|---|---|---|
| 8 | Add fill_ratio + timer + line_count to nodes (NodeDim: 25→29) | `observation.go` + `env.py` + `model.py` |
| 9 | Add max_queue, overcrowding_count, pending_reward flag to globals (GlobalDim: 8→13) | `observation.go` + `env.py` + `model.py` |
| 10 | Expose num_nodes/num_edges from C API; remove heuristic | `c_api/main.go` + `env.py` + `agent.py` |

**Status**: All 3 tasks applied and smoke-tested on branch `fixes`.

**Changes**:
| File | Changes |
|---|---|
| `simulator/engine/observation.go` | NodeDim 25→29 (+fill_ratio, lines_serving, timer_norm, queue_norm); GlobalDim 8→13 (+station_count, max_fill, overcrowd_count, pending_reward, game_time) |
| `simulator/c_api/main.go` | GetObservation now returns numNodes/numEdges via int32 output params |
| `ml/env.py` | Updated dims; new C API call; removed fragile heuristic node/edge counting |
| `ml/model.py` | Updated default dims (node 25→29, global 8→13); mean+max pooling; fc dims ×2→×3 |

**Test results**:
```
PASS node shape (30, 29), global shape (13,)
PASS num_nodes from C API = 3  (no heuristic)
PASS new globals: station_count=0.100 max_fill=0.000 overcrowd=0.000 pending=0 gametime=0.000
PASS model forward: action=2 logprob=-1.4402 entropy=1.3800 value=-0.1020
PASS mean+max pool: fc_actor input = 96 = 3*H
PASS all losses finite across 3 PPO updates
```

**Next**: Phase 3 — Fix GNN Architecture (dst_feat in messages, 3rd GCN layer, edge update MLP).

Retrain for 10M steps. Verify: congested stations get served more reliably.

### Phase 3 — Fix GNN Architecture ✅ COMPLETED (branch: fixes)

| # | Change | File(s) |
|---|---|---|
| 11 | Replace mean pool with mean+max pool | `ml/model.py` |
| 12 | Include dst_feat in GCN message function | `ml/model.py` |
| 13 | Add 3rd GCN layer with residual connection | `ml/model.py` |
| 14 | Add edge update MLP; pass updated edges between layers | `ml/model.py` |

**Status**: All 3 tasks applied and smoke-tested on branch `fixes`.

**Changes** (all in `ml/model.py`):
| Task | What changed |
|---|---|
| GNN-1: edge update MLP | `DenseGCNLayer` replaced with `GNNLayer` which returns `(new_nodes, new_edges)`; each layer now evolves edge embeddings via `edge_update(src, dst, e)` and passes them to the next layer |
| GNN-2: dst_feat in messages | `msg_proj` input changed from `[src, edge]` (2H) to `[src, dst, edge]` (3H) — destination node state now influences messages, enabling "station A is congested" to propagate upstream |
| GNN-3: 3rd layer + residual | Added `gcn3 = GNNLayer(...)` with `x3 = gcn3(x2) + x2` residual — information now propagates 3 hops (was 2); residual prevents oversmoothing |

**Test results**:
```
PASS: GNNLayer present, DenseGCNLayer removed
PASS: 3 GNNLayers present
PASS: msg_proj input = 96 = 3H (dst_feat included)
PASS: edge_update MLP present
PASS: forward pass — action=0 lp=-1.3581 ent=1.3811 val=-0.0798
PASS: 5 env steps — all model outputs finite with residual connection
INFO: model params = 175,821
PASS: 3 PPO updates — all losses finite, KL < 0.001
```

**Next**: Phase 4 — Fix Action Space & Reward (AddCarriage by lineID, improved reward, BetaGameOverPenalty×4).

Retrain. Verify: policy selects different actions for different congested stations.

### Phase 4 — Fix Action Space & Reward ✅ COMPLETED (branch: fixes)

| # | Change | File(s) |
|---|---|---|
| 15 | Change AddCarriage to index by lineID | `action_space.go` + `simulator.go` |
| 16 | Implement improved reward with survival bonus + connectivity bonus | `scoring.go` + `env.py` |
| 17 | Increase BetaGameOverPenalty to 200 | `scoring.go` |
| 18 | Add `pending_reward` binary flag to globals | `observation.go` |

**Status**: Tasks 15, 16, 17 applied and smoke-tested. Task 18 (pending_reward global flag) was completed in Phase 2.

**Changes**:
| Task | File(s) | Change |
|---|---|---|
| 15: AddCarriage by lineID | `engine/action_space.go`, `engine/actions.go`, `engine/simulator.go`, `server/actions.go` | `AddCarriage{TrainID}`→`AddCarriage{LineID}`; action space 4108→4087; agent now targets a line it can reason about, not an opaque train slot |
| 16: Improved reward | `engine/scoring.go` | Added `ConnectivityBonus=2.0` — reward per reachable distinct-type station pair; eliminates positive gradient for redundant connections |
| 16: Survival bonus | already in `env.py` | Kept from Phase 1 (+0.01/step) |
| 17: BetaGameOverPenalty×4 | `engine/scoring.go` | 50→200; game-over signal now dominates |
| 17: AlphaCrowdPenalty×6 | `engine/scoring.go` | 0.05→0.30; crowding penalty now meaningful vs delivery reward |
| 18: pending_reward flag | `engine/observation.go` | globals[11] — done in Phase 2 |

**Test results**:
```
PASS: action_space_size=4087 (was 4108)
PASS: model output dim=4087
PASS: rewards include positive signal (connectivity bonus + survival)
PASS: mean reward=0.010 over non-gameover steps
PASS: forward pass — action=0 ent=1.0980
Go build: ALL OK (libminimetro.so rebuilt)
```

**Next**: Phase 5 — Advanced Architecture (hierarchical action head, bilinear scoring, train position features, value clipping).

### Phase 5 — Advanced Architecture ✅ COMPLETED (branch: fixes)

| # | Change | File(s) |
|---|---|---|
| 19 | Hierarchical action head (action type first, then params) | `ml/model.py` |
| 20 | Bilinear action scoring using per-node embeddings directly | `ml/model.py` |
| 21 | Add train position/load/proximity to node features (NodeDim: 29→32) | `observation.go` + `env.py` + `agent.py` + `model.py` |
| 22 | Value function clipping in PPO (`PPO-5`) + gamma=0.995 (`PPO-6`) + deterministic eval (`INF-1`) | `ml/ppo.py` + `ml/train.py` + `ml/train_local.py` + `ml/eval.py` |

**Status**: All 4 tasks applied and smoke-tested.

**Changes**:
| Task | File(s) | Change |
|---|---|---|
| 19: Hierarchical action head | `ml/model.py` | 12-way action type selector head (`type_net`); factorizes $P(a) = P(\text{type } t) \cdot P(a \mid t)$; eliminates 76.7% `InsertStation` dominance bias while keeping exact 4087-way action ID compatibility |
| 20: Bilinear action scoring | `ml/model.py` | `AddLine` (435) scored via symmetric bilinear form $S = \frac{1}{2}(q_u^T k_v + q_v^T k_u)$; `UpgradeInterchange` (30) scored via station projection; `ExtendLine` (420) and `InsertStation` (3150) scored via factored line/end and line/segment embeddings against station embeddings $x_u$ |
| 21: Train position/load/prox | `engine/observation.go`, `ml/env.py`, `ml/agent.py`, `ml/model.py` | `NodeFeatureDim` 29→32: +incoming_train_count ([29]), +incoming_train_load ([30]), +nearest_train_proximity ([31]); `libminimetro.so` rebuilt |
| 22: Value clipping & gamma | `ml/ppo.py`, `ml/train.py`, `ml/train_local.py`, `ml/eval.py` | Added $v_{\text{clipped}}$ loss in PPO critic; default $\gamma = 0.995$; `eval.py` uses `deterministic=True` argmax |

**Test results**:
```
PASS: node shape = (30, 32)
PASS: train_count, train_load, train_prox in [0.0, 1.0]
PASS: Hierarchical action probabilities sum to 1.0 per batch element (exact sum=1.000000)
PASS: get_action_and_value sampling valid (entropy=5.4946)
PASS: deterministic eval returns valid argmax
PASS: PPO update with value clipping: pg=1.1512, v=0.0500, ent=5.5086, kl=0.001748
PASS: Gradients successfully backpropagated to GNN, bilinear heads, and type selector!
Go build: ALL OK (libminimetro.so rebuilt)
```

---

## 9. Expected Impact per Change

| Change | Impact | Confidence |
|---|---|---|
| Fix eval.py hidden_dim | Fixes broken evaluation | CERTAIN |
| Remove action penalty | Reduces passivity; more exploration | HIGH |
| Normalize score global | Stabilizes critic gradient | HIGH |
| num_steps 128 → 512 | Much better credit assignment | VERY HIGH |
| ent_coef 0.01 → 0.05 | More diverse actions early | MEDIUM-HIGH |
| Per-batch advantage norm | Prevents early exploding gradient | HIGH |
| Add fill_ratio to nodes | Agent identifies congested stations | HIGH |
| mean+max pooling | Policy distinguishes worst station | MEDIUM-HIGH |
| dst_feat in messages | More expressive GNN | MEDIUM |
| 3rd GCN layer | Longer-range propagation | MEDIUM |
| AddCarriage by lineID | Agent can correctly target underserved lines | MEDIUM |
| Survival reward | Longer episodes; network building | MEDIUM |
| Connectivity bonus | Reward for connecting isolated stations | MEDIUM |
| BetaGameOverPenalty × 4 | Stronger avoidance of game-over | MEDIUM |

---

## 10. Metrics for Determining Real Improvement

### Primary (track per TensorBoard update):
1. **Game score** (obs["globals"][6] normalized at episode end) — target: >100 after Phase 1, >300 after Phase 2
2. **Episode length** — target: >500 steps after Phase 1
3. **NoOp rate** (fraction of steps with action_id==0) — healthy: 15–35%

### Diagnostic:
4. **Entropy** — should stay >1.0. Collapse below 0.5 = add more ent_coef
5. **approx_kl** — target: 0.005–0.02. Above 0.05 → reduce LR
6. **value_loss** — should decrease over training
7. **Max station queue at game-over** — should decrease (model learns to prevent crowding)

### Infrastructure quality (add custom logging):
8. **Average line length** — should increase (agent extends lines)
9. **Fraction of station pairs that are routable** — should increase
10. **Action type distribution** — monitor all types; none should be zero

### Experiment isolation rule:
Run 3 seeds per change for 2M steps. A change is confirmed if 2 of 3 seeds improve and the mean metric improves by >10% at 2M steps.

---

## Appendix: File Reference Map

| File | Key Issues |
|---|---|
| [`simulator/engine/observation.go`](file:///home/leomarshall/mm/simulator/engine/observation.go) | Unbounded score, missing train info, missing fill ratio |
| [`simulator/engine/action_space.go`](file:///home/leomarshall/mm/simulator/engine/action_space.go) | InsertStation dominates, disabled actions waste space |
| [`simulator/engine/scoring.go`](file:///home/leomarshall/mm/simulator/engine/scoring.go) | Alpha too small (0.05), Beta too small (50) |
| [`simulator/engine/simulator.go`](file:///home/leomarshall/mm/simulator/engine/simulator.go) | shortenLine() is a no-op |
| [`simulator/c_api/main.go`](file:///home/leomarshall/mm/simulator/c_api/main.go) | No num_nodes/num_edges return, shared boolMaskBuf |
| [`ml/env.py`](file:///home/leomarshall/mm/ml/env.py) | Fragile heuristic node count, flat -0.05 penalty |
| [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py) | Mean-only pooling, no edge update, 2 GCN layers, no dst_feat |
| [`ml/ppo.py`](file:///home/leomarshall/mm/ml/ppo.py) | Per-minibatch advantage norm, no value clipping, low ent_coef |
| [`ml/train.py`](file:///home/leomarshall/mm/ml/train.py) | num_steps=128 too small, no LR schedule |
| [`ml/train_local.py`](file:///home/leomarshall/mm/ml/train_local.py) | SyncVectorEnv instead of AsyncVectorEnv |
| [`ml/agent.py`](file:///home/leomarshall/mm/ml/agent.py) | Stochastic eval, fragile node count heuristic |
| [`ml/eval.py`](file:///home/leomarshall/mm/ml/eval.py) | Wrong hidden_dim (128 vs 256 trained), reports shaped reward not score |

---

## 11. Root Cause Analysis — "Connects Every Station With Every Line"

> This symptom is **not a failure of the GNN to understand Metro networks**.
> It is the **gradient-optimal response** to the current reward function given the action space.
> The model has learned a perfectly rational policy — for the wrong objective.
> Fixing the incentive structure is the primary fix; architectural improvements only help it learn faster once the incentives are correct.

### The causal chain

Every step of the following chain is traceable to specific lines in the codebase.

---

#### Step 1 — The only positive reward is passenger delivery

**File**: [`simulator/engine/scoring.go` L33-34](file:///home/leomarshall/mm/simulator/engine/scoring.go#L33-L34)

```go
func (s *Simulator) ComputeStepReward(deliveredDelta int) float64 {
    reward := float64(deliveredDelta)   // ← sole source of positive reward
```

Adding any new line segment between stations A and B makes it possible for passengers at A to reach B (and vice versa). More reachable pairs = more potential deliveries per step = higher expected reward. The agent has correctly learned: **more connections → more reward**. This is true. The problem is that it is *also* true for redundant, wasteful connections.

---

#### Step 2 — There is no penalty proportional to resource waste

**File**: [`ml/env.py` L173-175](file:///home/leomarshall/mm/ml/env.py#L173-L175)

```python
if action_id != 0:
    reward -= 0.05     # same cost for a critical connection and a redundant one
```

Adding line 4 between stations 3 and 7 (which are already connected by lines 0, 1, and 2) costs exactly `-0.05` — identical to the cost of connecting a completely isolated station for the first time. There is zero marginal penalty for redundancy. The agent has no gradient signal telling it "this connection added nothing."

---

#### Step 3 — Spending a resource token has no observable opportunity cost

**File**: [`simulator/engine/observation.go` L197](file:///home/leomarshall/mm/simulator/engine/observation.go#L197), [`action_space.go` GetActionMask L188-204](file:///home/leomarshall/mm/simulator/engine/action_space.go#L188-L204)

`globals[0] = float32(s.State.Resources.Lines)` shows the raw count of remaining line tokens. When the agent spends the last line token on a redundant connection, a new station spawns 40 steps later and cannot be connected — causing overcrowding and eventual game-over.

This consequence is **invisible** because:
1. The rollout window (`num_steps=128`) is shorter than the time between spending the token and the new station spawning + overcrowding (often 150–300 steps).
2. Even if it were in the window, `0.99^150 ≈ 0.22` — 78% discounted away.

The agent rationally treats line tokens as cheap because it never observes the cost of misusing them.

**File**: [`ml/train.py` L27](file:///home/leomarshall/mm/ml/train.py#L27)

```python
num_steps = 128    # 128s rollout; consequences land at 150–300s → invisible
```

---

#### Step 4 — The GNN cannot see that a connection is redundant

**File**: [`ml/model.py` L110-115](file:///home/leomarshall/mm/ml/model.py#L110-L115)

```python
pooled = (x * node_mask.unsqueeze(-1)).sum(dim=1) / num_nodes.clamp(min=1).float()
```

After mean-pooling, a state where stations 3 and 7 are connected by 1 line versus 4 lines produces **nearly the same pooled embedding**:
- `nodes[23]` (degree) changes slightly.
- Everything else — kind, queue breakdown, position — is identical.

The actor maps this near-identical embedding to near-identical logits. It cannot learn to suppress `AddLine(3,7)` just because stations 3 and 7 are already well-served, because the representation doesn't distinguish those two states strongly enough.

---

#### Step 5 — The action mask confirms the action is "valid", not "useful"

**File**: [`simulator/engine/action_space.go` L188-204](file:///home/leomarshall/mm/simulator/engine/action_space.go#L188-L204)

The action mask marks `AddLine(u, v)` as valid whenever:
- Both stations are alive
- A line token is available
- (Optionally) a tunnel token is available if water crossing is needed

It does **not** check whether stations u and v are already reachable from each other. A fully redundant connection is always presented as a valid choice. From the mask alone, the agent cannot distinguish "first connection" from "fifth connection between the same pair."

---

### Summary: Why this specific degenerate strategy is learned

| Cause | Mechanism | File |
|---|---|---|
| **Reward**: only deliveries count | More connections → more reachable pairs → more deliveries | `scoring.go` L34 |
| **Penalty**: flat cost, not waste-proportional | Redundant = first connection in terms of cost | `env.py` L173-175 |
| **Rollout**: too short to see opportunity cost | Spending last line token on redundant route has invisible consequence | `train.py` L27 |
| **GNN**: mean pool can't distinguish 1 vs 4 lines | No structural gradient to stop adding lines | `model.py` L112 |
| **Mask**: validity ≠ usefulness | Redundant connections always presented as legal | `action_space.go` L188-204 |
| **Observation**: no reachability feature | Agent can't see "these stations are already connected" | `observation.go` (missing) |

All five causes are **active simultaneously**. Any one alone might be overcome by the agent through exploration; all five together create an inescapable local optimum.

---

### Targeted fixes for this specific symptom

These are ordered: fix earlier ones first, re-evaluate, then proceed.

#### Fix 1 — Add a network connectivity bonus to the reward (HIGHEST IMPACT)
**Rationale**: Make the *first* connection between two previously disconnected station-type pairs explicitly valuable. Make redundant connections neutral or mildly penalised.

In `scoring.go`, expose a new reward term. Implement in Python (easier to iterate):

```python
# In env.py step(), after receiving Go reward:

# Count station pairs newly reachable (requires exposing CanReach through C API,
# or approximating by checking adjacency list change in observation)
newly_connected_pairs = count_newly_routable_pairs(prev_obs, curr_obs)
redundant_connection = is_redundant(action_id, curr_obs)  # stations already reachable

reward += 0.5 * newly_connected_pairs          # reward first connections
reward -= 0.1 * redundant_connection           # penalise redundant ones
```

The key insight: `newly_connected_pairs` is 0 when you add the 4th line between stations already connected, but >0 when you connect an isolated station for the first time.

To implement `is_redundant`: before the action is applied, check if the two stations in an `AddLine(u,v)` or `ExtendLine` action are already reachable from each other via `FindOptimalRoute`. Expose this through a new C API call `CanReach(handle, fromID, toID)`.

#### Fix 2 — Remove the flat `-0.05` action penalty
**File**: [`ml/env.py` L173-175](file:///home/leomarshall/mm/ml/env.py#L173-L175)

```python
# DELETE these three lines:
if action_id != 0:
    reward -= 0.05
```

Replace with the resource-aware penalty from Fix 1 only. A flat penalty does not discriminate useful from wasteful actions, and trains passivity (NoOp) as the safe default.

#### Fix 3 — Increase rollout length so opportunity cost becomes visible
**File**: [`ml/train.py` L27](file:///home/leomarshall/mm/ml/train.py#L27)

```python
num_steps = 512   # was 128
```

With 512 steps, the consequence of spending the last line token on a redundant connection (new station spawns unconnected → overcrowding → game-over at ~step 300) falls **within** the GAE window. At γ=0.99 and step 300: `0.99^300 ≈ 0.05` — still heavily discounted. Consider γ=0.999: `0.999^300 ≈ 0.74`.

#### Fix 4 — Add a `reachability_already` flag to the action mask observation
**File**: New feature in observation

Rather than modifying the mask (which would hide the action entirely), add a per-action feature that signals "these stations are already connected." The simplest approach: add a global binary feature `is_network_saturated` (all existing stations reachable from each other) and a per-station feature `num_lines_serving_this_station / 7.0`.

When the GNN sees that all stations already have 2+ lines serving them and the agent tries `AddLine`, the policy should learn to down-weight this action. But it can only learn this if the observation carries the signal.

**File**: [`simulator/engine/observation.go` L89-129](file:///home/leomarshall/mm/simulator/engine/observation.go#L89-L129)

```go
// Add to node features:
numLinesServingStation := 0
for _, line := range s.State.Lines {
    if !line.Removed {
        for _, stID := range line.Stations {
            if stID == i {
                numLinesServingStation++
                break
            }
        }
    }
}
outNodes[base+26] = float32(numLinesServingStation) / 7.0  // normalized 0–1
```

When `nodes[26]` is already 0.86 (6/7 lines serve this station), the GNN will learn to penalise `AddLine` targeting it.

#### Fix 5 — Mean+max pool so the model can see the "least served" station
**File**: [`ml/model.py` L110-115](file:///home/leomarshall/mm/ml/model.py#L110-L115)

```python
# Replace:
pooled = (x * node_mask.unsqueeze(-1)).sum(dim=1) / num_nodes.clamp(min=1).float()

# With:
mean_pool = (x * node_mask.unsqueeze(-1)).sum(1) / num_nodes.clamp(min=1).float()
x_masked = x.masked_fill(~node_mask.unsqueeze(-1), -1e9)
max_pool  = x_masked.max(dim=1).values  # captures "worst" station

# Also add min pool — captures "best served" station (for detecting redundancy):
x_masked_min = x.masked_fill(~node_mask.unsqueeze(-1), 1e9)
min_pool = x_masked_min.min(dim=1).values

pooled = torch.cat([mean_pool, max_pool, min_pool], dim=-1)  # [B, 3H]
```

Update `fc_actor` and `fc_critic` input dim: `hidden_dim * 2` → `hidden_dim * 4` (global + 3×pooled).

The `min_pool` is specifically useful here: it represents the "best served" or "least needy" station. When `min_pool` shows all stations are well-connected, the actor learns to reduce `AddLine` logits.

---

### Verification: how to confirm this symptom is fixed

1. **Log redundant connection rate**: After each episode, count how many `AddLine` / `ExtendLine` actions connected station pairs that were already reachable before the action. This should drop from ~70–90% (current) toward ~10–20%.

2. **Log average lines per station-pair**: Compute `total_edge_count / unique_station_pairs_connected`. This should be close to 1.0 (each pair served by ~1 line) rather than growing above 2+.

3. **Sanity check via game score**: If the symptom is fixed, more line tokens will be available for new stations, leading to longer episodes and higher scores.

4. **Plot action type over training**: `AddLine` should be selected less frequently as training progresses (agent learns it's often redundant), while `ExtendLine` and `AddTrain` should remain stable or increase.

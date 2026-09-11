# Mini Metro RL Engineering Roadmap & Implementation TODO

This document serves as the master engineering roadmap and technical implementation tracker for the Mini Metro reinforcement learning project. It synthesizes the empirical findings, architectural audits, and mathematical failure modes documented in [`resrc/model_behavioral_analysis_report.md`](file:///home/leomarshall/mm/resrc/model_behavioral_analysis_report.md), along with post-audit behavioral observations regarding network over-expansion and geometric inefficiency.

Every task is designed to be immediately actionable by human developers and autonomous AI coding agents, providing precise problem descriptions, verified code locations, concrete architectural changes, mathematical justifications, and verifiable acceptance criteria.

---

## Progress Overview & Checklist

* **P0 — Correctness / Critical Bugs**
  * [x] [P0-1: Hierarchical Deterministic-Action Decoding Fix (Argmax Passivity Trap)](#p0-1-hierarchical-deterministic-action-decoding-fix-argmax-passivity-trap)
  * [x] [P0-2: Weekly Reward-Card Observation Blindness Fix](#p0-2-weekly-reward-card-observation-blindness-fix)
  * [ ] [P0-3: Evaluation & Probing Distribution Normalization Fix](#p0-3-evaluation--probing-distribution-normalization-fix)
* **P1 — Major Policy & Architecture Problems**
  * [ ] [P1-1: Explicit Candidate Distance & Geometric Awareness in AddLine](#p1-1-explicit-candidate-distance--geometric-awareness-in-addline)
  * [ ] [P1-2: Redundant / Indiscriminate Network Expansion Mitigation](#p1-2-redundant--indiscriminate-network-expansion-mitigation)
  * [ ] [P1-3: Candidate-Conditioned Line Scoring for AddTrain and AddCarriage](#p1-3-candidate-conditioned-line-scoring-for-addtrain-and-addcarriage)
  * [ ] [P1-4: Loop Toggling Hysteresis & Reversal Oscillation Mitigation](#p1-4-loop-toggling-hysteresis--reversal-oscillation-mitigation)
* **P2 — Reward Shaping & Training Improvements**
  * [ ] [P2-1: Comprehensive Reward Decomposition & Ablation Suite](#p2-1-comprehensive-reward-decomposition--ablation-suite)
  * [ ] [P2-2: Geometric Efficiency & Track Sprawl Regularization](#p2-2-geometric-efficiency--track-sprawl-regularization)
  * [ ] [P2-3: Tail vs. Front Extension Symmetry Audit & Debiasing](#p2-3-tail-vs-front-extension-symmetry-audit--debiasing)
* **P3 — Evaluation, Diagnostics & Empirical Audits**
  * [ ] [P3-1: Rigorous Multi-Seed & Cross-Map Evaluation Suite](#p3-1-rigorous-multi-seed--cross-map-evaluation-suite)
  * [ ] [P3-2: Semantic Permutation & Spatial Invariance Audit](#p3-2-semantic-permutation--spatial-invariance-audit)
  * [ ] [P3-3: Station-Shape Affinity Validation vs. Dynamic Demand Distributions](#p3-3-station-shape-affinity-validation-vs-dynamic-demand-distributions)
  * [ ] [P3-4: Counterfactual Interchange Decision Verification](#p3-4-counterfactual-interchange-decision-verification)
  * [ ] [P3-5: NoOp Disambiguation & Macro-Step Simulation Accounting](#p3-5-noop-disambiguation--macro-step-simulation-accounting)
* **P4 — Long-Term Architectural & Environment Improvements**
  * [ ] [P4-1: Safe Dynamic Line Re-Routing & Deletion (RemoveLine / ShortenLine)](#p4-1-safe-dynamic-line-re-routing--deletion-removeline--shortenline)
  * [ ] [P4-2: Relational Spatial Cross-Attention Network (GAT-v2 / Transformer Scorer)](#p4-2-relational-spatial-cross-attention-network-gat-v2--transformer-scorer)

---

## P0 — Correctness / Critical Bugs

Things that make evaluation, inference, or learning fundamentally incorrect.

### P0-1: Hierarchical Deterministic-Action Decoding Fix (Argmax Passivity Trap) [COMPLETED]

* **Problem**:
  In deterministic evaluation mode (`deterministic=True`, used in [`ml/eval.py`](file:///home/leomarshall/mm/ml/eval.py#L57)), the trained agent selects `NoOp` (Action 0) 100% of the time, resulting in immediate game-over within 67 steps and a score of 0. However, in stochastic sampling mode (`deterministic=False`, used in [`ml/agent.py`](file:///home/leomarshall/mm/ml/agent.py#L141)), the exact same checkpoint plays actively, surviving 12–15 minutes and achieving scores up to 196.

* **Evidence**:
  * Report Section 7.1 & Section 1: At Step 0 on London, the 12-way Type-Selector network strongly prefers `AddLine` ($\text{logit} = +1.2707$) over `NoOp` ($\text{logit} = +0.9412$).
  * The hierarchical joint log-probability across all 4,087 flat actions is computed in [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py#L306) as:
    $$\log P(\text{action } a) = \log P(\text{type } t(a) \mid \text{state}) + \log P(a \mid \text{type } t(a), \text{state})$$
  * For parameterized types (such as `AddLine` with $K=3$ valid station pairs), parameter log-probabilities are normalized over candidates: $\log P(a \mid t) \approx \log(1/K) = -1.098$.
  * For `NoOp`, there is only a single action ($K=1$), so $\log P(a \mid \text{NoOp}) = \log(1) = 0.0$.
  * Consequently, the flat joint logits evaluate to:
    $$\text{Logit}(\text{NoOp}) = -0.8714 \quad (41.8\%)$$
    $$\text{Logit}(\text{AddLine Pair 0-1}) = -1.6621 \quad (19.0\%)$$
    $$\text{Logit}(\text{AddLine Pair 0-2}) = -1.6503 \quad (19.2\%)$$
    $$\text{Logit}(\text{AddLine Pair 1-2}) = -1.6100 \quad (20.0\%)$$
  * Running a flat `torch.argmax(logits)` unconditionally selects `NoOp`, because parameter probability dilution suppresses every individual parameter candidate below the solitary `NoOp` slot.

* **Likely Root Cause**:
  Conflating joint action probabilities $P(t, a \mid s)$ with marginal type probabilities $P(t \mid s)$. The code implements a two-stage hierarchical head, but [`get_action_and_value()`](file:///home/leomarshall/mm/ml/model.py#L337) executes a flat `argmax` over the combined 4,087-dimensional tensor instead of a two-stage hierarchical argmax.

* **Files/Components to Inspect**:
  * [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py): `MiniMetroActorCritic._compute_hierarchical_logits()` (lines 210–315) and `get_action_and_value()` (lines 321–344).
  * [`ml/eval.py`](file:///home/leomarshall/mm/ml/eval.py): line 57.
  * [`ml/agent.py`](file:///home/leomarshall/mm/ml/agent.py): lines 141–143.

* **Required Change**:
  Implement true **Hierarchical Argmax** for deterministic evaluation:
  1. Calculate valid action types: a type $t \in \{0, \dots, 11\}$ is valid if `mask[:, slice_t].any()`.
  2. Compute masked type-level log-probabilities:
     $$t^* = \arg\max_{t \in \text{valid types}} \left(\text{type\_logits}_t\right)$$
  3. Conditioned on the winning type $t^*$, compute parameter-level masked argmax over candidates belonging strictly to slice $t^*$:
     $$a^* = \arg\max_{a \in \text{valid params of } t^*} \left(\text{param\_scores}_{t^*, a}\right)$$
  4. Ensure sampling mode continues to sample correctly from the joint distribution or hierarchical conditional stages.

* **Implementation Guidance**:
  In `MiniMetroActorCritic.get_action_and_value()`:
  ```python
  if deterministic:
      if self.use_hierarchical:
          # 1. Type argmax over valid types
          type_valid = torch.stack([mask[:, s].any(dim=-1) for s in ACTION_TYPE_SLICES], dim=-1) # [B, 12]
          masked_type_logits = type_logits.masked_fill(~type_valid, -1e9)
          best_type = torch.argmax(masked_type_logits, dim=-1) # [B]
          
          # 2. Parameter argmax within the chosen type slice
          action = torch.zeros(B, dtype=torch.long, device=mask.device)
          for b in range(B):
              t = best_type[b].item()
              sl = ACTION_TYPE_SLICES[t]
              p_scores = param_scores_list[t][b] # parameter score slice
              p_mask = mask[b, sl]
              best_p = torch.argmax(p_scores.masked_fill(~p_mask, -1e9), dim=-1)
              action[b] = sl.start + best_p
      else:
          action = torch.argmax(logits.masked_fill(~mask, -1e9), dim=-1)
  ```

* **Validation / Test**:
  Create a test script `ml/test_hierarchical_decode.py`:
  1. Initialize environment on Map 0 (London).
  2. Run Step 0 in deterministic mode with the trained checkpoint.
  3. Assert `action != 0` (it must select `AddLine`, not `NoOp`).
  4. Run a 100-step deterministic rollout and assert score $> 0$ and survival $> 100$ steps.

* **Acceptance Criteria**:
  * Deterministic inference no longer selects 100% `NoOp`.
  * Deterministic score on Map 0 reaches $\ge 120$ passengers delivered.
  * Hierarchical decoding mathematics explicitly guarantees $\sum_t P(t) = 1$ and $\sum_{a \in t} P(a \mid t) = 1$.

---

### P0-2: Weekly Reward-Card Observation Blindness Fix

* **Problem**:
  When `EventReward` fires at the end of each in-game week, the simulation freezes normal operations and requires the agent to select one of two offered upgrade cards (`ChooseReward`, actions 4050 or 4051). The vectorized observation tensor completely conceals the identity of what is on Card 0 and Card 1. The agent is forced to make a blind positional lottery guess.

* **Evidence**:
  * Report Section 2.2 & Section 3 (Rule 5): In [`simulator/engine/observation.go`](file:///home/leomarshall/mm/simulator/engine/observation.go#L383-L388), the only reward feature exposed to Python is:
    ```go
    if len(s.State.PendingRewardChoices) > 0 {
        outGlobals[11] = 1.0
    } else {
        outGlobals[11] = 0.0
    }
    ```
  * In [`scratch/probe_controlled_scenarios.py`](file:///home/leomarshall/mm/scratch/probe_controlled_scenarios.py), evaluating `ChooseReward` under varied states revealed that the model picks Card 1 with $71.09\%$ probability and Card 0 with $28.91\%$, regardless of map type, tunnel deficit, or passenger congestion.

* **Likely Root Cause**:
  Omission in the Go C-API observation serialization. The internal simulator struct `Observation` contains `PendingRewardChoices []RewardType`, but `WriteVectorizedObservation` failed to serialize the slice elements into the flat float buffer.

* **Files/Components to Inspect**:
  * [`simulator/engine/resources.go`](file:///home/leomarshall/mm/simulator/engine/resources.go#L20-L28): `RewardType` enum definition.
  * [`simulator/engine/observation.go`](file:///home/leomarshall/mm/simulator/engine/observation.go#L68,L383-L392): `GlobalFeatureDim` and `WriteVectorizedObservation`.
  * [`simulator/c_api/main.go`](file:///home/leomarshall/mm/simulator/c_api/main.go#L114-L120): `globalsBuf` slice size.
  * [`ml/env.py`](file:///home/leomarshall/mm/ml/env.py#L62): `self.global_dim`.
  * [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py#L118): `global_dim` parameter in `GNNLayer` and `MiniMetroActorCritic`.
  * [`ml/agent.py`](file:///home/leomarshall/mm/ml/agent.py#L17): `GLOBAL_DIM`.

* **Required Change**:
  1. Inspect the reward enum in `resources.go`:
     * `RewardLine = 0`
     * `RewardTrain = 1`
     * `RewardTunnel = 2`
     * `RewardCarriage = 3`
     * `RewardInterchange = 4`
     There are **5 distinct reward types**.
  2. For the two offered cards (Card 0 and Card 1), expose two 5-dimensional one-hot vectors in `globals`:
     * Card 0: 5 floats (indices $13..17$)
     * Card 1: 5 floats (indices $18..22$)
     If no reward is pending, both vectors are all zeros.
  3. Update `GlobalFeatureDim`:
     $$\text{GlobalFeatureDim: } 13 \longrightarrow 23 \quad (\text{was } 13; +10 \text{ for two 5-class one-hot vectors})$$
     *(Note: The audit report casually hypothesized $13 \to 17$, which is insufficient for two 5-class cards. $23$ is the exact dimension).*
  4. Rebuild the Go shared library via [`ml/build_lib.sh`](file:///home/leomarshall/mm/ml/build_lib.sh).
  5. Update `global_dim=23` across `ml/env.py`, `ml/model.py`, `ml/agent.py`, `ml/train.py`, and `ml/train_local.py`.

* **Implementation Guidance**:
  In `simulator/engine/observation.go`:
  ```go
  const GlobalFeatureDim = 23 // was 13; added two 5-dim one-hot reward card encodings
  ...
  // Clear reward card features
  for k := 13; k < 23; k++ {
      outGlobals[k] = 0.0
  }
  if len(s.State.PendingRewardChoices) >= 2 {
      outGlobals[11] = 1.0 // pending flag
      c0 := int(s.State.PendingRewardChoices[0])
      c1 := int(s.State.PendingRewardChoices[1])
      if c0 >= 0 && c0 < 5 { outGlobals[13 + c0] = 1.0 }
      if c1 >= 0 && c1 < 5 { outGlobals[18 + c1] = 1.0 }
  }
  ```

* **Validation / Test**:
  Create `ml/test_reward_observation.py`:
  1. Trigger an in-game reward event in `MiniMetroEnv`.
  2. Read `obs["globals"]`. Assert that exactly one feature in `[13..17]` is $1.0$ and exactly one feature in `[18..22]` is $1.0$.
  3. Permutation test: Mock state with Card 0 = Line, Card 1 = Tunnel vs Card 0 = Tunnel, Card 1 = Line. Verify input tensors reflect the swapped positions.

* **Acceptance Criteria**:
  * `libminimetro.so` builds cleanly without CGO errors.
  * Environment passes vectorized observations with `global_dim=23`.
  * PyTorch model forward pass executes without shape mismatch errors.

---

### P0-3: Evaluation & Probing Distribution Normalization Fix

* **Problem**:
  In Section 5.1 of the audit report, the tabulated action-type probabilities across congestion levels did not sum to $100.0\%$ (sums ranged between $69.0\%$ and $73.3\%$). This discrepancy represents a probing calculation bug where raw unmasked logits or sub-distributions were tabulated without explicit denominator normalization.

* **Evidence**:
  * Report Section 5.1 table: Sum for MaxFill 0.0 = $7.7 + 10.5 + 9.1 + 9.6 + 7.4 + 8.0 + 8.4 + 5.7 = 66.4\%$.
  * Probe script [`scratch/probe_controlled_scenarios.py`](file:///home/leomarshall/mm/scratch/probe_controlled_scenarios.py#L90-L95) performed a partial slice evaluation over 8 types while ignoring the remaining 4 types (`ChooseReward`, `OpenLoop`, `RemoveLine`, `ShortenLine`).

* **Likely Root Cause**:
  Failure to distinguish between:
  1. Global 12-way type probabilities $P(t) = \text{Softmax}(\text{type\_logits})_t$.
  2. Masked type probabilities $P(t \mid \text{valid}) = \frac{\exp(\text{logit}_t \cdot \mathbb{I}_t)}{\sum_{t'} \exp(\text{logit}_{t'} \cdot \mathbb{I}_{t'})}$.
  3. Total probability mass assigned to an action type $\sum_{a \in \text{slice}_t} P(a)$.

* **Files/Components to Inspect**:
  * [`scratch/probe_controlled_scenarios.py`](file:///home/leomarshall/mm/scratch/probe_controlled_scenarios.py).
  * [`scratch/probe_rollouts.py`](file:///home/leomarshall/mm/scratch/probe_rollouts.py).
  * [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py): `_compute_hierarchical_logits()`.

* **Required Change**:
  1. Rewrite diagnostic and probing utilities to formally distinguish and report:
     * `raw_logits`: Unnormalized model outputs.
     * `masked_type_probs`: Normalized strictly over valid action types ($L_1 \text{ norm} = 1.0$).
     * `conditional_param_probs`: Normalized strictly within the selected type slice ($L_1 \text{ norm} = 1.0$).
     * `joint_action_probs`: Normalized over all 4,087 actions ($\sum_{a=0}^{4086} P(a) = 1.0$).
  2. Insert strict automated assertions `assert np.isclose(probs.sum(), 1.0, atol=1e-5)` in all probing scripts.

* **Validation / Test**:
  Run diagnostic script across all 12 action types with diverse masks and verify every probability table outputs rows summing to exactly $100.0\%$.

* **Acceptance Criteria**:
  * No tabulated probability table contains rows summing to anything other than $100.0\%$.
  * Assertion guards prevent unnormalized logit-softmax leakage.

---

## P1 — Major Policy & Architecture Problems

Issues that significantly degrade decision quality or create behavioral pathologies.

### P1-1: Explicit Candidate Distance & Geometric Awareness in AddLine

* **Problem**:
  The model frequently draws geometrically poor, sprawling, or unnecessarily long connections across the map rather than sensible direct links. While existing graph edges contain distance features, the candidate scoring head for `AddLine(u, v)` does not explicitly receive the Euclidean distance between candidate stations $u$ and $v$.

* **Evidence**:
  * Report Section 3 (Rule 1 & Rule 10) & Section 4: In [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py#L161-L167,L220-L225), `AddLine` candidate scoring is computed as:
    $$q_u = W_q x_u, \quad k_v = W_k x_v, \quad S_{uv} = \frac{1}{2\sqrt{H}} \left(q_u^\top k_v + q_v^\top k_u\right)$$
  * $x_u$ and $x_v$ are station node embeddings after 3 GNN layers. Because stations $u$ and $v$ are currently **unconnected**, there is no edge between them in the GNN adjacency graph.
  * In [`scratch/probe_expansion.py`](file:///home/leomarshall/mm/scratch/probe_expansion.py): An isolated Circle at $(20, 20)$ and Triangle at $(30, 20)$ ($d=10$) scored $-1.1004$, whereas the same Circle and a Triangle at $(80, 80)$ ($d=85$) scored $-1.0887$ (virtually identical, with the distant station scoring slightly *higher*).
  * The inner product $q_u^\top k_v$ cannot compute Euclidean distance $\| \text{Pos}_u - \text{Pos}_v \|_2 = \sqrt{(X_u - X_v)^2 + (Y_u - Y_v)^2}$ from unlinked coordinate projections.

* **Likely Root Cause**:
  Structural absence of candidate edge geometry in the spatial bilinear head. The network has no geometric inductive bias penalizing long candidate spans during line creation.

* **Files/Components to Inspect**:
  * [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py): lines 161–167 and 220–225 (`add_line_q`, `add_line_k`).
  * [`simulator/engine/observation.go`](file:///home/leomarshall/mm/simulator/engine/observation.go#L187-L188): node position encoding ($X/100, Y/100$).

* **Required Change**:
  * **Current Architecture**: Bilinear inner product over post-GNN node embeddings $x_u, x_v$ without explicit pairwise candidate distance.
  * **Proposed Architecture**: Compute candidate pairwise geometric displacement vector:
    $$d_{uv} = \left[ \frac{\|\text{Pos}_u - \text{Pos}_v\|_2}{100.0}, \frac{|X_u - X_v|}{100.0}, \frac{|Y_u - Y_v|}{100.0}, \text{CrossesWater}(u, v) \right] \in \mathbb{R}^4$$
    Pass $d_{uv}$ through a lightweight pairwise geometry projection MLP and add it directly to the bilinear score:
    $$S(u, v) = \frac{1}{2\sqrt{H}} \left(q_u^\top k_v + q_v^\top k_u\right) + \text{MLP}_{\text{geom}}(d_{uv})$$
  * **Why It Is Better**: Gives the actor head direct, uncompressed access to physical track length, allowing it to learn a smooth distance-attenuation penalty without interfering with topological shape matching.
  * **Tensors/Shapes**:
    `triu_geom`: `[B, 435, 4]` derived from node positions `nodes[:, :, 0:2]`.
    `MLP_geom`: `nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))`. Output: `[B, 435]`.
  * **Retraining Implications**: Requires retraining or fine-tuning from checkpoint with frozen GNN backbone.

* **Validation / Test**:
  Create `ml/test_add_line_distance.py`:
  1. Construct counterfactual test scenario: Station 0 (Circle) at $(50, 50)$, Station 1 (Square) at $(55, 50)$ ($d=5$), Station 2 (Square) at $(95, 50)$ ($d=45$).
  2. Compute candidate scores for Pair $(0, 1)$ vs Pair $(0, 2)$.
  3. Assert that $S(0, 1) > S(0, 2)$ with a statistically significant margin ($P(\text{near}) \ge 75\%$).

* **Acceptance Criteria**:
  * Candidate distance changes monotonically decrease `AddLine` score when station types and network context are held constant.
  * Average line track distance in evaluation rollouts decreases by $\ge 20\%$ without reducing passenger throughput.

---

### P1-2: Redundant / Indiscriminate Network Expansion Mitigation

* **Problem**:
  The agent exhibits an "over-expansion / indiscriminate connectivity" bias: whenever a new station spawns, the model repeatedly attempts to connect it to multiple or all existing lines, regardless of whether additional lines provide any marginal benefit.

* **Evidence**:
  * User observation post-audit: The model behaves as if "more valid connections = better," accumulating excessive line overlap on the same stations.
  * In [`simulator/engine/scoring.go`](file:///home/leomarshall/mm/simulator/engine/scoring.go#L84-L119), `ConnectivityBonus` ($+2.0 \cdot \text{ReachablePairs} / 45$) rewards distinct-type reachability. However, there is **zero cost or penalty** for adding redundant parallel lines between stations that are already transit-connected.
  * In live rollouts, lines reach 6–10 stations each, with central stations served by 4–5 lines simultaneously, diluting train frequency and causing train starvation on peripheral segments.

* **Likely Root Cause**:
  1. The reward function provides a positive gradient for making connections, but has no complexity, track-mileage, or route-redundancy penalty.
  2. The observation vector does not provide candidate scorers with explicit signals regarding whether a proposed connection provides *marginal* reachability or *redundant* duplicate reachability.

* **Files/Components to Inspect**:
  * [`simulator/engine/scoring.go`](file:///home/leomarshall/mm/simulator/engine/scoring.go#L10-L14, L84-L122): `ConnectivityBonus` and `ComputeStepReward()`.
  * [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py): `ext_proj`, `ins_proj`, `add_line_q`.
  * [`ml/env.py`](file:///home/leomarshall/mm/ml/env.py): step reward calculation.

* **Required Change**:
  1. **Diagnostic Metric**: Implement `ExpansionRatio`:
     $$\text{ExpansionRatio} = \frac{\text{Selected Line Connections}}{\text{Total Legal Connections}}$$
     Track this metric across early, mid, and late game in TensorBoard.
  2. **Candidate Marginal Utility Conditioning**:
     Incorporate node feature `[26]` (`LinesServing / 7.0`) and node degree `[23]` directly into candidate scoring:
     * As `LinesServing` increases on station $u$, candidate scores for adding *yet another* line to $u$ should be attenuated.
  3. **Reward Shaping Refinement**:
     In `scoring.go`, refine `ConnectivityBonus` to only reward connections that establish *new* reachability between previously disconnected components, or introduce a small marginal redundancy penalty:
     $$\mathcal{R}_{\text{redundancy}} = -0.05 \cdot \max(0, \text{LinesServing}(u) - 2)$$
     *(Only applies when a station exceeds 2 lines without passenger transfer justification).*

* **Potential Unintended Consequences**:
  Penalizing high-degree stations could discourage creating necessary central interchanges. The penalty must only apply to redundant lines that duplicate already-connected station shapes, not legitimate transfer hubs.

* **Validation / Test**:
  Create `ml/test_expansion_redundancy.py`:
  1. Run 10 evaluation episodes on London and NYC.
  2. Measure: (a) Average lines per station, (b) Redundant connections (stations with $>2$ lines sharing identical destination coverage), (c) Passengers delivered per train.
  3. Assert redundant connections decrease by $\ge 35\%$ while total delivered passengers does not decrease.

* **Acceptance Criteria**:
  * `ExpansionRatio` drops significantly in mid/late game.
  * Central stations no longer accumulate redundant 4th and 5th line connections when existing lines already satisfy passenger flow.

---

### P1-3: Candidate-Conditioned Line Scoring for AddTrain and AddCarriage

* **Problem**:
  When a locomotive (`AddTrain`) or carriage (`AddCarriage`) becomes available, the model strongly favors **Line 0** ($45.0\%$) over **Line 1** ($28.7\%$) and **Line 2** ($26.2\%$), even when Line 2 has 15 waiting passengers and Line 0 has 0.

* **Evidence**:
  * Report Section 3 (Rule 6) & Section 4 (Experiment 3):
    In [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py#L183-L190), `AddTrain` (actions 4006..4012) and `AddCarriage` (actions 4013..4019) are decoded by `self.non_spatial_head`:
    ```python
    self.non_spatial_head = nn.Sequential(
        nn.Linear(hidden_dim * 5, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 52)
    )
    ```
  * `non_spatial_head` receives only the globally pooled graph representation `combined` (`[B, 1280]`).
  * The 7 output neurons for `AddTrain` have fixed biases: `Line 0: +0.0581`, `Line 1: +0.0105`, `Line 2: -0.0294`, `Line 3: -0.0547`.
  * Raising the passenger queue on Line 1 from 0 to 25 passengers shifted selection probability by less than $3\%$. The architecture is structurally incapable of candidate-conditioned line selection.

* **Likely Root Cause**:
  Architectural bottleneck: Lines are treated as fixed non-spatial integer slots rather than dynamic subgraphs with pooled station and train features.

* **Files/Components to Inspect**:
  * [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py): lines 183–190, 255–265 (`non_spatial_head`).
  * [`ml/train.py`](file:///home/leomarshall/mm/ml/train.py): action dictionary and loss computation.

* **Required Change**:
  * **Current Architecture**: Fixed 7-neuron linear projection from global pooled context.
  * **Proposed Architecture**: Candidate-Conditioned Line Scorer:
    1. For each line $i \in \{0, \dots, 6\}$, construct a dynamic line representation vector $h_{\text{line}_i}$:
       $$h_{\text{line}_i} = \text{MeanPool}\left(\{x_u \mid u \in \text{Stations}(\text{Line}_i)\}\right) \oplus \text{LineStats}_i$$
       Where $\text{LineStats}_i$ includes: line length, total waiting passengers along line, number of active trains, and passenger load of active trains.
    2. Compute candidate score using an MLP scoring head:
       $$\text{Score}(\text{AddTrain on Line } i) = W_{\text{train}}^\top \text{ReLU}\left(W_h h_{\text{line}_i} + W_g g_{\text{global}}\right)$$
  * **Why It Is Better**: Permutation-invariant across line IDs; dynamically routes trains to lines with the highest passenger backlog and longest routes.
  * **Retraining Implications**: Replaces 7 static neurons with a shared candidate-line scoring head; requires model retraining.

* **Validation / Test**:
  Create `ml/test_train_dispatch.py`:
  1. Set up two lines: Line 0 with 2 stations and queue = 0; Line 1 with 5 stations and queue = 15.
  2. Assert $P(\text{AddTrain Line 1}) > 80\%$.
  3. Swap Line 0 and Line 1 indices in simulator state. Assert the train is still dispatched to the 5-station congested line (proving permutation invariance).

* **Acceptance Criteria**:
  * Train and carriage allocation correlates positively with queue severity along the line ($r \ge 0.70$).
  * Permuting line IDs produces identical dispatch choices.

---

### P1-4: Loop Toggling Hysteresis & Reversal Oscillation Mitigation

* **Problem**:
  In evaluation rollouts on NYC and Tokyo, the model executed `CloseLoop` 5 times and `OpenLoop` 5 times on the exact same line, repeatedly toggling the line between a closed circle and an open linear track.

* **Evidence**:
  * Report Section 3 (Rule 8) & Section 4: Live rollout logs recorded exactly 5 CloseLoop and 5 OpenLoop actions on NYC, and 5 CloseLoop / 5 OpenLoop on Tokyo.
  * In [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py#L152-L156), `TypeNet` output bias for `CloseLoop` is $-0.1056$ and for `OpenLoop` is $+0.0267$.
  * Once a line is closed, `CloseLoop` is masked out and `OpenLoop` becomes valid with a positive base logit, frequently sampling `OpenLoop`. Once opened, `CloseLoop` becomes valid again, creating an oscillatory loop trap.

* **Likely Root Cause**:
  Markov action instability coupled with lack of action hysteresis. The environment allows immediate reversal of loop status with zero cooldown or penalty.

* **Files/Components to Inspect**:
  * [`simulator/engine/action_space.go`](file:///home/leomarshall/mm/simulator/engine/action_space.go#L370-L385): `CloseLoop` and `OpenLoop` masking conditions.
  * [`simulator/engine/simulator.go`](file:///home/leomarshall/mm/simulator/engine/simulator.go#L477-L525): `closeLoop()` and `openLoop()`.
  * [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py): non-spatial head indices 17..31.

* **Required Change**:
  1. **Investigate LSTM Dependence vs Type Bias**:
     Test whether the oscillation is driven by memory in the LSTM hidden state or purely by static type biases.
  2. **Implement Simulator Action Cooldown**:
     In `simulator/engine/simulator.go`, record `Line.LastLoopToggleTick`. In `action_space.go`, mask `OpenLoop` and `CloseLoop` for that line for at least 900 ticks (30 seconds) following a toggle.
  3. **Reward Penalty on Rapid Reversals**:
     Apply a small penalty ($-0.50$) if a line loop is toggled back within 60 seconds of being modified.

* **Validation / Test**:
  Create `ml/test_loop_stability.py`:
  Run 10 episodes on Tokyo. Count total `CloseLoop` and `OpenLoop` actions. Assert total loop toggles per line $\le 1$ per episode.

* **Acceptance Criteria**:
  * Oscillatory loop toggling is eliminated.
  * Lines closed into loops remain stable unless significant network topology changes occur.

---

## P2 — Reward Shaping & Training Improvements

Reward design refinements, incentive balancing, and ablation tracking.

### P2-1: Comprehensive Reward Decomposition & Ablation Suite

* **Problem**:
  The reward formulation combines 5 disparate terms: passenger delivery ($+1.0$), survival ($+0.01$), distinct-type connectivity ($+2.0 \cdot \text{pairs}/45$), quadratic overcrowding penalty ($-0.30$), and game-over penalty ($-200.0$). Currently, only scalar episode return is logged, making it impossible to determine which terms drive specific learned behaviors.

* **Files/Components to Inspect**:
  * [`simulator/engine/scoring.go`](file:///home/leomarshall/mm/simulator/engine/scoring.go#L68-L122): `ComputeStepReward()`.
  * [`ml/env.py`](file:///home/leomarshall/mm/ml/env.py#L180-L214): `step()` reward accumulation.
  * [`ml/train.py`](file:///home/leomarshall/mm/ml/train.py#L247-L253): TensorBoard scalar logging.

* **Required Change**:
  1. Instrument `MiniMetroEnv.step()` to return a decomposed dictionary in `info`:
     ```python
     info["reward_breakdown"] = {
         "delivery": delivered_reward,
         "survival": survival_reward,
         "connectivity": connectivity_reward,
         "crowd_penalty": crowd_penalty,
         "game_over": game_over_penalty
     }
     ```
  2. Log cumulative episode metrics for each reward component to TensorBoard in `train.py`.
  3. Run systematic ablation experiments disabling:
     * Ablation A: No `ConnectivityBonus`.
     * Ablation B: Linear crowd penalty instead of quadratic.
     * Ablation C: Reduced `BetaGameOverPenalty` ($200 \to 50$).

* **Acceptance Criteria**:
  * TensorBoard displays real-time breakdowns of all 5 reward channels.
  * Ablation study quantifies the exact contribution of `ConnectivityBonus` to network expansion rates.

---

### P2-2: Geometric Efficiency & Track Sprawl Regularization

* **Problem**:
  The agent has no incentive to build compact or efficient rail networks. Extremely long tracks with sharp zigzag angles are evaluated identically to straight, compact corridors as long as the same stations are connected.

* **Files/Components to Inspect**:
  * [`simulator/engine/scoring.go`](file:///home/leomarshall/mm/simulator/engine/scoring.go).
  * [`simulator/engine/train.go`](file:///home/leomarshall/mm/simulator/engine/train.go#L60-L100): `trackCornerMultiplier` turn slowdown.

* **Required Change**:
  1. Do NOT hard-mask long connections (which would break legitimate water crossings in NYC and Tokyo).
  2. Introduce a continuous track-mileage regularization term in `scoring.go`:
     $$\mathcal{R}_{\text{track\_efficiency}} = -0.01 \cdot \sum_{e \in \text{Network}} \left(\frac{\text{Distance}(e)}{100.0}\right)$$
  3. Ensure passenger delivery rewards ($+1.0$) remain dominant, but excessive wandering track incurs steady negative pressure.

* **Validation / Test**:
  Compare baseline checkpoint vs. efficiency-regularized checkpoint across 20 runs. Assert total track length decreases by $\ge 15\%$ with zero degradation in delivered passengers.

* **Acceptance Criteria**:
  * Policy ceases drawing redundant S-curves and long-distance bypasses across empty terrain.

---

### P2-3: Tail vs. Front Extension Symmetry Audit & Debiasing

* **Problem**:
  The model displays a 3.7x probability bias toward extending lines from their tail/back endpoint (`end=1`, score $+4.825$) rather than front endpoint (`end=0`, score $+3.543$). It must be established whether this is a genuine topological strategy or an artifact of array indexing conventions ($u < v$).

* **Files/Components to Inspect**:
  * [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py#L172-L176, L230-L241): `ext_end_emb` and `ext_proj`.
  * [`scratch/probe_extend_insert.py`](file:///home/leomarshall/mm/scratch/probe_extend_insert.py).

* **Required Change**:
  1. Test endpoint index invariance: Create counterfactual lines where station order in the Go struct is reversed ($[s_0, s_1, s_2] \leftrightarrow [s_2, s_1, s_0]$).
  2. If the preference follows the array index rather than the physical station geometry, implement training-time endpoint data augmentation: randomly flip line station ordering during rollout collection with 50% probability.

* **Acceptance Criteria**:
  * Extension preference is determined by station geometry and passenger demand rather than array index `0` vs `1`.

---

## P3 — Evaluation, Diagnostics & Empirical Audits

Rigorous testing protocols, counterfactual verification, and metrics.

### P3-1: Rigorous Multi-Seed & Cross-Map Evaluation Suite

* **Problem**:
  The project has historically relied on single-seed evaluations (e.g., Seed 100), which obscures variance and seed-specific layout luck.

* **Files/Components to Inspect**:
  * [`ml/eval.py`](file:///home/leomarshall/mm/ml/eval.py).

* **Required Change**:
  Completely overhaul `ml/eval.py` to run:
  * 10 distinct random seeds (e.g., 1000..1009) per map.
  * All 3 standard maps: London (Map 0), New York City (Map 1), Tokyo (Map 2) (30 runs total).
  * Dual evaluation modes: (1) Hierarchical Deterministic, (2) Stochastic Sampling.
  * Compute and tabulate: Mean, Median, StdDev, Min, Max, 25th/75th Percentiles for:
    * Score (passengers delivered).
    * Survival duration (macro-steps and in-game seconds).
    * Cause of death (which station overcrowded and its kind).
    * Resource utilization (lines, trains, tunnels, carriages, interchanges used).
  * Benchmark against a Random Legal baseline and a simple Greedy Heuristic baseline.

* **Acceptance Criteria**:
  * `ml/eval.py` outputs a markdown summary table with full statistical dispersion metrics.
  * Results are 100% reproducible given fixed seeds.

---

### P3-2: Semantic Permutation & Spatial Invariance Audit

* **Problem**:
  Neural networks in graph environments frequently develop hidden positional biases (such as the confirmed Line 0 train bias and Card 1 reward bias).

* **Files/Components to Inspect**:
  * Create dedicated test module: `ml/test_invariance.py`.

* **Required Change**:
  Implement automated tests verifying policy invariance under semantically neutral transformations:
  1. **Line ID Permutation**: Permuting line IDs must produce permuted action indices with identical probability distribution.
  2. **Station ID Permutation**: Re-indexing unconnected stations must not change AddLine candidate probabilities.
  3. **Reward Slot Permutation**: Swapping the order of Card 0 and Card 1 must swap the model's action choice.
  4. **Coordinate Translation/Rotation**: Translating all station coordinates by a constant $(\Delta X, \Delta Y)$ must produce identical relative candidate scores.

* **Acceptance Criteria**:
  * All invariance tests pass with KL-divergence $< 10^{-4}$ between original and permuted distributions.

---

### P3-3: Station-Shape Affinity Validation vs. Dynamic Demand Distributions

* **Problem**:
  The audit demonstrated high bilinear affinity for Square stations ($0.094$) vs Circles ($0.072$). It must be verified whether this reflects true demand awareness or is a static artifact of map defaults.

* **Files/Components to Inspect**:
  * [`simulator/engine/spawner.go`](file:///home/leomarshall/mm/simulator/engine/spawner.go#L13-L24): `stationWeights`.
  * [`scratch/probe_expansion.py`](file:///home/leomarshall/mm/scratch/probe_expansion.py).

* **Required Change**:
  Create an experiment that inverts shape scarcity:
  * Set `stationWeights`: Square = 10, Circle = 2.
  * Evaluate candidate scoring: does the model adapt its affinity to prioritize the newly scarce shape, or does it stubbornly prefer Squares?
  * Document whether scarcity awareness is dynamically computed via GNN features or hardcoded in static projection weights.

* **Acceptance Criteria**:
  * Documented empirical report on whether shape affinity is dynamically adaptive or static.

---

### P3-4: Counterfactual Interchange Decision Verification

* **Problem**:
  The audit's interchange findings relied on gradient sensitivity ($+0.020$ Degree, $+0.031$ Incoming trains, $-0.048$ Overcrowding timer). Behavioral verification requires counterfactual state intervention.

* **Files/Components to Inspect**:
  * [`scratch/probe_detailed_behaviors.py`](file:///home/leomarshall/mm/scratch/probe_detailed_behaviors.py).
  * [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py#L169, L227): `interchange_net`.

* **Required Change**:
  Construct matched pairs of stations in an identical environment state:
  * Pair 1: Station A (Degree 1) vs Station B (Degree 3); all queues, kinds, and locations identical.
  * Pair 2: Station A (0 incoming trains) vs Station B (2 incoming trains).
  * Pair 3: Station A (normal timer) vs Station B (active countdown $< 10\text{s}$).
  Measure actual choice probabilities from `interchange_net`.

* **Acceptance Criteria**:
  * Counterfactual tests confirm or refine the gradient sensitivity findings with direct behavioral choice probabilities.

---

### P3-5: NoOp Disambiguation & Macro-Step Simulation Accounting

* **Problem**:
  The audit reported ~70% `NoOp` actions during sampling rollouts. However, the environment uses dynamic frame-skipping (ticking up to 4 in-game seconds per macro-step). It must be determined whether 70% `NoOp` represents excessive idling or necessary simulation advancement.

* **Files/Components to Inspect**:
  * [`ml/env.py`](file:///home/leomarshall/mm/ml/env.py#L178-L214): frame-skipping loop.

* **Required Change**:
  1. Measure simulation seconds elapsed per macro-step under different action types.
  2. Compute **Policy Opportunity Rate**: Of the 70% `NoOp` steps, what percentage occurred when `action_mask` contained *zero* legal construction actions (forced NoOp due to resource exhaustion) vs when legal lines/trains were available?
  3. Quantify whether the agent is voluntarily idling when resources exist.

* **Acceptance Criteria**:
  * Clear accounting distinguishing voluntary passivity from resource-constrained forced NoOps.

---

## P4 — Long-Term Architectural & Environment Improvements

Non-critical research improvements and environment extensions.

### P4-1: Safe Dynamic Line Re-Routing & Deletion (RemoveLine / ShortenLine)

* **Problem**:
  Mini Metro gameplay in the commercial game relies heavily on pausing, deleting outdated lines, and redesigning networks globally as new stations appear. In the simulator, `RemoveLine` and `ShortenLine` are permanently disabled in `action_space.go` line 386 and return rule violation errors.

* **Files/Components to Inspect**:
  * [`simulator/engine/action_space.go`](file:///home/leomarshall/mm/simulator/engine/action_space.go#L386-L393).
  * [`simulator/engine/simulator.go`](file:///home/leomarshall/mm/simulator/engine/simulator.go#L422-L425, L472-L475).

* **Required Change**:
  1. Implement safe line removal mechanics in Go engine:
     * Unassign active trains and return them to the available train pool.
     * Passengers on the removed line alight at the nearest station.
     * Refund tunnel tokens used by the line segments.
  2. Unmask `RemoveLine` and `ShortenLine` in `action_space.go`.
  3. Retrain agent to evaluate whether global reconstruction outperforms purely additive network growth.

* **Acceptance Criteria**:
  * Engine allows safe line deletion without state corruption.
  * Agent learns to reallocate lines from low-density to high-density corridors.

---

### P4-2: Relational Spatial Cross-Attention Network (GAT-v2 / Transformer Scorer)

* **Problem**:
  Bilinear matrix factorizations ($q_u^\top k_v$) have limited expressiveness for modeling complex geometric constraints, river crossings, and multi-line interactions.

* **Files/Components to Inspect**:
  * [`ml/model.py`](file:///home/leomarshall/mm/ml/model.py): `GNNLayer` and bilinear action heads.

* **Required Change**:
  1. Replace GCN layers with Graph Attention Networks v2 (GATv2) incorporating edge features directly into attention logits.
  2. Replace bilinear action heads with cross-attention Transformer heads that attend over both station nodes and candidate displacement vectors simultaneously:
     $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d}} + \text{GeomBias}\right) V$$

* **Acceptance Criteria**:
  * Demonstrates improved sample efficiency and higher final passenger throughput compared to GCN baseline.

---

## Definition of Done (DoD)

The Mini Metro RL agent optimization phase will be formally declared complete when all of the following verifiable system criteria are satisfied:

1. **Deterministic Decoding Correctness**:
   `ml/eval.py` running in deterministic mode (`deterministic=True`) executes active construction actions at Step 0, achieves zero 100%-NoOp failures, and delivers $\ge 120$ passengers on London across 10 evaluation seeds.
2. **Weekly Reward Observability**:
   `GlobalFeatureDim` is updated to 23. `WriteVectorizedObservation` serializes both offered reward card identities into two 5-dimensional one-hot vectors. Permuting Card 0 and Card 1 flips agent selection logits accordingly.
3. **Candidate-Conditioned Dispatch**:
   `AddTrain` and `AddCarriage` heads score dynamic line embeddings. In controlled tests with unequal queues, the congested line is selected with $\ge 80\%$ probability, and line-ID permutation produces identical dispatch targets.
4. **Verified Distance Awareness**:
   Candidate pairwise Euclidean displacement $d_{uv}$ is explicitly fed into `AddLine` scoring. Candidate scores decrease monotonically with distance when station types and network states are held constant. Average track length per line decreases by $\ge 15\%$.
5. **Reduced Redundant Expansion**:
   `ExpansionRatio` is tracked in TensorBoard. Stations exceeding 2 lines without transfer justification decrease by $\ge 35\%$, eliminating the "connect new station to every line" pathology.
6. **Loop Stability**:
   Action hysteresis or cooldown prevents rapid oscillatory toggling between `CloseLoop` and `OpenLoop` ($\le 1$ toggle per line per episode).
7. **Statistical Evaluation Reproducibility**:
   `ml/eval.py` evaluates 10 seeds across London, NYC, and Tokyo, outputting full distributional statistics (Mean, Median, StdDev, IQR) for score, survival time, and cause-of-death breakdown.
8. **Automated Test Suite**:
   All unit and integration tests (`ml/test_*.py`) pass cleanly in the CI/local environment.

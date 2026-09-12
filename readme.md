# Mini Metro AI: Reinforcement Learning & Simulation Engine

A research and simulation environment for Mini Metro. The project combines a Go simulation engine, an interactive TypeScript/Canvas web visualization client, and a reinforcement learning pipeline powered by Proximal Policy Optimization (PPO), Graph Neural Networks (GNN), Recurrent LSTM architectures, Potential-Based Reward Shaping (PBRS), and a 4-Stage Progressive Curriculum.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Simulation Engine Internals](#simulation-engine-internals)
3. [Observation & Action Spaces](#observation--action-spaces)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Potential-Based Reward Shaping (PBRS)](#potential-based-reward-shaping-pbrs)
6. [Curriculum Learning: 4-Stage Progressive Pool](#curriculum-learning-4-stage-progressive-pool)
7. [Supported Maps & Geographic Mechanics](#supported-maps--geographic-mechanics)
8. [Prerequisites & Installation](#prerequisites--installation)
9. [Makefile Reference](#makefile-reference)
10. [Training Workflows & CLI Parameters](#training-workflows--cli-parameters)
11. [Checkpoint Management & Inference Hierarchy](#checkpoint-management--inference-hierarchy)
12. [Monitoring & TensorBoard Metrics](#monitoring--tensorboard-metrics)
13. [Project Directory Structure](#project-directory-structure)
14. [Troubleshooting & FAQ](#troubleshooting--faq)
15. [License & Disclaimers](#license--disclaimers)

---

## Architecture Overview

The system is decoupled into three independent subsystems communicating via in-memory C-bindings (during training) or WebSockets (during live visualization):

```
+-----------------------------------------------------------------------------+
|                               Web Frontend                                  |
|         (TypeScript / HTML5 Canvas / Vite Development Server)               |
|      - Real-time track rendering, train animations, passenger queues       |
|      - Interactive human controls: drag-to-draw lines, relocate trains      |
+---------------------------------------+-------------------------------------+
                                        | WebSocket JSON (Port 6969)
                                        v
+-----------------------------------------------------------------------------+
|                        Go Simulation Backend Server                         |
|                 (Kinematics, Routing, A*, Game Loop)                        |
|      - Physics loop, station crowding timers                |
|      - Deterministic RNG seeding, dynamic water intersection polygon checks  |
+---------------------------------------+-------------------------------------+
                                        | CGO C-Shared Library (libminimetro.so)
                                        v
+-----------------------------------------------------------------------------+
|                        Python Reinforcement Learning                        |
|                  (Gymnasium / PyTorch / PPO Vectorized Envs)                |
|      - Vectorized C-API bindings (SyncVectorEnv / AsyncVectorEnv)           |
|      - Graph Attention Network (GATv2) + Bilinear Affinity Scorer + LSTM    |
|      - Potential-Based Reward Shaping (PBRS) & 4-Stage Expanding Curriculum  |
+-----------------------------------------------------------------------------+
```

---

## Simulation Engine Internals

The Go engine (`simulator/`) models all core mechanics of Mini Metro with complete determinism:

### Kinematics & Train Scheduling
- **Continuous Path Parameterization**: Trains navigate piecewise polyline tracks parameterized by cumulative arc length.
- **Physics**: Constant acceleration, top cruising velocity, symmetric deceleration curves approaching stations, and minimum dwell times for passenger boarding/alighting.
- **Multi-Line Sharing & Reversal**: Support for circular loop tracks (`is_loop`), bidirectional shuttle tracks, train transfers between lines, and carriage attachments.

### Passenger Pathfinding & Flow Dynamics
- **Dynamic A* Graph Search**: Passengers evaluate global transit paths using an A* graph with transfer penalty weighting. Routes favor minimal train interchanges rather than purely geometric line length.
- **Multi-Line Boarding Equality**: Transfer stations serving multiple lines distribute boarding passengers fairly, preventing line starvation.
- **Stranded Passenger Rerouting**: When a line is shortened or removed, affected passengers disembark immediately at the nearest valid station and recalculate alternate paths.

### Overcrowding & Game-Over Mechanics
- **Capacity Thresholds**: Standard stations hold up to 6 passengers before entering an overcrowding state. Upgraded interchange hubs increase capacity to 18 passengers.
- **Radial Overcrowding Gauge**: When station capacity is exceeded, an overcrowding timer begins counting down. If unaddressed for a continuous period (default: 45 real-time simulation seconds), the game ends.

---

## Observation & Action Spaces

### Observation Space

Observations are structured as heterogeneous graph dictionaries returned by `ml/env.py`:

#### 1. Station Node Features (`obs["node_features"]`: `[max_stations, 32]`)
| Feature Indices | Description | Normalization / Encoding |
|:---:|:---|:---|
| `[0, 1]` | Station coordinates $(x, y)$ | Normalized to $[-1.0, 1.0]$ |
| `[2:7]` | Station shape type | One-hot vector: Circle, Triangle, Square, Star, Pentagon |
| `[7:12]` | Passenger queue counts by target shape | Scaled count of waiting passengers demanding each shape |
| `[12:19]` | Connected line membership | Boolean indicators for Line 0 through Line 6 |
| `[19]` | Overcrowding timer fraction | Continuous value in $[0.0, 1.0]$ (1.0 = imminent failure) |
| `[20]` | Overcrowding active boolean | 1.0 if overcrowding timer is running, 0.0 otherwise |
| `[21]` | Interchange hub status | 1.0 if upgraded to interchange, 0.0 otherwise |
| `[22]` | Station connection degree | Total number of line connections arriving at station |
| `[23:32]` | Structural context & padding | Local graph centrality and reserved expansion slots |

#### 2. Adjacency Matrix (`obs["adjacency"]`: `[max_stations, max_stations]`)
- Weighted adjacency representing direct transit tracks between stations, encoded with active line IDs.

#### 3. Global State Features (`obs["global_features"]`: `[23]`)
- Inventory counts: available lines, spare trains, carriages, tunnels, and interchange upgrades.
- Simulation metrics: total elapsed game days, total delivered passengers, active passengers in transit, and weekly reward offer states.

### Action Space & Two-Stage Hierarchical Factorization

To navigate the combinatorial action space (which includes lines, stations, trains, and upgrades), the policy factorizes decisions hierarchically:

1. **Stage 1: Action Type Categorization**:
   The primary policy head outputs a categorical distribution over 13 discrete action classes:
   - `0: No-Op` (Do nothing / advance simulation)
   - `1: AddLine` (Create new line between two stations)
   - `2: ExtendLineFront` (Extend line origin to station)
   - `3: ExtendLineBack` (Extend line terminus to station)
   - `4: CloseLoop` (Connect endpoints into a circular loop)
   - `5: ShortenLineFront` (Remove station from line origin)
   - `6: ShortenLineBack` (Remove station from line terminus)
   - `7: RemoveLine` (Delete line entirely and refund tokens)
   - `8: AddTrain` (Deploy spare locomotive to line)
   - `9: RepositionTrain` (Move locomotive to another line)
   - `10: AddCarriage` (Attach carriage to train)
   - `11: RemoveCarriage` (Return carriage to inventory)
   - `12: UpgradeInterchange` (Convert station to high-capacity hub)

2. **Stage 2: Parameter Selection Heads**:
   Conditioned on the selected action type, specialized sub-heads evaluate legal parameter combinations with strict action masking:
   - **Bilinear Station Pair Scorer**: Evaluates $(u, v)$ pairs for line creation via learned shape affinity:
     $$S(u, v) = h_u^T W h_v - \alpha \|x_u - x_v\|_2$$
   - **Station Target Head**: Evaluates candidate station extensions.
   - **Train / Line Allocation Head**: Allocates locomotives and carriages based on line passenger density.

---

## Machine Learning Pipeline

### Neural Network Architecture (`ml/model.py`)

The policy and value networks are parameterized by an actor-critic model:

```
                  Station Node Features [N, 32]
                               |
                               v
                     Node Projection (Linear)
                               |
                               v
             2-Layer Graph Attention Network (GATv2)
                 - Multi-head attention (4 heads)
                 - LeakyReLU activations
                 - Residual skip connections
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
       Mean Pooling                          Max Pooling
            |                                     |
            +------------------+------------------+
                               |
                               v
               Concatenated Graph Embedding [N, 2*H]
                               +
                    Global Features [23]
                               |
                               v
                Recurrent Memory Core (LSTM)
                               |
         +---------------------+---------------------+
         |                                           |
         v                                           v
    Policy Heads                                Value Head
- Stage 1: Action Type Logits              - Scalar State Value V(s)
- Stage 2: Bilinear Pair Matrix            - Used for GAE advantage
- Stage 2: Parameter Heads                   computation
```

### Reinforcement Learning Algorithm (`ml/ppo.py`)

- **Proximal Policy Optimization (PPO)**: Clipped surrogate objective preventing destructively large policy updates:
  $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$
- **Generalized Advantage Estimation (GAE)**: Balances bias and variance across temporal trajectory segments using $\gamma = 0.99$ and $\lambda = 0.95$.
- **Sequence-Based Recurrent Minibatches**: LSTM hidden and cell states are preserved across rollout steps and sliced into contiguous sequences during gradient descent.
- **Gradient Clipping & Value Function Clipping**: Standardized gradient norm clipping at $0.5$ and value loss clipping.

---

## Potential-Based Reward Shaping (PBRS)

Sparse episodic returns (passengers delivered before eventual game over) make early exploration difficult. The training pipeline integrates Potential-Based Reward Shaping (Ng, Harada, & Russell, 1999) to provide dense feedback while preserving optimal policy invariance:

$$R'(s, a, s') = R(s, a, s') + F(s, a, s')$$
$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

### Potential Function Formulation

The potential function $\Phi(s)$ is constructed from three state metrics:

1. **Passenger Path Distance Reduction ($\Phi_{dist}$)**:
   Measures cumulative remaining Manhattan/Euclidean distance for all waiting and traveling passengers relative to their target shape stations:
   $$\Phi_{dist}(s) = -w_1 \sum_{p \in \text{passengers}} D(p.\text{station}, p.\text{target})$$

2. **Overcrowding Mitigation ($\Phi_{crowd}$)**:
   Penalizes stations approaching their critical limit:
   $$\Phi_{crowd}(s) = -w_2 \sum_{i \in \text{stations}} \left( \frac{\text{timer}_i}{\text{max\_timer}} \right)^2$$

3. **Shape Diversity & Network Connectivity ($\Phi_{conn}$)**:
   Rewards networks where lines link complementary station shapes (e.g., Circle $\leftrightarrow$ Triangle $\leftrightarrow$ Square) in alternating sequences, minimizing unnecessary transfers.

Because $F(s, a, s')$ is expressed as the difference of potentials, the optimal policy $\pi^*$ under shaped rewards is identical to the optimal policy under the original environment reward.

---

## Curriculum Learning: 4-Stage Progressive Pool

Training reinforcement learning models across complex maps from scratch often causes policy collapse:
- If trained on simple maps alone, the agent never learns tunnel or water conservation.
- If trained on constrained maps (such as NYC) from step 0, early policies can quickly exhaust tunnels, causing station overflow before discovering basic routing.
- If maps are swapped in isolation (Map 3 $\rightarrow$ Map 0 $\rightarrow$ Map 1), the neural network suffers from **catastrophic forgetting**, overwriting earlier routing representations.

To resolve this, the curriculum implements a **Progressive Expanding Pool**:

```
[Stage 1: Berlin Fundamentals]
 Maps: [Berlin] (100%)
 Focus: Open grid routing, shape alternation, train dispatching.
   |
   | (Rolling Avg Score >= 100.0 AND Steps >= 30,000)
   v
[Stage 2: River Crossing]
 Maps: [Berlin, London] (50% / 50%)
 Focus: Horizontal River Thames crossing; tunnel budgeting.
   |
   | (Rolling Avg Score >= 150.0 AND Steps >= 60,000)
   v
[Stage 3: Coastal Islands]
 Maps: [Berlin, London, Tokyo] (33% / 33% / 34%)
 Focus: Tokyo bay archipelago; high-speed Shinkansen surges.
   |
   | (Rolling Avg Score >= 200.0 AND Steps >= 100,000)
   v
[Stage 4: All Maps]
 Maps: [Berlin, London, Tokyo, NYC] (25% each)
 Focus: Manhattan river bottlenecks; all four maps active simultaneously.
```

### Promotion Criteria

| Parameter | GPU Full Training (`train.py`) | Local CPU Fast Training (`train_local.py`) |
|:---|:---:|:---:|
| **Stage 1 $\rightarrow$ 2 Threshold** | Rolling Avg Score $\ge 100.0$ | Rolling Avg Score $\ge 35.0$ |
| **Stage 1 Min Steps** | 30,000 global steps | 8,000 global steps |
| **Stage 2 $\rightarrow$ 3 Threshold** | Rolling Avg Score $\ge 150.0$ | Rolling Avg Score $\ge 55.0$ |
| **Stage 2 Min Steps** | 60,000 global steps | 16,000 global steps |
| **Stage 3 $\rightarrow$ 4 Threshold** | Rolling Avg Score $\ge 200.0$ | Rolling Avg Score $\ge 75.0$ |
| **Stage 3 Min Steps** | 100,000 global steps | 26,000 global steps |

Both scripts support CLI configuration of these values via `--curriculum-thresholds` and `--curriculum-min-steps`.

---

## Supported Maps & Geographic Mechanics

The simulator models four cities:

### 1. London (Map ID: 0)
- **Geography**: Bisected horizontally by the meandering River Thames.
- **Constraints**: 1 starting tunnel token; stations spawn on north and south banks.
- **Routing Strategy**: Lines must bridge the river without depleting tunnel reserves, creating interchange connections across the water.

### 2. New York City (Map ID: 1)
- **Geography**: Narrow, elongated Manhattan island flanked by the Hudson River and East River, with outer boroughs.
- **Constraints**: 2 starting tunnels; higher station density.
- **Routing Strategy**: Tracks must traverse multiple river channels while handling station crowding across Manhattan.

### 3. Tokyo (Map ID: 2)
- **Geography**: Tokyo Bay coastal perimeter with offshore islands.
- **Constraints**: Accelerated passenger generation rates; specialized high-speed Shinkansen train models.
- **Routing Strategy**: Requires carriage deployment and loop topology to handle passenger surges.

### 4. Berlin (Map ID: 3)
- **Geography**: Continental open urban terrain bisected by the narrow Spree river.
- **Constraints**: Generous land area; zero tunnel requirements.
- **Routing Strategy**: Suitable for learning network layout, circular line balancing, and radial spoke architecture without water obstacles.

---

## Prerequisites & Installation

### System Dependencies

On Ubuntu / Debian:
```bash
sudo apt update
sudo apt install -y build-essential golang python3 python3-venv python3-pip nodejs npm
```

On Arch Linux:
```bash
sudo pacman -S base-devel go python nodejs npm
```

### Environment Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AniruthKarthik/mini-metro.git
   cd mini-metro
   ```

2. **Build the C-Shared Engine Library**:
   ```bash
   make build-lib
   ```
   This executes `ml/build_lib.sh` using `go build -buildmode=c-shared` to compile `simulator/c_api/` into `ml/libminimetro.so`.

3. **Install Python Dependencies**:
   ```bash
   cd ml
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cd ..
   ```

4. **Install Frontend Dependencies**:
   ```bash
   cd ui
   npm install
   cd ..
   ```

---

## Makefile Reference

The root [Makefile](file:///home/leomarshall/mm/Makefile) provides unified commands for building, testing, training, and playing:

| Target | Description | Underlying Execution |
|:---|:---|:---|
| `make build-lib` | Compiles Go engine into C-shared library | `bash ml/build_lib.sh` $\rightarrow$ `ml/libminimetro.so` |
| `make test` | Runs Go simulator test suite | `cd simulator && go test -v ./...` |
| `make game` | Default gameplay launcher (alias for `gpu-game`) | Starts UI, backend, and GPU AI agent |
| `make gpu-game` | Launches game with 256-dim GPU model | UI (port 3000), Go backend (port 6969), `ml/agent.py` |
| `make cpu-game` | Launches game with 32-dim local CPU model | UI (port 3000), Go backend (port 6969), `ml/agent.py --device cpu` |
| `make train` | Full multi-map PPO training on GPU | `train.py --maps 0 1 2 3 --map-mode stratified` |
| `make train-local` | Fast parallel multi-map training on CPU | `train_local.py --maps 0 1 2 3 --map-mode stratified` |
| `make finetune` | Fine-tune existing checkpoint across all maps | `train.py --fine-tune --maps 0 1 2 3 --map-mode stratified` |
| `make finetune-berlin`| Fine-tune emphasizing Berlin open grid | `train.py --fine-tune --maps 0 1 2 3 --map-weights 1 1 1 3` |
| `make back` | Run standalone Go backend server | `cd simulator && go run cmd/server/main.go -addr :6969` |
| `make front` | Run standalone Vite frontend UI | `cd ui && npm run dev -- --port 3000 --host` |
| `make clean` | Free development ports 6969 and 3000 | Terminates listening processes on ports 6969 and 3000 |
| `make fnlist` | Lists all exported Go functions | Parses Go sources and prints function signatures |

---

## Training Workflows & CLI Parameters

### 1. GPU Multi-Map Training (`ml/train.py`)

Used for GPU training runs. Supports CUDA, Automatic Mixed Precision (AMP), and multi-environment vectorized workers.

```bash
cd ml
./venv/bin/python train.py \
    --maps 0 1 2 3 \
    --map-mode stratified \
    --curriculum \
    --curriculum-thresholds 100.0 150.0 200.0 \
    --curriculum-min-steps 30000 60000 100000 \
    --pbrs \
    --hierarchical \
    --num-envs 32 \
    --num-steps 512 \
    --total-steps 10000000 \
    --learning-rate 0.0003
```

#### Complete Argument Reference:
- `--maps`: List of map IDs to train on (`0`: London, `1`: NYC, `2`: Tokyo, `3`: Berlin). Default: `[0, 1, 2, 3]`.
- `--map-mode`: Strategy for assigning maps to workers:
  - `stratified`: Evenly partitions worker environments across active maps.
  - `mixed`: Samples maps according to `--map-weights`.
  - `round_robin`: Cycles maps per episode reset.
- `--map-weights`: Relative probability weights when `--map-mode mixed` is active.
- `--curriculum / --no-curriculum`: Enables/disables procedural 4-stage curriculum progression.
- `--curriculum-thresholds`: Rolling average score promotion targets (3 floats).
- `--curriculum-min-steps`: Minimum global steps required per stage before promotion (3 ints).
- `--pbrs / --no-pbrs`: Enables Potential-Based Reward Shaping (Ng et al. 1999).
- `--hierarchical / --no-hierarchical`: Enables two-stage action factorization.
- `--num-envs`: Number of parallel vectorized environments (default: 32 on GPU, 16 on CPU).
- `--num-steps`: Trajectory steps gathered per environment per PPO rollout update (default: 512).
- `--total-steps`: Total global environment interaction steps before termination.
- `--learning-rate`: Base Adam optimizer learning rate (linearly decayed to 0).
- `--fine-tune`: Resumes from the latest checkpoint while reinitializing the learning rate schedule for adaptation.
- `--pretrained`: Explicit path to a `.pt` checkpoint file to resume from.

### 2. Fast CPU Local Training (`ml/train_local.py`)

Optimized for lightweight local development on consumer laptops or workstations without a dedicated GPU:

```bash
cd ml
./venv/bin/python train_local.py \
    --maps 0 1 2 3 \
    --map-mode stratified \
    --num-envs 16 \
    --total-steps 40000 \
    --curriculum-thresholds 35.0 55.0 75.0 \
    --curriculum-min-steps 8000 16000 26000
```

- Uses compact hidden dimension (`hidden_dim=32`) to achieve high step-per-second (SPS) throughput on standard CPUs.
- Automatically handles emergency checkpointing on unexpected termination or `SIGINT` (Ctrl+C).

---

## Checkpoint Management & Inference Hierarchy

Training runs automatically save checkpoints in versioned directory structures:
- GPU Runs: `ml/runs/minimetro_ppo/`
- Local CPU Runs: `ml/runs/minimetro_ppo_local/`

### Checkpoint File Types

1. **`model_best.pt`**:
   - Saved **strictly when the rolling average score reaches a new all-time high**.
   - Contains raw model weights (`state_dict`).
   - **Never pruned or deleted** by cleanup routines.

2. **`model_final.pt`**:
   - Saved at the conclusion of training when the final update completes.

3. **`checkpoint_NNNNN.pt`**:
   - Periodic full-state training snapshots (every 10 updates).
   - Contains model weights, optimizer states, curriculum progression, and step counters.
   - **Pruning Rule**: The cleanup routine automatically retains the last 5 numbered checkpoints and prunes older numbered snapshots to prevent storage exhaustion.

### Model Loading & Inference Priority

When starting the live game (`make game`) or running inference (`ml/agent.py`), the model loader evaluates all available checkpoint files and selects a candidate based on a four-tier sorting priority:

$$\text{Priority} = (\text{hidden\_dim}, \text{tier}, \text{model\_rank}, \text{mtime})$$

1. **`hidden_dim`**: Higher model capacity (256-dim GPU model preferred over 32-dim local model).
2. **`tier`**: Finetuned models (`tier 3`) over standard runs (`tier 2`) over local runs (`tier 1`).
3. **`model_rank`**:
   - `model_best.pt` $\rightarrow$ `rank = 2` (**Highest Priority**)
   - `model_final.pt` $\rightarrow$ `rank = 1`
   - `checkpoint_*.pt` $\rightarrow$ `rank = 0`
4. **`mtime`**: Most recently modified timestamp breaks ties.

---

## Monitoring & TensorBoard Metrics

Training runs log scalar metrics to TensorBoard:

```bash
tensorboard --logdir ml/runs
```

### Key Metrics Guide

#### Performance Metrics
- `charts/score_all`: Overall passenger delivery score across all active environments.
- `charts/score_<map_name>`: Isolated score per individual city map (e.g. `charts/score_london`).
- `charts/best_rolling_score`: Running peak score tracking promotion eligibility.
- `charts/curriculum_stage`: Current active curriculum stage (1 to 4).
- `charts/episodic_return`: Cumulative shaped episodic return.
- `charts/SPS`: Real-time steps per second throughput.

#### Policy & Action Behavior
- `charts/noop_rate`: Proportion of macro-steps where the policy elected to do nothing.
- `charts/expansion_action_rate`: Frequency of network modifications (adding/extending lines).
- `charts/avg_lines_per_station`: Average connectivity degree across all active stations.
- `charts/redundant_station_rate`: Percentage of stations connected redundantly by multiple identical lines.

#### Optimization Diagnostics
- `losses/policy_loss`: PPO surrogate clipping loss.
- `losses/value_loss`: Mean squared error of the critic's state value estimations.
- `losses/entropy`: Policy entropy (monitors exploration vs exploitation collapse).
- `losses/approx_kl`: Approximate KL divergence between old and updated policy distributions.

---

## Project Directory Structure

```
.
├── Makefile                     # Root build, test, run, and training automation
├── readme.md                    # System documentation and instructions
├── simulator/                   # Go Simulation Engine
│   ├── c_api/                   # CGO exports bridging Go engine to C shared library
│   │   ├── main.go              # Shared library exports (Step, Reset, GetObs, etc.)
│   │   └── minimetro.h          # Generated C header definitions
│   ├── cmd/                     # CLI entrypoints and engine test suites
│   │   ├── main.go              # Standalone simulator runner
│   │   ├── server/              # HTTP / WebSocket server entrypoint
│   │   ├── passenger_test.go    # Passenger demand, spawning, and destination tests
│   │   ├── physics_test.go      # Train kinematics, speed, and braking tests
│   │   ├── routing_test.go      # A* pathfinding and transfer routing tests
│   │   ├── station_test.go      # Station geometry, interchange, and crowding tests
│   │   ├── topology_test.go     # Line connection, loop closing, and topology tests
│   │   └── water_test.go        # River crossings and tunnel constraint tests
│   ├── engine/                  # Core simulation mechanics
│   │   ├── actions.go           # Action validation and state execution
│   │   ├── action_space.go      # Discrete action masking and mapping logic
│   │   ├── gameState.go         # Complete state container and serialization
│   │   ├── line.go              # Track lines, loop detection, and station order
│   │   ├── map.go               # Map configurations (London, NYC, Tokyo, Berlin)
│   │   ├── observation.go       # Tensor observation generation for Python
│   │   ├── routing.go           # A* pathfinding graph and transfer penalties
│   │   ├── scoring.go           # Passenger scoring and penalty calculations
│   │   ├── simulator.go         # Master simulation loop and tick coordination
│   │   ├── spawner.go           # Progressive station and passenger spawners
│   │   ├── station.go           # Station capacities, shapes, and crowd timers
│   │   └── train.go             # Train physics, carriages, and movement
│   └── server/                  # WebSocket communications
│       ├── handler.go           # WebSocket upgrader and pump goroutines
│       ├── hub.go               # Broadcast hub managing active clients
│       └── server.go            # Action parsing and dispatch router
├── ml/                          # Machine Learning & Reinforcement Learning
│   ├── agent.py                 # Live agent connecting to WebSocket server
│   ├── build_lib.sh             # Shell script compiling Go C-shared library
│   ├── curriculum.py            # 4-stage progressive expanding pool manager
│   ├── diagnostics.py           # Model introspection and latent space probing
│   ├── env.py                   # Vectorized Gymnasium C-API environment wrapper
│   ├── eval.py                  # Benchmark evaluation script generating reports
│   ├── intervention.py          # Strategic intervention and disruption accounting
│   ├── libminimetro.h           # C header for libminimetro.so
│   ├── mcts.py                  # Guided lookahead search for emergency states
│   ├── migrate_checkpoint.py    # Checkpoint schema migration utility
│   ├── model.py                 # Actor-Critic Graph Attention Network (GATv2+LSTM)
│   ├── ppo.py                   # Vectorized PPO trainer with GAE and sequence LSTM
│   ├── probing.py               # Feature attribution and attention analysis
│   ├── requirements.txt         # Python package dependencies
│   ├── run_deletion_comparison.py # Comparative study of line deletion strategies
│   ├── run_reward_ablations.py  # Ablation studies on reward shaping components
│   ├── train.py                 # Distributed GPU PPO training pipeline
│   ├── train_local.py           # Parallel CPU PPO training pipeline
│   └── runs/                    # Saved checkpoints and TensorBoard telemetry
├── ui/                          # TypeScript Frontend Visualization
│   ├── src/
│   │   ├── assets/              # SVG map icons and city assets
│   │   ├── interaction/         # Mouse drag-and-drop line and train manipulation
│   │   ├── render/              # Canvas track renderer and smooth interpolation
│   │   ├── ui/                  # HUD, resource bar, clock, and speed controls
│   │   └── ws/                  # WebSocket client and snapshot parser
│   ├── index.html               # Web entrypoint
│   ├── package.json             # Vite dependencies and build scripts
│   └── tsconfig.json            # TypeScript compiler configuration
└── resrc/                       # Historical research logs and documentation
    ├── improvements.md          # Architectural improvements audit log
    └── todo.md                  # Development tasks and milestone tracking
```

---

## Troubleshooting & FAQ

### 1. `OSError: ml/libminimetro.so: cannot open shared object file`
**Cause**: The Go simulation engine has not been compiled into the C-shared library required by Python.  
**Resolution**: Run `make build-lib` from the repository root. Ensure `gcc` and `go` are installed and accessible on your `PATH`.

### 2. `listen tcp :6969: bind: address already in use` (or port 3000)
**Cause**: A previous simulation server or Vite dev server instance is still running in the background.  
**Resolution**: Run `make clean` to terminate any existing processes bound to ports 6969 and 3000.

### 3. Training fails with `CUDA out of memory`
**Cause**: GPU VRAM is exhausted by high environment concurrency or large rollout buffers.  
**Resolution**: Decrease `--num-envs` (e.g. from 32 to 16) or reduce `--num-steps` (from 512 to 256):
```bash
./venv/bin/python train.py --num-envs 16 --num-steps 256
```

### 4. GPU Capability Warning (`sm_XX not supported`)
**Cause**: PyTorch was installed with CUDA runtime binaries incompatible with the current GPU compute capability.  
**Resolution**: `train.py` detects this automatically and falls back safely to CPU execution. To leverage the GPU, install matching PyTorch wheels from [pytorch.org](https://pytorch.org/).

### 5. Will running training delete my best model?
**Answer**: **No.** Peak-performing weights are stored in `model_best.pt`, which is never deleted or pruned by automated cleanup routines. The checkpoint cleanup function only prunes intermediate step snapshots matching `checkpoint_*.pt`.

### 6. Which model is loaded by default when I launch the game?
**Answer**: When you run `make game`, the agent introspects all available checkpoint folders and loads `model_best.pt` from the highest-capacity trained model available.

---

## License & Disclaimers

This project is an independent research and educational implementation created for artificial intelligence benchmarking. Mini Metro is a registered trademark of Dinosaur Polo Club. All original visual designs, brand assets, and game concepts belong to Dinosaur Polo Club.

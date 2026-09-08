# Mini Metro AI - Model Improvement Tasks

Based on the architectural audit, the following tasks are prioritized to achieve State-of-the-Art performance for the Mini Metro AI agent.

## Phase 1: Temporal & Sequence Modeling
Congestion in Mini Metro is highly temporal. The agent needs to understand the trajectory of stations (filling up vs. emptying out).
- [ ] **Implement Recurrent Policy (LSTM/GRU)**: 
  - Add an LSTM layer after the GNN embedding output and before the actor/critic heads.
  - Update the PPO training loop to handle hidden states during rollout collection and minibatch updates (e.g., using Truncated BPTT).
- [ ] **Dynamic Frame Skipping**:
  - Modify the step function in `ml/env.py` to tick every 3-5 in-game seconds instead of 1 second.
  - Add logic to interrupt the skip if an emergency (e.g., overcrowding timer > 0) is triggered, allowing the agent to react immediately.

## Phase 2: Advanced Graph Representation (GNN Improvements)
The current 3-layer GCN with Mean+Max pooling is good, but can be improved to capture long-range dependencies and critical bottlenecks.
- [ ] **Implement Graph Attention Networks (GAT / GATv2)**:
  - Replace `GNNLayer` with a GAT-based layer so the model can dynamically weigh critical edges (e.g., lines connecting to unique shape stations).
- [ ] **Add Virtual Global Nodes**:
  - Introduce a virtual "global" node connected to all other nodes to allow congestion signals to propagate across the entire map in a single hop, mitigating the 3-hop receptive field limit.
- [ ] **Implement Hierarchical Pooling**:
  - Replace the global Mean+Max pooling with DiffPool or SAGPool to cluster the graph into regions (e.g., North, South, Central). This preserves sub-graph topological information before passing it to the action heads.

## Phase 3: Training & RL Optimizations
Improve the PPO training regime to help the agent discover better strategies.
- [ ] **Entropy Annealing**:
  - Implement a decaying schedule for `ent_coef` in `ml/train.py` (e.g., start at 0.1 and decay linearly or exponentially to 0.01) to force early exploration and late exploitation.
- [ ] **Curriculum Learning Pipeline**:
  - Modify the Gym environment to support customizable spawn rates and map complexities.
  - Create a training script that starts with low passenger spawn rates and gradually increases them based on the agent's win rate/survival time.
- [ ] **Population Based Training (PBT)**:
  - Set up a PBT script to dynamically tune learning rates, clipping coefficients, and the entropy schedule across multiple concurrent runs.
- [ ] **Scale Network Capacity**:
  - Ensure the production run script uses `hidden_dim` between 256 and 512 for the GNN layers and MLP heads to capture late-game complexity.

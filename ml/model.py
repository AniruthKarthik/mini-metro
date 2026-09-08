import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PHASE-3: Improved GNN layer with:
#   1. dst_feat in message: "station A is congested" influences messages
#      arriving *at* A from neighbours, not just messages departing from A.
#   2. Edge update MLP: edge embeddings evolve across layers instead of using
#      the same stale raw features at every layer. Without this, stacking
#      two GCN layers adds almost no expressiveness for the edge modality.
# ---------------------------------------------------------------------------
class GNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # PHASE-3 fix GNN-2: message uses [src, dst, edge] — destination-aware
        self.msg_proj    = nn.Linear(hidden_dim * 3, hidden_dim)
        self.update_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # PHASE-3 fix GNN-1: edge update MLP — edges evolve between layers
        self.edge_update = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, nodes, edges, edge_feats, num_nodes, num_edges):
        """
        Args:
            nodes:      [B, N, node_dim]
            edges:      [B, 2, E]  — edges[:, 0] = src, edges[:, 1] = dst
            edge_feats: [B, E, edge_dim]  — raw OR updated edge features
            num_nodes:  [B] or [B, 1]
            num_edges:  [B] or [B, 1]
        Returns:
            new_nodes:  [B, N, H]
            new_edges:  [B, E, H]  — updated edge embeddings to pass to next layer
        """
        B, N, _ = nodes.shape
        _, _, E = edges.shape
        H = self.msg_proj.out_features

        x = F.relu(self.node_proj(nodes))       # [B, N, H]
        e = F.relu(self.edge_proj(edge_feats))  # [B, E, H]

        src = edges[:, 0, :].long().clamp(0, N - 1)  # [B, E]
        dst = edges[:, 1, :].long().clamp(0, N - 1)  # [B, E]

        src_feat = torch.gather(x, 1, src.unsqueeze(-1).expand(-1, -1, H))  # [B, E, H]
        dst_feat = torch.gather(x, 1, dst.unsqueeze(-1).expand(-1, -1, H))  # [B, E, H]

        # PHASE-3 GNN-2: destination-aware message
        msg = F.relu(self.msg_proj(torch.cat([src_feat, dst_feat, e], dim=-1)))  # [B, E, H]

        # Mask invalid edges
        edge_mask = (torch.arange(E, device=x.device).unsqueeze(0) < num_edges)  # [B, E]
        msg = msg * edge_mask.unsqueeze(-1)

        # Aggregate messages at destination nodes
        offsets  = torch.arange(B, device=x.device).unsqueeze(-1) * N  # [B, 1]
        flat_dst = (dst + offsets).view(-1)
        flat_msg = msg.view(-1, H)
        aggr = torch.zeros(B * N, H, device=x.device, dtype=flat_msg.dtype)
        aggr.index_add_(0, flat_dst, flat_msg)
        aggr = aggr.view(B, N, H)

        new_x = F.relu(self.update_proj(torch.cat([x, aggr], dim=-1)))
        no_edges = (num_edges == 0).unsqueeze(-1)
        new_nodes = torch.where(no_edges, x, new_x)

        # PHASE-3 GNN-1: update edge embeddings for the next layer
        new_edges = F.relu(self.edge_update(torch.cat([src_feat, dst_feat, e], dim=-1)))
        new_edges = new_edges * edge_mask.unsqueeze(-1)  # zero out invalid

        return new_nodes, new_edges


class MiniMetroActorCritic(nn.Module):
    def __init__(self, node_dim=29, edge_dim=10, global_dim=13, action_space_size=4108, hidden_dim=128):
        # PHASE-2: node_dim 25→29, global_dim 8→13 to match updated observation.go
        # PHASE-3: DenseGCNLayer→GNNLayer (dst_feat + edge update); 3rd layer + residual
        super().__init__()

        # PHASE-3: use improved GNNLayer for all three passes
        self.gcn1 = GNNLayer(node_dim, edge_dim, hidden_dim)
        self.gcn2 = GNNLayer(hidden_dim, hidden_dim, hidden_dim)  # edge_dim = H after layer 1
        self.gcn3 = GNNLayer(hidden_dim, hidden_dim, hidden_dim)  # PHASE-3: 3rd layer

        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fc_actor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # *3 for [mean_pool, max_pool, global]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size)
        )

        self.fc_critic = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # *3 for [mean_pool, max_pool, global]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_value(self, obs):
        logits, value = self.forward(obs)
        return value

    def get_action_and_value(self, obs, action=None, mask=None):
        logits, value = self.forward(obs)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value

    def forward(self, obs):
        nodes        = obs["nodes"]
        edges        = obs["edges"]
        edge_attrs   = obs["edge_attrs"]
        globals_feat = obs["globals"]
        num_nodes    = obs["num_nodes"]
        num_edges    = obs["num_edges"]

        # PHASE-3: thread updated edge embeddings between layers
        x, e   = self.gcn1(nodes, edges, edge_attrs, num_nodes, num_edges)
        x2, e2 = self.gcn2(x, edges, e, num_nodes, num_edges)

        # PHASE-3 GNN-3: 3rd layer with residual connection (prevents oversmoothing)
        x3, _  = self.gcn3(x2, edges, e2, num_nodes, num_edges)
        x3 = x3 + x2  # residual: skip-connect layer-2 output into layer-3 output

        B, N, H = x3.shape
        node_mask = torch.arange(N, device=x3.device).unsqueeze(0) < num_nodes  # [B, N]

        # PHASE-2: mean+max pooling so the actor can detect the single worst station.
        mean_pool = (x3 * node_mask.unsqueeze(-1)).sum(dim=1) / num_nodes.clamp(min=1).float()
        x_for_max = x3.masked_fill(~node_mask.unsqueeze(-1), -1e9)
        max_pool  = x_for_max.max(dim=1).values
        pooled    = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2H]

        g        = self.global_proj(globals_feat)
        combined = torch.cat([pooled, g], dim=-1)  # [B, 3H]

        logits = self.fc_actor(combined)
        value  = self.fc_critic(combined)

        return logits, value

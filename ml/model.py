import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGCNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.msg_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, nodes, edges, edge_attrs, num_nodes, num_edges):
        B, N, _ = nodes.shape
        _, _, E = edges.shape
        H = self.node_proj.out_features
        
        x = F.relu(self.node_proj(nodes)) # [B, N, H]
        e = F.relu(self.edge_proj(edge_attrs)) # [B, E, H]
        
        src = edges[:, 0, :].long() # [B, E]
        dst = edges[:, 1, :].long() # [B, E]
        
        # Clamp to avoid out of bounds on invalid edges
        src = src.clamp(min=0, max=N-1)
        dst = dst.clamp(min=0, max=N-1)
        
        src_feat = torch.gather(x, 1, src.unsqueeze(-1).expand(-1, -1, H)) # [B, E, H]
        
        msg = F.relu(self.msg_proj(torch.cat([src_feat, e], dim=-1))) # [B, E, H]
        
        # Mask out invalid edges
        edge_mask = torch.arange(E, device=x.device).unsqueeze(0) < num_edges # [B, E]
        msg = msg * edge_mask.unsqueeze(-1)
        
        batch_offsets = torch.arange(B, device=x.device).unsqueeze(-1) * N # [B, 1]
        flat_dst = (dst + batch_offsets).view(-1)
        flat_msg = msg.view(-1, H)
        
        aggr = torch.zeros(B * N, H, device=x.device)
        aggr.index_add_(0, flat_dst, flat_msg)
        aggr = aggr.view(B, N, H)
        
        new_x = F.relu(self.update_proj(torch.cat([x, aggr], dim=-1)))
        
        # Where n_edges == 0, old code just outputs x
        no_edges_mask = (num_edges == 0).unsqueeze(-1)
        out_nodes = torch.where(no_edges_mask, x, new_x)
        
        return out_nodes

class MiniMetroActorCritic(nn.Module):
    def __init__(self, node_dim=25, edge_dim=10, global_dim=8, action_space_size=4108, hidden_dim=128):
        super().__init__()
        
        self.gcn1 = DenseGCNLayer(node_dim, edge_dim, hidden_dim)
        self.gcn2 = DenseGCNLayer(hidden_dim, edge_dim, hidden_dim)
        
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.fc_actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size)
        )
        
        self.fc_critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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
            # Apply large negative number to invalid actions
            logits = logits.masked_fill(~mask, -1e9)
            
        probs = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), value

    def forward(self, obs):
        nodes = obs["nodes"]
        edges = obs["edges"]
        edge_attrs = obs["edge_attrs"]
        globals_feat = obs["globals"]
        num_nodes = obs["num_nodes"]
        num_edges = obs["num_edges"]
        
        x = self.gcn1(nodes, edges, edge_attrs, num_nodes, num_edges)
        x = self.gcn2(x, edges, edge_attrs, num_nodes, num_edges)
        
        B, N, H = x.shape
        node_mask = torch.arange(N, device=x.device).unsqueeze(0) < num_nodes # [B, N]
        pooled = (x * node_mask.unsqueeze(-1)).sum(dim=1) / num_nodes.clamp(min=1).float()
                
        g = self.global_proj(globals_feat)
        combined = torch.cat([pooled, g], dim=-1)
        
        logits = self.fc_actor(combined)
        value = self.fc_critic(combined)
        
        return logits, value

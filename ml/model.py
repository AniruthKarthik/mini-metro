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
        
        x = F.relu(self.node_proj(nodes)) # [B, N, H]
        e = F.relu(self.edge_proj(edge_attrs)) # [B, E, H]
        
        out_nodes = torch.zeros_like(x)
        
        for b in range(B):
            n_edges = int(num_edges[b, 0].item())
            if n_edges == 0:
                out_nodes[b] = x[b]
                continue
                
            e_idx = edges[b, :, :n_edges].long() # [2, e]
            e_feat = e[b, :n_edges] # [e, H]
            
            src = e_idx[0]
            dst = e_idx[1]
            
            src_feat = x[b, src] # [e, H]
            
            msg = F.relu(self.msg_proj(torch.cat([src_feat, e_feat], dim=-1))) # [e, H]
            
            aggr = torch.zeros(N, x.shape[-1], device=x.device)
            aggr.index_add_(0, dst, msg)
            
            new_x = F.relu(self.update_proj(torch.cat([x[b], aggr], dim=-1)))
            out_nodes[b] = new_x
            
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
        pooled = torch.zeros(B, H, device=x.device)
        for b in range(B):
            n_n = int(num_nodes[b, 0].item())
            if n_n > 0:
                pooled[b] = x[b, :n_n].mean(dim=0)
                
        g = self.global_proj(globals_feat)
        combined = torch.cat([pooled, g], dim=-1)
        
        logits = self.fc_actor(combined)
        value = self.fc_critic(combined)
        
        return logits, value

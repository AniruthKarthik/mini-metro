import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class DenseGCNLayerOld(nn.Module):
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

class DenseGCNLayerNew(nn.Module):
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

B, N, E, H = 512, 30, 200, 128
node_dim, edge_dim = 25, 10
old_layer = DenseGCNLayerOld(node_dim, edge_dim, H)
new_layer = DenseGCNLayerNew(node_dim, edge_dim, H)
new_layer.load_state_dict(old_layer.state_dict())

nodes = torch.randn(B, N, node_dim)
edges = torch.randint(0, N, (B, 2, E))
edge_attrs = torch.randn(B, E, edge_dim)
num_nodes = torch.randint(10, N, (B, 1))
num_edges = torch.randint(0, E, (B, 1))

out_old = old_layer(nodes, edges, edge_attrs, num_nodes, num_edges)
out_new = new_layer(nodes, edges, edge_attrs, num_nodes, num_edges)

diff = (out_old - out_new).abs().max()
print("Max diff:", diff.item())

start = time.time()
for _ in range(10):
    old_layer(nodes, edges, edge_attrs, num_nodes, num_edges)
print("Old time:", time.time() - start)

start = time.time()
for _ in range(10):
    new_layer(nodes, edges, edge_attrs, num_nodes, num_edges)
print("New time:", time.time() - start)

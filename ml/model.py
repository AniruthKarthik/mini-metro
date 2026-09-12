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
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__()
        self.node_proj = nn.Linear(node_dim + global_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        self.msg_proj    = nn.Linear(hidden_dim * 3, hidden_dim)
        self.attn_proj   = nn.Linear(hidden_dim * 3, 1)
        self.update_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.edge_update = nn.Linear(hidden_dim * 3, hidden_dim)
        
        self.global_update = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, nodes, edges, edge_feats, global_ctx, num_nodes, num_edges):
        """
        global_ctx: [B, global_dim]
        """
        B, N, _ = nodes.shape
        _, _, E = edges.shape
        H = self.msg_proj.out_features

        global_ctx_broadcast = global_ctx.unsqueeze(1).expand(-1, N, -1)
        nodes_with_ctx = torch.cat([nodes, global_ctx_broadcast], dim=-1)

        x = F.relu(self.node_proj(nodes_with_ctx))       # [B, N, H]
        e = F.relu(self.edge_proj(edge_feats))           # [B, E, H]

        src = edges[:, 0, :].long().clamp(0, N - 1)  # [B, E]
        dst = edges[:, 1, :].long().clamp(0, N - 1)  # [B, E]

        src_feat = torch.gather(x, 1, src.unsqueeze(-1).expand(-1, -1, H))  # [B, E, H]
        dst_feat = torch.gather(x, 1, dst.unsqueeze(-1).expand(-1, -1, H))  # [B, E, H]
        
        cat_feat = torch.cat([src_feat, dst_feat, e], dim=-1)

        msg = F.relu(self.msg_proj(cat_feat))  # [B, E, H]
        attn_score = F.leaky_relu(self.attn_proj(cat_feat).squeeze(-1), 0.2) # [B, E]

        edge_mask = (torch.arange(E, device=x.device).unsqueeze(0) < num_edges)  # [B, E]
        
        mask_val = -1e4 if attn_score.dtype in (torch.float16, torch.bfloat16) else -1e9
        attn_score = attn_score.masked_fill(~edge_mask, mask_val)
        exp_score = torch.exp(attn_score) * edge_mask.float()
        
        sum_exp = torch.zeros(B * N, device=x.device, dtype=exp_score.dtype)
        offsets = torch.arange(B, device=x.device).unsqueeze(-1) * N  # [B, 1]
        flat_dst = (dst + offsets).view(-1)
        sum_exp.index_add_(0, flat_dst, exp_score.view(-1))
        
        edge_sum_exp = torch.gather(sum_exp.view(B, N), 1, dst) + 1e-9
        alpha = exp_score / edge_sum_exp # [B, E]
        
        msg = msg * alpha.unsqueeze(-1)
        msg = msg * edge_mask.unsqueeze(-1)

        flat_msg = msg.view(-1, H)
        aggr = torch.zeros(B * N, H, device=x.device, dtype=flat_msg.dtype)
        aggr.index_add_(0, flat_dst, flat_msg)
        aggr = aggr.view(B, N, H)

        new_x = F.relu(self.update_proj(torch.cat([x, aggr], dim=-1)))
        no_edges = (num_edges == 0).unsqueeze(-1)
        new_nodes = torch.where(no_edges, x, new_x)

        new_edges = F.relu(self.edge_update(cat_feat))
        new_edges = new_edges * edge_mask.unsqueeze(-1)
        
        node_mask = (torch.arange(N, device=x.device).unsqueeze(0) < num_nodes)
        mean_pool = (new_nodes * node_mask.unsqueeze(-1)).sum(dim=1) / num_nodes.clamp(min=1).float()
        new_global_ctx = self.global_update(torch.cat([mean_pool, global_ctx], dim=-1))

        return new_nodes, new_edges, new_global_ctx


# ---------------------------------------------------------------------------
# P4-2: Graph Attention Network v2 (GATv2) with Edge Features
# ---------------------------------------------------------------------------
class GATv2Layer(nn.Module):
    """
    Graph Attention Network v2 (Brody et al., 2021) with edge features (P4-2).
    Computes dynamic multi-head attention:
        e_{ij}^k = a_k^T LeakyReLU( W [h_i || h_j || e_{ij}] )
        \alpha_{ij}^k = Softmax_j( e_{ij}^k )
    Overcomes static attention bottlenecks where ranking of neighbours cannot depend
    on the query node.
    """
    def __init__(self, in_node_dim, in_edge_dim, global_dim, out_dim, num_heads=4):
        super().__init__()
        self.in_node_dim = in_node_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.node_proj = nn.Linear(in_node_dim + global_dim, out_dim)
        self.edge_proj = nn.Linear(in_edge_dim, out_dim)

        # Dynamic GATv2 attention projections
        self.attn_linear = nn.Linear(out_dim * 3, out_dim)
        self.attn_vec = nn.Linear(out_dim, num_heads, bias=False)
        nn.init.orthogonal_(self.attn_vec.weight, gain=0.1)

        # Multi-head message projection
        self.msg_proj = nn.Linear(out_dim * 3, out_dim)

        # Node update projection
        self.update_proj = nn.Linear(out_dim * 2, out_dim)

        # Dynamic edge feature update
        self.edge_update = nn.Linear(out_dim * 3, out_dim)

        # Global context update
        self.global_update = nn.Sequential(
            nn.Linear(out_dim + global_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, nodes, edges, edge_feats, global_ctx, num_nodes, num_edges):
        B, N, _ = nodes.shape
        _, _, E = edges.shape
        H = self.out_dim
        K = self.num_heads
        D = self.head_dim

        num_nodes_col = num_nodes.view(B, 1)
        num_edges_col = num_edges.view(B, 1)

        global_ctx_broadcast = global_ctx.unsqueeze(1).expand(-1, N, -1)
        nodes_with_ctx = torch.cat([nodes, global_ctx_broadcast], dim=-1)

        x = F.relu(self.node_proj(nodes_with_ctx))  # [B, N, H]
        e = F.relu(self.edge_proj(edge_feats))      # [B, E, H]

        if E == 0 or (num_edges_col == 0).all():
            node_mask = (torch.arange(N, device=x.device).unsqueeze(0) < num_nodes_col)
            mean_pool = (x * node_mask.unsqueeze(-1)).sum(dim=1) / num_nodes_col.clamp(min=1).float()
            new_global_ctx = self.global_update(torch.cat([mean_pool, global_ctx], dim=-1))
            return x, e, new_global_ctx

        src = edges[:, 0, :].long().clamp(0, N - 1)  # [B, E]
        dst = edges[:, 1, :].long().clamp(0, N - 1)  # [B, E]

        src_feat = torch.gather(x, 1, src.unsqueeze(-1).expand(-1, -1, H))  # [B, E, H]
        dst_feat = torch.gather(x, 1, dst.unsqueeze(-1).expand(-1, -1, H))  # [B, E, H]

        cat_feat = torch.cat([src_feat, dst_feat, e], dim=-1)  # [B, E, 3*H]

        # GATv2 dynamic attention score: a^T LeakyReLU(W [src || dst || e])
        h_attn = F.leaky_relu(self.attn_linear(cat_feat), 0.2)  # [B, E, H]
        attn_logits = self.attn_vec(h_attn)                     # [B, E, K]

        edge_mask = (torch.arange(E, device=x.device).unsqueeze(0) < num_edges_col).unsqueeze(-1)  # [B, E, 1]
        mask_val = -1e4 if attn_logits.dtype in (torch.float16, torch.bfloat16) else -1e9
        attn_logits_masked = attn_logits.masked_fill(~edge_mask, mask_val)

        # Numerically stable softmax per destination node:
        max_val = attn_logits_masked.max(dim=1, keepdim=True)[0].detach()  # [B, 1, K]
        max_val = torch.where(max_val < -1e8, torch.zeros_like(max_val), max_val)
        exp_score = torch.exp(attn_logits_masked - max_val) * edge_mask.float()  # [B, E, K]

        sum_exp = torch.zeros(B * N, K, device=x.device, dtype=exp_score.dtype)
        offsets = torch.arange(B, device=x.device).unsqueeze(-1) * N
        flat_dst = (dst + offsets).view(-1)
        sum_exp.index_add_(0, flat_dst, exp_score.view(-1, K))

        edge_sum_exp = torch.gather(sum_exp.view(B, N, K), 1, dst.unsqueeze(-1).expand(-1, -1, K)) + 1e-9
        alpha = exp_score / edge_sum_exp  # [B, E, K]

        msg = F.relu(self.msg_proj(cat_feat))  # [B, E, H]
        msg_heads = msg.view(B, E, K, D)
        weighted_msg = (msg_heads * alpha.unsqueeze(-1)).view(B, E, H) * edge_mask.float()  # [B, E, H]

        flat_msg = weighted_msg.view(-1, H)
        aggr = torch.zeros(B * N, H, device=x.device, dtype=flat_msg.dtype)
        aggr.index_add_(0, flat_dst, flat_msg)
        aggr = aggr.view(B, N, H)

        new_x = F.relu(self.update_proj(torch.cat([x, aggr], dim=-1)))
        no_edges = (num_edges_col.unsqueeze(-1) == 0)  # [B, 1, 1]
        new_nodes = torch.where(no_edges, x, new_x)

        new_edges = F.relu(self.edge_update(cat_feat)) * edge_mask.float()

        node_mask = (torch.arange(N, device=x.device).unsqueeze(0) < num_nodes_col)
        mean_pool = (new_nodes * node_mask.unsqueeze(-1)).sum(dim=1) / num_nodes_col.clamp(min=1).float()
        new_global_ctx = self.global_update(torch.cat([mean_pool, global_ctx], dim=-1))

        return new_nodes, new_edges, new_global_ctx


# ---------------------------------------------------------------------------
# P4-2: Relational Spatial Cross-Attention Transformer Scorer
# ---------------------------------------------------------------------------
class SpatialCrossAttentionScorer(nn.Module):
    """
    Relational Spatial Cross-Attention Transformer Scorer (P4-2).
    Implements multi-head cross-attention over station nodes and candidate spatial
    displacements simultaneously:
        Attention(Q, K, V) = Softmax( Q K^T / sqrt(d) + GeomBias ) V
    Models complex river crossings, network bottlenecks, and multi-line interactions.
    """
    def __init__(self, hidden_dim, num_heads=4, geom_dim=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Multi-head geometric displacement bias MLP:
        self.geom_bias_mlp = nn.Sequential(
            nn.Linear(geom_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads),
        )

        # Relational scoring MLP combines queries, candidate keys, and attended contextual values:
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_geom_bias()

    def _init_geom_bias(self):
        # Inductive geometric prior: penalize candidate distance in attention logits
        l1 = self.geom_bias_mlp[0]
        l2 = self.geom_bias_mlp[2]
        with torch.no_grad():
            l1.weight.zero_()
            l1.bias.zero_()
            l2.weight.zero_()
            l2.bias.zero_()
            # Channel 0: Euclidean distance (first 16 hidden units)
            l1.weight[:16, 0] = 1.0
            l2.weight[:, :16] = -2.0 / 16.0
            # Channel 1: dx (next 8 units)
            l1.weight[16:24, 1] = 1.0
            l2.weight[:, 16:24] = -0.5 / 8.0
            # Channel 2: dy (next 8 units)
            l1.weight[24:32, 2] = 1.0
            l2.weight[:, 24:32] = -0.5 / 8.0

        for m in self.out_proj:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, q_input, k_input, geom_features, key_mask=None):
        """
        q_input: [B, N_q, H]
        k_input: [B, N_k, H]
        geom_features: [B, N_q, N_k, geom_dim]
        key_mask: [B, N_k] (optional bool mask)
        Returns:
            scores: [B, N_q, N_k]
            attn_weights: [B, num_heads, N_q, N_k]
        """
        B, N_q, H = q_input.shape
        _, N_k, _ = k_input.shape
        K = self.num_heads
        D = self.head_dim

        # Project Q, K, V
        Q = self.q_proj(q_input).view(B, N_q, K, D).permute(0, 2, 1, 3)    # [B, K, N_q, D]
        K_mat = self.k_proj(k_input).view(B, N_k, K, D).permute(0, 2, 1, 3) # [B, K, N_k, D]
        V = self.v_proj(k_input).view(B, N_k, K, D).permute(0, 2, 1, 3)    # [B, K, N_k, D]

        # Multi-head attention logits: Q K^T / sqrt(D) + GeomBias
        dot_logits = torch.matmul(Q, K_mat.transpose(-2, -1)) * self.scale  # [B, K, N_q, N_k]
        geom_bias = self.geom_bias_mlp(geom_features).permute(0, 3, 1, 2)  # [B, K, N_q, N_k]
        attn_logits = dot_logits + geom_bias                               # [B, K, N_q, N_k]

        if key_mask is not None:
            mask_val = -1e4 if attn_logits.dtype in (torch.float16, torch.bfloat16) else -1e9
            attn_logits = attn_logits.masked_fill(~key_mask.unsqueeze(1).unsqueeze(2), mask_val)

        attn_weights = F.softmax(attn_logits, dim=-1)                      # [B, K, N_q, N_k]

        # Contextual values: [B, K, N_q, N_k] @ [B, K, N_k, D] -> [B, K, N_q, D]
        ctx_val = torch.matmul(attn_weights, V)                            # [B, K, N_q, D]
        ctx_val = ctx_val.permute(0, 2, 1, 3).reshape(B, N_q, H)          # [B, N_q, H]

        q_exp = q_input.unsqueeze(2).expand(-1, -1, N_k, -1)               # [B, N_q, N_k, H]
        k_exp = k_input.unsqueeze(1).expand(-1, N_q, -1, -1)               # [B, N_q, N_k, H]
        ctx_exp = ctx_val.unsqueeze(2).expand(-1, -1, N_k, -1)             # [B, N_q, N_k, H]

        rel_feat = torch.cat([q_exp, k_exp, ctx_exp], dim=-1)             # [B, N_q, N_k, 3*H]
        # Candidate score combines relational projection with direct multi-head geometric bias:
        scores = self.out_proj(rel_feat).squeeze(-1) + geom_bias.mean(dim=1) # [B, N_q, N_k]

        return scores, attn_weights


# ---------------------------------------------------------------------------
# Action space layout constants matching simulator/engine/action_space.go
# TotalActionSpaceSize = 4087
# ---------------------------------------------------------------------------
ACTION_TYPE_SLICES = [
    slice(0, 1),       # 0: NoOp (1)
    slice(1, 436),     # 1: AddLine (435)
    slice(436, 856),   # 2: ExtendLine (420)
    slice(856, 4006),  # 3: InsertStation (3150)
    slice(4006, 4013), # 4: AddTrain (7)
    slice(4013, 4020), # 5: AddCarriage (7)
    slice(4020, 4050), # 6: UpgradeInterchange (30)
    slice(4050, 4052), # 7: ChooseReward (2)
    slice(4052, 4059), # 8: CloseLoop (7)
    slice(4059, 4066), # 9: OpenLoop (7)
    slice(4066, 4073), # 10: RemoveLine (7)
    slice(4073, 4087), # 11: ShortenLine (14)
]


class MiniMetroActorCritic(nn.Module):
    def __init__(
        self,
        node_dim=32,
        edge_dim=10,
        global_dim=23,
        action_space_size=4087,
        hidden_dim=128,
        use_hierarchical=True,
        gnn_type="gatv2",
        use_transformer_scorer=True,
        num_heads=4,
    ):
        # PHASE-2: global_dim 8→13 to match updated observation.go
        # P0-2: global_dim 13→23 for two 5-class weekly reward card one-hot encodings
        # PHASE-3: DenseGCNLayer→GNNLayer (dst_feat + edge update); 3rd layer + residual
        # PHASE-4: action_space_size 4108→4087 (AddCarriage now lineID-indexed, 28→7 slots)
        # PHASE-5: node_dim 29→32 (+incoming_train_count, incoming_train_load, nearest_train_proximity)
        # PHASE-5: hierarchical action head (Task 19) + bilinear action scoring (Task 20)
        # P4-2: Relational Spatial Cross-Attention Network (GAT-v2 / Transformer Scorer)
        super().__init__()

        self.use_hierarchical = use_hierarchical
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        self.use_transformer_scorer = use_transformer_scorer
        self.num_heads = num_heads
        self.is_legacy_checkpoint = False

        # GNN passes: legacy GCN trunk for backward compatibility
        self.gcn1 = GNNLayer(node_dim, edge_dim, global_dim, hidden_dim)
        self.gcn2 = GNNLayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        self.gcn3 = GNNLayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim)

        # P4-2: Graph Attention Network v2 (GATv2) layers with multi-head dynamic edge attention
        self.gatv2_1 = GATv2Layer(node_dim, edge_dim, global_dim, hidden_dim, num_heads=num_heads)
        self.gatv2_2 = GATv2Layer(hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_heads=num_heads)
        self.gatv2_3 = GATv2Layer(hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_heads=num_heads)

        # P4-2: Relational Spatial Cross-Attention Transformer heads with geometric bias
        self.add_line_transformer = SpatialCrossAttentionScorer(hidden_dim, num_heads=num_heads, geom_dim=4)
        self.extend_line_transformer = SpatialCrossAttentionScorer(hidden_dim, num_heads=num_heads, geom_dim=4)

        # PHASE-2: Hierarchical Pooling (DiffPool) assignment network
        self.diffpool_assign = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

        # PHASE-1: LSTM for recurrent policy (input: 4H DiffPool + 1H Global = 5H)
        self.lstm = nn.LSTM(hidden_dim * 5, hidden_dim * 5, batch_first=True)

        # -------------------------------------------------------------------
        # PHASE-5 Task 19: Hierarchical 12-way Action Type Selector
        # Eliminates the 76.7% InsertStation dominance bias.
        # -------------------------------------------------------------------
        self.type_net = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 12)
        )

        # -------------------------------------------------------------------
        # PHASE-5 Task 20: Bilinear Action Scoring using per-node embeddings
        # -------------------------------------------------------------------
        # AddLine (435): Symmetric bilinear form s(u, v) = (q_u^T k_v + q_v^T k_u) / 2
        self.add_line_q = nn.Linear(hidden_dim, hidden_dim)
        self.add_line_k = nn.Linear(hidden_dim, hidden_dim)
        triu_indices = torch.triu_indices(30, 30, offset=1)
        self.register_buffer("triu_u", triu_indices[0])
        self.register_buffer("triu_v", triu_indices[1])

        # P1-1: Explicit pairwise candidate distance & geometric awareness in AddLine
        # Triu geometry: [distance, dx, dy, distance^2] in R^4
        self.add_line_geom_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self._init_add_line_geom_mlp()

        # UpgradeInterchange (30): Linear projection directly on per-station embedding
        self.interchange_net = nn.Linear(hidden_dim, 1)

        # ExtendLine (420): Factored line-end embedding scored against station embeddings
        self.ext_line_emb = nn.Embedding(7, hidden_dim)
        self.ext_end_emb = nn.Embedding(2, hidden_dim)
        self.ext_context = nn.Linear(hidden_dim * 5, hidden_dim)
        self.ext_proj = nn.Linear(hidden_dim, hidden_dim)

        # InsertStation (3150): Factored line-segment embedding scored against station embeddings
        self.ins_line_emb = nn.Embedding(7, hidden_dim)
        self.ins_seg_emb = nn.Embedding(15, hidden_dim)
        self.ins_context = nn.Linear(hidden_dim * 5, hidden_dim)
        self.ins_proj = nn.Linear(hidden_dim, hidden_dim)

        # Non-spatial action heads (52 actions total):
        # NoOp(1), AddTrain(7), AddCarriage(7), ChooseReward(2),
        # CloseLoop(7), OpenLoop(7), RemoveLine(7), ShortenLine(14)
        self.non_spatial_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 52)
        )

        # P1-3: Candidate-conditioned dispatch heads for AddTrain and AddCarriage (7 lines)
        self.dispatch_line_ctx = nn.Linear(hidden_dim * 5, hidden_dim)
        self.dispatch_train_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 8 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.dispatch_carriage_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 8 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_dispatch_mlps()

        # P3-2: Symmetric candidate-conditioned ChooseReward head (Card 0 vs Card 1)
        self.reward_card_ctx = nn.Linear(hidden_dim * 5, hidden_dim)
        self.reward_card_mlp = nn.Sequential(
            nn.Linear(5 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_reward_card_mlp()

        # Strategic Intervention: Candidate-conditioned RemoveLine and ShortenLine heads
        self.remove_line_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 8 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.shorten_line_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 8 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self._init_intervention_mlps()

        # Fallback flat actor head (for baseline ablation)
        self.fc_actor = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size)
        )

        # Critic value head
        self.fc_critic = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def _init_add_line_geom_mlp(self):
        for m in self.add_line_geom_mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_dispatch_mlps(self):
        nn.init.orthogonal_(self.dispatch_line_ctx.weight, gain=0.1)
        if self.dispatch_line_ctx.bias is not None:
            self.dispatch_line_ctx.bias.data.zero_()

        for mlp in (self.dispatch_train_mlp, self.dispatch_carriage_mlp):
            for m in mlp:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _init_reward_card_mlp(self):
        nn.init.orthogonal_(self.reward_card_ctx.weight, gain=0.1)
        if self.reward_card_ctx.bias is not None:
            self.reward_card_ctx.bias.data.zero_()

        for m in self.reward_card_mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_intervention_mlps(self):
        """
        Initializes intervention heads with inductive stability biases:
        - Penalize unnecessary line deletion when line is healthy/active.
        - Encourage local line shortening over full deletion.
        """
        with torch.no_grad():
            self.remove_line_mlp[-1].bias.fill_(-1.0)
            self.shorten_line_mlp[-1].bias.fill_(-0.2)

    def debias_extension_embeddings(self):

        """
        P2-3 & P3-2: Symmetrize extension and line embeddings.
        - Symmetrizes ext_end_emb weights between front (end=0) and tail (end=1).
        - Symmetrizes ext_line_emb and ins_line_emb weights across lines 0..6.
        - Symmetrizes gcn1.edge_proj line one-hot channel weights across lines 0..6.
        - Symmetrizes non_spatial_head line action slices across lines 0..6.
        Eliminates arbitrary positional and indexing biases across lines and endpoints.
        """
        with torch.no_grad():
            # 1. Front vs Tail symmetry (P2-3)
            w_avg = 0.5 * (self.ext_end_emb.weight[0] + self.ext_end_emb.weight[1])
            self.ext_end_emb.weight[0] = w_avg
            self.ext_end_emb.weight[1] = w_avg

            # 2. Line ID permutation symmetry (P3-2)
            w_ext_line = self.ext_line_emb.weight.mean(dim=0, keepdim=True)
            self.ext_line_emb.weight.copy_(w_ext_line.expand(7, -1))

            w_ins_line = self.ins_line_emb.weight.mean(dim=0, keepdim=True)
            self.ins_line_emb.weight.copy_(w_ins_line.expand(7, -1))

            # Symmetrize gcn1 edge line one-hot features (first 7 channels of edge_dim=10)
            w_edge = self.gcn1.edge_proj.weight[:, 0:7].mean(dim=1, keepdim=True)
            self.gcn1.edge_proj.weight[:, 0:7].copy_(w_edge.expand(-1, 7))

            # Symmetrize gatv2_1 edge line one-hot features if present
            if hasattr(self, "gatv2_1"):
                w_gat_edge = self.gatv2_1.edge_proj.weight[:, 0:7].mean(dim=1, keepdim=True)
                self.gatv2_1.edge_proj.weight[:, 0:7].copy_(w_gat_edge.expand(-1, 7))

            # Symmetrize non_spatial_head line action heads
            # CloseLoop: 17:24 (7 lines)
            # OpenLoop: 24:31 (7 lines)
            # RemoveLine: 31:38 (7 lines)
            # ShortenLine: 38:52 (7 lines x 2 ends)
            ns_w = self.non_spatial_head[2].weight
            ns_b = self.non_spatial_head[2].bias
            for sl in [slice(17, 24), slice(24, 31), slice(31, 38)]:
                ns_w[sl] = ns_w[sl].mean(dim=0, keepdim=True).expand(7, -1)
                ns_b[sl] = ns_b[sl].mean(dim=0, keepdim=True).expand(7)

            sl_w = ns_w[38:52].view(7, 2, -1).mean(dim=0, keepdim=True).expand(7, 2, -1).reshape(14, -1)
            sl_b = ns_b[38:52].view(7, 2).mean(dim=0, keepdim=True).expand(7, 2).reshape(14)
            ns_w[38:52] = sl_w
            ns_b[38:52] = sl_b

    def _extract_line_representations(self, nodes, edges, edge_attrs, x):
        """
        Constructs dynamic candidate line representations for lines 0..6:
        h_line = [mean_pool(x_u for u in line), LineStats] in R^{H + 8}
        """
        B = x.shape[0]
        H = x.shape[-1]
        device = x.device
        dtype = x.dtype

        if edges is None or edge_attrs is None or edges.shape[-1] == 0:
            return torch.zeros(B, 7, H + 8, device=device, dtype=dtype)

        # edges: [B, 2, E], edge_attrs: [B, E, 10]
        # edge_attrs[:, :, 0:7]: one-hot line ID for each edge segment
        edge_lines = edge_attrs[:, :, 0:7].to(dtype=dtype)  # [B, E, 7]
        src = edges[:, 0, :].long().clamp(min=0, max=29)  # [B, E]
        dst = edges[:, 1, :].long().clamp(min=0, max=29)  # [B, E]

        # Scatter line indicators to stations: station_line[b, u, i] > 0 if station u is in line i
        station_line = torch.zeros(B, 30, 7, device=device, dtype=dtype)
        station_line.scatter_add_(1, src.unsqueeze(-1).expand(-1, -1, 7), edge_lines)
        station_line.scatter_add_(1, dst.unsqueeze(-1).expand(-1, -1, 7), edge_lines)
        station_in_line = (station_line > 0).to(dtype=dtype).permute(0, 2, 1)  # [B, 7, 30]

        # Station count per line: [B, 7, 1]
        num_stations = station_in_line.sum(dim=-1, keepdim=True)
        is_active = (num_stations >= 2.0).to(dtype=dtype)

        # 1. Mean-pool node embeddings x: [B, 7, H]
        line_x = torch.bmm(station_in_line, x) / torch.clamp(num_stations, min=1.0)
        line_x = line_x * is_active

        # 2. Extract LineStats [B, 7, 8]:
        stat_station_count = num_stations / 10.0

        # Physical track length: sum forward edge distances (direction > 0)
        fwd_mask = (edge_attrs[:, :, 8:9] > 0).to(dtype=dtype)
        fwd_dist = edge_attrs[:, :, 7:8].to(dtype=dtype) * fwd_mask  # [B, E, 1]
        stat_track_len = torch.bmm(edge_lines.permute(0, 2, 1), fwd_dist)  # [B, 7, 1]

        if nodes is not None:
            nodes_dt = nodes.to(dtype=dtype)
            st_queue = nodes_dt[:, :, 12:22].sum(dim=-1, keepdim=True)  # [B, 30, 1]
            stat_total_queue = torch.bmm(station_in_line, st_queue) / 20.0  # [B, 7, 1]
            stat_avg_queue = (stat_total_queue * 20.0) / (torch.clamp(num_stations, min=1.0) * 10.0)

            st_overcrowd = nodes_dt[:, :, 22:23]  # [B, 30, 1]
            masked_overcrowd = station_in_line.unsqueeze(-1) * st_overcrowd.unsqueeze(1)
            stat_max_overcrowd = masked_overcrowd.max(dim=2)[0]  # [B, 7, 1]

            st_train_count = nodes_dt[:, :, 29:30] * 4.0  # [B, 30, 1]
            stat_train_count = torch.bmm(station_in_line, st_train_count) / 4.0  # [B, 7, 1]

            st_train_load = nodes_dt[:, :, 30:31]  # [B, 30, 1]
            stat_train_load = torch.bmm(station_in_line, st_train_load) / torch.clamp(num_stations, min=1.0)
        else:
            stat_total_queue = torch.zeros(B, 7, 1, device=device, dtype=dtype)
            stat_avg_queue = torch.zeros(B, 7, 1, device=device, dtype=dtype)
            stat_max_overcrowd = torch.zeros(B, 7, 1, device=device, dtype=dtype)
            stat_train_count = torch.zeros(B, 7, 1, device=device, dtype=dtype)
            stat_train_load = torch.zeros(B, 7, 1, device=device, dtype=dtype)

        stat_is_active = is_active

        line_stats = torch.cat([
            stat_station_count,
            stat_track_len,
            stat_total_queue,
            stat_avg_queue,
            stat_max_overcrowd,
            stat_train_count,
            stat_train_load,
            stat_is_active,
        ], dim=-1)  # [B, 7, 8]

        h_line = torch.cat([line_x, line_stats], dim=-1)  # [B, 7, H + 8]
        return h_line

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Handle backward compatibility: adapt legacy checkpoint weights (e.g. global_dim 13 -> 23)
        node_proj_key = prefix + "gcn1.node_proj.weight"
        if node_proj_key in state_dict:
            w = state_dict[node_proj_key]
            curr_w = self.gcn1.node_proj.weight
            if w.shape[0] == curr_w.shape[0] and w.shape[1] < curr_w.shape[1]:
                diff = curr_w.shape[1] - w.shape[1]
                state_dict[node_proj_key] = torch.cat([w, torch.zeros(w.shape[0], diff, device=w.device, dtype=w.dtype)], dim=1)

        glob_up_key = prefix + "gcn1.global_update.0.weight"
        if glob_up_key in state_dict:
            w = state_dict[glob_up_key]
            curr_w = self.gcn1.global_update[0].weight
            if w.shape[0] == curr_w.shape[0] and w.shape[1] < curr_w.shape[1]:
                diff = curr_w.shape[1] - w.shape[1]
                state_dict[glob_up_key] = torch.cat([w, torch.zeros(w.shape[0], diff, device=w.device, dtype=w.dtype)], dim=1)

        # Handle backward compatibility: populate add_line_geom_mlp if missing from legacy checkpoint
        for gk, gv in self.add_line_geom_mlp.state_dict().items():
            full_k = prefix + "add_line_geom_mlp." + gk
            if full_k not in state_dict:
                state_dict[full_k] = gv.clone()

        # Handle backward compatibility: populate dispatch and reward card MLPs if missing
        dispatch_named_modules = [
            ("dispatch_line_ctx", self.dispatch_line_ctx),
            ("dispatch_train_mlp", self.dispatch_train_mlp),
            ("dispatch_carriage_mlp", self.dispatch_carriage_mlp),
            ("reward_card_ctx", self.reward_card_ctx),
            ("reward_card_mlp", self.reward_card_mlp),
            ("remove_line_mlp", self.remove_line_mlp),
            ("shorten_line_mlp", self.shorten_line_mlp),
        ]
        for mod_name, mod in dispatch_named_modules:
            for k, v in mod.state_dict().items():
                full_k = prefix + f"{mod_name}." + k
                if full_k not in state_dict:
                    state_dict[full_k] = v.clone()

        # P4-2: Dual architecture compatibility for GATv2 and Transformer Scorer
        is_legacy = (prefix + "gatv2_1.node_proj.weight") not in state_dict
        self.is_legacy_checkpoint = is_legacy
        if is_legacy:
            self.gnn_type = "gcn"
            self.use_transformer_scorer = False
            gat_named_modules = [
                ("gatv2_1", self.gatv2_1),
                ("gatv2_2", self.gatv2_2),
                ("gatv2_3", self.gatv2_3),
                ("add_line_transformer", self.add_line_transformer),
                ("extend_line_transformer", self.extend_line_transformer),
            ]
            for mod_name, mod in gat_named_modules:
                for k, v in mod.state_dict().items():
                    full_k = prefix + f"{mod_name}." + k
                    if full_k not in state_dict:
                        state_dict[full_k] = v.clone()
        else:
            self.gnn_type = "gatv2"
            self.use_transformer_scorer = True
            # Populate legacy GCN keys if missing from a pure GATv2 checkpoint
            for gcn_name, gcn_mod in [("gcn1", self.gcn1), ("gcn2", self.gcn2), ("gcn3", self.gcn3)]:
                for k, v in gcn_mod.state_dict().items():
                    full_k = prefix + f"{gcn_name}." + k
                    if full_k not in state_dict:
                        state_dict[full_k] = v.clone()

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _compute_hierarchical_logits(self, x, combined, mask=None, nodes=None, edges=None, edge_attrs=None, globals=None):
        """
        Compute hierarchical action log-probabilities:
            log P(a) = log P(type t(a)) + log P(a | type t(a))
        Directly returns log-probabilities over all 4087 actions.
        """
        B, N, H = x.shape
        scale = 1.0 / (H ** 0.5)

        # 1. AddLine scores (435 actions)
        if self.use_transformer_scorer:
            if nodes is not None:
                pos = nodes[:, :, 0:2].to(dtype=x.dtype)  # [B, 30, 2]
                delta = (pos.unsqueeze(2) - pos.unsqueeze(1)).abs()  # [B, 30, 30, 2]
                dist = torch.sqrt(delta[..., 0]**2 + delta[..., 1]**2 + 1e-8).unsqueeze(-1)  # [B, 30, 30, 1]
                dist_sq = dist ** 2  # [B, 30, 30, 1]
                geom_features = torch.cat([dist, delta, dist_sq], dim=-1)  # [B, 30, 30, 4]
            else:
                geom_features = torch.zeros(B, 30, 30, 4, device=x.device, dtype=x.dtype)

            scores_mat, attn_weights = self.add_line_transformer(x, x, geom_features)  # [B, 30, 30]
            self._last_add_line_attn = attn_weights
            scores_sym = 0.5 * (scores_mat + scores_mat.transpose(1, 2))
            scores_add_line = scores_sym[:, self.triu_u, self.triu_v]  # [B, 435]

            if nodes is not None:
                lines_serving = nodes[:, :, 26] * 7.0  # [B, 30]
                is_interchange = nodes[:, :, 24]       # [B, 30]
                redundant_lines = torch.relu(lines_serving - 2.0) * (1.0 - is_interchange)  # [B, 30]
                redundancy_uv = redundant_lines[:, self.triu_u] + redundant_lines[:, self.triu_v]  # [B, 435]
                scores_add_line = scores_add_line - 0.75 * redundancy_uv
            else:
                redundant_lines = None
        else:
            q = self.add_line_q(x)
            k = self.add_line_k(x)
            S = torch.bmm(q, k.transpose(1, 2)) * scale  # [B, 30, 30]
            S_sym = 0.5 * (S + S.transpose(1, 2))
            scores_add_line = S_sym[:, self.triu_u, self.triu_v]  # [B, 435]

            # P1-1 & P1-2: Candidate geometry and redundancy attenuation
            if nodes is not None:
                # P1-1: Distance & geometric displacement penalty
                pos_u = nodes[:, self.triu_u, 0:2]  # [B, 435, 2]
                pos_v = nodes[:, self.triu_v, 0:2]  # [B, 435, 2]
                delta = (pos_u - pos_v).abs()       # [B, 435, 2]
                dist = torch.sqrt(delta[:, :, 0]**2 + delta[:, :, 1]**2 + 1e-8).unsqueeze(-1)  # [B, 435, 1]
                dist_sq = dist ** 2
                triu_geom = torch.cat([dist, delta, dist_sq], dim=-1)  # [B, 435, 4]
                geom_bias = self.add_line_geom_mlp(triu_geom).squeeze(-1)  # [B, 435]

                # P1-2: Redundant expansion attenuation: penalize stations with >2 lines unless interchange
                lines_serving = nodes[:, :, 26] * 7.0  # [B, 30]
                is_interchange = nodes[:, :, 24]       # [B, 30]
                redundant_lines = torch.relu(lines_serving - 2.0) * (1.0 - is_interchange)  # [B, 30]
                redundancy_uv = redundant_lines[:, self.triu_u] + redundant_lines[:, self.triu_v]  # [B, 435]

                scores_add_line = scores_add_line + geom_bias - 0.75 * redundancy_uv
            else:
                redundant_lines = None

        # 2. Per-station UpgradeInterchange scores (30 actions)
        scores_interchange = self.interchange_net(x).squeeze(-1)  # [B, 30]

        # 3. ExtendLine scores (420 actions)
        lines7 = torch.arange(7, device=x.device)
        ends2 = torch.arange(2, device=x.device)
        l_emb = self.ext_line_emb(lines7)  # [7, H]
        e_emb = self.ext_end_emb(ends2)    # [2, H]
        # [7, 2, H] -> [14, H]
        le_base = (l_emb.unsqueeze(1) + e_emb.unsqueeze(0)).view(14, H)
        # Condition on global graph context
        ext_ctx = self.ext_context(combined).unsqueeze(1)  # [B, 1, H]
        line_ends = F.relu(self.ext_proj(le_base.unsqueeze(0) + ext_ctx))  # [B, 14, H]

        if self.use_transformer_scorer:
            ext_geom_features = torch.zeros(B, 14, 30, 4, device=x.device, dtype=x.dtype)
            if nodes is not None and edges is not None and edge_attrs is not None and edges.shape[-1] > 0:
                dtype = x.dtype
                fwd_mask = (edge_attrs[:, :, 8:9] > 0).to(dtype=dtype)
                edge_lines = edge_attrs[:, :, 0:7].to(dtype=dtype)
                fwd_lines = edge_lines * fwd_mask
                src = edges[:, 0, :].long().clamp(min=0, max=29)
                dst = edges[:, 1, :].long().clamp(min=0, max=29)

                fwd_out = torch.zeros(B, 30, 7, device=x.device, dtype=dtype)
                fwd_in = torch.zeros(B, 30, 7, device=x.device, dtype=dtype)
                fwd_out.scatter_add_(1, src.unsqueeze(-1).expand(-1, -1, 7), fwd_lines)
                fwd_in.scatter_add_(1, dst.unsqueeze(-1).expand(-1, -1, 7), fwd_lines)

                is_front = ((fwd_out > 0.5) & (fwd_in < 0.5)).to(dtype=dtype)
                is_back = ((fwd_in > 0.5) & (fwd_out < 0.5)).to(dtype=dtype)

                is_end = torch.stack([is_front.permute(0, 2, 1), is_back.permute(0, 2, 1)], dim=2)
                has_endpoint = is_end.any(dim=-1, keepdim=True).to(dtype=dtype)

                pos = nodes[:, :, 0:2].to(dtype=dtype)
                pos_end = torch.einsum("bler,brc->blec", is_end, pos)

                delta = (pos.unsqueeze(1).unsqueeze(2) - pos_end.unsqueeze(3)).abs()
                dist_cand = torch.sqrt(delta[..., 0]**2 + delta[..., 1]**2 + 1e-8).unsqueeze(-1)
                dist_cand = dist_cand * has_endpoint.unsqueeze(-1)
                dist_sq = dist_cand ** 2
                delta = delta * has_endpoint.unsqueeze(-1)

                geom_cand = torch.cat([dist_cand, delta, dist_sq], dim=-1)  # [B, 7, 2, 30, 4]
                ext_geom_features = geom_cand.view(B, 14, 30, 4)

            ext_scores_mat, ext_attn_weights = self.extend_line_transformer(line_ends, x, ext_geom_features)  # [B, 14, 30]
            self._last_extend_line_attn = ext_attn_weights
            if redundant_lines is not None:
                ext_scores_mat = ext_scores_mat - 0.75 * redundant_lines.unsqueeze(1)
            scores_extend = ext_scores_mat.view(B, 7, 2, 30).permute(0, 1, 3, 2).reshape(B, 420)
        else:
            ext_mat = torch.bmm(line_ends, x.transpose(1, 2)) * scale          # [B, 14, 30]
            if redundant_lines is not None:
                ext_mat = ext_mat - 0.75 * redundant_lines.unsqueeze(1)
            # Permute (lineID, end, stID) -> (lineID, stID, end) then flatten to [B, 420]
            scores_extend = ext_mat.view(B, 7, 2, 30).permute(0, 1, 3, 2).reshape(B, 420)

            # P2-3: Candidate-to-endpoint geometric distance conditioning
            if nodes is not None and edges is not None and edge_attrs is not None and edges.shape[-1] > 0:
                dtype = x.dtype
                fwd_mask = (edge_attrs[:, :, 8:9] > 0).to(dtype=dtype)  # [B, E, 1]
                edge_lines = edge_attrs[:, :, 0:7].to(dtype=dtype)      # [B, E, 7]
                fwd_lines = edge_lines * fwd_mask                       # [B, E, 7]
                src = edges[:, 0, :].long().clamp(min=0, max=29)               # [B, E]
                dst = edges[:, 1, :].long().clamp(min=0, max=29)               # [B, E]

                fwd_out = torch.zeros(B, 30, 7, device=x.device, dtype=dtype)
                fwd_in = torch.zeros(B, 30, 7, device=x.device, dtype=dtype)
                fwd_out.scatter_add_(1, src.unsqueeze(-1).expand(-1, -1, 7), fwd_lines)
                fwd_in.scatter_add_(1, dst.unsqueeze(-1).expand(-1, -1, 7), fwd_lines)

                is_front = ((fwd_out > 0.5) & (fwd_in < 0.5)).to(dtype=dtype)  # [B, 30, 7]
                is_back = ((fwd_in > 0.5) & (fwd_out < 0.5)).to(dtype=dtype)   # [B, 30, 7]

                is_end = torch.stack([is_front.permute(0, 2, 1), is_back.permute(0, 2, 1)], dim=2)
                has_endpoint = is_end.any(dim=-1, keepdim=True).to(dtype=dtype)  # [B, 7, 2, 1]

                pos = nodes[:, :, 0:2].to(dtype=dtype)  # [B, 30, 2]
                pos_end = torch.einsum("bler,brc->blec", is_end, pos)  # [B, 7, 2, 2]

                delta = pos.unsqueeze(1).unsqueeze(2) - pos_end.unsqueeze(3)  # [B, 7, 2, 30, 2]
                dist_cand = torch.sqrt(delta[..., 0]**2 + delta[..., 1]**2 + 1e-8)  # [B, 7, 2, 30]
                dist_cand = dist_cand * has_endpoint

                # Distance penalty (2.0 per 100 distance units)
                ext_geom_bias = 2.0 * dist_cand  # [B, 7, 2, 30]
                ext_geom_bias_flat = ext_geom_bias.permute(0, 1, 3, 2).reshape(B, 420)
                scores_extend = scores_extend - ext_geom_bias_flat

        # 4. Bilinear InsertStation scores (3150 actions)
        segs15 = torch.arange(15, device=x.device)
        l_ins = self.ins_line_emb(lines7)   # [7, H]
        s_ins = self.ins_seg_emb(segs15)    # [15, H]
        # [7, 15, H] -> [105, H]
        ls_base = (l_ins.unsqueeze(1) + s_ins.unsqueeze(0)).view(105, H)
        ins_ctx = self.ins_context(combined).unsqueeze(1)  # [B, 1, H]
        line_segs = F.relu(self.ins_proj(ls_base.unsqueeze(0) + ins_ctx))  # [B, 105, H]
        ins_mat = torch.bmm(line_segs, x.transpose(1, 2)) * scale          # [B, 105, 30]
        if redundant_lines is not None:
            ins_mat = ins_mat - 0.75 * redundant_lines.unsqueeze(1)
        # Permute (lineID, seg, stID) -> (lineID, stID, seg) then flatten to [B, 3150]
        scores_insert = ins_mat.view(B, 7, 15, 30).permute(0, 1, 3, 2).reshape(B, 3150)

        # 5. Non-spatial scores (52 actions)
        ns = self.non_spatial_head(combined)  # [B, 52]
        scores_noop          = ns[:, 0:1]     # 1

        # P1-3: Candidate-conditioned dispatch scoring for AddTrain and AddCarriage
        if edges is not None and edge_attrs is not None:
            h_line = self._extract_line_representations(nodes, edges, edge_attrs, x)  # [B, 7, H + 8]
            g_ctx = self.dispatch_line_ctx(combined).unsqueeze(1).expand(-1, 7, -1)   # [B, 7, H]
            dispatch_feat = torch.cat([h_line, g_ctx], dim=-1)                        # [B, 7, 2H + 8]
            scores_add_train = self.dispatch_train_mlp(dispatch_feat).squeeze(-1)     # [B, 7]
            scores_add_carriage = self.dispatch_carriage_mlp(dispatch_feat).squeeze(-1) # [B, 7]
        else:
            scores_add_train     = ns[:, 1:8]     # 7
            scores_add_carriage  = ns[:, 8:15]    # 7

        # P3-2: Symmetric candidate-conditioned ChooseReward scoring (Card 0 vs Card 1)
        if globals is not None and hasattr(self, "reward_card_mlp"):
            c0 = globals[:, 13:18].to(dtype=combined.dtype)  # [B, 5]
            c1 = globals[:, 18:23].to(dtype=combined.dtype)  # [B, 5]
            card_ctx = self.reward_card_ctx(combined)        # [B, H]
            feat0 = torch.cat([c0, card_ctx], dim=-1)        # [B, 5 + H]
            feat1 = torch.cat([c1, card_ctx], dim=-1)        # [B, 5 + H]
            score0 = self.reward_card_mlp(feat0)             # [B, 1]
            score1 = self.reward_card_mlp(feat1)             # [B, 1]
            scores_choose_reward = torch.cat([score0, score1], dim=-1) # [B, 2]
        else:
            scores_choose_reward = ns[:, 15:17]   # 2
        scores_close_loop    = ns[:, 17:24]   # 7
        scores_open_loop     = ns[:, 24:31]   # 7

        if edges is not None and edge_attrs is not None and hasattr(self, "remove_line_mlp"):
            scores_remove_line = self.remove_line_mlp(dispatch_feat).squeeze(-1)  # [B, 7]
            scores_shorten_line = self.shorten_line_mlp(dispatch_feat).reshape(B, 14)  # [B, 14]
        else:
            scores_remove_line   = ns[:, 31:38]   # 7
            scores_shorten_line  = ns[:, 38:52]   # 14

        # Pack parameter scores matching ACTION_TYPE_SLICES
        param_scores_list = [
            scores_noop,             # Type 0: NoOp (1)
            scores_add_line,         # Type 1: AddLine (435)
            scores_extend,           # Type 2: ExtendLine (420)
            scores_insert,           # Type 3: InsertStation (3150)
            scores_add_train,        # Type 4: AddTrain (7)
            scores_add_carriage,     # Type 5: AddCarriage (7)
            scores_interchange,      # Type 6: UpgradeInterchange (30)
            scores_choose_reward,    # Type 7: ChooseReward (2)
            scores_close_loop,       # Type 8: CloseLoop (7)
            scores_open_loop,        # Type 9: OpenLoop (7)
            scores_remove_line,      # Type 10: RemoveLine (7)
            scores_shorten_line,     # Type 11: ShortenLine (14)
        ]

        # 6. Type-level selection
        type_logits = self.type_net(combined)  # [B, 12]

        mask_val = -1e4 if combined.dtype in (torch.float16, torch.bfloat16) else -1e9

        if mask is not None:
            # Mask action types that have zero valid actions
            type_valid = torch.stack([
                mask[:, s].any(dim=-1) for s in ACTION_TYPE_SLICES
            ], dim=-1)  # [B, 12] bool

            # Fallback for all-zero masks: treat all types as valid
            all_invalid = (~type_valid).all(dim=-1, keepdim=True)
            type_valid = torch.where(all_invalid, torch.ones_like(type_valid), type_valid)

            type_log_probs = F.log_softmax(type_logits.masked_fill(~type_valid, mask_val), dim=-1)  # [B, 12]

            action_log_probs = torch.full((B, 4087), mask_val, device=combined.device, dtype=combined.dtype)
            for k, (s, p_scores) in enumerate(zip(ACTION_TYPE_SLICES, param_scores_list)):
                p_mask = mask[:, s]
                p_valid = p_mask.any(dim=-1, keepdim=True)
                safe_mask = torch.where(p_valid, p_mask, torch.ones_like(p_mask))
                p_log_probs = F.log_softmax(p_scores.masked_fill(~safe_mask, mask_val), dim=-1)

                joint = type_log_probs[:, k:k+1] + p_log_probs
                action_log_probs[:, s] = torch.where(p_mask, joint, mask_val)
        else:
            type_log_probs = F.log_softmax(type_logits, dim=-1)
            action_log_probs = torch.zeros(B, 4087, device=combined.device, dtype=combined.dtype)
            for k, (s, p_scores) in enumerate(zip(ACTION_TYPE_SLICES, param_scores_list)):
                p_log_probs = F.log_softmax(p_scores, dim=-1)
                action_log_probs[:, s] = type_log_probs[:, k:k+1] + p_log_probs

        self._last_type_logits = type_logits
        self._last_type_log_probs = type_log_probs
        self._last_param_scores = param_scores_list
        return action_log_probs

    def get_value(self, obs, lstm_state=None):
        logits, value, _ = self.forward(obs, lstm_state=lstm_state)
        return value

    def get_action_and_value(self, obs, lstm_state=None, action=None, mask=None, deterministic=False):
        if mask is None and isinstance(obs, dict) and "action_mask" in obs:
            mask = obs["action_mask"].bool()

        if mask is not None and getattr(self, "is_legacy_checkpoint", False):
            # Legacy checkpoints were trained strictly under pure additive expansion (Phases 1-5).
            # Mask out untrained dynamic demolition heads (RemoveLine, ShortenLine)
            # to prevent untrained random logits from triggering infinite deletion loops.
            mask = mask.clone()
            mask[:, ACTION_TYPE_SLICES[10]] = False
            mask[:, ACTION_TYPE_SLICES[11]] = False
            if not mask.any(dim=-1).all():
                mask[:, 0] = True

        logits, value, next_lstm_state = self.forward(obs, lstm_state=lstm_state, mask=mask)

        if mask is not None and not self.use_hierarchical:
            mask_val = -1e4 if logits.dtype in (torch.float16, torch.bfloat16) else -1e9
            logits = logits.masked_fill(~mask, mask_val)

        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            if deterministic:
                if self.use_hierarchical and hasattr(self, "_last_type_log_probs") and self._last_type_log_probs is not None:
                    # P0-1 FIX: Two-stage Hierarchical Argmax
                    # 1. Select the action type with maximum probability among valid types
                    type_lp = self._last_type_log_probs
                    is_seq = logits.dim() == 3
                    if is_seq:
                        B, T, _ = logits.shape
                        flat_type_lp = type_lp.view(B * T, 12)
                        flat_logits = logits.view(B * T, 4087)
                        flat_mask = mask.view(B * T, 4087) if mask is not None else None
                        best_type = torch.argmax(flat_type_lp, dim=-1)  # [B*T]
                        flat_actions = torch.zeros(B * T, dtype=torch.long, device=logits.device)
                        for b in range(B * T):
                            t = best_type[b].item()
                            sl = ACTION_TYPE_SLICES[t]
                            sl_logits = flat_logits[b, sl]
                            if flat_mask is not None:
                                sl_logits = sl_logits.masked_fill(~flat_mask[b, sl], -1e9)
                            flat_actions[b] = sl.start + torch.argmax(sl_logits, dim=-1)
                        action = flat_actions.view(B, T)
                    else:
                        B = logits.shape[0]
                        best_type = torch.argmax(type_lp, dim=-1)  # [B]
                        action = torch.zeros(B, dtype=torch.long, device=logits.device)
                        for b in range(B):
                            t = best_type[b].item()
                            sl = ACTION_TYPE_SLICES[t]
                            sl_logits = logits[b, sl]
                            if mask is not None:
                                sl_logits = sl_logits.masked_fill(~mask[b, sl], -1e9)
                            action[b] = sl.start + torch.argmax(sl_logits, dim=-1)
                else:
                    if mask is not None:
                        action = torch.argmax(logits.masked_fill(~mask, -1e9), dim=-1)
                    else:
                        action = torch.argmax(logits, dim=-1)
            else:
                action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value, next_lstm_state

    def forward(self, obs, lstm_state=None, mask=None):
        nodes        = obs["nodes"]
        edges        = obs["edges"]
        edge_attrs   = obs["edge_attrs"]
        globals_feat = obs["globals"]
        num_nodes    = obs["num_nodes"]
        num_edges    = obs["num_edges"]

        if mask is None and "action_mask" in obs:
            mask = obs["action_mask"].bool()

        is_sequence = nodes.dim() == 4
        if is_sequence:
            B, T, N, Feat = nodes.shape
            nodes = nodes.view(B * T, N, Feat)
            edges = edges.view(B * T, 2, -1)
            edge_attrs = edge_attrs.view(B * T, -1, edge_attrs.shape[-1])
            globals_feat = globals_feat.view(B * T, -1)
            num_nodes = num_nodes.view(B * T, 1)
            num_edges = num_edges.view(B * T, 1)
            if mask is not None:
                mask = mask.view(B * T, -1)
        else:
            B = nodes.shape[0]
            T = 1
            num_nodes = num_nodes.view(B, 1)
            num_edges = num_edges.view(B, 1)

        # GNN trunk: GATv2 with dynamic multi-head attention (P4-2) or legacy GCN
        if self.gnn_type == "gatv2":
            x, e, g   = self.gatv2_1(nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges)
            x2, e2, g2 = self.gatv2_2(x, edges, e, g, num_nodes, num_edges)
            x3, _, g3  = self.gatv2_3(x2, edges, e2, g2, num_nodes, num_edges)
            x3 = x3 + x2
        else:
            # Legacy GCN trunk with residual connection
            x, e, g   = self.gcn1(nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges)
            x2, e2, g2 = self.gcn2(x, edges, e, g, num_nodes, num_edges)
            x3, _, g3  = self.gcn3(x2, edges, e2, g2, num_nodes, num_edges)
            x3 = x3 + x2

        B_flat, N, H = x3.shape
        node_mask = torch.arange(N, device=x3.device).unsqueeze(0) < num_nodes  # [B_flat, N]

        # PHASE-2: Hierarchical Pooling (DiffPool)
        assign_logits = self.diffpool_assign(x3) # [B_flat, N, 4]
        assign_mask_val = -1e4 if assign_logits.dtype in (torch.float16, torch.bfloat16) else -1e9
        assign_logits = assign_logits.masked_fill(~node_mask.unsqueeze(-1), assign_mask_val)
        S = F.softmax(assign_logits, dim=1) # Softmax over nodes [B_flat, N, 4]
        
        # cluster features: S^T * X
        cluster_feats = torch.bmm(S.transpose(1, 2), x3) # [B_flat, 4, H]
        pooled = cluster_feats.view(B_flat, 4 * H)
        
        combined = torch.cat([pooled, g3], dim=-1)  # [B_flat, 5H]

        # LSTM pass
        combined_seq = combined.view(B, T, -1)
        if lstm_state is None:
            hx = torch.zeros(1, B, combined_seq.shape[-1], device=combined.device, dtype=combined.dtype)
            cx = torch.zeros(1, B, combined_seq.shape[-1], device=combined.device, dtype=combined.dtype)
            lstm_state = (hx, cx)

        lstm_out, next_lstm_state = self.lstm(combined_seq, lstm_state)
        lstm_out_flat = lstm_out.reshape(B_flat, -1)

        if self.use_hierarchical:
            # PHASE-5: Hierarchical action head + Bilinear parameter scoring
            # P1-1: Pass nodes for explicit candidate pairwise geometric awareness
            # P1-3: Pass edges and edge_attrs for candidate-conditioned line dispatch
            # P3-2: Pass globals for symmetric ChooseReward card evaluation
            logits = self._compute_hierarchical_logits(
                x3, lstm_out_flat, mask=mask, nodes=nodes, edges=edges, edge_attrs=edge_attrs, globals=globals_feat
            )
        else:
            logits = self.fc_actor(lstm_out_flat)
            self._last_type_log_probs = None

        value = self.fc_critic(lstm_out_flat)
        
        # If input was a sequence, return sequence-shaped outputs?
        # Typically PPO expects flat logits for categorical dist, so we keep it flat.
        # We will reshape in ppo.py if needed, or leave flat.
        if is_sequence:
            logits = logits.view(B, T, -1)
            value = value.view(B, T, -1)
            if self._last_type_log_probs is not None:
                self._last_type_log_probs = self._last_type_log_probs.view(B, T, -1)

        return logits, value, next_lstm_state

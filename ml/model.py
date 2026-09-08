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
        global_dim=13,
        action_space_size=4087,
        hidden_dim=128,
        use_hierarchical=True,
    ):
        # PHASE-2: global_dim 8→13 to match updated observation.go
        # PHASE-3: DenseGCNLayer→GNNLayer (dst_feat + edge update); 3rd layer + residual
        # PHASE-4: action_space_size 4108→4087 (AddCarriage now lineID-indexed, 28→7 slots)
        # PHASE-5: node_dim 29→32 (+incoming_train_count, incoming_train_load, nearest_train_proximity)
        # PHASE-5: hierarchical action head (Task 19) + bilinear action scoring (Task 20)
        super().__init__()

        self.use_hierarchical = use_hierarchical
        self.hidden_dim = hidden_dim

        # GNN passes
        self.gcn1 = GNNLayer(node_dim, edge_dim, hidden_dim)
        self.gcn2 = GNNLayer(hidden_dim, hidden_dim, hidden_dim)
        self.gcn3 = GNNLayer(hidden_dim, hidden_dim, hidden_dim)

        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # -------------------------------------------------------------------
        # PHASE-5 Task 19: Hierarchical 12-way Action Type Selector
        # Eliminates the 76.7% InsertStation dominance bias.
        # -------------------------------------------------------------------
        self.type_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
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

        # UpgradeInterchange (30): Linear projection directly on per-station embedding
        self.interchange_net = nn.Linear(hidden_dim, 1)

        # ExtendLine (420): Factored line-end embedding scored against station embeddings
        self.ext_line_emb = nn.Embedding(7, hidden_dim)
        self.ext_end_emb = nn.Embedding(2, hidden_dim)
        self.ext_context = nn.Linear(hidden_dim * 3, hidden_dim)
        self.ext_proj = nn.Linear(hidden_dim, hidden_dim)

        # InsertStation (3150): Factored line-segment embedding scored against station embeddings
        self.ins_line_emb = nn.Embedding(7, hidden_dim)
        self.ins_seg_emb = nn.Embedding(15, hidden_dim)
        self.ins_context = nn.Linear(hidden_dim * 3, hidden_dim)
        self.ins_proj = nn.Linear(hidden_dim, hidden_dim)

        # Non-spatial action heads (52 actions total):
        # NoOp(1), AddTrain(7), AddCarriage(7), ChooseReward(2),
        # CloseLoop(7), OpenLoop(7), RemoveLine(7), ShortenLine(14)
        self.non_spatial_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 52)
        )

        # Fallback flat actor head (for baseline ablation)
        self.fc_actor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size)
        )

        # Critic value head
        self.fc_critic = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def _compute_hierarchical_logits(self, x, combined, mask=None):
        """
        Compute hierarchical action log-probabilities:
            log P(a) = log P(type t(a)) + log P(a | type t(a))
        Directly returns log-probabilities over all 4087 actions.
        """
        B, N, H = x.shape
        scale = 1.0 / (H ** 0.5)

        # 1. Bilinear AddLine scores (435 actions)
        q = self.add_line_q(x)
        k = self.add_line_k(x)
        S = torch.bmm(q, k.transpose(1, 2)) * scale  # [B, 30, 30]
        S_sym = 0.5 * (S + S.transpose(1, 2))
        scores_add_line = S_sym[:, self.triu_u, self.triu_v]  # [B, 435]

        # 2. Per-station UpgradeInterchange scores (30 actions)
        scores_interchange = self.interchange_net(x).squeeze(-1)  # [B, 30]

        # 3. Bilinear ExtendLine scores (420 actions)
        lines7 = torch.arange(7, device=x.device)
        ends2 = torch.arange(2, device=x.device)
        l_emb = self.ext_line_emb(lines7)  # [7, H]
        e_emb = self.ext_end_emb(ends2)    # [2, H]
        # [7, 2, H] -> [14, H]
        le_base = (l_emb.unsqueeze(1) + e_emb.unsqueeze(0)).view(14, H)
        # Condition on global graph context
        ext_ctx = self.ext_context(combined).unsqueeze(1)  # [B, 1, H]
        line_ends = F.relu(self.ext_proj(le_base.unsqueeze(0) + ext_ctx))  # [B, 14, H]
        ext_mat = torch.bmm(line_ends, x.transpose(1, 2)) * scale          # [B, 14, 30]
        # Permute (lineID, end, stID) -> (lineID, stID, end) then flatten to [B, 420]
        scores_extend = ext_mat.view(B, 7, 2, 30).permute(0, 1, 3, 2).reshape(B, 420)

        # 4. Bilinear InsertStation scores (3150 actions)
        segs15 = torch.arange(15, device=x.device)
        l_ins = self.ins_line_emb(lines7)   # [7, H]
        s_ins = self.ins_seg_emb(segs15)    # [15, H]
        # [7, 15, H] -> [105, H]
        ls_base = (l_ins.unsqueeze(1) + s_ins.unsqueeze(0)).view(105, H)
        ins_ctx = self.ins_context(combined).unsqueeze(1)  # [B, 1, H]
        line_segs = F.relu(self.ins_proj(ls_base.unsqueeze(0) + ins_ctx))  # [B, 105, H]
        ins_mat = torch.bmm(line_segs, x.transpose(1, 2)) * scale          # [B, 105, 30]
        # Permute (lineID, seg, stID) -> (lineID, stID, seg) then flatten to [B, 3150]
        scores_insert = ins_mat.view(B, 7, 15, 30).permute(0, 1, 3, 2).reshape(B, 3150)

        # 5. Non-spatial scores (52 actions)
        ns = self.non_spatial_head(combined)  # [B, 52]
        scores_noop          = ns[:, 0:1]     # 1
        scores_add_train     = ns[:, 1:8]     # 7
        scores_add_carriage  = ns[:, 8:15]    # 7
        scores_choose_reward = ns[:, 15:17]   # 2
        scores_close_loop    = ns[:, 17:24]   # 7
        scores_open_loop     = ns[:, 24:31]   # 7
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

        if mask is not None:
            # Mask action types that have zero valid actions
            type_valid = torch.stack([
                mask[:, s].any(dim=-1) for s in ACTION_TYPE_SLICES
            ], dim=-1)  # [B, 12] bool

            # Fallback for all-zero masks: treat all types as valid
            all_invalid = (~type_valid).all(dim=-1, keepdim=True)
            type_valid = torch.where(all_invalid, torch.ones_like(type_valid), type_valid)

            type_log_probs = F.log_softmax(type_logits.masked_fill(~type_valid, -1e9), dim=-1)  # [B, 12]

            action_log_probs = torch.full((B, 4087), -1e9, device=combined.device, dtype=combined.dtype)
            for k, (s, p_scores) in enumerate(zip(ACTION_TYPE_SLICES, param_scores_list)):
                p_mask = mask[:, s]
                p_valid = p_mask.any(dim=-1, keepdim=True)
                safe_mask = torch.where(p_valid, p_mask, torch.ones_like(p_mask))
                p_log_probs = F.log_softmax(p_scores.masked_fill(~safe_mask, -1e9), dim=-1)

                joint = type_log_probs[:, k:k+1] + p_log_probs
                action_log_probs[:, s] = torch.where(p_mask, joint, -1e9)

            # Compute true hierarchical argmax for greedy deterministic evaluation
            safe_type_logits = type_logits.masked_fill(~type_valid, -1e9)
            best_types = torch.argmax(safe_type_logits, dim=-1)  # [B]
            hier_actions = torch.zeros(B, dtype=torch.long, device=combined.device)
            for b in range(B):
                t_idx = best_types[b].item()
                s = ACTION_TYPE_SLICES[t_idx]
                p_scores = param_scores_list[t_idx][b:b+1]
                p_mask = mask[b:b+1, s]
                p_valid = p_mask.any(dim=-1, keepdim=True)
                safe_p_mask = torch.where(p_valid, p_mask, torch.ones_like(p_mask))
                safe_scores = p_scores.masked_fill(~safe_p_mask, -1e9)
                best_param = torch.argmax(safe_scores, dim=-1).item()
                hier_actions[b] = s.start + best_param
            self._last_hierarchical_actions = hier_actions
        else:
            type_log_probs = F.log_softmax(type_logits, dim=-1)
            action_log_probs = torch.zeros(B, 4087, device=combined.device, dtype=combined.dtype)
            for k, (s, p_scores) in enumerate(zip(ACTION_TYPE_SLICES, param_scores_list)):
                p_log_probs = F.log_softmax(p_scores, dim=-1)
                action_log_probs[:, s] = type_log_probs[:, k:k+1] + p_log_probs

            best_types = torch.argmax(type_logits, dim=-1)
            hier_actions = torch.zeros(B, dtype=torch.long, device=combined.device)
            for b in range(B):
                t_idx = best_types[b].item()
                s = ACTION_TYPE_SLICES[t_idx]
                best_param = torch.argmax(param_scores_list[t_idx][b:b+1], dim=-1).item()
                hier_actions[b] = s.start + best_param
            self._last_hierarchical_actions = hier_actions

        return action_log_probs

    def get_value(self, obs):
        logits, value = self.forward(obs)
        return value

    def get_action_and_value(self, obs, action=None, mask=None, deterministic=False):
        if mask is None and isinstance(obs, dict) and "action_mask" in obs:
            mask = obs["action_mask"].bool()

        logits, value = self.forward(obs, mask=mask)

        if mask is not None and not self.use_hierarchical:
            logits = logits.masked_fill(~mask, -1e9)

        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            if deterministic:
                # Hierarchical argmax: select highest-scoring valid type, then highest-scoring valid parameter
                if self.use_hierarchical and hasattr(self, "_last_hierarchical_actions"):
                    action = self._last_hierarchical_actions
                elif mask is not None:
                    action = torch.argmax(logits.masked_fill(~mask, -1e9), dim=-1)
                else:
                    action = torch.argmax(logits, dim=-1)
            else:
                action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value

    def forward(self, obs, mask=None):
        nodes        = obs["nodes"]
        edges        = obs["edges"]
        edge_attrs   = obs["edge_attrs"]
        globals_feat = obs["globals"]
        num_nodes    = obs["num_nodes"]
        num_edges    = obs["num_edges"]

        if mask is None and "action_mask" in obs:
            mask = obs["action_mask"].bool()

        # PHASE-3: thread updated edge embeddings between layers
        x, e   = self.gcn1(nodes, edges, edge_attrs, num_nodes, num_edges)
        x2, e2 = self.gcn2(x, edges, e, num_nodes, num_edges)

        # PHASE-3 GNN-3: 3rd layer with residual connection
        x3, _  = self.gcn3(x2, edges, e2, num_nodes, num_edges)
        x3 = x3 + x2

        B, N, H = x3.shape
        node_mask = torch.arange(N, device=x3.device).unsqueeze(0) < num_nodes  # [B, N]

        # PHASE-2: mean+max pooling
        mean_pool = (x3 * node_mask.unsqueeze(-1)).sum(dim=1) / num_nodes.clamp(min=1).float()
        x_for_max = x3.masked_fill(~node_mask.unsqueeze(-1), -1e9)
        max_pool  = x_for_max.max(dim=1).values
        pooled    = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2H]

        g        = self.global_proj(globals_feat)
        combined = torch.cat([pooled, g], dim=-1)  # [B, 3H]

        if self.use_hierarchical:
            # PHASE-5: Hierarchical action head + Bilinear parameter scoring
            logits = self._compute_hierarchical_logits(x3, combined, mask=mask)
        else:
            logits = self.fc_actor(combined)

        value = self.fc_critic(combined)

        return logits, value

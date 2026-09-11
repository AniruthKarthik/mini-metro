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
        self.gcn1 = GNNLayer(node_dim, edge_dim, global_dim, hidden_dim)
        self.gcn2 = GNNLayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        self.gcn3 = GNNLayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim)

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

        self._last_type_log_probs = type_log_probs
        return action_log_probs

    def get_value(self, obs, lstm_state=None):
        logits, value, _ = self.forward(obs, lstm_state=lstm_state)
        return value

    def get_action_and_value(self, obs, lstm_state=None, action=None, mask=None, deterministic=False):
        if mask is None and isinstance(obs, dict) and "action_mask" in obs:
            mask = obs["action_mask"].bool()

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
            num_nodes = num_nodes.view(B * T, -1)
            num_edges = num_edges.view(B * T, -1)
            if mask is not None:
                mask = mask.view(B * T, -1)
        else:
            B = nodes.shape[0]
            T = 1

        # PHASE-3: thread updated edge embeddings between layers
        x, e, g   = self.gcn1(nodes, edges, edge_attrs, globals_feat, num_nodes, num_edges)
        x2, e2, g2 = self.gcn2(x, edges, e, g, num_nodes, num_edges)

        # PHASE-3 GNN-3: 3rd layer with residual connection
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
            logits = self._compute_hierarchical_logits(x3, lstm_out_flat, mask=mask)
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

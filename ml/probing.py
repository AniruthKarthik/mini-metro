"""
Probing and diagnostic distribution utilities for Mini Metro RL.

Fixes P0-3: Evaluation & Probing Distribution Normalization.
Formally distinguishes and computes:
1. raw_type_logits: Unnormalized TypeNet outputs.
2. unmasked_type_probs: Softmax over all 12 types (sum = 1.0).
3. masked_type_probs: Softmax normalized strictly over valid action types (L1 = 1.0).
4. conditional_param_probs: Normalized strictly within each valid action type slice (L1 = 1.0).
5. joint_action_probs: Hierarchical joint action probabilities across all 4,087 actions (sum = 1.0).
6. type_action_mass: Marginal sum of joint action probabilities per type (matches masked_type_probs).

Enforces automated assertion guards:
assert np.isclose(probs.sum(), 1.0, atol=1e-5)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from model import ACTION_TYPE_SLICES

ACTION_NAMES = [
    "NoOp",
    "AddLine",
    "ExtendLine",
    "InsertStation",
    "AddTrain",
    "AddCarriage",
    "UpgradeInterchange",
    "ChooseReward",
    "CloseLoop",
    "OpenLoop",
    "RemoveLine",
    "ShortenLine",
]


@dataclass
class ActionDistributionDiagnostics:
    """Structured container for all hierarchy levels of action distributions."""
    raw_type_logits: np.ndarray             # [12]
    unmasked_type_probs: np.ndarray         # [12], sum == 1.0
    valid_type_mask: np.ndarray             # [12] bool
    masked_type_probs: np.ndarray           # [12], sum == 1.0 over valid types
    param_raw_scores: Dict[int, np.ndarray] # type_idx -> [slice_len]
    param_valid_masks: Dict[int, np.ndarray]# type_idx -> [slice_len] bool
    conditional_param_probs: Dict[int, np.ndarray] # type_idx -> [slice_len], sum == 1.0 if valid
    joint_action_log_probs: np.ndarray      # [4087]
    joint_action_probs: np.ndarray          # [4087], sum == 1.0
    type_action_mass: np.ndarray            # [12], sum_{a in slice_t} P(a), equals masked_type_probs

    def __post_init__(self):
        # Strict automated normalization invariants
        assert np.isclose(self.unmasked_type_probs.sum(), 1.0, atol=1e-5), \
            f"unmasked_type_probs must sum to 1.0, got {self.unmasked_type_probs.sum()}"

        if self.valid_type_mask.any():
            assert np.isclose(self.masked_type_probs.sum(), 1.0, atol=1e-5), \
                f"masked_type_probs must sum to 1.0, got {self.masked_type_probs.sum()}"
            assert np.isclose(self.joint_action_probs.sum(), 1.0, atol=1e-5), \
                f"joint_action_probs must sum to 1.0, got {self.joint_action_probs.sum()}"

            for t, is_valid in enumerate(self.valid_type_mask):
                if is_valid:
                    p = self.conditional_param_probs[t]
                    assert np.isclose(p.sum(), 1.0, atol=1e-5), \
                        f"conditional_param_probs for type {ACTION_NAMES[t]} must sum to 1.0, got {p.sum()}"

            # Invariance: marginal sum of joint actions in type slice must match type probability
            assert np.allclose(self.type_action_mass, self.masked_type_probs, atol=1e-5), \
                f"type_action_mass does not match masked_type_probs: diff={np.abs(self.type_action_mass - self.masked_type_probs).max()}"


def compute_action_diagnostics(
    model,
    obs: Dict[str, torch.Tensor],
    lstm_state=None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[ActionDistributionDiagnostics, any]:
    """
    Extracts, normalizes, and validates all hierarchical action distributions for an observation.
    Returns (ActionDistributionDiagnostics, next_lstm_state).
    """
    model.eval()

    # Prepare inputs
    obs_tensor = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
        elif isinstance(v, torch.Tensor):
            t = v
        else:
            t = torch.tensor(v)
        if t.dim() == 0 or (k in ("nodes", "edge_attrs", "edges", "globals") and t.dim() == len(v.shape)):
            if t.dim() == 1 and k == "globals":
                t = t.unsqueeze(0)
            elif t.dim() == 2 and k == "nodes":
                t = t.unsqueeze(0)
            elif t.dim() == 2 and k == "edges":
                t = t.unsqueeze(0)
            elif t.dim() == 2 and k == "edge_attrs":
                t = t.unsqueeze(0)
            elif t.dim() == 1 and k in ("num_nodes", "num_edges"):
                t = t.unsqueeze(0)
        obs_tensor[k] = t

    if mask is None:
        if "action_mask" in obs_tensor:
            mask = obs_tensor["action_mask"].bool()
        else:
            mask = torch.ones((1, 4087), dtype=torch.bool, device=obs_tensor["globals"].device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    device = obs_tensor["globals"].device
    obs_tensor["action_mask"] = mask

    with torch.no_grad():
        action_log_probs, value, next_lstm_state = model(obs_tensor, lstm_state=lstm_state, mask=mask)

        # Retrieve saved type logits and param scores
        if hasattr(model, "_last_type_logits") and model._last_type_logits is not None:
            raw_type_logits_t = model._last_type_logits[0].cpu()
        else:
            raise RuntimeError("Model does not have _last_type_logits populated.")

        param_scores_list = [p[0].cpu() for p in model._last_param_scores]

    raw_type_logits = raw_type_logits_t.numpy()
    unmasked_type_probs = F.softmax(raw_type_logits_t, dim=-1).numpy()

    mask_cpu = mask[0].cpu().bool()
    valid_type_mask = np.array([mask_cpu[s].any().item() for s in ACTION_TYPE_SLICES], dtype=bool)

    if not valid_type_mask.any():
        valid_type_mask[:] = True

    # Masked type probabilities
    mask_val = -1e9
    masked_logits_t = raw_type_logits_t.clone()
    masked_logits_t[~torch.from_numpy(valid_type_mask)] = mask_val
    masked_type_probs = F.softmax(masked_logits_t, dim=-1).numpy()

    # Zero out explicitly for invalid types to prevent -1e9 precision residue
    masked_type_probs[~valid_type_mask] = 0.0
    if masked_type_probs.sum() > 0:
        masked_type_probs /= masked_type_probs.sum()

    # Parameter conditional probabilities
    param_raw_scores = {}
    param_valid_masks = {}
    conditional_param_probs = {}

    for t, (sl, p_scores) in enumerate(zip(ACTION_TYPE_SLICES, param_scores_list)):
        p_mask = mask_cpu[sl].numpy()
        param_raw_scores[t] = p_scores.numpy()
        param_valid_masks[t] = p_mask

        if p_mask.any():
            p_scores_masked = p_scores.clone()
            p_scores_masked[~torch.from_numpy(p_mask)] = mask_val
            p_probs = F.softmax(p_scores_masked, dim=-1).numpy()
            p_probs[~p_mask] = 0.0
            if p_probs.sum() > 0:
                p_probs /= p_probs.sum()
            conditional_param_probs[t] = p_probs
        else:
            conditional_param_probs[t] = np.zeros(sl.stop - sl.start, dtype=np.float32)

    # Joint action probabilities
    joint_action_log_probs_arr = action_log_probs[0].cpu().numpy()
    joint_action_probs = np.zeros(4087, dtype=np.float32)

    for t, sl in enumerate(ACTION_TYPE_SLICES):
        if valid_type_mask[t]:
            joint_action_probs[sl] = masked_type_probs[t] * conditional_param_probs[t]

    if joint_action_probs.sum() > 0:
        joint_action_probs /= joint_action_probs.sum()

    # Type action mass: sum of joint action probs across type slices
    type_action_mass = np.array([joint_action_probs[sl].sum() for sl in ACTION_TYPE_SLICES], dtype=np.float32)

    diag = ActionDistributionDiagnostics(
        raw_type_logits=raw_type_logits,
        unmasked_type_probs=unmasked_type_probs,
        valid_type_mask=valid_type_mask,
        masked_type_probs=masked_type_probs,
        param_raw_scores=param_raw_scores,
        param_valid_masks=param_valid_masks,
        conditional_param_probs=conditional_param_probs,
        joint_action_log_probs=joint_action_log_probs_arr,
        joint_action_probs=joint_action_probs,
        type_action_mass=type_action_mass,
    )

    return diag, next_lstm_state


def format_probability_table(
    rows: List[Tuple[str, np.ndarray]],
    column_names: List[str],
    title: str = "Action Type Distribution (%)",
    row_label_header: str = "Scenario",
    normalize_subset: bool = False,
) -> str:
    """
    Formats a probability distribution table with an enforced Sum column of 100.0%.
    - If len(column_names) == 12, displays all 12 types.
    - If a subset of types is passed:
        - normalize_subset=True: renormalizes the subset to 100.0% with an explicit notice.
        - normalize_subset=False: includes an 'Other' column for omitted types so total is 100.0%.
    """
    lines = []
    lines.append(f"\n### {title}")
    if normalize_subset:
        lines.append("*Note: Table shows normalized sub-distribution conditioned on the displayed subset.*")

    headers = [row_label_header] + column_names
    has_other = False

    # Check if this is an unnormalized partial subset
    if len(column_names) < 12 and not normalize_subset:
        headers.append("Other")
        has_other = True
    headers.append("Sum")

    col_widths = [max(len(h), 8) for h in headers]
    for label, probs in rows:
        col_widths[0] = max(col_widths[0], len(label))

    header_str = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_str = "-|-".join("-" * col_widths[i] for i in range(len(headers)))
    lines.append(header_str)
    lines.append(sep_str)

    for label, probs in rows:
        probs = np.asarray(probs, dtype=np.float64)

        if normalize_subset:
            subset_sum = probs.sum()
            assert subset_sum > 0, f"Cannot normalize zero sum for row '{label}'"
            norm_probs = probs / subset_sum * 100.0
            row_sum = norm_probs.sum()
            assert np.isclose(row_sum, 100.0, atol=0.01), f"Row '{label}' sum must be 100%, got {row_sum:.2f}%"
            val_strs = [f"{v:6.2f}%" for v in norm_probs]
            row_vals = [label.ljust(col_widths[0])] + [v.rjust(col_widths[i+1]) for i, v in enumerate(val_strs)]
            row_vals.append(f"{row_sum:6.1f}%".rjust(col_widths[-1]))
        else:
            disp_sum = probs.sum() * 100.0
            val_strs = [f"{v * 100.0:6.2f}%" for v in probs]
            row_vals = [label.ljust(col_widths[0])] + [v.rjust(col_widths[i+1]) for i, v in enumerate(val_strs)]
            if has_other:
                other_val = max(0.0, 100.0 - disp_sum)
                row_vals.append(f"{other_val:6.2f}%".rjust(col_widths[-2]))
                total_sum = disp_sum + other_val
            else:
                total_sum = disp_sum

            assert np.isclose(total_sum, 100.0, atol=0.01), f"Row '{label}' sum must be 100%, got {total_sum:.2f}%"
            row_vals.append(f"{total_sum:6.1f}%".rjust(col_widths[-1]))

        lines.append(" | ".join(row_vals))

    return "\n".join(lines)


@dataclass
class ExpansionMetrics:
    """Diagnostic metrics tracking network expansion and station redundancy."""
    expansion_ratio: float
    expansion_action_rate: float
    lines_per_station_mean: float
    redundant_stations_count: float
    redundant_station_rate: float
    total_legal_connections: float
    selected_connections: float

    def __getitem__(self, item: str) -> float:
        return getattr(self, item)

    def to_dict(self) -> Dict[str, float]:
        return {
            "expansion_ratio": self.expansion_ratio,
            "expansion_action_rate": self.expansion_action_rate,
            "lines_per_station_mean": self.lines_per_station_mean,
            "redundant_stations_count": self.redundant_stations_count,
            "redundant_station_rate": self.redundant_station_rate,
            "total_legal_connections": self.total_legal_connections,
            "selected_connections": self.selected_connections,
        }


def compute_expansion_metrics(
    obs: Dict[str, Union[np.ndarray, torch.Tensor]],
    actions: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> ExpansionMetrics:
    """
    Computes network expansion and redundancy metrics:
    1. expansion_ratio: selected line connections / total legal connections
    2. expansion_action_rate: fraction of actions that expand lines (AddLine, ExtendLine, InsertStation)
    3. lines_per_station_mean: average lines serving per active station
    4. redundant_stations_count: average number of non-interchange stations served by >2 lines
    5. redundant_station_rate: fraction of active stations that are redundant (>2 lines, not interchange)
    """
    nodes = obs.get("nodes")
    if nodes is None:
        raise ValueError("obs must contain 'nodes' tensor/array")

    if isinstance(nodes, torch.Tensor):
        nodes_np = nodes.detach().cpu().numpy()
    else:
        nodes_np = np.asarray(nodes)

    orig_shape = nodes_np.shape
    if len(orig_shape) == 2:
        nodes_np = nodes_np[None, ...]
    elif len(orig_shape) > 3:
        nodes_np = nodes_np.reshape(-1, orig_shape[-2], orig_shape[-1])

    B = nodes_np.shape[0]

    mask_np = None
    if "action_mask" in obs:
        mask_raw = obs["action_mask"]
        if isinstance(mask_raw, torch.Tensor):
            mask_np = mask_raw.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask_raw)
        if len(mask_np.shape) == 1:
            mask_np = mask_np[None, ...]
        elif len(mask_np.shape) > 2:
            mask_np = mask_np.reshape(-1, mask_np.shape[-1])

    act_np = None
    if actions is not None:
        if isinstance(actions, torch.Tensor):
            act_np = actions.detach().cpu().numpy()
        else:
            act_np = np.asarray(actions)
        act_np = act_np.reshape(-1)

    num_nodes_np = None
    if "num_nodes" in obs:
        nn_raw = obs["num_nodes"]
        if isinstance(nn_raw, torch.Tensor):
            num_nodes_np = nn_raw.detach().cpu().numpy()
        else:
            num_nodes_np = np.asarray(nn_raw)
        num_nodes_np = num_nodes_np.reshape(-1)

    total_stations = 0
    total_lines_serving = 0.0
    total_redundant_stations = 0.0
    total_legal_connections = 0.0
    total_selected_connections = 0.0
    total_expansion_actions = 0.0

    for b in range(B):
        if num_nodes_np is not None and b < len(num_nodes_np):
            n_st = min(30, max(1, int(num_nodes_np[b])))
        else:
            # Active if kind one-hot is nonzero or coords != 0
            has_kind = (nodes_np[b, :, 2:12].sum(axis=-1) > 0)
            n_st = int(has_kind.sum())
            if n_st == 0:
                n_st = 30

        st_nodes = nodes_np[b, :n_st]
        lines_serving = np.round(st_nodes[:, 26] * 7.0)
        is_interchange = st_nodes[:, 24] > 0.5
        is_redundant = (lines_serving > 2.0) & (~is_interchange)

        total_stations += n_st
        total_lines_serving += float(lines_serving.sum())
        total_redundant_stations += float(is_redundant.sum())

        if mask_np is not None:
            # Slices: AddLine (1:436), ExtendLine (436:856), InsertStation (856:4006)
            legal_conn = float(mask_np[b, 1:4006].sum())
            total_legal_connections += legal_conn

        if act_np is not None and b < len(act_np):
            act = act_np[b]
            if 1 <= act < 4006:
                total_expansion_actions += 1.0
                total_selected_connections += 1.0

    avg_lines_per_st = total_lines_serving / max(total_stations, 1)
    redundant_count_avg = total_redundant_stations / max(B, 1)
    redundant_st_rate = total_redundant_stations / max(total_stations, 1)

    if act_np is not None:
        expansion_action_rate = total_expansion_actions / max(len(act_np), 1)
        expansion_ratio = total_selected_connections / max(total_legal_connections, 1.0)
    else:
        expansion_action_rate = 0.0
        # If no actions provided, ratio is existing connections / legal connection candidates
        expansion_ratio = total_lines_serving / max(total_legal_connections, 1.0)
        total_selected_connections = total_lines_serving

    return ExpansionMetrics(
        expansion_ratio=float(expansion_ratio),
        expansion_action_rate=float(expansion_action_rate),
        lines_per_station_mean=float(avg_lines_per_st),
        redundant_stations_count=float(redundant_count_avg),
        redundant_station_rate=float(redundant_st_rate),
        total_legal_connections=float(total_legal_connections),
        selected_connections=float(total_selected_connections),
    )


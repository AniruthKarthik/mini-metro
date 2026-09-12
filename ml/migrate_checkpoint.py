#!/usr/bin/env python3
"""
Checkpoint Migration Utility for Mini Metro RL Models.
Upgrades legacy checkpoints (Phases 1-5) to current architecture metadata and weight shapes.
"""

import argparse
import os
import sys
import time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import MiniMetroActorCritic


def migrate_checkpoint(input_path: str, output_path: str = None) -> str:
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_migrated{ext}"

    print(f"Loading checkpoint from: {input_path}")
    raw_data = torch.load(input_path, map_location="cpu", weights_only=False)
    state_dict = raw_data["model_state_dict"] if isinstance(raw_data, dict) and "model_state_dict" in raw_data else raw_data

    # Introspect hidden_dim from node projection weight
    node_w = state_dict.get("gcn1.node_proj.weight", state_dict.get("gatv2_1.node_proj.weight", None))
    hidden_dim = int(node_w.shape[0]) if node_w is not None else 256
    print(f"Detected hidden_dim: {hidden_dim}")

    # Instantiate target model to get canonical shapes and module keys
    model = MiniMetroActorCritic(hidden_dim=hidden_dim)
    model.load_state_dict(state_dict, strict=False)

    migrated_state_dict = model.state_dict()

    migrated_ckpt = {
        "version": "2.0",
        "migrated_at": time.time(),
        "source_checkpoint": os.path.abspath(input_path),
        "hidden_dim": hidden_dim,
        "is_legacy_source": getattr(model, "is_legacy_checkpoint", True),
        "model_state_dict": migrated_state_dict,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(migrated_ckpt, output_path)
    print(f"✓ Saved migrated checkpoint to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Migrate Mini Metro RL model checkpoints.")
    parser.add_argument("--input", "-i", type=str, default="ml/runs/minimetro_ppo/model_final.pt", help="Path to input checkpoint")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to output migrated checkpoint")
    args = parser.parse_args()

    migrate_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()

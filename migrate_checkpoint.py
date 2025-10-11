#!/usr/bin/env python3
"""
Migrate checkpoint from old architecture to new architecture.

Changes:
1. MLP layers: fc1/fc2 -> 0/2
2. Head: multi-layer (0,2,4) -> single layer (weight, bias)
3. Add elevation_alpha and H_scale for physics mask
"""

import torch
import argparse
from pathlib import Path


def migrate_checkpoint(old_ckpt_path, new_ckpt_path):
    """Migrate checkpoint to new architecture."""

    print(f"Loading old checkpoint from: {old_ckpt_path}")
    checkpoint = torch.load(old_ckpt_path, map_location='cpu')

    state_dict = checkpoint['state_dict']
    new_state_dict = {}

    print("\n=== Migrating state_dict ===")

    for key, value in state_dict.items():
        new_key = key

        # 1. Migrate MLP layers: fc1 -> 0, fc2 -> 2
        if '.mlp.fc1.' in key:
            new_key = key.replace('.mlp.fc1.', '.mlp.0.')
            print(f"  MLP: {key} -> {new_key}")
        elif '.mlp.fc2.' in key:
            new_key = key.replace('.mlp.fc2.', '.mlp.2.')
            print(f"  MLP: {key} -> {new_key}")

        # 2. Migrate head: take only the first layer (head.0) -> head
        elif '.head.0.' in key:
            new_key = key.replace('.head.0.', '.head.')
            print(f"  HEAD: {key} -> {new_key}")
        elif '.head.2.' in key or '.head.4.' in key:
            # Skip intermediate head layers (we only keep the first one)
            print(f"  HEAD: Skipping {key} (old multi-layer head)")
            continue

        new_state_dict[new_key] = value

    # 3. Add new physics mask parameters (elevation_alpha, H_scale)
    # These will be initialized randomly - they're new parameters
    print("\n=== Adding new physics mask parameters ===")

    # Count blocks to add elevation_alpha and H_scale for each
    num_blocks = 0
    for key in new_state_dict.keys():
        if 'model.climax.blocks.' in key:
            block_idx = int(key.split('blocks.')[1].split('.')[0])
            num_blocks = max(num_blocks, block_idx + 1)

    print(f"Found {num_blocks} transformer blocks")

    for block_idx in range(num_blocks):
        # Add elevation_alpha (learnable parameter for elevation mask)
        alpha_key = f"model.climax.blocks.{block_idx}.attn.elevation_alpha"
        new_state_dict[alpha_key] = torch.tensor(1.0)  # Initialize to 1.0
        print(f"  Added: {alpha_key} = 1.0")

        # Add H_scale (learnable scale for height)
        h_scale_key = f"model.climax.blocks.{block_idx}.attn.H_scale"
        new_state_dict[h_scale_key] = torch.tensor(1.0)  # Initialize to 1.0
        print(f"  Added: {h_scale_key} = 1.0")

    # Update checkpoint with new state_dict
    checkpoint['state_dict'] = new_state_dict

    # Save migrated checkpoint
    print(f"\nSaving migrated checkpoint to: {new_ckpt_path}")
    torch.save(checkpoint, new_ckpt_path)

    print("\n=== Migration Summary ===")
    print(f"Original keys: {len(state_dict)}")
    print(f"Migrated keys: {len(new_state_dict)}")
    print(f"Added keys: {len(new_state_dict) - len([k for k in state_dict.keys() if '.head.0.' in k or '.mlp.fc' in k])}")
    print(f"Removed keys: {len([k for k in state_dict.keys() if '.head.2.' in k or '.head.4.' in k])}")

    return new_ckpt_path


def verify_checkpoint(ckpt_path):
    """Verify checkpoint can be loaded."""
    print(f"\n=== Verifying checkpoint ===")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    # Check for expected keys
    has_elevation_alpha = any('elevation_alpha' in k for k in state_dict.keys())
    has_h_scale = any('H_scale' in k for k in state_dict.keys())
    has_mlp_0 = any('.mlp.0.' in k for k in state_dict.keys())
    has_mlp_fc1 = any('.mlp.fc1.' in k for k in state_dict.keys())
    has_head_weight = any('.head.weight' in k for k in state_dict.keys())
    has_head_0 = any('.head.0.' in k for k in state_dict.keys())

    print(f"  ✓ Has elevation_alpha: {has_elevation_alpha}")
    print(f"  ✓ Has H_scale: {has_h_scale}")
    print(f"  ✓ Has mlp.0 (new): {has_mlp_0}")
    print(f"  ✗ Has mlp.fc1 (old): {has_mlp_fc1} (should be False)")
    print(f"  ✓ Has head.weight (new): {has_head_weight}")
    print(f"  ✗ Has head.0 (old): {has_head_0} (should be False)")

    if has_elevation_alpha and has_h_scale and has_mlp_0 and has_head_weight:
        print("\n✅ Checkpoint migration successful!")
        return True
    else:
        print("\n❌ Checkpoint migration incomplete!")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate checkpoint to new architecture")
    parser.add_argument("--input", type=str, required=True, help="Path to old checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to save migrated checkpoint")
    parser.add_argument("--verify", action="store_true", help="Verify migrated checkpoint")

    args = parser.parse_args()

    # Migrate
    new_ckpt_path = migrate_checkpoint(args.input, args.output)

    # Verify
    if args.verify:
        verify_checkpoint(new_ckpt_path)

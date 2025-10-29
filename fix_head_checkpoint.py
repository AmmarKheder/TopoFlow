#!/usr/bin/env python3
"""
Fix checkpoint HEAD keys to match new architecture.

OLD (checkpoint): head.0, head.2, head.4
NEW (model): head_fc1, head_fc2, head_fc3
"""

import torch
import sys

def fix_checkpoint_head_keys(checkpoint_path, output_path):
    """Rename HEAD keys in checkpoint to match new architecture."""

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Rename keys
    state_dict = checkpoint['state_dict']
    new_state_dict = {}

    # Mapping: old key -> new key
    key_mapping = {
        'model.climax.head.0.weight': 'model.climax.head_fc1.weight',
        'model.climax.head.0.bias': 'model.climax.head_fc1.bias',
        'model.climax.head.2.weight': 'model.climax.head_fc2.weight',
        'model.climax.head.2.bias': 'model.climax.head_fc2.bias',
        'model.climax.head.4.weight': 'model.climax.head_fc3.weight',
        'model.climax.head.4.bias': 'model.climax.head_fc3.bias',
    }

    renamed_count = 0
    for old_key, value in state_dict.items():
        if old_key in key_mapping:
            new_key = key_mapping[old_key]
            new_state_dict[new_key] = value
            print(f"  ✅ Renamed: {old_key} -> {new_key}")
            renamed_count += 1
        else:
            new_state_dict[old_key] = value

    # Update checkpoint
    checkpoint['state_dict'] = new_state_dict

    print(f"\n✅ Renamed {renamed_count} keys")
    print(f"Saving fixed checkpoint: {output_path}")

    torch.save(checkpoint, output_path)
    print("✅ Done!")

    return renamed_count

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python fix_head_checkpoint.py <input_checkpoint> <output_checkpoint>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    fix_checkpoint_head_keys(input_path, output_path)

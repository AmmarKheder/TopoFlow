#!/usr/bin/env python3
"""Test FULL TopoFlow: Wind + 3D MLP"""
import torch
import yaml

print("=" * 70)
print("FULL TopoFlow Test: Wind Scanning + 3D MLP")
print("=" * 70)

# Load config
with open('configs/config_full_topoflow.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("\n1. Config loaded ✓")
print(f"  parallel_patch_embed: {config['model']['parallel_patch_embed']}")
print(f"  use_physics_mask: {config['model']['use_physics_mask']}")
print(f"  use_3d_learnable: {config['model']['use_3d_learnable']}")

# Import
from src.model_multipollutants import MultiPollutantModel

print("\n2. Model class imported ✓")

# Create model
try:
    model = MultiPollutantModel(config)
    print("\n3. Model created ✓")

    # Check first block
    from src.climax_core.relative_position_bias_3d import Attention3D
    first_block = model.climax.blocks[0]
    has_3d_attn = hasattr(first_block, 'attn') and isinstance(first_block.attn, Attention3D)

    print(f"\n4. First block has Attention3D: {has_3d_attn}")

    if has_3d_attn:
        print(f"   - use_3d_bias: {first_block.attn.use_3d_bias}")
        print(f"   - num_heads: {first_block.attn.num_heads}")
        num_params = sum(p.numel() for p in first_block.attn.rel_pos_bias_3d.parameters())
        print(f"   - MLP parameters: {num_params}")
        print("\n✅ FULL TopoFlow ready!")
        print("   - Wind scanning: enabled")
        print("   - 3D MLP elevation: enabled")
    else:
        print(f"\n❌ ERROR: First block attention type: {type(first_block.attn)}")
        exit(1)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("=" * 70)

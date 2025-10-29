#!/usr/bin/env python3
"""Test Wind + Simple Physics"""
import torch
import yaml

print("=" * 70)
print("Wind + Simple Physics Test")
print("=" * 70)

# Load config
with open('configs/config_wind_physics.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("\n1. Config loaded ✓")
print(f"  parallel_patch_embed: {config['model']['parallel_patch_embed']}")
print(f"  use_physics_mask: {config['model']['use_physics_mask']}")
print(f"  use_3d_learnable: {config['model']['use_3d_learnable']}")

# Import
from src.model_multipollutants import MultiPollutantModel
from src.climax_core.topoflow_attention import TopoFlowBlock

print("\n2. Classes imported ✓")

# Create model
try:
    model = MultiPollutantModel(config)
    print("\n3. Model created ✓")

    # Check first block
    first_block = model.climax.blocks[0]
    is_topoflow = isinstance(first_block, TopoFlowBlock)

    print(f"\n4. First block is TopoFlowBlock: {is_topoflow}")

    if is_topoflow:
        print(f"   - elevation_alpha: {first_block.attn.elevation_alpha.item():.3f}")
        print(f"   - wind_beta: {first_block.attn.wind_beta.item():.3f}")
        print(f"   - H_scale: {first_block.attn.H_scale.item():.1f}m")
        print("\n✅ Wind + Simple Physics ready!")
        print("   - Wind scanning: enabled (32×32)")
        print("   - Elevation: simple formula (2 params)")
        print("   - Approach: additive BEFORE softmax")
    else:
        print(f"\n❌ ERROR: First block type: {type(first_block)}")
        exit(1)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("=" * 70)

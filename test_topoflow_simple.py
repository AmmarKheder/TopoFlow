#!/usr/bin/env python3
"""Quick test: Can we create the model?"""
import torch
import yaml

print("=" * 70)
print("TopoFlow Quick Test")
print("=" * 70)

# Load config
with open('configs/config_topoflow_elevation.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("\n1. Config loaded ✓")

# Import model
from src.model_multipollutants import MultiPollutantModel

print("2. Model class imported ✓")

# Create model
try:
    model = MultiPollutantModel(config)
    print("3. Model created ✓")

    # Check first block
    from src.climax_core.topoflow_attention import TopoFlowBlock
    first_block = model.climax.blocks[0]  # model.climax!
    is_topoflow = isinstance(first_block, TopoFlowBlock)

    print(f"4. First block is TopoFlowBlock: {is_topoflow}")

    if is_topoflow:
        print(f"   - elevation_alpha: {first_block.attn.elevation_alpha.item():.3f}")
        print(f"   - wind_beta: {first_block.attn.wind_beta.item():.3f}")
        print(f"   - H_scale: {first_block.attn.H_scale.item():.1f}")
        print("\n✅ ALL GOOD - Model is ready!")
    else:
        print("\n❌ ERROR: First block is NOT TopoFlow!")
        print(f"   Type: {type(first_block)}")
        exit(1)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("=" * 70)

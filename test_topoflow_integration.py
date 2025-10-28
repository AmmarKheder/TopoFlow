#!/usr/bin/env python3
"""
Quick test to verify TopoFlow integration works before launching 400 GPU job.
"""
import torch
import yaml
from src.model_multipollutants import load_model_from_config

print("=" * 70)
print("TopoFlow Integration Test")
print("=" * 70)

# Load config
with open('configs/config_topoflow_elevation.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("\n✓ Config loaded")
print(f"  use_physics_mask: {config['model']['use_physics_mask']}")
print(f"  parallel_patch_embed: {config['model']['parallel_patch_embed']}")

# Try to create model
try:
    print("\n✓ Creating model...")
    model = load_model_from_config(config)
    print(f"  Model created successfully")
    print(f"  Model type: {type(model).__name__}")

    # Check if first block is TopoFlow
    from src.climax_core.topoflow_attention import TopoFlowBlock
    first_block = model.net.blocks[0]
    is_topoflow = isinstance(first_block, TopoFlowBlock)
    print(f"  First block is TopoFlowBlock: {is_topoflow}")

    if is_topoflow:
        print(f"    elevation_alpha: {first_block.attn.elevation_alpha.item():.3f}")
        print(f"    wind_beta: {first_block.attn.wind_beta.item():.3f}")

    # Test forward pass with dummy data
    print("\n✓ Testing forward pass...")
    B, V, H, W = 2, 14, 128, 256  # 2 batch, 14 variables
    x = torch.randn(B, V, H, W)
    y = torch.randn(B, 6, H, W)  # 6 target variables
    lead_times = torch.tensor([12, 24])
    variables = ['u', 'v', 'temp', 'rh', 'psfc', 'pm10', 'so2', 'no2', 'co', 'o3', 'lat2d', 'lon2d', 'pm25', 'elevation']
    out_variables = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']

    with torch.no_grad():
        output = model(x, y, lead_times, variables, out_variables, metric=None, lat=None)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output['preds'].shape}")
    print(f"  Forward pass successful!")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - READY TO TRAIN!")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 70)
    print("❌ TESTS FAILED - DO NOT SUBMIT JOB!")
    print("=" * 70)
    exit(1)

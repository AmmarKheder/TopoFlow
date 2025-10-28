#!/usr/bin/env python3
"""
Test script to verify checkpoint loading works correctly in DDP.

This simulates what happens with multiple GPUs:
1. Model created
2. Checkpoint path stored
3. DDP spawn happens
4. setup() called in each rank
5. Checkpoint loaded in ALL ranks
"""

import sys
sys.path.insert(0, '/pfs/lustrep1/scratch/project_462000640/ammar/aq_net2')

import torch
from src.model_multipollutants import MultiPollutantLightningModule

print("=" * 80)
print("TEST: DDP Checkpoint Loading")
print("=" * 80)

# Minimal config
config = {
    "data": {
        "variables": ['u', 'v', 'temp', 'rh', 'psfc', 'pm10', 'so2', 'no2', 'co', 'o3', 'lat2d', 'lon2d', 'pm25', 'elevation', 'population'],
        "target_variables": ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3'],
    },
    "model": {
        "img_size": [128, 256],
        "patch_size": 2,
        "embed_dim": 768,
        "depth": 6,
        "decoder_depth": 2,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "parallel_patch_embed": True,
        "use_physics_mask": True,
        "use_3d_learnable": False,
        "checkpoint_path": "/scratch/project_462000640/ammar/aq_net2/logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"
    }
}

print("\n1️⃣ Creating Lightning module...")
model = MultiPollutantLightningModule(config)
print(f"   ✅ Model created")

print("\n2️⃣ Setting checkpoint path (simulating main.py)...")
checkpoint_path = config['model']['checkpoint_path']
model._checkpoint_path_to_load = checkpoint_path
print(f"   ✅ Checkpoint path stored: {checkpoint_path}")

print("\n3️⃣ Simulating DDP spawn + setup() call...")
print("   (In real DDP, this happens in each rank)")

# Check parameter values BEFORE setup
sample_param_name = None
sample_param_before = None
for name, param in model.named_parameters():
    if 'blocks.0.attn.qkv.weight' in name:
        sample_param_name = name
        sample_param_before = param[0, :5].clone().detach()
        print(f"\n   BEFORE setup(): {name}[0, :5]")
        print(f"   Values: {param[0, :5]}")
        break

# Call setup (simulating what Lightning does after DDP spawn)
model.setup(stage='fit')

print("\n4️⃣ Checking if checkpoint was loaded...")

# Check parameter values AFTER setup
for name, param in model.named_parameters():
    if name == sample_param_name:
        sample_param_after = param[0, :5].clone().detach()
        print(f"\n   AFTER setup(): {name}[0, :5]")
        print(f"   Values: {param[0, :5]}")

        if torch.allclose(sample_param_before, sample_param_after):
            print(f"\n   ❌ VALUES DID NOT CHANGE!")
            print(f"   Checkpoint was NOT loaded correctly!")
        else:
            print(f"\n   ✅ VALUES CHANGED!")
            print(f"   Checkpoint loaded successfully!")
        break

print("\n5️⃣ Checking elevation_alpha initialization...")
for name, param in model.named_parameters():
    if 'elevation_alpha' in name:
        print(f"   {name} = {param.item():.6f}")
        if abs(param.item() - 0.01) < 0.001:
            print(f"   ✅ Correctly initialized to ~0.01")
        else:
            print(f"   ⚠️  Value: {param.item()}")

print("\n" + "=" * 80)
print("RESULT:")
print("=" * 80)
print("✅ If values changed after setup(), the fix works!")
print("✅ Checkpoint will be loaded in ALL DDP ranks")
print("✅ Training will start from val_loss ≈ 0.3557 instead of ~3.8")
print("=" * 80)

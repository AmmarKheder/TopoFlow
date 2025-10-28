#!/usr/bin/env python3
"""
Debug script to understand why val_loss starts at 1.75 instead of 0.3557
"""

import sys
sys.path.insert(0, '/pfs/lustrep1/scratch/project_462000640/ammar/aq_net2')

import torch
import torch.nn as nn
from src.climax_core.arch import ClimaX

# Checkpoint path
CHECKPOINT_PATH = "/scratch/project_462000640/ammar/aq_net2/logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"

# Config (matching the checkpoint)
DEFAULT_VARS = ['u', 'v', 'temp', 'rh', 'psfc', 'pm10', 'so2', 'no2', 'co', 'o3', 'lat2d', 'lon2d', 'pm25', 'elevation', 'population']
IMG_SIZE = [128, 256]
PATCH_SIZE = 2
EMBED_DIM = 768
DEPTH = 6
NUM_HEADS = 8

print("=" * 80)
print("🔍 DEBUG: Why val_loss starts at 1.75 instead of 0.3557?")
print("=" * 80)

# Simplified Lightning Module structure
class SimpleMultiPollutantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.climax = ClimaX(
            default_vars=DEFAULT_VARS,
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM,
            depth=DEPTH,
            num_heads=NUM_HEADS,
            parallel_patch_embed=True,
            use_physics_mask=True,  # TopoFlow on block 0
            use_3d_learnable=False,
        )
        # Dummy target indices
        self.register_buffer("target_indices", torch.tensor([5, 6, 7, 8, 9, 12], dtype=torch.long))

class SimpleLightningModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleMultiPollutantModel()

print("\n1️⃣ Creating model...")
lightning_model = SimpleLightningModule()
total_params = sum(p.numel() for p in lightning_model.parameters())
print(f"   ✅ Model created: {total_params:,} parameters")

# Check a few parameter values BEFORE loading
print("\n2️⃣ Checking parameter values BEFORE loading checkpoint...")
sample_params_before = {}
for name, param in lightning_model.named_parameters():
    if 'blocks.0.attn.qkv.weight' in name:
        sample_params_before['qkv'] = param[0, :5].clone().detach()
        print(f"   blocks.0.attn.qkv.weight[0, :5] = {param[0, :5]}")
        break

print("\n3️⃣ Loading checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
print(f"   ✅ Checkpoint loaded from disk")
print(f"   - Keys in checkpoint['state_dict']: {len(checkpoint['state_dict'])}")

# Check what keys are in checkpoint
model_keys = [k for k in checkpoint['state_dict'].keys() if k.startswith('model.')]
print(f"   - Keys starting with 'model.': {len(model_keys)}")
print(f"   - Sample keys: {list(checkpoint['state_dict'].keys())[:5]}")

print("\n4️⃣ Loading state_dict with strict=False...")
result = lightning_model.load_state_dict(checkpoint['state_dict'], strict=False)

print(f"\n   📊 Load result:")
print(f"   - Missing keys: {len(result.missing_keys)}")
if result.missing_keys:
    print(f"     {result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}")
print(f"   - Unexpected keys: {len(result.unexpected_keys)}")
if result.unexpected_keys:
    print(f"     {result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}")

print("\n5️⃣ Checking parameter values AFTER loading checkpoint...")
for name, param in lightning_model.named_parameters():
    if 'blocks.0.attn.qkv.weight' in name:
        sample_params_after = param[0, :5].clone().detach()
        print(f"   blocks.0.attn.qkv.weight[0, :5] = {param[0, :5]}")

        # Check if values changed
        if 'qkv' in sample_params_before:
            if torch.allclose(sample_params_before['qkv'], sample_params_after):
                print(f"   ⚠️  VALUES DID NOT CHANGE! Checkpoint not loaded correctly!")
            else:
                print(f"   ✅ VALUES CHANGED! Checkpoint loaded successfully!")
        break

print("\n6️⃣ Checking if elevation_alpha was initialized...")
for name, param in lightning_model.named_parameters():
    if 'elevation_alpha' in name:
        print(f"   {name} = {param.item()}")
        if param.item() == 0.01:
            print(f"   ✅ Correctly initialized to 0.01")
        else:
            print(f"   ⚠️  NOT initialized to 0.01! Value: {param.item()}")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)

if len(result.missing_keys) == 2 and len(result.unexpected_keys) == 0:
    print("✅ Checkpoint structure matches model")
elif len(result.unexpected_keys) > 50:
    print("❌ PROBLEM: Too many unexpected keys - checkpoint not matching!")
elif len(result.missing_keys) > 10:
    print("❌ PROBLEM: Too many missing keys - checkpoint not matching!")
else:
    print("⚠️  Unexpected load result")

print("=" * 80)

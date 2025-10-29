#!/usr/bin/env python3
"""
Test script to verify checkpoint loading with fixed architecture.
Should show only 2 missing keys: elevation_alpha and H_scale (TopoFlow params).
"""

import sys
sys.path.insert(0, '/pfs/lustrep1/scratch/project_462000640/ammar/aq_net2')

import torch
from src.climax_core.arch import ClimaX

# Configuration matching the checkpoint (from config_all_pollutants.yaml)
DEFAULT_VARS = ['u', 'v', 'temp', 'rh', 'psfc', 'pm10', 'so2', 'no2', 'co', 'o3', 'lat2d', 'lon2d', 'pm25', 'elevation', 'population']  # 15 variables
IMG_SIZE = [128, 256]  # From config
PATCH_SIZE = 2
EMBED_DIM = 768
DEPTH = 6  # From config (not 8!)
NUM_HEADS = 8  # From config (not 16!)

# Checkpoint path (ORIGINAL - correct format!)
CHECKPOINT_PATH = "/scratch/project_462000640/ammar/aq_net2/logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"

print("=" * 80)
print("🔍 CHECKPOINT LOADING TEST - TopoFlow Architecture")
print("=" * 80)

# Create model with TopoFlow on block 0
print("\n1️⃣ Creating model with TopoFlow on block 0...")
model = ClimaX(
    default_vars=DEFAULT_VARS,
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    parallel_patch_embed=True,  # Must match checkpoint!
    use_physics_mask=True,  # Enables TopoFlow on block 0
    use_3d_learnable=False,  # Use simple elevation formula, not 3D MLP
)

print(f"   ✅ Model created")
print(f"   - Total blocks: {DEPTH}")
print(f"   - TopoFlow blocks: [0]")
print(f"   - Embed dim: {EMBED_DIM}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"   - Total parameters: {total_params:,}")

# Load checkpoint
print(f"\n2️⃣ Loading checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

# Extract model state dict (remove 'model.' and 'climax.' prefixes)
state_dict = checkpoint['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    # Remove prefixes
    new_k = k
    if new_k.startswith('model.'):
        new_k = new_k[6:]  # Remove 'model.'
    if new_k.startswith('climax.'):
        new_k = new_k[7:]  # Remove 'climax.'

    # Skip masks and indices (not model parameters)
    if new_k in ['china_mask', 'target_indices']:
        continue

    new_state_dict[new_k] = v

print(f"   ✅ Checkpoint loaded")
print(f"   - Checkpoint keys: {len(new_state_dict)}")
print(f"   - Sample keys: {list(new_state_dict.keys())[:5]}")

# Load state dict into model
print(f"\n3️⃣ Loading state dict into model...")
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

print(f"\n   📊 RESULTS:")
print(f"   - Missing keys: {len(missing_keys)}")
print(f"   - Unexpected keys: {len(unexpected_keys)}")

# Analyze missing keys
if missing_keys:
    print(f"\n   ❌ MISSING KEYS (should be only 2 TopoFlow params):")
    for i, key in enumerate(missing_keys, 1):
        print(f"      {i}. {key}")

    # Check if missing keys are only TopoFlow params (without 'climax.' prefix after processing)
    topoflow_keys = {'blocks.0.attn.elevation_alpha', 'blocks.0.attn.H_scale'}
    missing_set = set(missing_keys)

    if missing_set == topoflow_keys:
        print(f"\n   ✅ PERFECT! Only TopoFlow parameters are missing (expected)")
    else:
        unexpected_missing = missing_set - topoflow_keys
        if unexpected_missing:
            print(f"\n   ⚠️  UNEXPECTED MISSING KEYS:")
            for key in unexpected_missing:
                print(f"      - {key}")

        extra_missing = topoflow_keys - missing_set
        if extra_missing:
            print(f"\n   ⚠️  EXPECTED MISSING KEYS NOT FOUND:")
            for key in extra_missing:
                print(f"      - {key}")

# Analyze unexpected keys
if unexpected_keys:
    print(f"\n   ⚠️  UNEXPECTED KEYS (from checkpoint but not in model):")
    for i, key in enumerate(unexpected_keys, 1):
        print(f"      {i}. {key}")

# Count loaded parameters
loaded_params = sum(p.numel() for name, p in model.named_parameters()
                   if not any(mk in name for mk in missing_keys))
missing_params = sum(p.numel() for name, p in model.named_parameters()
                    if any(mk in name for mk in missing_keys))

print(f"\n   📈 PARAMETER STATISTICS:")
print(f"   - Total parameters: {total_params:,}")
print(f"   - Loaded from checkpoint: {loaded_params:,} ({100*loaded_params/total_params:.2f}%)")
print(f"   - Random initialized: {missing_params:,} ({100*missing_params/total_params:.2f}%)")

# Summary
print("\n" + "=" * 80)
if len(missing_keys) == 2 and set(missing_keys) == topoflow_keys:
    print("✅ SUCCESS! Architecture is correctly configured.")
    print("   Only TopoFlow parameters (elevation_alpha, H_scale) are missing.")
    print("   All other weights will load from checkpoint.")
else:
    print("❌ FAILED! Architecture mismatch detected.")
    print("   Please check the missing/unexpected keys above.")
print("=" * 80)

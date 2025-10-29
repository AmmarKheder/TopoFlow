#!/usr/bin/env python3
"""
Script to analyze checkpoint keys vs current model architecture
"""
import sys
import torch

# Load checkpoint
ckpt_path = "/scratch/project_462000640/ammar/aq_net2/logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"

print("Loading checkpoint...")
ckpt = torch.load(ckpt_path, map_location='cpu')

print("\n" + "="*80)
print("CHECKPOINT STATE_DICT KEYS")
print("="*80)

state_dict = ckpt['state_dict']
keys = sorted(state_dict.keys())

# Filter for relevant keys
print("\n### BLOCK 0 KEYS (attention + MLP):")
block0_keys = [k for k in keys if 'blocks.0' in k]
for i, key in enumerate(block0_keys, 1):
    shape = tuple(state_dict[key].shape) if hasattr(state_dict[key], 'shape') else 'scalar'
    print(f"{i}. {key}")
    print(f"   Shape: {shape}")

print("\n### HEAD KEYS:")
head_keys = [k for k in keys if 'head' in k]
for i, key in enumerate(head_keys, 1):
    shape = tuple(state_dict[key].shape) if hasattr(state_dict[key], 'shape') else 'scalar'
    print(f"{i}. {key}")
    print(f"   Shape: {shape}")

print("\n### BLOCK 1 KEYS (for comparison):")
block1_keys = [k for k in keys if 'blocks.1.' in k]
for i, key in enumerate(block1_keys, 1):
    shape = tuple(state_dict[key].shape) if hasattr(state_dict[key], 'shape') else 'scalar'
    print(f"{i}. {key}")
    print(f"   Shape: {shape}")

print("\n" + "="*80)
print(f"TOTAL KEYS IN CHECKPOINT: {len(keys)}")
print("="*80)

# Now let's check what the current model expects
print("\n\n" + "="*80)
print("EXPECTED KEYS BY CURRENT MODEL")
print("="*80)

expected_block0_keys = [
    "model.climax.blocks.0.attn.elevation_alpha",  # TopoFlow
    "model.climax.blocks.0.attn.H_scale",  # TopoFlow
    "model.climax.blocks.0.mlp.0.weight",  # Sequential MLP
    "model.climax.blocks.0.mlp.0.bias",
    "model.climax.blocks.0.mlp.2.weight",
    "model.climax.blocks.0.mlp.2.bias",
]

print("\n### EXPECTED BLOCK 0 KEYS:")
for i, key in enumerate(expected_block0_keys, 1):
    in_ckpt = key in keys
    status = "✅ IN CKPT" if in_ckpt else "❌ MISSING (random init)"
    print(f"{i}. {key} - {status}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Missing keys
missing = [k for k in expected_block0_keys if k not in keys]
print(f"\n❌ MISSING KEYS (will be randomly initialized): {len(missing)}")
for k in missing:
    print(f"   - {k}")

# Unexpected keys (in checkpoint but not used)
checkpoint_block0_mlp_keys = [k for k in block0_keys if 'mlp' in k]
print(f"\n⚠️  UNUSED CHECKPOINT KEYS (will be ignored): {len(checkpoint_block0_mlp_keys)}")
for k in checkpoint_block0_mlp_keys:
    print(f"   - {k}")

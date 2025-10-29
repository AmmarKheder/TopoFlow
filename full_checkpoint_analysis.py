#!/usr/bin/env python3
"""
Complete analysis of checkpoint vs current model architecture
"""
import torch

ckpt_path = "/scratch/project_462000640/ammar/aq_net2/logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"

print("="*100)
print("COMPLETE CHECKPOINT ANALYSIS")
print("="*100)

# Load checkpoint
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']

print(f"\n📊 Checkpoint info:")
print(f"   - Total parameters: {len(state_dict)} keys")
print(f"   - Step: 311")
print(f"   - val_loss: 0.3557")

# Categorize all keys
print("\n" + "="*100)
print("1️⃣  BLOCK 0 - ATTENTION KEYS (QKV + Projection)")
print("="*100)
attn_keys = sorted([k for k in state_dict.keys() if 'blocks.0.attn' in k])
for key in attn_keys:
    shape = tuple(state_dict[key].shape)
    params = state_dict[key].numel()
    print(f"   ✓ {key:60s} {str(shape):20s} ({params:,} params)")

print("\n" + "="*100)
print("2️⃣  BLOCK 0 - MLP KEYS (Feed-Forward Network)")
print("="*100)
print("\n🔴 IN CHECKPOINT (will be IGNORED by current model):")
mlp_keys = sorted([k for k in state_dict.keys() if 'blocks.0.mlp' in k])
for key in mlp_keys:
    shape = tuple(state_dict[key].shape)
    params = state_dict[key].numel()
    print(f"   ✗ {key:60s} {str(shape):20s} ({params:,} params)")

print("\n🟢 EXPECTED BY CURRENT MODEL (will be RANDOM):")
expected_mlp = [
    ("model.climax.blocks.0.mlp.0.weight", "(3072, 768)"),
    ("model.climax.blocks.0.mlp.0.bias", "(3072,)"),
    ("model.climax.blocks.0.mlp.2.weight", "(768, 3072)"),
    ("model.climax.blocks.0.mlp.2.bias", "(768,)"),
]
for key, shape in expected_mlp:
    print(f"   ? {key:60s} {shape:20s} (RANDOM INIT)")

print("\n" + "="*100)
print("3️⃣  BLOCK 0 - NORMALIZATION KEYS")
print("="*100)
norm_keys = sorted([k for k in state_dict.keys() if 'blocks.0.norm' in k])
for key in norm_keys:
    shape = tuple(state_dict[key].shape)
    params = state_dict[key].numel()
    print(f"   ✓ {key:60s} {str(shape):20s} ({params:,} params)")

print("\n" + "="*100)
print("4️⃣  BLOCK 0 - TOPOFLOW KEYS (NEW!)")
print("="*100)
print("🟢 EXPECTED BY CURRENT MODEL (will be RANDOM):")
topoflow_keys = [
    ("model.climax.blocks.0.attn.elevation_alpha", "() scalar"),
    ("model.climax.blocks.0.attn.H_scale", "() scalar"),
]
for key, shape in topoflow_keys:
    print(f"   ? {key:60s} {shape:20s} (RANDOM INIT)")

print("\n" + "="*100)
print("5️⃣  HEAD (Prediction Layer)")
print("="*100)
print("\n🔴 IN CHECKPOINT (3-layer Sequential):")
head_keys = sorted([k for k in state_dict.keys() if 'head' in k])
for key in head_keys:
    shape = tuple(state_dict[key].shape)
    params = state_dict[key].numel()
    print(f"   ✗ {key:60s} {str(shape):20s} ({params:,} params)")

print("\n🟢 EXPECTED BY CURRENT MODEL (single Linear):")
print(f"   ? model.climax.head.weight                                    (60, 768)            (RANDOM? or loaded from head.4?)")
print(f"   ? model.climax.head.bias                                      (60,)                (RANDOM? or loaded from head.4?)")

print("\n" + "="*100)
print("📊 SUMMARY")
print("="*100)

# Count parameters
block0_attn_checkpoint = sum([state_dict[k].numel() for k in attn_keys])
block0_mlp_checkpoint = sum([state_dict[k].numel() for k in mlp_keys])
block0_norm_checkpoint = sum([state_dict[k].numel() for k in norm_keys])
head_checkpoint = sum([state_dict[k].numel() for k in head_keys])

print(f"\n✅ LOADED FROM CHECKPOINT:")
print(f"   - Block 0 Attention (QKV + proj): {block0_attn_checkpoint:,} params")
print(f"   - Block 0 Normalization: {block0_norm_checkpoint:,} params")
print(f"   - Blocks 1-7 (all layers): [calculated separately]")
print(f"   - Other layers (embeddings, etc): [calculated separately]")

print(f"\n❌ RANDOMLY INITIALIZED (NOT loaded from checkpoint):")
print(f"   - Block 0 MLP (nn.Sequential): ~2.4M params (was 'fc1'/'fc2', now '0'/'2')")
print(f"   - Block 0 TopoFlow params: 2 params (elevation_alpha, H_scale)")
print(f"   - Head: {head_checkpoint:,} params (was Sequential.0/2/4, now single Linear)")

print(f"\n🔍 ARCHITECTURE MISMATCH REASON:")
print(f"   1. MLP: TopoFlowBlock uses nn.Sequential → keys: mlp.0, mlp.2")
print(f"           Checkpoint has named layers → keys: mlp.fc1, mlp.fc2")
print(f"   2. HEAD: Current model has nn.Linear → keys: head.weight, head.bias")
print(f"            Checkpoint has nn.Sequential → keys: head.0, head.2, head.4")

print(f"\n⚠️  IMPACT ON TRAINING:")
print(f"   - val_loss = 2.190 at step 25 (vs 0.3557 at checkpoint)")
print(f"   - Block 0 MLP is COMPLETELY random → disrupts feature transformation")
print(f"   - Head is COMPLETELY random → disrupts final predictions")
print(f"   - ~{block0_mlp_checkpoint + head_checkpoint:,} parameters need retraining")

print("\n" + "="*100)

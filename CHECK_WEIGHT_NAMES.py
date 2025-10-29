#!/usr/bin/env python3
"""
Vérifier les noms des poids dans le checkpoint vs le modèle.
"""
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule

print("="*100)
print("CHECK: WEIGHT NAMES IN CHECKPOINT VS MODEL")
print("="*100)

# Config
config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config

ckpt_path = config['model']['checkpoint_path']
print(f"\nCheckpoint: {ckpt_path}\n")

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print("📦 CHECKPOINT STATE_DICT KEYS (first 20):")
ckpt_keys = list(ckpt['state_dict'].keys())
for i, key in enumerate(ckpt_keys[:20]):
    print(f"  {i}: {key}")

# Create model
model = MultiPollutantLightningModule(config=config)

print("\n📦 MODEL PARAMETER NAMES (first 20):")
model_keys = [name for name, _ in model.named_parameters()]
for i, key in enumerate(model_keys[:20]):
    print(f"  {i}: {key}")

# Check if prefixes match
print("\n" + "="*100)
print("🔍 ANALYSIS:")
print("="*100)

# Find common prefix
if len(ckpt_keys) > 0 and len(model_keys) > 0:
    ckpt_prefix = ckpt_keys[0].split('.')[0] if '.' in ckpt_keys[0] else ""
    model_prefix = model_keys[0].split('.')[0] if '.' in model_keys[0] else ""

    print(f"\nCheckpoint first key: {ckpt_keys[0]}")
    print(f"Model first key:      {model_keys[0]}")
    print(f"\nCheckpoint prefix: '{ckpt_prefix}'")
    print(f"Model prefix:      '{model_prefix}'")

    if ckpt_prefix != model_prefix:
        print(f"\n❌ PREFIX MISMATCH!")
        print(f"   Checkpoint uses prefix '{ckpt_prefix}' but model expects '{model_prefix}'")
        print(f"\n🔧 SOLUTION:")
        print(f"   When loading checkpoint, we need to either:")
        print(f"   1. Strip '{ckpt_prefix}.' prefix from checkpoint keys")
        print(f"   2. Add '{ckpt_prefix}.' prefix to model keys")
        print(f"   3. Use strict=False and handle mismatches")
    else:
        print(f"\n✅ Prefixes match!")

# Check how many keys match
matches = 0
ckpt_only = []
model_only = []

ckpt_set = set(ckpt_keys)
model_set = set(model_keys)

for key in model_set:
    if key in ckpt_set:
        matches += 1
    else:
        model_only.append(key)

for key in ckpt_set:
    if key not in model_set:
        ckpt_only.append(key)

print(f"\n📊 KEY MATCHING STATISTICS:")
print(f"  Checkpoint keys: {len(ckpt_keys)}")
print(f"  Model keys:      {len(model_keys)}")
print(f"  Matches:         {matches}")
print(f"  Checkpoint only: {len(ckpt_only)}")
print(f"  Model only:      {len(model_only)}")

if ckpt_only:
    print(f"\n📦 Keys in CHECKPOINT but not in MODEL (first 10):")
    for key in ckpt_only[:10]:
        print(f"    {key}")

if model_only:
    print(f"\n📦 Keys in MODEL but not in CHECKPOINT (first 10):")
    for key in model_only[:10]:
        print(f"    {key}")

print("\n" + "="*100)
if matches == 0:
    print("❌❌❌ AUCUNE CLÉ NE CORRESPOND!")
    print("C'est pourquoi le checkpoint ne se charge pas correctement!")
elif matches < len(model_keys) * 0.9:
    print(f"⚠️ Seulement {matches}/{len(model_keys)} clés correspondent ({100*matches/len(model_keys):.1f}%)")
    print("Certains poids ne seront pas chargés!")
else:
    print(f"✅ {matches}/{len(model_keys)} clés correspondent ({100*matches/len(model_keys):.1f}%)")
print("="*100)

#!/usr/bin/env python3
"""
Vérifier si les poids du checkpoint sont vraiment chargés.
"""
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule

print("="*100)
print("DEBUG: CHECKPOINT WEIGHTS LOADING")
print("="*100)

# Config
config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config

ckpt_path = config['model']['checkpoint_path']
print(f"\nCheckpoint: {ckpt_path}\n")

# 1. Load checkpoint raw to see what weights it has
print("📦 Loading checkpoint dict...")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Get some sample weights from checkpoint
sample_ckpt_weights = {}
for i, (name, tensor) in enumerate(ckpt['state_dict'].items()):
    if i < 10:
        sample_ckpt_weights[name] = {
            'mean': tensor.float().mean().item(),
            'std': tensor.float().std().item(),
            'shape': tuple(tensor.shape)
        }

print("\n📊 Sample weights FROM CHECKPOINT (state_dict):")
for name, stats in sample_ckpt_weights.items():
    print(f"  {name}:")
    print(f"    shape={stats['shape']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}")

# 2. Create a fresh model (no checkpoint)
print("\n" + "="*100)
print("🆕 Creating FRESH model (no checkpoint)...")
print("="*100)

model_fresh = MultiPollutantLightningModule(config=config)

sample_fresh_weights = {}
for name, param in list(model_fresh.named_parameters())[:10]:
    sample_fresh_weights[name] = {
        'mean': param.data.mean().item(),
        'std': param.data.std().item(),
        'shape': tuple(param.shape)
    }

print("\n📊 Sample weights from FRESH model:")
for name, stats in sample_fresh_weights.items():
    print(f"  {name}:")
    print(f"    shape={stats['shape']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}")

# 3. Load model from checkpoint
print("\n" + "="*100)
print("📥 Loading model FROM CHECKPOINT...")
print("="*100)

model_loaded = MultiPollutantLightningModule.load_from_checkpoint(
    ckpt_path,
    config=config,
    strict=False
)

sample_loaded_weights = {}
for name, param in list(model_loaded.named_parameters())[:10]:
    sample_loaded_weights[name] = {
        'mean': param.data.mean().item(),
        'std': param.data.std().item(),
        'shape': tuple(param.shape)
    }

print("\n📊 Sample weights from LOADED model:")
for name, stats in sample_loaded_weights.items():
    print(f"  {name}:")
    print(f"    shape={stats['shape']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}")

# 4. Compare
print("\n" + "="*100)
print("🔍 COMPARISON: FRESH vs LOADED")
print("="*100)

all_match = True
for name in sample_fresh_weights:
    fresh_mean = sample_fresh_weights[name]['mean']
    loaded_mean = sample_loaded_weights[name]['mean']
    diff = abs(fresh_mean - loaded_mean)

    if diff < 0.001:
        print(f"❌ {name}: IDENTICAL (diff={diff:.6f}) - WEIGHTS NOT LOADED!")
        all_match = False
    else:
        print(f"✅ {name}: DIFFERENT (diff={diff:.6f}) - weights loaded correctly")

print("\n" + "="*100)
if all_match:
    print("❌❌❌ PROBLÈME MAJEUR!")
    print("Les poids du checkpoint ne sont PAS chargés!")
    print("Le modèle utilise les poids aléatoires d'initialisation.")
else:
    print("✅ Les poids semblent chargés correctement.")
    print("Le problème doit être ailleurs (normalisation? autre chose?).")
print("="*100)

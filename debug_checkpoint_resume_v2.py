#!/usr/bin/env python3
"""
Test plus précis: compare un VRAI paramètre entraînable (pas china_mask qui est un buffer)
"""

import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule

print("="*80)
print("🔍 TEST PRÉCIS: Le checkpoint charge-t-il les vrais poids du modèle?")
print("="*80)

config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config

ckpt_path = config['model']['checkpoint_path']

# Load checkpoint raw
print("\n1️⃣ Chargement du checkpoint brut...")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Trouver un vrai paramètre entraînable (pas un buffer)
state_dict = ckpt['state_dict']

# Prenons un poids du ClimaX (certainement entraînable)
test_params = [
    'model.climax.encoder_blocks.0.attn.qkv.weight',
    'model.climax.var_embed',
    'model.head.weight'
]

test_param = None
for param_name in test_params:
    if param_name in state_dict:
        test_param = param_name
        break

if test_param is None:
    # Fallback: prendre n'importe quel poids qui contient "weight"
    for key in state_dict.keys():
        if 'weight' in key and 'climax' in key:
            test_param = key
            break

print(f"\n📊 Paramètre de test: {test_param}")
checkpoint_weights = state_dict[test_param].clone()
print(f"   Checkpoint: mean={checkpoint_weights.float().mean():.6f}, std={checkpoint_weights.float().std():.6f}")
print(f"   Shape: {checkpoint_weights.shape}")

# Créer 2 modèles: un neuf, un chargé
print("\n2️⃣ Création de 2 modèles...")

print("   - Modèle 1: random init")
model_random = MultiPollutantLightningModule(config=config)

print("\n   - Modèle 2: chargé depuis checkpoint")
model_loaded = MultiPollutantLightningModule.load_from_checkpoint(
    ckpt_path,
    config=config,
    strict=False
)

# Comparer les poids
print("\n3️⃣ Comparaison des poids...")

random_weights = model_random.state_dict()[test_param]
loaded_weights = model_loaded.state_dict()[test_param]

print(f"\n📊 {test_param}:")
print(f"   Random init:  mean={random_weights.float().mean():.6f}, std={random_weights.float().std():.6f}")
print(f"   Checkpoint:   mean={checkpoint_weights.float().mean():.6f}, std={checkpoint_weights.float().std():.6f}")
print(f"   Loaded model: mean={loaded_weights.float().mean():.6f}, std={loaded_weights.float().std():.6f}")

# Test d'égalité
print("\n4️⃣ Tests d'égalité...")

if torch.allclose(random_weights, loaded_weights, atol=1e-6):
    print("   ❌ ÉCHEC: Loaded == Random (checkpoint ne s'est PAS chargé!)")
elif torch.allclose(checkpoint_weights, loaded_weights, atol=1e-6):
    print("   ✅ SUCCÈS: Loaded == Checkpoint (bon chargement)")
else:
    print("   ⚠️  BIZARRE: Loaded ≠ Random ET Loaded ≠ Checkpoint")
    print(f"      Diff Random vs Loaded: {(random_weights - loaded_weights).abs().max():.6f}")
    print(f"      Diff Checkpoint vs Loaded: {(checkpoint_weights - loaded_weights).abs().max():.6f}")

# Test: le modèle chargé a-t-il le bon global_step?
print("\n5️⃣ Vérification du global_step...")
if hasattr(model_loaded, 'global_step'):
    print(f"   Loaded model global_step: {model_loaded.global_step}")
    print(f"   Checkpoint global_step: {ckpt['global_step']}")
    if model_loaded.global_step == ckpt['global_step']:
        print("   ✅ global_step correct")
    else:
        print("   ❌ global_step différent!")
else:
    print("   ⚠️  global_step non disponible sur le modèle")

print("\n" + "="*80)

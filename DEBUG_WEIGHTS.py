#!/usr/bin/env python3
"""Vérifier si les poids du checkpoint sont VRAIMENT chargés"""
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule

config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config
ckpt_path = config['model']['checkpoint_path']

print("="*100)
print("VÉRIFICATION POIDS CHECKPOINT")
print("="*100)

# 1. Charger checkpoint brut
print("\n1. Chargement checkpoint brut...")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict_ckpt = ckpt['state_dict']

# Prendre 5 paramètres critiques du modèle
test_params = [
    'model.climax.encoder_blocks.0.attn.qkv.weight',
    'model.climax.var_embed',
    'model.head.weight',
    'model.head.bias',
    'model.climax.pos_embed'
]

print("\n2. Poids dans le checkpoint:")
ckpt_weights = {}
for pname in test_params:
    if pname in state_dict_ckpt:
        w = state_dict_ckpt[pname]
        ckpt_weights[pname] = w.clone()
        print(f"   {pname}: mean={w.float().mean():.6f}, std={w.float().std():.6f}")

# 2. Charger modèle avec Lightning
print("\n3. Chargement modèle avec Lightning...")
model = MultiPollutantLightningModule.load_from_checkpoint(
    ckpt_path,
    config=config,
    strict=False
)

# 3. Comparer
print("\n4. COMPARAISON POIDS:")
all_match = True
for pname, ckpt_w in ckpt_weights.items():
    if pname in model.state_dict():
        loaded_w = model.state_dict()[pname]

        match = torch.allclose(ckpt_w, loaded_w, atol=1e-6)
        diff = (ckpt_w - loaded_w).abs().max().item()

        symbol = "✅" if match else "❌"
        print(f"\n   {symbol} {pname}:")
        print(f"      Checkpoint: mean={ckpt_w.float().mean():.6f}")
        print(f"      Loaded:     mean={loaded_w.float().mean():.6f}")
        print(f"      Max diff:   {diff:.9f}")

        if not match:
            all_match = False
            print(f"      ⚠️  LES POIDS NE CORRESPONDENT PAS!")

print("\n" + "="*100)
if all_match:
    print("✅ TOUS LES POIDS SONT CORRECTEMENT CHARGÉS")
else:
    print("❌ CERTAINS POIDS NE SE SONT PAS CHARGÉS CORRECTEMENT!")
print("="*100)

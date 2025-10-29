#!/usr/bin/env python3
"""
Verify that MLP is REALLY loaded from checkpoint in the actual model
"""
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

import torch
import yaml
from src.model_multipollutants import MultiPollutantLightningModule

print("="*100)
print("VÉRIFICATION COMPLÈTE : Est-ce que le MLP est VRAIMENT chargé ?")
print("="*100)

# Load config
config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Create model
print("\n1️⃣  Création du modèle...")
model = MultiPollutantLightningModule(config)

# Load checkpoint
ckpt_path = config['model']['checkpoint_path']
print(f"\n2️⃣  Chargement du checkpoint: {ckpt_path.split('/')[-1]}")
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Load with strict=False
result = model.load_state_dict(checkpoint['state_dict'], strict=False)

print(f"\n3️⃣  Résultat du chargement:")
print(f"   Missing keys: {len(result.missing_keys)}")
for k in result.missing_keys:
    print(f"      - {k}")

print(f"\n   Unexpected keys: {len(result.unexpected_keys)}")
for k in result.unexpected_keys[:10]:
    print(f"      - {k}")

# CRITICAL: Verify MLP weights are loaded
print("\n" + "="*100)
print("4️⃣  VÉRIFICATION CRITIQUE : MLP Block 0")
print("="*100)

# Get checkpoint MLP weights
ckpt_mlp_fc1_weight = checkpoint['state_dict']['model.climax.blocks.0.mlp.fc1.weight']
ckpt_mlp_fc1_bias = checkpoint['state_dict']['model.climax.blocks.0.mlp.fc1.bias']
ckpt_mlp_fc2_weight = checkpoint['state_dict']['model.climax.blocks.0.mlp.fc2.weight']
ckpt_mlp_fc2_bias = checkpoint['state_dict']['model.climax.blocks.0.mlp.fc2.bias']

print(f"\n📦 Checkpoint MLP weights:")
print(f"   fc1.weight: {tuple(ckpt_mlp_fc1_weight.shape)} - sum: {ckpt_mlp_fc1_weight.sum():.6f}")
print(f"   fc1.bias:   {tuple(ckpt_mlp_fc1_bias.shape)} - sum: {ckpt_mlp_fc1_bias.sum():.6f}")
print(f"   fc2.weight: {tuple(ckpt_mlp_fc2_weight.shape)} - sum: {ckpt_mlp_fc2_weight.sum():.6f}")
print(f"   fc2.bias:   {tuple(ckpt_mlp_fc2_bias.shape)} - sum: {ckpt_mlp_fc2_bias.sum():.6f}")

# Get loaded MLP weights from model
try:
    loaded_mlp_fc1_weight = model.model.climax.blocks[0].mlp.fc1.weight
    loaded_mlp_fc1_bias = model.model.climax.blocks[0].mlp.fc1.bias
    loaded_mlp_fc2_weight = model.model.climax.blocks[0].mlp.fc2.weight
    loaded_mlp_fc2_bias = model.model.climax.blocks[0].mlp.fc2.bias

    print(f"\n📥 Loaded MLP weights (in model):")
    print(f"   fc1.weight: {tuple(loaded_mlp_fc1_weight.shape)} - sum: {loaded_mlp_fc1_weight.sum():.6f}")
    print(f"   fc1.bias:   {tuple(loaded_mlp_fc1_bias.shape)} - sum: {loaded_mlp_fc1_bias.sum():.6f}")
    print(f"   fc2.weight: {tuple(loaded_mlp_fc2_weight.shape)} - sum: {loaded_mlp_fc2_weight.sum():.6f}")
    print(f"   fc2.bias:   {tuple(loaded_mlp_fc2_bias.shape)} - sum: {loaded_mlp_fc2_bias.sum():.6f}")

    # Compare
    print(f"\n🔍 Comparaison (torch.allclose):")
    fc1_w_match = torch.allclose(ckpt_mlp_fc1_weight, loaded_mlp_fc1_weight)
    fc1_b_match = torch.allclose(ckpt_mlp_fc1_bias, loaded_mlp_fc1_bias)
    fc2_w_match = torch.allclose(ckpt_mlp_fc2_weight, loaded_mlp_fc2_weight)
    fc2_b_match = torch.allclose(ckpt_mlp_fc2_bias, loaded_mlp_fc2_bias)

    print(f"   fc1.weight: {fc1_w_match} {'✅' if fc1_w_match else '❌'}")
    print(f"   fc1.bias:   {fc1_b_match} {'✅' if fc1_b_match else '❌'}")
    print(f"   fc2.weight: {fc2_w_match} {'✅' if fc2_w_match else '❌'}")
    print(f"   fc2.bias:   {fc2_b_match} {'✅' if fc2_b_match else '❌'}")

    all_match = fc1_w_match and fc1_b_match and fc2_w_match and fc2_b_match

except AttributeError as e:
    print(f"\n❌ ERREUR : Impossible d'accéder au MLP !")
    print(f"   {str(e)}")
    print("\n   Cela signifie que l'architecture n'a PAS de fc1/fc2 (encore Sequential?)")
    all_match = False

# Check HEAD
print("\n" + "="*100)
print("5️⃣  HEAD (Prediction Layer)")
print("="*100)

try:
    loaded_head_weight = model.model.climax.head.weight
    loaded_head_bias = model.model.climax.head.bias

    print(f"\n📥 Loaded HEAD:")
    print(f"   head.weight: {tuple(loaded_head_weight.shape)} - sum: {loaded_head_weight.sum():.6f}")
    print(f"   head.bias:   {tuple(loaded_head_bias.shape)} - sum: {loaded_head_bias.sum():.6f}")

    # Check if it matches checkpoint head.4 (last layer of Sequential)
    if 'model.climax.head.4.weight' in checkpoint['state_dict']:
        ckpt_head4_weight = checkpoint['state_dict']['model.climax.head.4.weight']
        ckpt_head4_bias = checkpoint['state_dict']['model.climax.head.4.bias']

        print(f"\n📦 Checkpoint HEAD.4 (last layer):")
        print(f"   head.4.weight: {tuple(ckpt_head4_weight.shape)} - sum: {ckpt_head4_weight.sum():.6f}")
        print(f"   head.4.bias:   {tuple(ckpt_head4_bias.shape)} - sum: {ckpt_head4_bias.sum():.6f}")

        head_w_match = torch.allclose(ckpt_head4_weight, loaded_head_weight)
        head_b_match = torch.allclose(ckpt_head4_bias, loaded_head_bias)

        print(f"\n🔍 HEAD matches checkpoint head.4?")
        print(f"   weight: {head_w_match} {'✅' if head_w_match else '❌ RANDOM'}")
        print(f"   bias:   {head_b_match} {'✅' if head_b_match else '❌ RANDOM'}")
    else:
        print("   ⚠️  Checkpoint n'a pas head.4")
        head_w_match = False
        head_b_match = False

except AttributeError as e:
    print(f"\n❌ ERREUR : Impossible d'accéder à la HEAD !")
    print(f"   {str(e)}")
    head_w_match = False
    head_b_match = False

# Final verdict
print("\n" + "="*100)
print("VERDICT FINAL")
print("="*100)

if all_match:
    print("\n✅✅✅ MLP BLOCK 0 : PARFAITEMENT CHARGÉ !")
    print("   → 4,722,432 paramètres du checkpoint")
    print("   → val_loss initiale attendue : ~0.40-0.50")
else:
    print("\n❌❌❌ MLP BLOCK 0 : PAS CHARGÉ !")
    print("   → Architecture n'a pas fc1/fc2")
    print("   → val_loss initiale attendue : ~2.0+")

if head_w_match and head_b_match:
    print("\n✅ HEAD : Chargée du checkpoint")
else:
    print("\n❌ HEAD : Random (46,140 params)")
    print("   → Impact modéré sur val_loss")

print("\n" + "="*100)

# Estimate val_loss
if all_match and (head_w_match and head_b_match):
    print("🎯 Estimation val_loss initiale : ~0.36-0.40 (quasi-checkpoint)")
elif all_match and not (head_w_match or head_b_match):
    print("🎯 Estimation val_loss initiale : ~0.40-0.60 (MLP ok, HEAD random)")
else:
    print("🎯 Estimation val_loss initiale : ~2.0+ (MLP random)")

print("="*100)

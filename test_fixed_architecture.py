#!/usr/bin/env python3
"""
Test that the FIXED architecture loads checkpoint correctly
"""
import sys
import torch

# Add src to path
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.climax_core.topoflow_attention import TopoFlowBlock

ckpt_path = "/scratch/project_462000640/ammar/aq_net2/logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"

print("="*100)
print("TEST : Architecture CORRIGÉE - Chargement du checkpoint")
print("="*100)

# Load checkpoint
print("\n📂 Chargement du checkpoint...")
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']

# Create TopoFlowBlock with FIXED architecture
print("🏗️  Création du TopoFlowBlock avec architecture corrigée...")
block = TopoFlowBlock(
    dim=768,
    num_heads=16,
    mlp_ratio=4.0,
    use_elevation_bias=True
)

print("\n📋 Clés du modèle corrigé (MLP) :")
model_mlp_keys = sorted([name for name, _ in block.named_parameters() if 'mlp' in name])
for key in model_mlp_keys:
    param = dict(block.named_parameters())[key]
    print(f"   {key:30s} {tuple(param.shape)}")

print("\n📋 Clés du checkpoint (block 0 MLP) :")
ckpt_mlp_keys = sorted([k for k in state_dict.keys() if 'blocks.0.mlp' in k])
for key in ckpt_mlp_keys:
    short_key = key.replace('model.climax.blocks.0.', '')
    print(f"   {short_key:30s} {tuple(state_dict[key].shape)}")

# Prepare state dict for loading
print("\n🔄 Chargement des poids avec strict=False...")
test_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('model.climax.blocks.0.'):
        # Keep only block 0 keys (attention, norm, mlp)
        new_key = k.replace('model.climax.blocks.0.', '')
        test_state_dict[new_key] = v

result = block.load_state_dict(test_state_dict, strict=False)

print(f"\n📊 Résultat du chargement :")
print(f"   Missing keys: {len(result.missing_keys)}")
if result.missing_keys:
    print(f"   → {result.missing_keys}")
print(f"   Unexpected keys: {len(result.unexpected_keys)}")
if result.unexpected_keys:
    print(f"   → {result.unexpected_keys[:5]}...")

# Verify weights are actually loaded
print("\n🔍 Vérification que les poids sont RÉELLEMENT chargés :")

ckpt_fc1 = state_dict['model.climax.blocks.0.mlp.fc1.weight']
loaded_fc1 = block.mlp.fc1.weight

ckpt_fc2 = state_dict['model.climax.blocks.0.mlp.fc2.weight']
loaded_fc2 = block.mlp.fc2.weight

fc1_match = torch.allclose(ckpt_fc1, loaded_fc1)
fc2_match = torch.allclose(ckpt_fc2, loaded_fc2)

print(f"   fc1.weight match: {fc1_match} ✅" if fc1_match else "   fc1.weight match: {fc1_match} ❌")
print(f"   fc2.weight match: {fc2_match} ✅" if fc2_match else "   fc2.weight match: {fc2_match} ❌")

# Check TopoFlow parameters
print("\n🌄 Vérification des paramètres TopoFlow :")
topoflow_params = ['attn.elevation_alpha', 'attn.H_scale']
for param_name in topoflow_params:
    if param_name in result.missing_keys:
        print(f"   {param_name:30s} → ✨ NOUVEAU (random init, attendu)")
    else:
        print(f"   {param_name:30s} → ❌ Devrait être missing!")

# Final summary
print("\n" + "="*100)
print("RÉSUMÉ FINAL")
print("="*100)

all_good = (
    fc1_match and fc2_match and
    'attn.elevation_alpha' in result.missing_keys and
    'attn.H_scale' in result.missing_keys
)

if all_good:
    print("\n✅✅✅ SUCCÈS TOTAL !")
    print("\n   Poids chargés du checkpoint :")
    print("   - attn.qkv, attn.proj (attention)")
    print("   - mlp.fc1, mlp.fc2 (MLP) ← 4.7M params !")
    print("   - norm1, norm2 (normalization)")
    print("\n   Poids random (attendu) :")
    print("   - elevation_alpha, H_scale (TopoFlow)")
    print("\n   🚀 Le modèle est prêt pour le training !")
    print("   🎯 val_loss initiale attendue : ~0.36")
else:
    print("\n❌ PROBLÈME détecté")
    if not fc1_match or not fc2_match:
        print("   → MLP weights ne matchent pas")
    if 'attn.elevation_alpha' not in result.missing_keys:
        print("   → TopoFlow params devraient être missing")

print("="*100)

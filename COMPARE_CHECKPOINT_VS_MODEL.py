"""
Comparer les clés exactes du checkpoint vs le modèle actuel
"""
import torch
import sys
sys.path.insert(0, 'src')
import yaml

print("="*80)
print("🔍 COMPARAISON: Checkpoint vs Modèle actuel")
print("="*80)

# Load checkpoint
ckpt_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
ckpt_state_dict = ckpt['state_dict']

print(f"\n1️⃣ CHECKPOINT:")
print(f"   Total keys: {len(ckpt_state_dict)}")

# Remove 'model.' prefix
ckpt_keys_fixed = set()
for key in ckpt_state_dict.keys():
    if key.startswith('model.'):
        ckpt_keys_fixed.add(key[6:])
    else:
        ckpt_keys_fixed.add(key)

print(f"   Keys après suppression du préfixe 'model.': {len(ckpt_keys_fixed)}")

# Create current model
from model_multipollutants import MultiPollutantModel
with open('configs/config_all_pollutants.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = MultiPollutantModel(config)
model_keys = set(dict(model.named_parameters()).keys())

print(f"\n2️⃣ MODÈLE ACTUEL:")
print(f"   Total keys: {len(model_keys)}")

# Find differences
only_in_ckpt = ckpt_keys_fixed - model_keys
only_in_model = model_keys - ckpt_keys_fixed
common = ckpt_keys_fixed & model_keys

print(f"\n3️⃣ DIFFÉRENCES:")
print(f"   Clés communes: {len(common)}")
print(f"   Seulement dans checkpoint: {len(only_in_ckpt)}")
print(f"   Seulement dans modèle actuel: {len(only_in_model)}")

if only_in_ckpt:
    print(f"\n   ❌ Clés SEULEMENT dans le checkpoint (unexpected):")
    for k in sorted(only_in_ckpt)[:20]:
        print(f"      {k}")

if only_in_model:
    print(f"\n   ⚠️  Clés SEULEMENT dans le modèle actuel (missing):")
    for k in sorted(only_in_model)[:20]:
        print(f"      {k}")

# Check for architecture differences
print(f"\n4️⃣ ANALYSE ARCHITECTURE:")

# Check patch embedding
ckpt_patch_embed = [k for k in ckpt_keys_fixed if 'token_embeds' in k or 'patch_embed' in k]
model_patch_embed = [k for k in model_keys if 'token_embeds' in k or 'patch_embed' in k]

print(f"\n   Patch Embedding:")
print(f"      Checkpoint: {len(ckpt_patch_embed)} keys")
for k in sorted(ckpt_patch_embed):
    print(f"         {k}")
print(f"      Modèle actuel: {len(model_patch_embed)} keys")
for k in sorted(model_patch_embed):
    print(f"         {k}")

# Check head
ckpt_head = [k for k in ckpt_keys_fixed if 'head' in k]
model_head = [k for k in model_keys if 'head' in k]

print(f"\n   Head:")
print(f"      Checkpoint: {len(ckpt_head)} keys")
for k in sorted(ckpt_head)[:10]:
    print(f"         {k}")
print(f"      Modèle actuel: {len(model_head)} keys")
for k in sorted(model_head)[:10]:
    print(f"         {k}")

print("\n" + "="*80)
print("💡 HYPOTHÈSES:")
print("="*80)
if only_in_ckpt or only_in_model:
    print("❌ L'architecture a changé entre septembre 2024 et maintenant!")
    print("   Le checkpoint version_47 n'est PAS compatible avec le code actuel.")
    print("")
    print("   OPTIONS:")
    print("   1. Trouver le code de septembre 2024 (git tag/commit)")
    print("   2. Créer un nouveau checkpoint baseline avec le code actuel")
    print("   3. Adapter le code actuel pour être compatible avec le checkpoint")
else:
    print("✅ L'architecture semble identique")
    print("   Le problème est ailleurs (normalisation? forward pass?)")
print("="*80)

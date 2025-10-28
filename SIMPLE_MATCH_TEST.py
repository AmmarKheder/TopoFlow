"""
TEST SIMPLE: Est-ce que le modèle actuel match parfaitement le checkpoint version_47 ?
"""
import torch

print("="*80)
print("TEST: COMPARAISON MODÈLE vs CHECKPOINT VERSION_47")
print("="*80)

# 1. Load checkpoint
ckpt_path = 'logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt'
print(f"\n1. Chargement checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Get all model keys from checkpoint
ckpt_keys = [k for k in checkpoint['state_dict'].keys() if k.startswith('model.climax')]
ckpt_keys_clean = [k.replace('model.', '') for k in ckpt_keys]

print(f"   Checkpoint a {len(ckpt_keys)} clés modèle")
print(f"   Premières clés:")
for k in sorted(ckpt_keys_clean)[:5]:
    print(f"     - {k}")

# 2. Create current model and get its keys
print("\n2. Création du modèle actuel...")
import sys
sys.path.insert(0, 'src')
import yaml
from model_multipollutants import MultiPollutantModel

with open('configs/config_all_pollutants.yaml') as f:
    config = yaml.safe_load(f)

model = MultiPollutantModel(config)

# Get model state dict keys
model_keys = list(model.state_dict().keys())
climax_keys = [k for k in model_keys if k.startswith('climax')]

print(f"   Modèle actuel a {len(climax_keys)} clés climax")
print(f"   Premières clés:")
for k in sorted(climax_keys)[:5]:
    print(f"     - {k}")

# 3. Compare keys
print("\n3. Comparaison des clés...")

# Keys in checkpoint but NOT in model
missing_in_model = set(ckpt_keys_clean) - set(climax_keys)
# Keys in model but NOT in checkpoint
extra_in_model = set(climax_keys) - set(ckpt_keys_clean)

print(f"\n   Clés dans checkpoint ABSENTES du modèle: {len(missing_in_model)}")
if missing_in_model:
    for k in sorted(list(missing_in_model))[:10]:
        print(f"     - {k}")

print(f"\n   Clés dans modèle ABSENTES du checkpoint: {len(extra_in_model)}")
if extra_in_model:
    for k in sorted(list(extra_in_model))[:10]:
        print(f"     - {k}")

# 4. Try loading
print("\n4. Test de chargement...")
state_dict = {}
for key, value in checkpoint['state_dict'].items():
    if key.startswith('model.'):
        state_dict[key[6:]] = value
    else:
        state_dict[key] = value

result = model.load_state_dict(state_dict, strict=False)

print(f"   Missing keys: {len(result.missing_keys)}")
if result.missing_keys:
    print(f"   Premières missing:")
    for k in result.missing_keys[:5]:
        print(f"     - {k}")

print(f"   Unexpected keys: {len(result.unexpected_keys)}")
if result.unexpected_keys:
    print(f"   Premières unexpected:")
    for k in result.unexpected_keys[:5]:
        print(f"     - {k}")

# 5. Conclusion
print("\n" + "="*80)
print("DIAGNOSTIC:")
print("="*80)

if len(result.missing_keys) == 0 and len(result.unexpected_keys) <= 1:
    print("✅ LE MODÈLE MATCH PARFAITEMENT LE CHECKPOINT!")
    print("   - Tous les poids du checkpoint peuvent être chargés")
    print("   - L'architecture est identique")
    print("")
    print("❓ ALORS POURQUOI LA TRAIN_LOSS EST HAUTE (3.2-3.8) ?")
    print("")
    print("Possibilités:")
    print("  1. L'optimizer est réinitialisé (pas de resume, juste fine-tuning)")
    print("  2. Le learning rate repart du début (pas celui du checkpoint)")
    print("  3. Les données ne sont pas normalisées de la même façon")
    print("  4. Il y a une différence dans le forward pass (variables, ordre, etc.)")
    print("")
    print("PROCHAINE ÉTAPE: Faire un forward pass et comparer la loss")

else:
    print("❌ LE MODÈLE NE MATCH PAS PARFAITEMENT")
    print(f"   - {len(result.missing_keys)} clés manquantes")
    print(f"   - {len(result.unexpected_keys)} clés inattendues")
    print("")
    print("PROBLÈME D'ARCHITECTURE!")
    print("Il faut corriger l'architecture avant de continuer.")

print("="*80)

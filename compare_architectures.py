"""Compare architecture actuelle vs checkpoint pour trouver les différences."""
import torch
import yaml
import sys
sys.path.insert(0, 'src')

# Force clean import
for module in list(sys.modules.keys()):
    if 'climax' in module or 'model_multi' in module:
        del sys.modules[module]

from model_multipollutants import MultiPollutantModel

print("="*70)
print("COMPARAISON ARCHITECTURE ACTUELLE vs CHECKPOINT")
print("="*70)

# Load config
with open('configs/config_all_pollutants.yaml') as f:
    config = yaml.safe_load(f)

# Create current model
print("\n1. Création du modèle ACTUEL...")
model_current = MultiPollutantModel(config)

# Load checkpoint
print("\n2. Chargement du CHECKPOINT...")
checkpoint_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict_ckpt = checkpoint['state_dict']

print(f"\n{'='*70}")
print("COMPARAISON DES CLÉS")
print(f"{'='*70}")

# Get keys from both
keys_current = set(dict(model_current.named_parameters()).keys())
keys_current.update(dict(model_current.named_buffers()).keys())

keys_ckpt = set(state_dict_ckpt.keys())

print(f"\nNombre de paramètres:")
print(f"  Modèle actuel: {len(keys_current)} clés")
print(f"  Checkpoint: {len(keys_ckpt)} clés")

# Find differences
only_in_current = keys_current - keys_ckpt
only_in_ckpt = keys_ckpt - keys_current

print(f"\n{'='*70}")
print("CLÉS UNIQUEMENT DANS LE MODÈLE ACTUEL (nouveaux paramètres):")
print(f"{'='*70}")
if only_in_current:
    for key in sorted(only_in_current):
        print(f"  + {key}")
else:
    print("  (aucune)")

print(f"\n{'='*70}")
print("CLÉS UNIQUEMENT DANS LE CHECKPOINT (paramètres manquants):")
print(f"{'='*70}")
if only_in_ckpt:
    for key in sorted(only_in_ckpt):
        # Check if it's just a prefix issue
        key_without_prefix = key.replace('model.', '') if key.startswith('model.') else key
        if key_without_prefix in keys_current:
            print(f"  - {key} → EXISTE DANS ACTUEL COMME: {key_without_prefix}")
        else:
            print(f"  - {key}")
else:
    print("  (aucune)")

# Check specific important components
print(f"\n{'='*70}")
print("VÉRIFICATION DES COMPOSANTS CRITIQUES:")
print(f"{'='*70}")

# Check decoder head
print("\n📌 DECODER HEAD:")
head_keys_current = [k for k in keys_current if 'head' in k]
head_keys_ckpt = [k for k in keys_ckpt if 'head' in k]

print(f"  Modèle actuel ({len(head_keys_current)} paramètres):")
for k in sorted(head_keys_current)[:10]:
    param = dict(model_current.named_parameters()).get(k)
    if param is not None:
        print(f"    {k}: shape={param.shape}")

print(f"\n  Checkpoint ({len(head_keys_ckpt)} paramètres):")
for k in sorted(head_keys_ckpt)[:10]:
    print(f"    {k}: shape={state_dict_ckpt[k].shape}")

# Check if head is MLP or single Linear
print("\n  Type de head:")
if len(head_keys_current) > 2:
    print(f"    Modèle actuel: MLP (multi-layer, {len(head_keys_current)} params)")
else:
    print(f"    Modèle actuel: Linear simple ({len(head_keys_current)} params)")

if len(head_keys_ckpt) > 2:
    print(f"    Checkpoint: MLP (multi-layer, {len(head_keys_ckpt)} params)")
else:
    print(f"    Checkpoint: Linear simple ({len(head_keys_ckpt)} params)")

# Check TopoFlow
print("\n📌 TOPOFLOW:")
topo_keys_current = [k for k in keys_current if 'elevation' in k or 'H_scale' in k]
topo_keys_ckpt = [k for k in keys_ckpt if 'elevation' in k or 'H_scale' in k]

print(f"  Modèle actuel: {len(topo_keys_current)} paramètres TopoFlow")
for k in sorted(topo_keys_current):
    print(f"    {k}")

print(f"  Checkpoint: {len(topo_keys_ckpt)} paramètres TopoFlow")
if topo_keys_ckpt:
    for k in sorted(topo_keys_ckpt):
        print(f"    {k}")
else:
    print(f"    (aucun - normal, TopoFlow est nouveau)")

# Try to load checkpoint
print(f"\n{'='*70}")
print("TEST DE CHARGEMENT DU CHECKPOINT:")
print(f"{'='*70}")

result = model_current.load_state_dict(state_dict_ckpt, strict=False)

print(f"\nRésultat:")
print(f"  Missing keys: {len(result.missing_keys)}")
if result.missing_keys:
    print(f"    {result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}")

print(f"  Unexpected keys: {len(result.unexpected_keys)}")
if result.unexpected_keys:
    print(f"    {result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}")

print(f"\n{'='*70}")
print("CONCLUSION:")
print(f"{'='*70}")

if len(result.unexpected_keys) > 0:
    print("\n⚠️  PROBLÈME: Unexpected keys détectées!")
    print(f"   Le checkpoint a des paramètres que le modèle actuel n'attend pas.")
    print(f"   Cela suggère une incompatibilité d'architecture.")

    # Check if it's a prefix issue
    unexpected_without_prefix = [k.replace('model.', '') for k in result.unexpected_keys if k.startswith('model.')]
    if all(k in keys_current for k in unexpected_without_prefix):
        print(f"\n   ✅ C'EST JUSTE UN PROBLÈME DE PRÉFIXE 'model.'")
        print(f"   Toutes les clés existent sans le préfixe.")
    else:
        print(f"\n   ❌ Ce n'est PAS qu'un problème de préfixe.")
        print(f"   Il y a de vraies différences d'architecture.")

if len(result.missing_keys) > 2:  # Plus que elevation_alpha et H_scale
    print("\n⚠️  PROBLÈME: Trop de missing keys!")
    print(f"   Le modèle actuel attend des paramètres que le checkpoint n'a pas.")

if len(result.unexpected_keys) == 0 and len(result.missing_keys) <= 2:
    print("\n✅ PAS DE PROBLÈME DE COMPATIBILITÉ!")
    print(f"   Le checkpoint se charge correctement.")
    print(f"   Les {len(result.missing_keys)} missing keys sont attendues (TopoFlow).")

print(f"\n{'='*70}")

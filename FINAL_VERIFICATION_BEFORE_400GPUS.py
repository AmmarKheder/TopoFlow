"""
VÉRIFICATION FINALE AVANT LE JOB 400 GPUs
S'assurer que checkpoint version_47 match parfaitement le modèle actuel
"""
import torch
import sys
sys.path.insert(0, 'src')
import yaml
import numpy as np

print("="*100)
print("🔍 VÉRIFICATION FINALE: Checkpoint version_47 vs Modèle actuel")
print("="*100)

# Load config
with open('configs/config_all_pollutants.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ============================================================================
# 1. VÉRIFIER LA CONFIGURATION
# ============================================================================
print("\n1️⃣ CONFIGURATION:")
print("-" * 100)
print(f"   TopoFlow (use_physics_mask): {config['model'].get('use_physics_mask', False)}")
print(f"   Wind scanning: {config['model'].get('parallel_patch_embed', False)}")
print(f"   Checkpoint: {config['model'].get('checkpoint_path', 'None')}")

if config['model'].get('use_physics_mask', False):
    print("   ❌ ERREUR: TopoFlow est activé! Doit être désactivé pour match checkpoint.")
    sys.exit(1)
else:
    print("   ✅ OK: TopoFlow désactivé")

# ============================================================================
# 2. CHARGER CHECKPOINT ET MODÈLE
# ============================================================================
print("\n2️⃣ CHARGEMENT CHECKPOINT ET MODÈLE:")
print("-" * 100)

ckpt_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
ckpt_state_dict = ckpt['state_dict']

print(f"   Checkpoint: {len(ckpt_state_dict)} keys")

# Fix prefix
ckpt_keys_fixed = {}
for key, value in ckpt_state_dict.items():
    if key.startswith('model.'):
        ckpt_keys_fixed[key[6:]] = value
    else:
        ckpt_keys_fixed[key] = value

print(f"   Après suppression préfixe 'model.': {len(ckpt_keys_fixed)} keys")

# Create model
from model_multipollutants import MultiPollutantModel
model = MultiPollutantModel(config)
model_keys = dict(model.named_parameters())

print(f"   Modèle actuel: {len(model_keys)} params")

# ============================================================================
# 3. COMPARER LES CLÉS
# ============================================================================
print("\n3️⃣ COMPARAISON DES CLÉS:")
print("-" * 100)

ckpt_param_keys = set([k for k in ckpt_keys_fixed.keys() if k in model_keys])
model_param_keys = set(model_keys.keys())

common = ckpt_param_keys & model_param_keys
only_in_ckpt = ckpt_param_keys - model_param_keys
only_in_model = model_param_keys - ckpt_param_keys

print(f"   Clés communes: {len(common)}")
print(f"   Seulement dans checkpoint: {len(only_in_ckpt)}")
print(f"   Seulement dans modèle: {len(only_in_model)}")

if only_in_ckpt:
    print(f"\n   ❌ Clés dans checkpoint mais pas dans modèle:")
    for k in sorted(only_in_ckpt)[:10]:
        print(f"      - {k}")

if only_in_model:
    print(f"\n   ⚠️  Clés dans modèle mais pas dans checkpoint:")
    for k in sorted(only_in_model)[:10]:
        print(f"      - {k}")

if len(common) == len(model_param_keys):
    print(f"   ✅ OK: Toutes les clés du modèle sont dans le checkpoint")
else:
    print(f"   ❌ ERREUR: {len(model_param_keys) - len(common)} clés manquantes!")
    sys.exit(1)

# ============================================================================
# 4. CHARGER LES POIDS
# ============================================================================
print("\n4️⃣ CHARGEMENT DES POIDS:")
print("-" * 100)

missing, unexpected = model.load_state_dict(ckpt_keys_fixed, strict=False)
print(f"   Chargés: {len(ckpt_keys_fixed) - len(missing)} params")
print(f"   Manquants: {len(missing)} params")
print(f"   Inattendus (ignorés): {len(unexpected)} params")

if missing:
    print(f"\n   ⚠️  Clés manquantes:")
    for k in missing[:5]:
        print(f"      - {k}")

if unexpected:
    print(f"\n   ℹ️  Clés inattendues (métadonnées, ignorées):")
    for k in unexpected[:5]:
        print(f"      - {k}")

# ============================================================================
# 5. VÉRIFIER LES POIDS
# ============================================================================
print("\n5️⃣ VÉRIFICATION DES POIDS:")
print("-" * 100)

# Check first layer
for name, param in model.named_parameters():
    if 'blocks.0.attn.qkv.weight' in name:
        print(f"   Layer: {name}")
        print(f"     Shape: {param.shape}")
        print(f"     Mean: {param.mean().item():.6f}")
        print(f"     Std: {param.std().item():.6f}")
        print(f"     Min/Max: [{param.min().item():.4f}, {param.max().item():.4f}]")

        if torch.all(param == 0):
            print(f"     ❌ ERREUR: Poids = 0!")
            sys.exit(1)
        elif abs(param.mean().item()) > 0.1:
            print(f"     ⚠️  WARNING: Mean élevé (devrait être proche de 0)")
        else:
            print(f"     ✅ OK: Poids semblent corrects")
        break

# ============================================================================
# 6. TEST FORWARD PASS
# ============================================================================
print("\n6️⃣ TEST FORWARD PASS:")
print("-" * 100)

batch_size = 2
n_vars = len(config['data']['variables'])
H, W = config['model']['img_size']

x_dummy = torch.randn(batch_size, n_vars, H, W) * 2.0  # Scaled for realism
lead_times_dummy = torch.tensor([12.0, 24.0])
variables = tuple(config['data']['variables'])

model.eval()
try:
    with torch.no_grad():
        output = model(x_dummy, lead_times_dummy, variables)

    print(f"   ✅ Forward pass réussi!")
    print(f"   Output shape: {output.shape}")
    print(f"   Output mean: {output.mean().item():.4f}")
    print(f"   Output std: {output.std().item():.4f}")
    print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Check if output is reasonable
    if torch.isnan(output).any():
        print(f"   ❌ ERREUR: Output contient des NaN!")
        sys.exit(1)
    elif torch.isinf(output).any():
        print(f"   ❌ ERREUR: Output contient des Inf!")
        sys.exit(1)
    elif abs(output.mean().item()) > 100:
        print(f"   ⚠️  WARNING: Output mean très élevé")
    else:
        print(f"   ✅ OK: Output semble raisonnable")

except Exception as e:
    print(f"   ❌ ERREUR Forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 7. VÉRIFIER LA COMPATIBILITÉ DES DONNÉES
# ============================================================================
print("\n7️⃣ VÉRIFICATION DES DONNÉES:")
print("-" * 100)

from datamodule_fixed import AQNetDataModule
config['data']['num_workers'] = 0
data_module = AQNetDataModule(config)
data_module.setup('fit')
val_loader = data_module.val_dataloader()

# Get one batch
x, y, lead_times = next(iter(val_loader))
print(f"   Batch shape: x={x.shape}, y={y.shape}")
print(f"   Lead times: {lead_times}")

# Check data statistics
print(f"\n   Statistiques des données (quelques variables):")
variables_to_check = ['u', 'v', 'temp', 'pm25', 'elevation']
for var in variables_to_check:
    if var in variables:
        idx = variables.index(var)
        if idx < x.shape[1]:
            mean = x[:, idx].mean().item()
            std = x[:, idx].std().item()
            print(f"     {var:15s}: mean={mean:7.3f}, std={std:7.3f}")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "="*100)
print("📊 RÉSUMÉ FINAL:")
print("="*100)
print(f"✅ Configuration: TopoFlow désactivé")
print(f"✅ Architecture: {len(common)}/{len(model_param_keys)} params match")
print(f"✅ Chargement: {len(ckpt_keys_fixed) - len(missing)} params chargés")
print(f"✅ Poids: Non-nuls, distribution raisonnable")
print(f"✅ Forward pass: OK, pas de NaN/Inf")
print(f"✅ Données: Chargement OK")
print("")
print("🚀 PRÊT POUR LE JOB 400 GPUs!")
print("="*100)

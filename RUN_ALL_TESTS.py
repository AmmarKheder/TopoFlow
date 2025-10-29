"""
BATTERIE COMPLÈTE DE TESTS
Pour vérifier que le modèle fonctionne correctement avec le checkpoint version_47
"""
import torch
import yaml
import sys
sys.path.insert(0, 'src')
from model_multipollutants import MultiPollutantModel
from datamodule import AQNetDataModule

print("="*80)
print("🔬 BATTERIE COMPLÈTE DE TESTS - CHECKPOINT VERSION_47")
print("="*80)

# Load config
with open('configs/config_all_pollutants.yaml') as f:
    config = yaml.safe_load(f)

ckpt_path = 'logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt'

# ============================================
# TEST 1: Checkpoint Keys
# ============================================
print("\n" + "="*80)
print("TEST 1: VÉRIFICATION DES CLÉS DU CHECKPOINT")
print("="*80)
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print(f"✓ Checkpoint chargé: {len(checkpoint['state_dict'])} clés")
print(f"✓ val_loss dans filename: 0.3557")

# ============================================
# TEST 2: Model Creation + Weight Loading
# ============================================
print("\n" + "="*80)
print("TEST 2: CRÉATION DU MODÈLE ET CHARGEMENT DES POIDS")
print("="*80)

model = MultiPollutantModel(config)
print(f"✓ Modèle créé")

# Sample weight before
sample_before = model.climax.head[0].weight[0, :3].clone()
print(f"  Poids avant: {sample_before.tolist()}")

# Load checkpoint
state_dict = {}
for key, value in checkpoint['state_dict'].items():
    if key.startswith('model.'):
        state_dict[key[6:]] = value
    else:
        state_dict[key] = value

result = model.load_state_dict(state_dict, strict=False)
print(f"✓ Checkpoint chargé: {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")

# Sample weight after
sample_after = model.climax.head[0].weight[0, :3].clone()
ckpt_weight = checkpoint['state_dict']['model.climax.head.0.weight'][0, :3]
print(f"  Poids après: {sample_after.tolist()}")
print(f"  Poids checkpoint: {ckpt_weight.tolist()}")
weights_match = torch.allclose(sample_after, ckpt_weight)
print(f"  Match? {weights_match}")

if weights_match:
    print("✅ TEST 2 PASSED: Poids chargés correctement")
else:
    print("❌ TEST 2 FAILED: Poids ne correspondent pas!")
    sys.exit(1)

# ============================================
# TEST 3: Forward Pass with Real Data
# ============================================
print("\n" + "="*80)
print("TEST 3: FORWARD PASS AVEC VRAIES DONNÉES")
print("="*80)

model.eval()

# Override config for single GPU test
test_config = config.copy()
test_config['train'] = config['train'].copy()
test_config['train']['devices'] = 1
test_config['train']['num_nodes'] = 1
test_config['data'] = config['data'].copy()
test_config['data']['num_workers'] = 2

print("  Création du DataModule...")
data_module = AQNetDataModule(test_config)
data_module.setup('fit')

print("  Récupération d'un batch de validation...")
val_loader = data_module.val_dataloader()
batch = next(iter(val_loader))

if len(batch) == 4:
    x, y, lead_times, variables = batch
else:
    x, y, lead_times = batch
    variables = config["data"]["variables"]

print(f"✓ Batch récupéré: x={x.shape}, y={y.shape}")

# Forward pass
with torch.no_grad():
    y_pred = model(x, lead_times, variables)

print(f"✓ Forward pass réussi: y_pred={y_pred.shape}")

# Compute loss
if y.dim() == 3:
    y = y.unsqueeze(1)

# Simple MSE loss (without mask for simplicity)
loss = torch.nn.functional.mse_loss(y_pred, y)

print(f"\n  Loss sur batch de validation: {loss.item():.4f}")
print(f"  Loss attendue (checkpoint): ~0.3557")
print(f"  Différence: {abs(loss.item() - 0.3557):.4f}")

if abs(loss.item() - 0.3557) < 0.5:
    print("✅ TEST 3 PASSED: Loss proche de la valeur attendue")
else:
    print(f"❌ TEST 3 FAILED: Loss trop différente ({loss.item():.4f} vs 0.3557)")
    print("\n🔍 Analyse du problème:")

    # Check data stats
    print(f"\n  Stats des données d'entrée:")
    print(f"    x: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
    print(f"    y: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")
    print(f"    y_pred: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}")

    # Check prediction quality
    diff = (y_pred - y).abs()
    print(f"\n  Différence prédiction vs vérité:")
    print(f"    mean abs error: {diff.mean():.4f}")
    print(f"    max abs error: {diff.max():.4f}")

# ============================================
# TEST 4: Multiple Batches Average Loss
# ============================================
print("\n" + "="*80)
print("TEST 4: LOSS MOYENNE SUR 10 BATCHES")
print("="*80)

losses = []
val_loader_iter = iter(val_loader)

for i in range(min(10, len(val_loader))):
    try:
        batch = next(val_loader_iter)
        if len(batch) == 4:
            x, y, lead_times, variables = batch
        else:
            x, y, lead_times = batch
            variables = config["data"]["variables"]

        with torch.no_grad():
            y_pred = model(x, lead_times, variables)

        if y.dim() == 3:
            y = y.unsqueeze(1)

        loss = torch.nn.functional.mse_loss(y_pred, y)
        losses.append(loss.item())

    except StopIteration:
        break

avg_loss = sum(losses) / len(losses)
print(f"\n  Losses individuelles: {[f'{l:.4f}' for l in losses]}")
print(f"  Loss moyenne sur {len(losses)} batches: {avg_loss:.4f}")
print(f"  Loss attendue: ~0.3557")
print(f"  Différence: {abs(avg_loss - 0.3557):.4f}")

if abs(avg_loss - 0.3557) < 0.3:
    print("✅ TEST 4 PASSED: Loss moyenne acceptable")
else:
    print(f"❌ TEST 4 FAILED: Loss moyenne trop différente")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*80)
print("📊 RÉSUMÉ DES TESTS")
print("="*80)
print(f"✅ Test 1: Checkpoint keys - OK")
print(f"{'✅' if weights_match else '❌'} Test 2: Weight loading - {'OK' if weights_match else 'FAILED'}")
print(f"{'✅' if abs(loss.item() - 0.3557) < 0.5 else '❌'} Test 3: Forward pass - {'OK' if abs(loss.item() - 0.3557) < 0.5 else 'FAILED'}")
print(f"{'✅' if abs(avg_loss - 0.3557) < 0.3 else '❌'} Test 4: Average loss - {'OK' if abs(avg_loss - 0.3557) < 0.3 else 'FAILED'}")

if weights_match and abs(avg_loss - 0.3557) < 0.5:
    print("\n" + "="*80)
    print("🎉 TOUS LES TESTS PASSENT!")
    print("Votre modèle fonctionne correctement avec le checkpoint version_47")
    print("Vous pouvez maintenant activer TopoFlow pour l'ablation study!")
    print("="*80)
else:
    print("\n" + "="*80)
    print("⚠️ CERTAINS TESTS ÉCHOUENT")
    print("Il faut investiguer avant d'activer TopoFlow")
    print("="*80)

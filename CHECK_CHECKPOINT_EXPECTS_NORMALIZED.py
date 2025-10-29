"""
Vérifier si le checkpoint attend des coordonnées normalisées ou pas
On va tester les deux cas et voir lequel donne val_loss ≈ 0.356
"""
import torch
import numpy as np
import sys
sys.path.insert(0, 'src')

print("="*80)
print("🔍 TEST: Le checkpoint attend-il lat2d/lon2d NORMALISÉS?")
print("="*80)

from model_multipollutants import MultiPollutantLightningModule
import yaml

# Load config
with open('configs/config_all_pollutants.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create Lightning module
print("\n1️⃣ Creating model and loading checkpoint...")
lightning_module = MultiPollutantLightningModule(config=config)
ckpt_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"
lightning_module._checkpoint_path_to_load = ckpt_path
lightning_module.setup(stage='fit')
lightning_module.eval()

# Get one validation batch
print("\n2️⃣ Loading validation data...")
from datamodule_fixed import AQNetDataModule
config['data']['num_workers'] = 0
data_module = AQNetDataModule(config)
data_module.setup('fit')
val_loader = data_module.val_dataloader()

# Get first batch
x, y, lead_times = next(iter(val_loader))
print(f"   Batch shape: x={x.shape}, y={y.shape}")

variables = tuple(config['data']['variables'])
lat_idx = variables.index('lat2d')
lon_idx = variables.index('lon2d')

print(f"\n3️⃣ Current data statistics:")
print(f"   lat2d (index {lat_idx}): mean={x[:, lat_idx].mean():.4f}, std={x[:, lat_idx].std():.4f}")
print(f"   lon2d (index {lon_idx}): mean={x[:, lon_idx].mean():.4f}, std={x[:, lon_idx].std():.4f}")

# TEST 1: With current data (coordinates NOT normalized)
print(f"\n4️⃣ TEST 1: Avec coordonnées NON-NORMALISÉES (actuelles)")
with torch.no_grad():
    y_pred = lightning_module.model(x, lead_times, variables)
    china_mask = lightning_module.china_mask.to(dtype=torch.bool).unsqueeze(0).unsqueeze(0)
    loss_1 = lightning_module._masked_mse(y_pred, y, china_mask)
print(f"   Loss: {loss_1.item():.4f}")

# TEST 2: Normalize lat2d and lon2d manually
print(f"\n5️⃣ TEST 2: Avec coordonnées NORMALISÉES")
x_normalized = x.clone()

# Normalize lat2d: (lat - 32.0) / 12.0
x_normalized[:, lat_idx] = (x[:, lat_idx] - 32.0) / 12.0

# Normalize lon2d: (lon - 106.0) / 16.0
x_normalized[:, lon_idx] = (x[:, lon_idx] - 106.0) / 16.0

print(f"   lat2d après normalisation: mean={x_normalized[:, lat_idx].mean():.4f}, std={x_normalized[:, lat_idx].std():.4f}")
print(f"   lon2d après normalisation: mean={x_normalized[:, lon_idx].mean():.4f}, std={x_normalized[:, lon_idx].std():.4f}")

with torch.no_grad():
    y_pred_2 = lightning_module.model(x_normalized, lead_times, variables)
    loss_2 = lightning_module._masked_mse(y_pred_2, y, china_mask)
print(f"   Loss: {loss_2.item():.4f}")

# Compare
print(f"\n" + "="*80)
print(f"RÉSULTATS:")
print(f"="*80)
print(f"Loss sans normalisation des coordonnées:  {loss_1.item():.4f}")
print(f"Loss avec normalisation des coordonnées:  {loss_2.item():.4f}")
print(f"Loss attendu du checkpoint:               0.3557")
print(f"")

if abs(loss_1.item() - 0.3557) < abs(loss_2.item() - 0.3557):
    print(f"✅ CONCLUSION: Le checkpoint attend des coordonnées NON-NORMALISÉES!")
    print(f"   Différence: {abs(loss_1.item() - 0.3557):.4f}")
    print(f"   ACTION: Garder le dataloader actuel (pas de normalisation de lat2d/lon2d)")
elif abs(loss_2.item() - 0.3557) < abs(loss_1.item() - 0.3557):
    print(f"✅ CONCLUSION: Le checkpoint attend des coordonnées NORMALISÉES!")
    print(f"   Différence: {abs(loss_2.item() - 0.3557):.4f}")
    print(f"   ACTION: Modifier le dataloader pour normaliser lat2d/lon2d")
else:
    print(f"❌ PROBLÈME: Aucun des deux ne donne 0.3557!")
    print(f"   Il y a un autre bug dans le modèle ou les données")

print(f"="*80)

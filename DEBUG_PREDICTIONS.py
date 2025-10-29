#!/usr/bin/env python3
"""Vérifier ce que prédit le modèle"""
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule
from src.datamodule_fixed import AQNetDataModule

config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config
config['data']['num_workers'] = 0
ckpt_path = config['model']['checkpoint_path']

print("="*100)
print("VÉRIFICATION PRÉDICTIONS DU MODÈLE")
print("="*100)

# Load model
model = MultiPollutantLightningModule.load_from_checkpoint(ckpt_path, config=config, strict=False)
model.eval()

# Load data
data_module = AQNetDataModule(config)
data_module.setup('fit')
train_loader = data_module.train_dataloader()
batch = next(iter(train_loader))

if len(batch) == 3:
    x, y, lead_times = batch
    variables = config["data"]["variables"]
elif len(batch) == 4:
    x, y, lead_times, variables = batch
else:
    x, y, lead_times, variables, _ = batch

print(f"\n📊 INPUT/TARGET:")
print(f"   Y (target) mean: {y.mean().item():.4f}, std: {y.std().item():.4f}")

# Forward
with torch.no_grad():
    preds = model(x, lead_times, variables)

print(f"\n📊 PRÉDICTIONS:")
print(f"   Shape: {preds.shape}")
print(f"   Mean:  {preds.mean().item():.4f}")
print(f"   Std:   {preds.std().item():.4f}")
print(f"   Min:   {preds.min().item():.4f}")
print(f"   Max:   {preds.max().item():.4f}")

# Calculer MSE directe
mse_direct = ((preds - y) ** 2).mean().item()
print(f"\n📊 MSE DIRECTE (sans masque):")
print(f"   MSE: {mse_direct:.4f}")

# Avec masque China
china_mask = model.china_mask.to(y.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
mse_masked = (((preds - y) ** 2) * china_mask).sum() / china_mask.sum()
print(f"\n📊 MSE AVEC MASQUE CHINA:")
print(f"   MSE: {mse_masked.item():.4f}")

# Comparer avec expected
print(f"\n📊 COMPARAISON:")
print(f"   Loss attendue (checkpoint): 0.3557")
print(f"   Loss actuelle (avec masque): {mse_masked.item():.4f}")
print(f"   Ratio: {mse_masked.item() / 0.3557:.2f}x")

# Vérifier prédictions par polluant
print(f"\n📊 PRÉDICTIONS PAR POLLUANT:")
target_vars = config['data']['target_variables']
for i, name in enumerate(target_vars):
    if i < preds.shape[1]:
        pred_i = preds[:, i]
        y_i = y[:, i]
        mse_i = ((pred_i - y_i) ** 2).mean().item()
        print(f"   {name:6s}: pred_mean={pred_i.mean().item():8.4f}, target_mean={y_i.mean().item():8.4f}, MSE={mse_i:.4f}")

print("\n" + "="*100)

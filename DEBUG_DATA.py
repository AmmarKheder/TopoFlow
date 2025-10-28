#!/usr/bin/env python3
"""Vérifier les stats des données - normalisation a peut-être changé"""
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.datamodule_fixed import AQNetDataModule

config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config
config['data']['num_workers'] = 0

print("="*100)
print("VÉRIFICATION STATS DES DONNÉES")
print("="*100)

data_module = AQNetDataModule(config)
data_module.setup('fit')
train_loader = data_module.train_dataloader()

# Prendre 1 batch
batch = next(iter(train_loader))

if len(batch) == 3:
    x, y, lead_times = batch
elif len(batch) == 4:
    x, y, lead_times, variables = batch
else:
    x, y, lead_times, variables, _ = batch

print(f"\n📊 BATCH INFO:")
print(f"   Input shape:  {x.shape}")
print(f"   Target shape: {y.shape}")
print(f"   Input dtype:  {x.dtype}")
print(f"   Target dtype: {y.dtype}")

print(f"\n📊 INPUT STATISTICS (X):")
print(f"   Mean: {x.mean().item():.6f}")
print(f"   Std:  {x.std().item():.6f}")
print(f"   Min:  {x.min().item():.6f}")
print(f"   Max:  {x.max().item():.6f}")

print(f"\n📊 TARGET STATISTICS (Y):")
print(f"   Mean: {y.mean().item():.6f}")
print(f"   Std:  {y.std().item():.6f}")
print(f"   Min:  {y.min().item():.6f}")
print(f"   Max:  {y.max().item():.6f}")

# Check par variable
print(f"\n📊 TARGET PAR POLLUANT:")
target_vars = config['data']['target_variables']
for i, name in enumerate(target_vars):
    if i < y.shape[1]:
        y_i = y[:, i]
        print(f"   {name:6s}: mean={y_i.mean().item():8.4f}, std={y_i.std().item():7.4f}, min={y_i.min().item():8.4f}, max={y_i.max().item():8.4f}")

print("\n" + "="*100)
print("💡 ANALYSE:")
print("   Si mean/std sont très différents de 0/1, il y a peut-être un problème de normalisation")
print("   Si min/max contiennent des valeurs extrêmes, ça peut causer une loss élevée")
print("="*100)

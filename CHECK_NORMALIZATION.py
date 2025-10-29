"""
Vérifier que les NORM_STATS d'aujourd'hui correspondent exactement à ceux de septembre
"""
import torch
import sys

# 1. Charger le checkpoint version_47
print("="*80)
print("VÉRIFICATION DES NORM_STATS")
print("="*80)

ckpt_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# 2. Extraire les variables utilisées dans le checkpoint
checkpoint_variables = ckpt['hyper_parameters']['config']['data']['variables']
print(f"\n✅ Variables dans checkpoint version_47:")
for i, var in enumerate(checkpoint_variables):
    print(f"  {i:2d}. {var}")

# 3. Charger les NORM_STATS actuels
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')
from src.dataloader import NORM_STATS

print(f"\n✅ NORM_STATS actuels (src/dataloader.py):")
for var in checkpoint_variables:
    if var in NORM_STATS:
        mean, std = NORM_STATS[var]
        print(f"  {var:12s}: mean={mean:12.2f}, std={std:12.2f}")
    else:
        print(f"  {var:12s}: ❌ MANQUANT!")

# 4. Test de normalisation sur des valeurs réelles
print("\n" + "="*80)
print("TEST DE NORMALISATION")
print("="*80)

# Charger un petit échantillon de données
import xarray as xr
from pathlib import Path

data_path = Path("data_processed/data_2013_china_masked.zarr")
if data_path.exists():
    ds = xr.open_zarr(data_path, consolidated=True)
    
    print(f"\n✅ Fichier chargé: {data_path}")
    print(f"   Résolution: {ds.dims['lat']} × {ds.dims['lon']}")
    print(f"   Variables: {list(ds.data_vars)}")
    
    # Prendre un timestep
    idx = 1000
    print(f"\n✅ Test sur timestep {idx}:")
    
    for var in ['pm25', 'pm10', 'elevation', 'population', 'u', 'v']:
        if var in ds:
            data = ds[var].isel(time=idx).values if 'time' in ds[var].dims else ds[var].values
            
            # Stats réelles
            data_finite = data[~np.isnan(data)]
            if len(data_finite) > 0:
                real_mean = float(data_finite.mean())
                real_std = float(data_finite.std())
                real_min = float(data_finite.min())
                real_max = float(data_finite.max())
                
                # Normalisation avec NORM_STATS
                if var in NORM_STATS:
                    norm_mean, norm_std = NORM_STATS[var]
                    normalized = (data - norm_mean) / norm_std
                    norm_finite = normalized[~np.isnan(normalized)]
                    norm_mean_result = float(norm_finite.mean())
                    norm_std_result = float(norm_finite.std())
                    
                    print(f"\n  {var}:")
                    print(f"    Raw data:    min={real_min:10.2f}, max={real_max:10.2f}, mean={real_mean:10.2f}, std={real_std:10.2f}")
                    print(f"    NORM_STATS:  mean={norm_mean:10.2f}, std={norm_std:10.2f}")
                    print(f"    Normalized:  mean={norm_mean_result:10.4f}, std={norm_std_result:10.4f}")
else:
    print(f"❌ Fichier non trouvé: {data_path}")

print("\n" + "="*80)

import numpy as np

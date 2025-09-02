import os
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import random

# Statistiques de normalisation prédéfinies
NORM_STATS = {
    "u": (0.0, 10.0), "v": (0.0, 10.0), "temp": (273.15, 30.0),
    "rh": (50.0, 30.0), "psfc": (101325.0, 1000.0), "pm10": (50.0, 25.0),
    "so2": (5.0, 5.0), "no2": (20.0, 15.0), "co": (200.0, 100.0),
    "o3": (40.0, 20.0), "pm25": (25.0, 15.0), 
    "lat2d": (32.0, 12.0), "lon2d": (106.0, 16.0)
}

class AQNetDataset(Dataset):
    """Dataset optimisé pour Zarr avec consolidated=True et single timestep"""
    
    def __init__(self, data_path: str, years: List[int], variables: List[str], 
                 target_variables: List[str], forecast_days: List[int], 
                 normalize: bool = True, mode: str = 'train', consolidated: bool = True):
        
        self.data_path = Path(data_path)
        self.years = years
        self.variables = variables
        self.target_variables = target_variables
        self.forecast_days = forecast_days
        self.normalize = normalize
        self.mode = mode
        self.consolidated = consolidated
        
        print(f"🚀 AQNetDataset (Zarr Optimized - consolidated={consolidated}):")
        print(f"   Mode: {mode}")
        print(f"   Years: {years}")
        print(f"   Variables ({len(variables)}): {variables}")
        print(f"   Targets: {target_variables}")
        print(f"   Forecast days: {forecast_days}")
        print(f"   Consolidated: {consolidated}")
        
        # Chargement des datasets Zarr avec consolidated=True
        self.datasets = {}
        self.all_samples = []
        
        self._load_datasets()
        self._create_samples()
    
    def _load_datasets(self):
        """Charge les datasets Zarr avec optimisation consolidated"""
        for year in self.years:
            zarr_file = self.data_path / f"data_{year}.zarr"
            
            if zarr_file.exists():
                print(f"   📁 Loading {year} with consolidated={self.consolidated}...")
                
                # OPTIMISATION: consolidated=True pour accès rapide
                ds = xr.open_zarr(zarr_file, consolidated=self.consolidated)
                
                # Crop spatial dimension si nécessaire (339->336)
                if 'latitude' in ds.dims and ds.dims['latitude'] == 339:
                    ds = ds.isel(latitude=slice(0, 336))
                
                self.datasets[year] = ds
                print(f"   ✅ Loaded {year}: {ds.dims}")
            else:
                print(f"   ❌ Missing: {zarr_file}")
    
    def _create_samples(self):
        """Créer les indices d'échantillons valides"""
        for year, ds in self.datasets.items():
            n_timesteps = len(ds.time)
            max_forecast_hours = max(self.forecast_days) * 24
            
            # Indices valides: besoin de max_forecast_hours après t
            valid_end = n_timesteps - max_forecast_hours
            
            for t in range(0, valid_end):
                for forecast_day in self.forecast_days:
                    self.all_samples.append((year, t, forecast_day))
        
        print(f"   📊 Total samples: {len(self.all_samples)}")
    
    def _normalize(self, data, var_name):
        """Normaliser les données"""
        if not self.normalize or var_name not in NORM_STATS:
            return data
        
        mean, std = NORM_STATS[var_name]
        return (data - mean) / (std + 1e-8)
    
    def _denormalize(self, data, var_name):
        """Dénormaliser les données"""
        if not self.normalize or var_name not in NORM_STATS:
            return data
        
        mean, std = NORM_STATS[var_name]
        return data * (std + 1e-8) + mean
    
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        year, t, forecast_day = self.all_samples[idx]
        forecast_hours = forecast_day * 24
        
        ds = self.datasets[year]
        
        # Input: single timestep t
        input_vars = []
        for var in self.variables:
            if var in ds.data_vars:
                # OPTIMISATION: Accès direct au chunk Zarr
                var_data = ds[var].isel(time=t).values  # [H, W]
                var_data = self._normalize(var_data, var)
                input_vars.append(var_data)
        
        input_data = np.stack(input_vars, axis=0)  # [C, H, W]
        
        # Target: à t + forecast_hours
        target_t = t + forecast_hours
        target_data = ds[self.target_variables[0]].isel(time=target_t).values  # [H, W]
        target_data = self._normalize(target_data, self.target_variables[0])
        
        # Conversion en tenseurs
        input_tensor = torch.from_numpy(input_data).float()
        target_tensor = torch.from_numpy(target_data).float().unsqueeze(0)  # Add variable dimension
        lead_time_tensor = torch.tensor(forecast_hours, dtype=torch.float32)
        
        return input_tensor, target_tensor, lead_time_tensor, tuple(self.variables)

def create_dataset(config, mode='train'):
    """Créer le dataset depuis la config"""
    data_config = config['data']
    
    if mode == 'train':
        years = data_config['train_years']
    elif mode == 'val':
        years = data_config['val_years']
    else:
        years = data_config['test_years']
    
    dataset = AQNetDataset(
        data_path=data_config['data_path'],
        years=years,
        variables=data_config['variables'],
        target_variables=data_config['target_variables'],
        forecast_days=data_config['forecast_days'],
        normalize=data_config['normalize'],
        mode=mode,
        consolidated=data_config.get('consolidated', True)
    )
    
    return dataset

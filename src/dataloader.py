import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
import yaml
import os
import gc
from pathlib import Path
from datetime import datetime, timedelta

class PM25AirQualityDataset(Dataset):
    """
    Dataset pour la prédiction de PM2.5 avec données d'observation météo et air quality
    Utilise downsampling avec interpolation au lieu de cropping
    """
    
    def __init__(self, 
                 data_path: str,
                 variables: list,
                 target_variables: list = ['pm25'],
                 years: list = [2013, 2014],
                 forecast_hours: list = [24],
                 time_step: int = 1,
                 normalize: bool = True,
                 target_resolution: tuple = (128, 256)):
        
        self.data_path = Path(data_path)
        self.variables = tuple(variables)
        self.target_variables = target_variables
        self.years = years
        self.forecast_hours = forecast_hours
        self.time_step = time_step
        self.normalize = normalize
        self.target_h, self.target_w = target_resolution
        
        # Stats pour normalisation
        self.stats = {}
        
        # Charger et préparer les données
        print(f"🔄 Initializing dataset for years {years}...")
        self._load_all_data()
        self._prepare_indices()
        
        print(f"✅ Dataset ready: {len(self)} samples")

    def _load_all_data(self):
        """Charger tous les fichiers zarr pour les années demandées"""
        self.datasets = []
        self.time_offsets = []
        current_offset = 0
        
        for year in self.years:
            zarr_path = self.data_path / f"data_{year}.zarr"
            if zarr_path.exists():
                print(f"   📂 Loading {zarr_path}")
                ds = xr.open_zarr(zarr_path, consolidated=True)
                
                # Préparer le downsampling et les grilles de coordonnées
                if len(self.datasets) == 0:  # Premier dataset
                    self._prepare_downsampling(ds)
                
                self.datasets.append(ds)
                self.time_offsets.append(current_offset)
                current_offset += len(ds.time)
                
                print(f"   ✅ Loaded {len(ds.time)} timesteps from {year}")
            else:
                print(f"   ❌ Missing: {zarr_path}")

    def _prepare_downsampling(self, ds):
        """Préparer automatiquement le downsampling et les grilles de coordonnées"""
        # Récupérer la résolution actuelle
        sample_var = None
        for var in self.variables:
            if var in ds.data_vars:
                sample_var = var
                break
        
        if sample_var is None:
            raise ValueError("No valid variables found in dataset")
        
        sample_data = ds[sample_var].isel(time=0)
        self.current_h, self.current_w = sample_data.shape
        
        print(f"   📐 Original resolution: {self.current_h}×{self.current_w}")
        print(f"   📐 Target resolution: {self.target_h}×{self.target_w}")
        print(f"   🔄 Using bilinear interpolation for downsampling")
        
        # Créer les grilles de coordonnées à la résolution cible
        self._create_coordinate_grids(ds)

    def _create_coordinate_grids(self, ds):
        """Créer les grilles de coordonnées automatiquement à la taille cible"""
        if 'lat2d' in ds.coords and 'lon2d' in ds.coords:
            # Extraire les coordonnées 1D originales
            lat_1d_orig = ds.coords['lat2d'].values  # (current_h,)
            lon_1d_orig = ds.coords['lon2d'].values  # (current_w,)
            
            # Downsampler les coordonnées par interpolation linéaire
            lat_indices = np.linspace(0, len(lat_1d_orig)-1, self.target_h)
            lon_indices = np.linspace(0, len(lon_1d_orig)-1, self.target_w)
            
            lat_1d = np.interp(lat_indices, np.arange(len(lat_1d_orig)), lat_1d_orig)
            lon_1d = np.interp(lon_indices, np.arange(len(lon_1d_orig)), lon_1d_orig)
            
            # Créer les grilles 2D à la résolution cible
            lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)  # (target_h, target_w)
            
            self.lat_grid = lat_grid.astype(np.float32)
            self.lon_grid = lon_grid.astype(np.float32)
            
            print(f"   🌍 Coordinate grids created: lat=({self.lat_grid.shape}), lon=({self.lon_grid.shape})")
        else:
            self.lat_grid = None
            self.lon_grid = None
            print("   ⚠️  No coordinate grids (lat2d/lon2d not found)")

    def _prepare_indices(self):
        """Préparer les indices des échantillons valides"""
        self.valid_indices = []
        
        for ds_idx, ds in enumerate(self.datasets):
            max_forecast = max(self.forecast_hours)
            max_time_idx = len(ds.time) - max_forecast - 1
            
            for t in range(0, max_time_idx, self.time_step):
                for forecast_h in self.forecast_hours:
                    self.valid_indices.append((ds_idx, t, forecast_h))
        
        print(f"   📊 Prepared {len(self.valid_indices)} valid samples")

    def _normalize(self, data, var_name):
        """Normalisation Z-score simplifiée"""
        if not self.normalize:
            return data
        
        if var_name not in self.stats:
            # Calculer stats à la volée (pour éviter le stockage)
            finite_data = data[np.isfinite(data)]
            if len(finite_data) > 0:
                self.stats[var_name] = {
                    'mean': np.mean(finite_data),
                    'std': np.std(finite_data) + 1e-8
                }
            else:
                self.stats[var_name] = {'mean': 0.0, 'std': 1.0}
        
        mean, std = self.stats[var_name]['mean'], self.stats[var_name]['std']
        return (data - mean) / std

    def _downsample_tensor(self, data):
        """Downsample un tensor 2D en utilisant l'interpolation bilinéaire"""
        # Convertir en tensor PyTorch si nécessaire
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        # Ajouter dimensions batch et channel pour interpolate: [1, 1, H, W]
        data = data.unsqueeze(0).unsqueeze(0)
        
        # Interpolation bilinéaire
        downsampled = interpolate(data, size=(self.target_h, self.target_w), 
                                mode='bilinear', align_corners=False)
        
        # Retirer les dimensions ajoutées: [H, W]
        return downsampled.squeeze(0).squeeze(0)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        ds_idx, t, forecast_hours = self.valid_indices[idx]
        ds = self.datasets[ds_idx]
        
        # Variables d'entrée
        input_vars = []
        for var in self.variables:
            if var in ds.data_vars:
                # Variable météo: charger et downsampler
                var_data = ds[var].isel(time=t).values  # [H, W]
                var_data = self._normalize(var_data, var)
                var_data = self._downsample_tensor(var_data)  # [target_h, target_w]
                input_vars.append(var_data.numpy())
                
            elif var == 'lat2d' and self.lat_grid is not None:
                # Grille latitude (déjà à la bonne taille)
                input_vars.append(self.lat_grid)
                
            elif var == 'lon2d' and self.lon_grid is not None:
                # Grille longitude (déjà à la bonne taille)
                input_vars.append(self.lon_grid)
        
        # Stack des variables: [n_vars, target_h, target_w]
        input_data = np.stack(input_vars, axis=0)
        
        # Variables cibles avec même downsampling - SUPPORT MULTI-POLLUANTS
        target_t = t + forecast_hours
        target_data_list = []
        for target_var in self.target_variables:
            target_data = ds[target_var].isel(time=target_t).values
            target_data = self._normalize(target_data, target_var)
            target_data = self._downsample_tensor(target_data)  # [target_h, target_w]
            target_data_list.append(target_data)
        
        # Stack des variables cibles: [n_targets, target_h, target_w]
        target_data_stacked = torch.stack(target_data_list, dim=0)
        
        # Conversion en tenseurs
        input_tensor = torch.from_numpy(input_data).float()
        target_tensor = target_data_stacked.float()
        lead_time_tensor = torch.tensor(forecast_hours, dtype=torch.float32)
        
        return input_tensor, target_tensor, lead_time_tensor

def create_dataset(config, mode='train'):
    """Créer le dataset depuis la config"""
    data_config = config['data']
    
    if mode == 'train':
        years = data_config['train_years']
    elif mode == 'val':
        years = data_config['val_years']
    elif mode == 'test':
        years = data_config['test_years']
    else:
        raise ValueError(f"Mode '{mode}' non supporté")
    
    return PM25AirQualityDataset(
        data_path=data_config['data_path'],
        variables=data_config['variables'],
        target_variables=data_config['target_variables'],
        years=years,
        forecast_hours=data_config['forecast_hours'],
        time_step=data_config['time_step'],
        normalize=data_config['normalize'],
        target_resolution=tuple(data_config['target_resolution'])
    )

def create_dataloader(dataset, batch_size, num_workers=4, shuffle=True):
    """Créer un DataLoader optimisé"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

# Alias pour compatibilité avec datamodule existant
AQNetDataset = PM25AirQualityDataset

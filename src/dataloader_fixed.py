"""
Dataloader avec indices fixes pour évaluation cohérente
"""

import json
import numpy as np
from pathlib import Path
from dataloader import PM25AirQualityDataset

class PM25AirQualityDatasetFixed(PM25AirQualityDataset):
    """Dataset avec indices fixes pour évaluation reproductible"""
    
    def __init__(self, fixed_indices_file=None, **kwargs):
        self.fixed_indices_file = fixed_indices_file
        super().__init__(**kwargs)
    
    def _prepare_indices(self):
        """Utiliser les indices fixes ou générer normalement"""
        if self.fixed_indices_file and Path(self.fixed_indices_file).exists():
            print(f"📌 Loading fixed indices from {self.fixed_indices_file}")
            
            with open(self.fixed_indices_file, 'r') as f:
                fixed_indices_by_horizon = json.load(f)
            
            # Générer d'abord tous les indices normalement
            super()._prepare_indices()
            all_indices = self.valid_indices.copy()
            
            # Filtrer selon les indices fixes
            self.valid_indices = []
            total_fixed = 0
            
            for horizon_str, fixed_idx_list in fixed_indices_by_horizon.items():
                horizon = int(horizon_str)
                for idx in fixed_idx_list:
                    if idx < len(all_indices):
                        self.valid_indices.append(all_indices[idx])
                        total_fixed += 1
            
            print(f"   📊 Using {total_fixed} fixed samples instead of {len(all_indices)}")
            
        else:
            print("⚠️  No fixed indices file found, using all samples")
            super()._prepare_indices()

def create_dataset_fixed(config, mode='train', fixed_indices_file=None):
    """Créer le dataset avec indices fixes pour l'évaluation"""
    data_config = config['data']
    
    if mode == 'train':
        years = data_config['train_years']
    elif mode == 'val':
        years = data_config['val_years']
    else:  # test
        years = data_config['test_years']
    
    return PM25AirQualityDatasetFixed(
        data_path=data_config['data_path'],
        variables=data_config['variables'],
        target_variables=data_config['target_variables'],
        years=years,
        forecast_hours=data_config['forecast_hours'],
        time_step=data_config['time_step'],
        normalize=data_config['normalize'],
        target_resolution=tuple(data_config['target_resolution']),
        fixed_indices_file=fixed_indices_file
    )

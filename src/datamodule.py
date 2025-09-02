import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.dataloader import PM25AirQualityDataset
from pathlib import Path

class AQNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule pour AQ_Net2
    Version qui reproduit EXACTEMENT l'ancien comportement
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        self.data_config = config['data']
        self.train_config = config['train']
        
        # Paramètres du dataset depuis la config
        self.data_path = self.data_config['data_path']
        self.variables = self.data_config['variables']
        self.target_variables = self.data_config.get('target_variables', ['pm25'])
        
        # IMPORTANT: L'ancien code utilisait forecast_days DIRECTEMENT comme forecast_hours
        # C'était un bug mais c'est ce qui fonctionnait !
        # Il passait [1, 3, 5, 7] au lieu de [24, 72, 120, 168]
        self.forecast_hours = self.data_config.get('forecast_days', [1, 3, 5, 7])
        
        self.normalize = self.data_config.get('normalize', True)
        self.target_resolution = tuple(self.data_config.get('target_resolution', [128, 256]))
        
        # Paramètres d'entraînement
        self.batch_size = self.train_config['batch_size']
        self.val_batch_size = self.train_config.get('val_batch_size', self.batch_size)
        self.num_workers = self.data_config.get('num_workers', 8)
        
        # Années pour train/val/test
        self.train_years = self.data_config.get('train_years', [2013, 2014, 2015, 2016])
        self.val_years = self.data_config.get('val_years', [2017])
        self.test_years = self.data_config.get('test_years', [2018])
        
    def setup(self, stage=None):
        """Prépare les datasets pour l'entraînement et la validation"""
        
        if stage == 'fit' or stage is None:
            # Dataset d'entraînement
            self.train_dataset = PM25AirQualityDataset(
                data_path=self.data_path,
                variables=self.variables,
                target_variables=self.target_variables,
                years=self.train_years,
                forecast_hours=self.forecast_hours,  # Passe [1, 3, 5, 7] comme l'ancien
                normalize=self.normalize,
                target_resolution=self.target_resolution
            )
            
            # Dataset de validation  
            self.val_dataset = PM25AirQualityDataset(
                data_path=self.data_path,
                variables=self.variables,
                target_variables=self.target_variables,
                years=self.val_years,
                forecast_hours=self.forecast_hours,  # Passe [1, 3, 5, 7] comme l'ancien
                normalize=self.normalize,
                target_resolution=self.target_resolution
            )
            
            # Partager les stats de normalisation du train vers le val
            if self.normalize and hasattr(self.train_dataset, 'stats'):
                self.val_dataset.stats = self.train_dataset.stats
            
            print(f"Training dataset: {len(self.train_dataset)} samples")
            print(f"Validation dataset: {len(self.val_dataset)} samples")
        
        if stage == 'test':
            # Dataset de test
            self.test_dataset = PM25AirQualityDataset(
                data_path=self.data_path,
                variables=self.variables,
                target_variables=self.target_variables,
                years=self.test_years,
                forecast_hours=self.forecast_hours,  # Passe [1, 3, 5, 7] comme l'ancien
                normalize=self.normalize,
                target_resolution=self.target_resolution
            )
            
            # Utiliser les stats du train si disponibles
            if self.normalize and hasattr(self, 'train_dataset') and hasattr(self.train_dataset, 'stats'):
                self.test_dataset.stats = self.train_dataset.stats
            
            print(f"Test dataset: {len(self.test_dataset)} samples")
    
    def train_dataloader(self):
        """Retourne le DataLoader d'entraînement"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,
            prefetch_factor=self.data_config.get('prefetch_factor', 2)
        )
    
    def val_dataloader(self):
        """Retourne le DataLoader de validation"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=False,
            prefetch_factor=self.data_config.get('prefetch_factor', 2)
        )
    
    def test_dataloader(self):
        """Retourne le DataLoader de test"""
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False,
                drop_last=False
            )
        return None
    
    def predict_dataloader(self):
        """Retourne le DataLoader pour la prédiction"""
        return self.val_dataloader()

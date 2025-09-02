#!/usr/bin/env python3
"""Test simple d'évaluation"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config_manager import ConfigManager
from datamodule import AQNetDataModule
from model import PM25LightningModule

def main():
    print("="*60)
    print("🚀 TEST SIMPLE ÉVALUATION")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Configuration avec batch_size=1
    config_manager = ConfigManager("configs/config.yaml")
    config = config_manager.config
    config['data']['batch_size'] = 1  # Batch size minimal
    config['data']['num_workers'] = 0  # Pas de multiprocessing
    
    # Charger le modèle
    checkpoint = "logs/climax_china_masked/version_19/checkpoints/epoch_epoch=07-val_loss_val_loss=0.0604.ckpt"
    print(f"\n📁 Chargement: {checkpoint}")
    
    model = PM25LightningModule.load_from_checkpoint(checkpoint, config=config)
    model = model.to(device)
    model.eval()
    print("✅ Modèle chargé")
    
    # Data module
    print("\n📊 Chargement des données...")
    data_module = AQNetDataModule(config)
    data_module.setup('test')
    test_dataloader = data_module.test_dataloader()
    print(f"✅ {len(test_dataloader.dataset)} échantillons de test")
    
    # Test sur quelques batches seulement
    print("\n🔬 Test sur 10 premiers batches...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= 10:
                break
                
            print(f"  Batch {i+1}/10...", end="")
            
            inputs, targets, lead_times, variables = batch
            inputs = inputs.to(device)
            lead_times = lead_times.to(device)
            
            predictions = model(inputs, lead_times, variables)
            
            # Calculer une erreur simple
            mse = ((predictions.cpu() - targets) ** 2).mean().item()
            print(f" MSE={mse:.4f}")
            
            del predictions
            torch.cuda.empty_cache()
    
    print("\n✅ Test terminé avec succès!")

if __name__ == "__main__":
    main()

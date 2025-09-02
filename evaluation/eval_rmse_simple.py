#!/usr/bin/env python3
"""
ÉVALUATION RMSE SIMPLE - Sans accumulation mémoire
Calcule RMSE par horizon directement
"""

import os
import sys
import numpy as np
import torch
import pytorch_lightning as pl
from datetime import datetime

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config_manager import ConfigManager
from datamodule_fixed import AQNetDataModule
from model_multipollutants import MultiPollutantLightningModule
from dataloader_zarr_optimized import NORM_STATS

def denormalize_data(data, variable):
    """Dénormaliser selon les stats"""
    if variable in NORM_STATS:
        mean, std = NORM_STATS[variable]
        return data * std + mean
    return data

def main():
    print("🚀 ÉVALUATION RMSE SIMPLE")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/eval_rmse_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Configuration
    config_manager = ConfigManager("configs/config.yaml")
    config = config_manager.config
    
    # Data module
    data_module = AQNetDataModule(config)
    data_module.setup('test')
    test_dataloader = data_module.test_dataloader()
    print(f"Dataset de test: {len(test_dataloader.dataset)} échantillons")
    
    # Charger le modèle
    print("\n🔄 Chargement du modèle...")
    best_checkpoint = "logs/climax_from_scratch/version_27/checkpoints/epoch_epoch=05-val_loss_val_loss=0.0431.ckpt"
    
    if os.path.exists(best_checkpoint):
        model = MultiPollutantLightningModule.load_from_checkpoint(best_checkpoint, config=config)
        print("✅ Checkpoint chargé avec succès")
    else:
        print("❌ Checkpoint non trouvé!")
        return
    
    model = model.to(device)
    model.eval()
    
    # Évaluation avec accumulation RMSE par horizon
    print("\n🔬 Début de l'évaluation...")
    
    sum_squared_errors = {}
    counts = {}
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i % 500 == 0:
                print(f"Batch {i}/{len(test_dataloader)}")
            
            inputs, targets, lead_times, variables = batch
            inputs = inputs.to(device)
            lead_times = lead_times.to(device)
            
            predictions = model(inputs, lead_times, variables)
            
            # CPU pour économiser mémoire
            predictions = predictions.cpu()
            targets = targets.cpu()
            
            # Dénormaliser
            pred_denorm = denormalize_data(predictions, 'pm25')
            target_denorm = denormalize_data(targets, 'pm25')
            
            # Calculer erreurs par horizon
            for j in range(len(lead_times)):
                horizon = float(lead_times[j].item())
                
                # Erreur quadratique pour cet échantillon
                error_sq = ((pred_denorm[j] - target_denorm[j]) ** 2).sum().item()
                n_pixels = pred_denorm[j].numel()
                
                if horizon not in sum_squared_errors:
                    sum_squared_errors[horizon] = 0.0
                    counts[horizon] = 0
                
                sum_squared_errors[horizon] += error_sq
                counts[horizon] += n_pixels
    
    # Calcul RMSE finales
    print("\n📊 Résultats RMSE par horizon:")
    print("=" * 50)
    
    for horizon in sorted(sum_squared_errors.keys()):
        mse = sum_squared_errors[horizon] / counts[horizon]
        rmse = np.sqrt(mse)
        print(f"🎯 Horizon {int(horizon)}j: RMSE = {rmse:.2f} µg/m³")
    
    print("=" * 50)
    print("✅ Évaluation terminée!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ÉVALUATION RMSE EPOCH 7 (CHINE FOCUS) - Sans accumulation mémoire
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
from datamodule import AQNetDataModule
from model import PM25LightningModule
from dataloader_zarr_optimized import NORM_STATS

def denormalize_data(data, variable):
    """Dénormaliser selon les stats"""
    if variable in NORM_STATS:
        mean, std = NORM_STATS[variable]
        return data * std + mean
    return data

def main():
    print("🚀 ÉVALUATION RMSE EPOCH 7 (CHINE FOCUS)")
    
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
    best_checkpoint = "logs/climax_china_masked/version_19/checkpoints/epoch_epoch=07-val_loss_val_loss=0.0604.ckpt"
    
    if os.path.exists(best_checkpoint):
        model = PM25LightningModule.load_from_checkpoint(best_checkpoint, config=config)
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

def evaluate_with_china_mask():
    """Évaluation supplémentaire avec masque Chine"""
    print("\n" + "="*60)
    print("🇨🇳 ÉVALUATION SUPPLÉMENTAIRE AVEC MASQUE CHINE")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    config_manager = ConfigManager("configs/config.yaml")
    config = config_manager.config
    
    # Data module
    data_module = AQNetDataModule(config)
    data_module.setup('test')
    test_dataloader = data_module.test_dataloader()
    
    # Charger le modèle
    best_checkpoint = "logs/climax_china_masked/version_19/checkpoints/epoch_epoch=07-val_loss_val_loss=0.0604.ckpt"
    model = PM25LightningModule.load_from_checkpoint(best_checkpoint, config=config)
    model = model.to(device)
    model.eval()
    
    # Charger le masque Chine
    china_mask = torch.load("china_mask_fixed.pt", map_location="cpu").to(device)
    print(f"✅ Masque Chine: {china_mask.sum().item():.0f} pixels actifs sur {china_mask.numel()}")
    
    # Évaluation avec masque
    sum_squared_errors_china = {}
    counts_china = {}
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i % 500 == 0:
                print(f"Batch {i}/{len(test_dataloader)} (masque Chine)")
            
            inputs, targets, lead_times, variables = batch
            inputs = inputs.to(device)
            lead_times = lead_times.to(device)
            targets = targets.to(device)
            
            predictions = model(inputs, lead_times, variables)
            
            # Dénormaliser
            pred_denorm = denormalize_data(predictions, 'pm25')
            target_denorm = denormalize_data(targets, 'pm25')
            
            # Calculer erreurs par horizon AVEC MASQUE CHINE
            for j in range(len(lead_times)):
                horizon = float(lead_times[j].item())
                
                # Erreur quadratique avec masque Chine
                error_sq = ((pred_denorm[j] - target_denorm[j]) ** 2)
                
                if error_sq.dim() == 2:  # [H, W]
                    mask_exp = china_mask
                elif error_sq.dim() == 3:  # [C, H, W]
                    mask_exp = china_mask.unsqueeze(0).expand_as(error_sq)
                
                # Appliquer le masque
                error_masked = error_sq * mask_exp
                error_sum = error_masked.sum().item()
                n_pixels_china = mask_exp.sum().item()
                
                if horizon not in sum_squared_errors_china:
                    sum_squared_errors_china[horizon] = 0.0
                    counts_china[horizon] = 0
                
                sum_squared_errors_china[horizon] += error_sum
                counts_china[horizon] += n_pixels_china
    
    # Résultats avec masque Chine
    print("\n🇨🇳 Résultats RMSE CHINE (région masquée):")
    print("=" * 50)
    
    for horizon in sorted(sum_squared_errors_china.keys()):
        mse = sum_squared_errors_china[horizon] / counts_china[horizon]
        rmse = np.sqrt(mse)
        print(f"🎯 Horizon {int(horizon)}j (CHINE): RMSE = {rmse:.2f} µg/m³")
    
    print("=" * 50)
    print("✅ Évaluation avec masque Chine terminée!")


if __name__ == "__main__":
    evaluate_with_china_mask()

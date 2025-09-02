#!/usr/bin/env python3
"""
ÉVALUATION RMSE + CARTES VISUELLES
Calcule RMSE par horizon ET génère des cartes de comparaison réel vs prédiction
"""

import os
import sys
import numpy as np
import torch
import pytorch_lightning as pl
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import random

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

def create_pm25_colormap():
    """Créer colormap personnalisée pour PM2.5"""
    colors = ['#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#4B0082']
    levels = [0, 10, 25, 50, 75, 100]
    return LinearSegmentedColormap.from_list('pm25', colors, N=256)

def save_comparison_maps(predictions_dict, targets_dict, horizons, output_dir):
    """Sauvegarder les cartes de comparaison pour chaque horizon"""
    print("🗺️ Génération des cartes de comparaison...")
    
    cmap = create_pm25_colormap()
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    
    for horizon in horizons:
        if horizon in predictions_dict and horizon in targets_dict:
            print(f"  📊 Carte horizon {int(horizon)}h...")
            
            # Sélectionner un échantillon aléatoire
            preds = predictions_dict[horizon]
            targs = targets_dict[horizon]
            
            if len(preds) > 0:
                idx = random.randint(0, len(preds) - 1)
                pred_sample = preds[idx]
                target_sample = targs[idx]
                
                # Créer la figure avec 3 sous-graphiques
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                vmin, vmax = 0, 100  # Limites pour PM2.5
                
                # 1. Prédiction
                im1 = axes[0].imshow(pred_sample, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
                axes[0].set_title(f'Prédiction - H{int(horizon)}h', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Longitude')
                axes[0].set_ylabel('Latitude')
                
                # 2. Réel (cible)
                im2 = axes[1].imshow(target_sample, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
                axes[1].set_title(f'Réel - H{int(horizon)}h', fontsize=14, fontweight='bold')
                axes[1].set_xlabel('Longitude')
                axes[1].set_ylabel('Latitude')
                
                # 3. Différence
                diff = pred_sample - target_sample
                vmin_diff, vmax_diff = -50, 50
                im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=vmin_diff, vmax=vmax_diff, aspect='auto')
                axes[2].set_title(f'Différence (Pred - Réel) - H{int(horizon)}h', fontsize=14, fontweight='bold')
                axes[2].set_xlabel('Longitude')
                axes[2].set_ylabel('Latitude')
                
                # Barres de couleur
                plt.colorbar(im1, ax=axes[0], label='PM2.5 (µg/m³)')
                plt.colorbar(im2, ax=axes[1], label='PM2.5 (µg/m³)')
                plt.colorbar(im3, ax=axes[2], label='Différence (µg/m³)')
                
                plt.suptitle(f'Comparaison Réel vs Prédiction - Horizon {int(horizon)}h', 
                            fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                # Sauvegarder
                filename = f"{output_dir}/visualizations/comparison_map_H{int(horizon)}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"    ✅ Sauvegardé: {filename}")

def main():
    print("🚀 ÉVALUATION RMSE + CARTES VISUELLES")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/eval_rmse_maps_{timestamp}"
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
        model = PM25LightningModule.load_from_checkpoint(best_checkpoint, config=config)
        print("✅ Checkpoint chargé avec succès")
    else:
        print("❌ Checkpoint non trouvé!")
        return
    
    model = model.to(device)
    model.eval()
    
    # Évaluation avec accumulation RMSE par horizon + sauvegarde d'échantillons
    print("\n🔬 Début de l'évaluation...")
    
    sum_squared_errors = {}
    counts = {}
    
    # Pour sauvegarder quelques échantillons par horizon
    predictions_dict = {}
    targets_dict = {}
    max_samples_per_horizon = 10
    
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
                    predictions_dict[horizon] = []
                    targets_dict[horizon] = []
                
                sum_squared_errors[horizon] += error_sq
                counts[horizon] += n_pixels
                
                # Sauvegarder quelques échantillons pour les cartes
                if len(predictions_dict[horizon]) < max_samples_per_horizon:
                    predictions_dict[horizon].append(pred_denorm[j].squeeze().numpy())
                    targets_dict[horizon].append(target_denorm[j].squeeze().numpy())
    
    # Calcul RMSE finales
    print("\n📊 Résultats RMSE par horizon:")
    print("=" * 50)
    
    horizons = []
    for horizon in sorted(sum_squared_errors.keys()):
        mse = sum_squared_errors[horizon] / counts[horizon]
        rmse = np.sqrt(mse)
        print(f"🎯 Horizon {int(horizon)}h: RMSE = {rmse:.2f} µg/m³")
        horizons.append(horizon)
    
    print("=" * 50)
    
    # Générer les cartes de comparaison
    save_comparison_maps(predictions_dict, targets_dict, horizons, output_dir)
    
    print(f"\n🎨 Cartes sauvegardées dans: {output_dir}/visualizations/")
    print("✅ Évaluation terminée!")

if __name__ == "__main__":
    main()

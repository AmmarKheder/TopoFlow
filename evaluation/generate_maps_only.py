#!/usr/bin/env python3
"""
GÉNÉRATION DES CARTES VISUELLES SEULEMENT
Génère des cartes de comparaison réel vs prédiction sans recalculer RMSE
Utilise juste quelques échantillons pour créer les visualisations
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
    return LinearSegmentedColormap.from_list('pm25', colors, N=256)

def generate_comparison_map(pred_sample, target_sample, horizon, output_dir):
    """Générer une carte de comparaison pour un horizon donné"""
    
    cmap = create_pm25_colormap()
    
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
    
    # Statistiques sur la différence
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)
    
    # Ajouter texte avec statistiques
    stats_text = f'RMSE: {rmse:.2f} µg/m³\nMAE: {mae:.2f} µg/m³\nBias: {bias:.2f} µg/m³'
    axes[2].text(0.02, 0.98, stats_text, transform=axes[2].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Barres de couleur
    plt.colorbar(im1, ax=axes[0], label='PM2.5 (µg/m³)')
    plt.colorbar(im2, ax=axes[1], label='PM2.5 (µg/m³)')
    plt.colorbar(im3, ax=axes[2], label='Différence (µg/m³)')
    
    plt.suptitle(f'Comparaison Réel vs Prédiction - Horizon {int(horizon)}h', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder
    filename = f"{output_dir}/comparison_map_H{int(horizon)}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Carte sauvegardée: {filename}")
    return rmse, mae, bias

def main():
    print("🗺️ GÉNÉRATION DES CARTES VISUELLES SEULEMENT")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/maps_only_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Configuration
    config_manager = ConfigManager("configs/config.yaml")
    config = config_manager.config
    
    print(f"Horizons de prévision: {config['forecast_horizons']} heures")
    
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
    
    # Générer quelques cartes seulement (pas tout le dataset)
    print(f"\n🎨 Génération des cartes pour {len(config['forecast_horizons'])} horizons...")
    
    horizons_found = {h: False for h in config['forecast_horizons']}
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i % 100 == 0:
                print(f"Recherche échantillons... Batch {i}")
            
            # Arrêter si on a trouvé tous les horizons
            if all(horizons_found.values()):
                break
                
            inputs, targets, lead_times, variables = batch
            inputs = inputs.to(device)
            lead_times = lead_times.to(device)
            
            predictions = model(inputs, lead_times, variables)
            
            # CPU pour traitement
            predictions = predictions.cpu()
            targets = targets.cpu()
            
            # Dénormaliser
            pred_denorm = denormalize_data(predictions, 'pm25')
            target_denorm = denormalize_data(targets, 'pm25')
            
            # Traiter chaque échantillon du batch
            for j in range(len(lead_times)):
                horizon = float(lead_times[j].item())
                
                # Si c'est un horizon qu'on veut et qu'on n'a pas encore traité
                if horizon in config['forecast_horizons'] and not horizons_found[horizon]:
                    print(f"📊 Génération carte pour horizon {int(horizon)}h...")
                    
                    pred_sample = pred_denorm[j].squeeze().numpy()
                    target_sample = target_denorm[j].squeeze().numpy()
                    
                    rmse, mae, bias = generate_comparison_map(pred_sample, target_sample, horizon, output_dir)
                    
                    print(f"  └─ RMSE: {rmse:.2f}, MAE: {mae:.2f}, Bias: {bias:.2f} µg/m³")
                    
                    horizons_found[horizon] = True
    
    print(f"\n🎨 Cartes sauvegardées dans: {output_dir}/")
    print("✅ Génération des cartes terminée!")

if __name__ == "__main__":
    main()

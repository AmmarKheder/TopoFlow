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
    
    # 1. Prediction
    im1 = axes[0].imshow(pred_sample, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title(f'Prediction - Day {int(horizon)}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    
    # 2. Ground Truth (cible)
    im2 = axes[1].imshow(target_sample, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title(f'Ground Truth - Day {int(horizon)}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    
    # 3. Différence
    diff = pred_sample - target_sample
    vmin_diff, vmax_diff = -50, 50
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=vmin_diff, vmax=vmax_diff, aspect='auto')
    axes[2].set_title(f'Difference (Pred - Ground Truth) - Day {int(horizon)}', fontsize=14, fontweight='bold')
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
    
    plt.suptitle(f'Comparison Ground Truth vs Prediction - Day {int(horizon)}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder
    filename = f"{output_dir}/comparison_map_Day{int(horizon)}.png"
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
    
    # Récupérer les horizons depuis la config
    forecast_horizons = [1.0, 3.0, 5.0, 7.0]  # EN JOURS, pas en jours !
    print(f"Horizons de prévision: {forecast_horizons} jours")
    
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
    print(f"\n🎨 Génération des cartes pour {len(forecast_horizons)} horizons...")
    
    # LOGIQUE INTELLIGENTE - VERSION ULTRA-RAPIDE ⚡
    max_batches = 100  # Limite stricte pour éviter des jours d'attente
    samples_per_horizon = 1  # Une seule carte par horizon suffit
    
    horizons_found = {h: False for h in forecast_horizons}
    backup_samples = {}  # Plan B si horizons exacts pas trouvés
    
    print(f"\n🚀 RECHERCHE INTELLIGENTE (max {max_batches} batches)...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # ✅ CONDITION D'ARRÊT INTELLIGENTE
            if i >= max_batches:
                print(f"🛑 Limite de {max_batches} batches atteinte")
                break
                
            if all(horizons_found.values()):
                print("🎯 Tous les horizons trouvés !")
                break
            
            if i % 20 == 0:  # Log plus fréquent
                print(f"⚡ Batch {i} - Trouvés: {sum(horizons_found.values())}/4")
                
            inputs, targets, lead_times, variables = batch
            inputs = inputs.to(device)
            lead_times = lead_times.to(device)
            
            predictions = model(inputs, lead_times, variables)
            predictions = predictions.cpu()
            targets = targets.cpu()
            
            pred_denorm = denormalize_data(predictions, 'pm25')
            target_denorm = denormalize_data(targets, 'pm25')
            
            # 🧠 LOGIQUE INTELLIGENTE : traiter tous les échantillons du batch
            for j in range(len(lead_times)):
                horizon = float(lead_times[j].item())
                
                # Garder des échantillons de backup (horizons proches)
                if horizon not in backup_samples:
                    backup_samples[horizon] = {
                        'pred': pred_denorm[j].squeeze().numpy(),
                        'target': target_denorm[j].squeeze().numpy()
                    }
                
                # Si c'est exactement ce qu'on cherche
                if horizon in forecast_horizons and not horizons_found[horizon]:
                    print(f"🎯 FOUND! Day {int(horizon)} - Generating map...")
                    
                    pred_sample = pred_denorm[j].squeeze().numpy()
                    target_sample = target_denorm[j].squeeze().numpy()
                    
                    rmse, mae, bias = generate_comparison_map(pred_sample, target_sample, horizon, output_dir)
                    print(f"  └─ RMSE: {rmse:.2f}, MAE: {mae:.2f}, Bias: {bias:.2f} µg/m³")
                    
                    horizons_found[horizon] = True
    
    # 🛡️ PLAN B : Utiliser les échantillons les plus proches si nécessaire
    missing_horizons = [h for h in forecast_horizons if not horizons_found[h]]
    
    if missing_horizons:
        print(f"\n🔄 Plan B: Horizons manquants {missing_horizons}")
        available_horizons = sorted(backup_samples.keys())
        
        for missing_h in missing_horizons:
            # Trouver l'horizon le plus proche
            closest_h = min(available_horizons, key=lambda x: abs(x - missing_h))
            print(f"📊 Utilisation horizon {closest_h}h pour remplacer {missing_h}h")
            
            sample = backup_samples[closest_h]
            rmse, mae, bias = generate_comparison_map(
                sample['pred'], sample['target'], missing_h, output_dir
            )
            print(f"  └─ RMSE: {rmse:.2f}, MAE: {mae:.2f}, Bias: {bias:.2f} µg/m³")

    print(f"\n🎨 Cartes sauvegardées dans: {output_dir}/")

if __name__ == "__main__":
    main()

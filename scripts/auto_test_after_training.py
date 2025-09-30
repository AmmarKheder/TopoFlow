#!/usr/bin/env python3
"""
Script d'évaluation automatique après entraînement.
Lance le test sur 2018 avec le meilleur checkpoint.
"""

import os
import sys
import yaml
import glob
import pytorch_lightning as pl
from pathlib import Path

# Ajouter src au path
sys.path.append('src')
from model_multipollutants import MultiPollutantLightningModule
from datamodule import AQNetDataModule

def find_best_checkpoint(log_dir):
    """Trouver le meilleur checkpoint basé sur val_rmse."""
    checkpoint_patterns = [
        f"{log_dir}/**/checkpoints/*.ckpt",
        f"{log_dir}/checkpoints/*.ckpt"
    ]
    
    checkpoints = []
    for pattern in checkpoint_patterns:
        checkpoints.extend(glob.glob(pattern, recursive=True))
    
    if not checkpoints:
        raise FileNotFoundError(f"Aucun checkpoint trouvé dans {log_dir}")
    
    print(f"📁 Checkpoints trouvés: {len(checkpoints)}")
    
    # Chercher le meilleur basé sur le nom (val_loss ou val_rmse)
    best_checkpoint = None
    best_metric = float('inf')
    
    for ckpt in checkpoints:
        filename = os.path.basename(ckpt)
        
        # Extraire métrique du nom
        if 'val_loss' in filename:
            try:
                metric_str = filename.split('val_loss_')[1].split('.ckpt')[0]
                metric_val = float(metric_str)
                if metric_val < best_metric:
                    best_metric = metric_val
                    best_checkpoint = ckpt
            except:
                continue
        elif 'val_rmse' in filename:
            try:
                metric_str = filename.split('val_rmse_')[1].split('.ckpt')[0]
                metric_val = float(metric_str)
                if metric_val < best_metric:
                    best_metric = metric_val
                    best_checkpoint = ckpt
            except:
                continue
    
    if best_checkpoint is None:
        # Prendre le plus récent
        best_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"⚠️ Impossible de parser les métriques, prise du plus récent")
    
    print(f"🏆 Meilleur checkpoint: {best_checkpoint}")
    print(f"🎯 Métrique: {best_metric:.4f}")
    
    return best_checkpoint

def run_test_evaluation(config_path, checkpoint_path, gpus=None):
    """Lance l'évaluation test."""
    
    # Charger config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("🎯 LANCEMENT ÉVALUATION TEST (2018)")
    print("="*50)
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"GPUs: {gpus}")
    print("="*50)
    
    # Charger modèle depuis checkpoint
    model = MultiPollutantLightningModule.load_from_checkpoint(
        checkpoint_path, 
        config=config
    )
    
    # DataModule
    data_module = AQNetDataModule(config)
    
    # Trainer pour test
    trainer = pl.Trainer(
        devices=gpus if gpus else 1,
        accelerator='gpu' if gpus else 'cpu',
        logger=False,  # Pas besoin de logger pour test
        enable_checkpointing=False,
        enable_model_summary=True
    )
    
    # Lancer test
    print("\n🚀 Début évaluation test...")
    trainer.test(model, data_module)
    print("\n✅ Évaluation test terminée!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Évaluation automatique post-entraînement')
    parser.add_argument('--config', required=True, help='Chemin vers config YAML')
    parser.add_argument('--log_dir', required=True, help='Dossier des logs/checkpoints')
    parser.add_argument('--gpus', type=int, default=1, help='Nombre de GPUs')
    parser.add_argument('--checkpoint', help='Checkpoint spécifique (optionnel)')
    
    args = parser.parse_args()
    
    # Trouver meilleur checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_best_checkpoint(args.log_dir)
    
    # Lancer évaluation
    run_test_evaluation(args.config, checkpoint_path, args.gpus)

if __name__ == "__main__":
    main()

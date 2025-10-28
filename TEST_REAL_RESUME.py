#!/usr/bin/env python3
"""
Test RÉEL de resume avec trainer.fit(ckpt_path=...) et affichage de la loss
"""
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import Callback
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.datamodule_fixed import AQNetDataModule
from src.model_multipollutants import MultiPollutantLightningModule

# Callback pour afficher la loss immédiatement
class LossLogger(Callback):
    def __init__(self):
        self.batch_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_count += 1
        if self.batch_count <= 5:  # Afficher les 5 premiers batches
            loss = outputs['loss'].item() if isinstance(outputs, dict) else outputs.item()
            print(f"\n🔥 BATCH {self.batch_count}: train_loss={loss:.4f}\n")

            if self.batch_count == 5:
                print("\n" + "="*100)
                print("✅ 5 PREMIERS BATCHES AFFICHÉS - ARRÊT DU TEST")
                print("="*100)
                trainer.should_stop = True  # Arrêter après 5 batches

print("="*100)
print("TEST RESUME RÉEL avec trainer.fit()")
print("="*100)

# Config
config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config

# Override pour test rapide
config['train']['max_steps'] = 5
config['data']['num_workers'] = 0  # Simplifier
ckpt_path = config['model']['checkpoint_path']

print(f"\nCheckpoint: {ckpt_path}")
print(f"Expected loss: ~0.35-0.40 (checkpoint val_loss=0.3557)")
print(f"\nSi la loss est > 1.0, il y a un problème de chargement!\n")

# Data
data_module = AQNetDataModule(config)

# Model (WITHOUT loading checkpoint - let trainer do it)
model = MultiPollutantLightningModule(config=config)

# Logger
tb_logger = pl_loggers.TensorBoardLogger(
    save_dir="logs/",
    name="test_resume_debug"
)

# Callbacks
loss_logger = LossLogger()

# Trainer
trainer = pl.Trainer(
    max_steps=5,
    accelerator='gpu',
    devices=1,
    num_nodes=1,
    strategy='auto',
    precision=32,
    logger=tb_logger,
    enable_checkpointing=False,
    log_every_n_steps=1,
    callbacks=[loss_logger],
    enable_model_summary=False
)

print("🔥 Démarrage du training avec RESUME depuis checkpoint...")
print("="*100)

# RESUME depuis checkpoint
trainer.fit(model, data_module, ckpt_path=ckpt_path)

print("\n" + "="*100)
print("✅ TEST TERMINÉ")
print("="*100)
print("\nSi les 5 batches affichent loss ~0.3-0.5:")
print("  ✅ Le checkpoint se charge correctement")
print("\nSi les 5 batches affichent loss > 2.0:")
print("  ❌ Le checkpoint NE se charge PAS correctement")
print("="*100)

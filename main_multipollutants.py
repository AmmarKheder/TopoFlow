import os
import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from src.config_manager import ConfigManager
from src.datamodule_fixed import AQNetDataModule
from src.model_multipollutants import MultiPollutantLightningModule


def main(config_path):
    print("🚀 DÉMARRAGE AQ_NET2 - PRÉDICTION MULTI-POLLUANTS")
    
    # Initial setup
    cfg_mgr = ConfigManager(config_path)
    config = cfg_mgr.config
    
    print(f"📋 Configuration: {config_path}")
    print(f"📊 Résolution: {config['model']['img_size']}")
    print(f"🔧 Variables: {len(config['data']['variables'])}")
    print(f"🎯 Cibles: {config['data']['target_variables']} ({len(config['data']['target_variables'])} polluants)")
    print("🇨🇳 MASQUE CHINE ACTIVÉ dans la loss function")
    
    # Initialize Data Module
    print("📁 Initialisation du DataModule...")
    data_module = AQNetDataModule(config)
    
    # Initialize Model from scratch (no checkpoint for multi-pollutant yet)
    print("🧠 Initialisation du modèle multi-polluants...")
    model = MultiPollutantLightningModule(config)
    print("✅ Modèle multi-polluants initialisé")
    
    # Loggers (TensorBoard + CSV)
    print("📈 Configuration des loggers (TensorBoard + CSV)...")
    
    # TensorBoard Logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="logs/",
        name="multipollutants_climax_ddp",
        log_graph=False
    )
    
    # CSV Logger
    csv_logger = pl_loggers.CSVLogger(
        save_dir="logs/",
        name="multipollutants_csv"
    )
    
    # Callbacks
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['callbacks']['early_stopping']['patience'],
            mode=config['callbacks']['early_stopping']['mode']
        ),
        pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode=config['callbacks']['model_checkpoint']['mode'],
            save_top_k=config['callbacks']['model_checkpoint']['save_top_k'],
            filename=config['callbacks']['model_checkpoint']['filename']
        )
    ]
    
    # Trainer
    trainer = pl.Trainer(
        num_nodes=4,
        devices=config['train']['devices'],
        accelerator=config['train']['accelerator'],
        strategy=config['train']['strategy'],
        precision=config['train']['precision'],
        max_epochs=config['train']['epochs'],
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        gradient_clip_val=config['train']['gradient_clip_val'],
        accumulate_grad_batches=config['train']['accumulate_grad_batches'],
        log_every_n_steps=config['train']['log_every_n_steps'],
        val_check_interval=config['train']['val_check_interval'],
        default_root_dir=config['lightning']['trainer']['default_root_dir'],
        enable_checkpointing=config['lightning']['trainer']['enable_checkpointing'],
        enable_model_summary=config['lightning']['trainer']['enable_model_summary']
    )
    
    print("\n" + "="*60)
    print("🏃‍♂️ DÉMARRAGE DE L'ENTRAÎNEMENT MULTI-POLLUANTS")
    print("="*60)
    print(f"🎯 Polluants: {', '.join(config['data']['target_variables'])}")
    print(f"🚀 Horizons: {config['data']['forecast_days']} jours")
    print(f"⚡ GPUs: {config['train']['devices']}")
    print(f"📦 Batch size: {config['train']['batch_size']} par GPU")
    print("="*60 + "\n")
    
    # Start training
    trainer.fit(model, data_module)
    
    print("\n🎉 ENTRAÎNEMENT TERMINÉ!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQ_Net2 Multi-Pollutant Training")
    parser.add_argument("--config", type=str, default="configs/config_all_pollutants.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)

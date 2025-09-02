import os
import sys
import argparse
import glob
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from src.config_manager import ConfigManager
from src.datamodule_fixed import AQNetDataModule
from src.model_multipollutants import MultiPollutantLightningModule


def find_latest_checkpoint(log_dir="logs/multipollutants_climax_ddp"):
    """Find the latest checkpoint from multipollutant training"""
    checkpoint_pattern = os.path.join(log_dir, "version_*/checkpoints/*.ckpt")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print("⚠️  Aucun checkpoint trouvé dans", log_dir)
        return None
    
    # Sort by modification time to get the latest
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    latest_checkpoint = checkpoints[0]
    
    print(f"🔄 Dernier checkpoint trouvé: {latest_checkpoint}")
    return latest_checkpoint


def main(config_path, checkpoint_path=None):
    print("🔄 REPRISE AQ_NET2 - PRÉDICTION MULTI-POLLUANTS DEPUIS CHECKPOINT")
    
    # Initial setup
    cfg_mgr = ConfigManager(config_path)
    config = cfg_mgr.config
    
    print(f"📋 Configuration: {config_path}")
    print(f"📊 Résolution: {config['model']['img_size']}")
    print(f"🔧 Variables: {len(config['data']['variables'])}")
    print(f"🎯 Cibles: {config['data']['target_variables']} ({len(config['data']['target_variables'])} polluants)")
    print("🇨🇳 MASQUE CHINE ACTIVÉ dans la loss function")
    
    # Find latest checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        
    if checkpoint_path is None:
        print("❌ Aucun checkpoint trouvé. Impossible de reprendre l'entraînement.")
        sys.exit(1)
        
    print(f"🔄 Reprise depuis: {checkpoint_path}")
    
    # Initialize Data Module
    print("📁 Initialisation du DataModule...")
    data_module = AQNetDataModule(config)
    
    # Initialize Model from checkpoint
    print("🧠 Chargement du modèle multi-polluants depuis checkpoint...")
    model = MultiPollutantLightningModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        strict=False
    )
    print("✅ Modèle multi-polluants chargé depuis checkpoint")
    
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
    print("🔄 REPRISE DE L'ENTRAÎNEMENT MULTI-POLLUANTS")
    print("="*60)
    print(f"🔄 Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"🎯 Polluants: {', '.join(config['data']['target_variables'])}")
    print(f"🚀 Horizons: {config['data']['forecast_days']} jours")
    print(f"⚡ GPUs: {config['train']['devices']}")
    print(f"📦 Batch size: {config['train']['batch_size']} par GPU")
    print("="*60 + "\n")
    
    # Resume training from checkpoint
    trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    
    print("\n🎉 ENTRAÎNEMENT TERMINÉ!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQ_Net2 Multi-Pollutant Training Resume")
    parser.add_argument("--config", type=str, default="configs/config_all_pollutants.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from (auto-detect if not specified)")
    args = parser.parse_args()
    
    main(args.config, args.checkpoint)

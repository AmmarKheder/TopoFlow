import os
import sys
import argparse
import subprocess
from datetime import timedelta
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy
from src.config_manager import ConfigManager
from src.datamodule_fixed import AQNetDataModule
from src.model_multipollutants import MultiPollutantLightningModule as PM25LightningModule


def main(config_path):
    # # # # #  FIX DEVICE MISMATCH - CRITICAL FOR DDP (LUMI/SLURM compatible)
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_LOCALID" in os.environ:  # fallback SLURM
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        local_rank = 0  # default (single-GPU run)

    # SLURM handles GPU binding - don't restrict visibility per-process
    # (Setting ROCR_VISIBLE_DEVICES per-process breaks DDP on LUMI)

    torch.cuda.set_device(local_rank)
    print(f"# # # #  Bound process to cuda:{local_rank} "
          f"(LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
          f"SLURM_LOCALID={os.environ.get('SLURM_LOCALID')})")
    
    print("# # # #  D�# MARRAGE AQ_NET2 - PR�# DICTION MULTI-POLLUANTS")
    
    # Initial setup
    cfg_mgr = ConfigManager(config_path)
    config = cfg_mgr.config
    
    print(f"# # # #  Configuration: {config_path}")
    print(f"# # # #  Résolution: {config['model']['img_size']}")
    print(f"# # # #  Variables: {len(config['data']['variables'])}")
    print(f"# # # #  Cibles: {config['data']['target_variables']} ({len(config['data']['target_variables'])} polluants)")
    print("# # # # # # # #  MASQUE CHINE ACTIV�#  dans la loss function")
    
    # Initialize Data Module
    print("# # # #  Initialisation du DataModule...")
    data_module = AQNetDataModule(config)
    
    # Initialize Model (with optional checkpoint loading)
    print("# # # #  Initialisation du modèle multi-polluants...")

    # Create model (Lightning will load checkpoint automatically via trainer.fit(ckpt_path=...))
    model = PM25LightningModule(config=config)

    checkpoint_path = config.get('model', {}).get('checkpoint_path', None)
    if checkpoint_path:
        print(f"# # # #  Will resume from checkpoint: {checkpoint_path}")
    else:
        print("# # # #  Training from scratch (no checkpoint)")

    print("# # # #  Modèle multi-polluants initialisé")
    
    # Loggers (TensorBoard + CSV)
    print("# # # #  Configuration des loggers (TensorBoard + CSV)...")
    
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

    # ============================================
    # CRITICAL FIX: Properly configure DDP strategy
    # ============================================
    # Create DDPStrategy with find_unused_parameters from config
    find_unused = config['train'].get('find_unused_parameters', False)

    if config['train']['strategy'] == 'ddp' and config["train"]["num_nodes"] > 1:
        strategy = DDPStrategy(
            find_unused_parameters=find_unused,
            timeout=timedelta(seconds=7200)  # 2 hours timeout for 400 GPUs (must be timedelta)
        )
        print(f"# # # #  DDPStrategy configured: find_unused_parameters={find_unused}, timeout=2h")
    else:
        strategy = config['train']['strategy']

    # Trainer
    trainer = pl.Trainer(
        num_nodes=config["train"]["num_nodes"],
        devices=config['train']['devices'],
        accelerator=config['train']['accelerator'],
        strategy=strategy,  # Use configured DDPStrategy object
        precision=config['train']['precision'],
        max_epochs=config['train']['epochs'],
        max_steps=config['train'].get('max_steps', -1),  # Use max_steps if specified
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        gradient_clip_val=config['train']['gradient_clip_val'],
        accumulate_grad_batches=config['train']['accumulate_grad_batches'],
        log_every_n_steps=config['train']['log_every_n_steps'],
        val_check_interval=config['train']['val_check_interval'],
        default_root_dir=config['lightning']['trainer']['default_root_dir'],
        enable_checkpointing=config['lightning']['trainer']['enable_checkpointing'],
        enable_model_summary=config['lightning']['trainer']['enable_model_summary'],
        num_sanity_val_steps=2  # Increased from 1 to 2 for better stability check
    )
    
    print("\n" + "="*60)
    print("# # # �# # �# # # # # #  D�# MARRAGE DE L'ENTRA�# NEMENT MULTI-POLLUANTS")
    print("="*60)
    print(f"# # # #  Polluants: {', '.join(config['data']['target_variables'])}")
    print(f"# # # #  Horizons: {config['data']['forecast_hours']} heures")
    print(f"# # � GPUs: {config['train']['devices']}")
    print(f"# # # #  Batch size: {config['train']['batch_size']} par GPU")
    print("# # # #  SANITY CHECK: 1 step seulement (debug device mismatch)")

    print("="*60 + "\n")

    # Resume training from checkpoint if specified
    ckpt_path = config['model'].get('checkpoint_path', None)

    if ckpt_path:
        print(f"\n# # # #  RESUME training from checkpoint: {ckpt_path}")
        print("# # # #  This will load: model weights + optimizer state + LR scheduler + step counter")
        print("# # # #  Training will continue EXACTLY where it left off")

        # RESUME: Pass ckpt_path to trainer.fit() to load everything
        # PyTorch Lightning will handle loading model + optimizer + scheduler
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
    else:
        print("# # # #  Training from scratch (no checkpoint)")
        trainer.fit(model, data_module)
    
    print("\n# # # #  ENTRA�# NEMENT TERMIN�# !")
    
    # ============================================
    # # # # #  LANCEMENT AUTOMATIQUE DES TESTS
    # ============================================
    print("\n" + "="*60)
    print("# # # #  LANCEMENT AUTOMATIQUE DU TEST (2018)")
    print("="*60)
    print("# # # #  Recherche du meilleur checkpoint...")
    print("# # # #  �# valuation sur l'année test 2018...")
    print("="*60 + "\n")
    
    try:
        # Déterminer le répertoire de logs
        log_dir = "logs/multipollutants_climax_ddp"
        
        # Lancer le test automatiquement
        test_cmd = [
            "python", "scripts/auto_test_after_training.py",
            "--config", config_path,
            "--log_dir", log_dir,
            "--gpus", str(config['train']['devices']) if isinstance(config['train']['devices'], int) else "1"
        ]
        
        print(f"# # # #  Commande de test: {' '.join(test_cmd)}")
        
        # Exécuter le test
        result = subprocess.run(
            test_cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print("# # #  �# VALUATION TEST R�# USSIE!")
            print("# # # #  Résultats du test:")
            print(result.stdout)
        else:
            print("# # # # # #  ERREUR LORS DU TEST:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            print(f"Code de retour: {result.returncode}")
            
    except Exception as e:
        print(f"# �#  ERREUR lors du lancement automatique du test: {str(e)}")
        print("# # # #  Vous pouvez lancer le test manuellement avec:")
        print(f"python scripts/auto_test_after_training.py --config {config_path} --log_dir logs/multipollutants_climax_ddp")
    
    print("\n" + "="*60)
    print("# # # #  PIPELINE COMPLET TERMIN�#  (ENTRA�# NEMENT + TEST)")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQ_Net2 Multi-Pollutant Training")
    parser.add_argument("--config", type=str, default="configs/config_all_pollutants.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)

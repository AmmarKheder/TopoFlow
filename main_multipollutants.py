import os
import sys
import argparse
import subprocess
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
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

    torch.cuda.set_device(local_rank)
    print(f"# # # #  Bound process to cuda:{local_rank} "
          f"(LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
          f"SLURM_LOCALID={os.environ.get('SLURM_LOCALID')})")
    
    print("# # # #  D√# MARRAGE AQ_NET2 - PR√# DICTION MULTI-POLLUANTS")
    
    # Initial setup
    cfg_mgr = ConfigManager(config_path)
    config = cfg_mgr.config
    
    print(f"# # # #  Configuration: {config_path}")
    print(f"# # # #  R√©solution: {config['model']['img_size']}")
    print(f"# # # #  Variables: {len(config['data']['variables'])}")
    print(f"# # # #  Cibles: {config['data']['target_variables']} ({len(config['data']['target_variables'])} polluants)")
    print("# # # # # # # #  MASQUE CHINE ACTIV√#  dans la loss function")
    
    # Initialize Data Module
    print("# # # #  Initialisation du DataModule...")
    data_module = AQNetDataModule(config)
    
    # TRANSFER LEARNING: Initialize Model from EO checkpoint
    print("# # # #  Initialisation du mod√# le multi-polluants...")
    # Load EO checkpoint for transfer learning
    checkpoint_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"
    print(f"# # # #  Loading EO checkpoint: {checkpoint_path}")
    model = PM25LightningModule.load_from_checkpoint(checkpoint_path, config=config, strict=False)
    print("# # #  Mod√# le multi-polluants initialis√©")
    
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
    
    # Trainer
    trainer = pl.Trainer(
        num_nodes=config["train"]["num_nodes"],
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
        enable_model_summary=config['lightning']['trainer']['enable_model_summary'],
        num_sanity_val_steps=1  # # # # #  TEST: 1 seul step pour debug device mismatch
    )
    
    print("\n" + "="*60)
    print("# # # É# # ç# # # # # #  D√# MARRAGE DE L'ENTRA√# NEMENT MULTI-POLLUANTS")
    print("="*60)
    print(f"# # # #  Polluants: {', '.join(config['data']['target_variables'])}")
    print(f"# # # #  Horizons: {config['data']['forecast_hours']} heures")
    print(f"# # ° GPUs: {config['train']['devices']}")
    print(f"# # # #  Batch size: {config['train']['batch_size']} par GPU")
    print("# # # #  SANITY CHECK: 1 step seulement (debug device mismatch)")
    print("="*60 + "\n")
    
    # Start training
    trainer.fit(model, data_module)
    
    print("\n# # # #  ENTRA√# NEMENT TERMIN√# !")
    
    # ============================================
    # # # # #  LANCEMENT AUTOMATIQUE DES TESTS
    # ============================================
    print("\n" + "="*60)
    print("# # # #  LANCEMENT AUTOMATIQUE DU TEST (2018)")
    print("="*60)
    print("# # # #  Recherche du meilleur checkpoint...")
    print("# # # #  √# valuation sur l'ann√©e test 2018...")
    print("="*60 + "\n")
    
    try:
        # D√©terminer le r√©pertoire de logs
        log_dir = "logs/multipollutants_climax_ddp"
        
        # Lancer le test automatiquement
        test_cmd = [
            "python", "scripts/auto_test_after_training.py",
            "--config", config_path,
            "--log_dir", log_dir,
            "--gpus", str(config['train']['devices']) if isinstance(config['train']['devices'], int) else "1"
        ]
        
        print(f"# # # #  Commande de test: {' '.join(test_cmd)}")
        
        # Ex√©cuter le test
        result = subprocess.run(
            test_cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print("# # #  √# VALUATION TEST R√# USSIE!")
            print("# # # #  R√©sultats du test:")
            print(result.stdout)
        else:
            print("# # # # # #  ERREUR LORS DU TEST:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            print(f"Code de retour: {result.returncode}")
            
    except Exception as e:
        print(f"# ù#  ERREUR lors du lancement automatique du test: {str(e)}")
        print("# # # #  Vous pouvez lancer le test manuellement avec:")
        print(f"python scripts/auto_test_after_training.py --config {config_path} --log_dir logs/multipollutants_climax_ddp")
    
    print("\n" + "="*60)
    print("# # # #  PIPELINE COMPLET TERMIN√#  (ENTRA√# NEMENT + TEST)")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQ_Net2 Multi-Pollutant Training")
    parser.add_argument("--config", type=str, default="configs/config_all_pollutants.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)

#!/bin/bash
#SBATCH --job-name=TEST_RESUME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001079
#SBATCH --output=logs/TEST_RESUME_%j.out
#SBATCH --error=logs/TEST_RESUME_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export PL_TORCH_DISTRIBUTED_BACKEND=nccl
export PL_DISABLE_FORK_DETECTION=1

source venv_pytorch_rocm/bin/activate

echo "=========================================="
echo "🔥 TEST RESUME - 1 NODE, 8 GPUs"
echo "=========================================="
echo "Checkpoint: version_47 val_loss=0.3557"
echo "Test: 10 steps d'entraînement pour voir la loss"
echo "=========================================="

# Créer un script Python de test
cat > /tmp/test_resume_quick_${SLURM_JOB_ID}.py << 'ENDPYTHON'
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy
import sys
import os
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

# Fix device binding for SLURM
if "SLURM_LOCALID" in os.environ:
    local_rank = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(local_rank)

from src.config_manager import ConfigManager
from src.datamodule_fixed import AQNetDataModule
from src.model_multipollutants import MultiPollutantLightningModule

# Config
config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config

# Override pour test rapide
config['train']['max_steps'] = 10  # Seulement 10 steps
config['train']['val_check_interval'] = 5  # Validation après 5 steps
config['train']['log_every_n_steps'] = 1  # Log chaque step
config['train']['num_nodes'] = 1
config['train']['devices'] = 8

print("\n" + "="*100)
print("📊 TEST RESUME CONFIGURATION")
print("="*100)
print(f"Checkpoint: {config['model']['checkpoint_path']}")
print(f"Max steps: {config['train']['max_steps']}")
print(f"Batch size: {config['train']['batch_size']}")
print(f"Devices: {config['train']['devices']}")
print("="*100 + "\n")

# Data
data_module = AQNetDataModule(config)

# Model
model = MultiPollutantLightningModule(config=config)

# Logger
tb_logger = pl_loggers.TensorBoardLogger(
    save_dir="logs/",
    name="test_resume_quick"
)

# Trainer
strategy = DDPStrategy(
    find_unused_parameters=config['train'].get('find_unused_parameters', False),
    timeout=7200
)

trainer = pl.Trainer(
    max_steps=config['train']['max_steps'],
    accelerator='gpu',
    devices=config['train']['devices'],
    num_nodes=1,
    strategy=strategy,
    precision=32,
    logger=tb_logger,
    enable_checkpointing=False,  # Pas de checkpoints pour le test
    log_every_n_steps=1,
    val_check_interval=5
)

# RESUME depuis checkpoint
ckpt_path = config['model'].get('checkpoint_path', None)

if ckpt_path:
    print(f"\n🔥 RESUME depuis: {ckpt_path}\n")
    trainer.fit(model, data_module, ckpt_path=ckpt_path)
else:
    print("\n⚠️  Pas de checkpoint - training from scratch\n")
    trainer.fit(model, data_module)

print("\n" + "="*100)
print("✅ TEST TERMINÉ")
print("="*100)
print("\n📊 VÉRIFIER LES LOGS:")
print("   - Si train_loss commence autour de 0.3-0.5: ✅ RESUME OK")
print("   - Si train_loss commence autour de 3-4:     ❌ PROBLÈME")
print("="*100 + "\n")
ENDPYTHON

# Lancer le test
srun bash -c "source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate && python /tmp/test_resume_quick_${SLURM_JOB_ID}.py"

echo ""
echo "=========================================="
echo "✅ TEST SLURM TERMINÉ - Check logs/TEST_RESUME_*.out"
echo "=========================================="

#!/bin/bash
#SBATCH --job-name=TEST_FIX
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001079
#SBATCH --output=logs/TEST_FIX_%j.out
#SBATCH --error=logs/TEST_FIX_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2
source venv_pytorch_rocm/bin/activate

python3 << 'EOF'
import torch
import sys
import os
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

# Fix GPU
if "SLURM_LOCALID" in os.environ:
    torch.cuda.set_device(int(os.environ["SLURM_LOCALID"]))

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule
from src.datamodule_fixed import AQNetDataModule

print("="*100)
print("TEST RESUME: 10 BATCHES")
print("="*100)

config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config
config['data']['num_workers'] = 0  # Pas de multiprocessing
ckpt_path = config['model']['checkpoint_path']

# Load model
model = MultiPollutantLightningModule.load_from_checkpoint(ckpt_path, config=config, strict=False)
model.cuda()
model.train()

# Load data
data_module = AQNetDataModule(config)
data_module.setup('fit')
train_loader = data_module.train_dataloader()

# Test sur 10 batches
losses = []
for i, batch in enumerate(train_loader):
    if i >= 10:
        break

    # Move to GPU
    if len(batch) == 3:
        x, y, lead_times = batch
        x = x.cuda()
        y = y.cuda()
    elif len(batch) == 4:
        x, y, lead_times, variables = batch
        x = x.cuda()
        y = y.cuda()
    else:
        x, y, lead_times, variables, _ = batch
        x = x.cuda()
        y = y.cuda()

    # Forward
    result = model.training_step(batch, i)
    if isinstance(result, dict):
        loss = result['loss'].item()
    else:
        loss = result.item()

    losses.append(loss)
    print(f"Batch {i+1}/10: loss={loss:.4f}")

print("\n" + "="*100)
print("RÉSULTATS")
print("="*100)
avg_loss = sum(losses) / len(losses)
print(f"Loss moyenne sur 10 batches: {avg_loss:.4f}")
print(f"Loss attendue (checkpoint):   0.3557")
print(f"Ratio: {avg_loss / 0.3557:.2f}x")

if avg_loss < 0.6:
    print("\n✅✅✅ RESUME FONCTIONNE!")
else:
    print(f"\n❌❌❌ PROBLÈME - Loss {avg_loss / 0.3557:.1f}x trop élevée")

print("="*100)
EOF

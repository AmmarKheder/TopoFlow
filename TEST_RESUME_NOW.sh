#!/bin/bash
# TEST RAPIDE SANS MULTIPROCESSING

cd /scratch/project_462000640/ammar/aq_net2
source venv_pytorch_rocm/bin/activate

# Disable multiprocessing
export OMP_NUM_THREADS=1

python3 << 'ENDPYTHON'
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule
from src.datamodule_fixed import AQNetDataModule

print("="*100)
print("🔥 TEST: LOSS SUR 1 BATCH DEPUIS CHECKPOINT")
print("="*100)

# Load config
config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config
ckpt_path = config['model']['checkpoint_path']

# Override num_workers to avoid multiprocessing issues
config['data']['num_workers'] = 0

print(f"\n✅ Checkpoint: {ckpt_path}")

# Load checkpoint
print("\n1️⃣ Loading checkpoint...")
model = MultiPollutantLightningModule.load_from_checkpoint(
    ckpt_path,
    config=config,
    strict=False
)

# Load data
print("\n2️⃣ Loading ONE batch...")
data_module = AQNetDataModule(config)
data_module.setup('fit')
train_loader = data_module.train_dataloader()
batch = next(iter(train_loader))

# Test in EVAL mode
print("\n3️⃣ Forward pass (EVAL mode)...")
model.eval()

with torch.no_grad():
    outputs = model.training_step(batch, 0)
    loss_eval = outputs['loss'].item()
    print(f"   Loss (eval): {loss_eval:.6f}")

# Test in TRAIN mode
print("\n4️⃣ Forward pass (TRAIN mode)...")
model.train()
outputs = model.training_step(batch, 0)
loss_train = outputs['loss'].item()
print(f"   Loss (train): {loss_train:.6f}")

# Verdict
print("\n" + "="*100)
print("📊 VERDICT")
print("="*100)

expected = 0.3557
diff = abs(loss_train - expected)

print(f"Loss checkpoint (expected): {expected:.4f}")
print(f"Loss actuelle (train):      {loss_train:.4f}")
print(f"Différence:                 {diff:.4f}")

if diff < 0.05:
    print("\n✅✅✅ CHECKPOINT OK - Le modèle fonctionne normalement!")
    print("Le problème est ailleurs (scheduler? data order? autre?)")
elif diff < 0.2:
    print("\n⚠️  LOSS UN PEU PLUS ÉLEVÉE - Acceptable")
else:
    print(f"\n❌❌❌ LOSS TROP ÉLEVÉE - PROBLÈME CRITIQUE!")
    print(f"Ratio: {loss_train/expected:.2f}x trop élevé")

print("="*100)
ENDPYTHON

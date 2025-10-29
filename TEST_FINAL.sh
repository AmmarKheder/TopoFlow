#!/bin/bash
cd /scratch/project_462000640/ammar/aq_net2
source venv_pytorch_rocm/bin/activate

python3 << 'ENDPYTHON'
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule
from src.datamodule_fixed import AQNetDataModule

print("\n" + "="*100)
print("🔥 TEST FINAL - LOSS DU CHECKPOINT")
print("="*100)

config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config
config['data']['num_workers'] = 0  # No multiprocessing
ckpt_path = config['model']['checkpoint_path']

print(f"\nCheckpoint: {ckpt_path}")

# Load model
model = MultiPollutantLightningModule.load_from_checkpoint(ckpt_path, config=config, strict=False)

# Load data
data_module = AQNetDataModule(config)
data_module.setup('fit')
train_loader = data_module.train_dataloader()

# Get batch
batch = next(iter(train_loader))

# Training step
model.train()
try:
    result = model.training_step(batch, 0)
    if isinstance(result, dict) and 'loss' in result:
        loss = result['loss']
        if hasattr(loss, 'item'):
            loss_val = loss.item()
        else:
            loss_val = float(loss)
    else:
        loss_val = float(result)

    expected = 0.3557
    print(f"\n📊 RÉSULTATS:")
    print(f"   Loss checkpoint (attendue): {expected:.4f}")
    print(f"   Loss actuelle:               {loss_val:.4f}")
    print(f"   Différence:                  {abs(loss_val - expected):.4f}")

    if abs(loss_val - expected) < 0.1:
        print(f"\n✅ CHECKPOINT BON - Loss normale!")
    else:
        print(f"\n❌ PROBLÈME - Loss {loss_val/expected:.2f}x trop élevée")

except Exception as e:
    print(f"\n❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()

print("="*100 + "\n")
ENDPYTHON

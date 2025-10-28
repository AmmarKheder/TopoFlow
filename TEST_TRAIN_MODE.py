#!/usr/bin/env python3
"""
Test en mode TRAIN (pas eval) pour voir si c'est le problème
"""
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule
from src.datamodule_fixed import AQNetDataModule

print("="*100)
print("TEST: Checkpoint loading in TRAIN MODE (not eval)")
print("="*100)

# Config
config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config
config['data']['num_workers'] = 0

ckpt_path = config['model']['checkpoint_path']

# Load model
print("📥 Loading model from checkpoint...")
model = MultiPollutantLightningModule.load_from_checkpoint(
    ckpt_path,
    config=config,
    strict=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# TEST 1: EVAL MODE
print("\n" + "="*100)
print("🔥 TEST 1: EVAL MODE (Dropout OFF)")
print("="*100)
model.eval()

data_module = AQNetDataModule(config)
data_module.setup('fit')
train_loader = data_module.train_dataloader()

losses_eval = []
with torch.no_grad():
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        result = model.training_step(batch, i)
        loss = result['loss'].item() if isinstance(result, dict) else result.item()
        losses_eval.append(loss)
        print(f"  Batch {i+1}: loss={loss:.4f}")

avg_loss_eval = sum(losses_eval) / len(losses_eval)
print(f"\n📊 Average loss (EVAL mode): {avg_loss_eval:.4f}")

# TEST 2: TRAIN MODE
print("\n" + "="*100)
print("🔥 TEST 2: TRAIN MODE (Dropout ON)")
print("="*100)
model.train()

losses_train = []
with torch.no_grad():  # Still no grad to avoid memory issues
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        result = model.training_step(batch, i)
        loss = result['loss'].item() if isinstance(result, dict) else result.item()
        losses_train.append(loss)
        print(f"  Batch {i+1}: loss={loss:.4f}")

avg_loss_train = sum(losses_train) / len(losses_train)
print(f"\n📊 Average loss (TRAIN mode): {avg_loss_train:.4f}")

# COMPARISON
print("\n" + "="*100)
print("📊 COMPARISON")
print("="*100)
print(f"EVAL mode:     {avg_loss_eval:.4f}")
print(f"TRAIN mode:    {avg_loss_train:.4f}")
print(f"Expected:      0.35")
print(f"Ratio (EVAL):  {avg_loss_eval / 0.35:.2f}x")
print(f"Ratio (TRAIN): {avg_loss_train / 0.35:.2f}x")

print("\n" + "="*100)
if avg_loss_train < 0.6:
    print("✅✅✅ SUCCÈS ! TRAIN mode donne la bonne loss!")
    print("     Le problème était qu'on testait en EVAL mode.")
elif avg_loss_eval < 0.6:
    print("✅✅✅ SUCCÈS ! EVAL mode donne la bonne loss!")
    print("     (Surprenant, mais le checkpoint fonctionne!)")
else:
    print("❌❌❌ PROBLÈME persiste dans les deux modes.")
    print(f"     Les loss sont toujours trop élevées.")
print("="*100)

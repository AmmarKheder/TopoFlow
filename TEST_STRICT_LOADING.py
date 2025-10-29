#!/usr/bin/env python3
"""
Test de chargement avec strict=True pour voir si tous les poids sont chargés
"""
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule
from src.datamodule_fixed import AQNetDataModule

print("="*100)
print("TEST STRICT LOADING")
print("="*100)

# Config
config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config
config['data']['num_workers'] = 0

ckpt_path = config['model']['checkpoint_path']
print(f"\nCheckpoint: {ckpt_path}\n")

# Test 1: Load with strict=True
print("🔥 TEST 1: Loading with strict=True")
try:
    model_strict = MultiPollutantLightningModule.load_from_checkpoint(
        ckpt_path,
        config=config,
        strict=True  # STRICT!
    )
    print("✅ Loaded successfully with strict=True!")
    print("   All weights match perfectly!")
except Exception as e:
    print(f"❌ Failed with strict=True:")
    print(f"   {str(e)[:500]}")
    print("\n   This means some keys don't match between checkpoint and model.")

# Test 2: Load with strict=False (current behavior)
print("\n🔥 TEST 2: Loading with strict=False")
try:
    model_loose = MultiPollutantLightningModule.load_from_checkpoint(
        ckpt_path,
        config=config,
        strict=False
    )
    print("✅ Loaded successfully with strict=False!")
except Exception as e:
    print(f"❌ Failed even with strict=False: {e}")

# Test 3: Test on actual data
print("\n🔥 TEST 3: Testing on actual batches")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model_loose = model_loose.to(device)
model_loose.eval()

# Create data
data_module = AQNetDataModule(config)
data_module.setup('fit')
train_loader = data_module.train_dataloader()

# Test on 5 batches
losses = []
with torch.no_grad():
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break

        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        result = model_loose.training_step(batch, i)
        loss = result['loss'].item() if isinstance(result, dict) else result.item()
        losses.append(loss)
        print(f"  Batch {i+1}: loss={loss:.4f}")

avg_loss = sum(losses) / len(losses)
print(f"\n📊 Average loss: {avg_loss:.4f}")
print(f"📊 Expected:     ~0.35")
print(f"📊 Ratio:        {avg_loss / 0.35:.2f}x")

if avg_loss < 0.6:
    print("\n✅✅✅ CHECKPOINT LOADS CORRECTLY!")
else:
    print(f"\n❌❌❌ PROBLEM! Loss is {avg_loss / 0.35:.1f}x too high")

print("="*100)

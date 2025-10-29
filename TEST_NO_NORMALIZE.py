#!/usr/bin/env python3
"""
Test SANS normalization pour voir si c'est le problème
"""
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule
from src.datamodule_fixed import AQNetDataModule

print("="*100)
print("TEST: Checkpoint loading WITHOUT NORMALIZATION")
print("="*100)
print("\nHypothèse: Le problème vient des stats de normalisation qui ont changé")
print("Si ce test donne une loss complètement différente, ça confirme que")
print("le problème est la normalisation.\n")

# Config
config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config
config['data']['num_workers'] = 0

# DISABLE NORMALIZATION
config['data']['normalize'] = False
print("⚠️  NORMALIZATION DISABLED for this test\n")

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
model.eval()

# Create data WITHOUT normalization
print("📊 Creating datamodule (normalize=False)...")
data_module = AQNetDataModule(config)
data_module.setup('fit')
train_loader = data_module.train_dataloader()

# Test on 5 batches
losses = []
print("\n🧪 Testing on 5 batches WITHOUT normalization:")
with torch.no_grad():
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break

        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        result = model.training_step(batch, i)
        loss = result['loss'].item() if isinstance(result, dict) else result.item()
        losses.append(loss)
        print(f"  Batch {i+1}: loss={loss:.4f}")

avg_loss = sum(losses) / len(losses)
print(f"\n📊 Average loss WITHOUT normalization: {avg_loss:.4f}")
print(f"📊 Previous loss WITH normalization:    4.10")
print(f"📊 Expected loss (checkpoint):          0.35")

print("\n" + "="*100)
print("CONCLUSION:")
if abs(avg_loss - 4.10) > 2.0:
    print("✅ La loss a CHANGÉ significativement!")
    print("   Cela CONFIRME que le problème est la normalisation.")
    print("\n🔧 SOLUTION:")
    print("   Il faut sauvegarder les stats de normalisation dans le checkpoint")
    print("   et les recharger lors du resume.")
else:
    print("❌ La loss est similaire même sans normalisation.")
    print("   Le problème doit être ailleurs.")
print("="*100)

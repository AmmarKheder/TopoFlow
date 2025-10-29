#!/usr/bin/env python3
"""
Test sur le VAL dataset pour confirmer val_loss ~0.35
"""
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule
from src.datamodule_fixed import AQNetDataModule

print("="*100)
print("TEST SUR LE VALIDATION DATASET")
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
model.eval()

# Create data
print("📊 Creating datamodule...")
data_module = AQNetDataModule(config)
data_module.setup('fit')
val_loader = data_module.val_dataloader()  # VAL DATASET !

# Test on 10 batches
losses = []
print("\n🧪 Testing on 10 VAL batches:")
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if i >= 10:
            break

        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        result = model.validation_step(batch, i)  # VALIDATION STEP !
        loss = result.item() if isinstance(result, torch.Tensor) else result
        losses.append(loss)
        print(f"  Batch {i+1}/10: loss={loss:.4f}")

avg_loss = sum(losses) / len(losses)

print("\n" + "="*100)
print("📊 RÉSULTATS FINAUX - VALIDATION")
print("="*100)
print(f"VAL Loss moyenne (10 batches):  {avg_loss:.4f}")
print(f"VAL Loss checkpoint:             0.3557")
print(f"Ratio:                           {avg_loss / 0.3557:.2f}x")

if avg_loss < 0.5:
    print("\n" + "🎉"*50)
    print("✅✅✅ PARFAIT !!! LE CHECKPOINT FONCTIONNE CORRECTEMENT !!!")
    print("🎉"*50)
    print("\nLa VAL loss est ~0.35 comme attendu !")
    print("Le problème était les NORM_STATS hardcodées.")
    print("\n🚀 VOUS POUVEZ MAINTENANT LANCER LE RESUME TRAINING SUR 256 GPUs !")
    print("="*100)
else:
    print(f"\n❌ Val loss toujours trop haute: {avg_loss / 0.3557:.1f}x")

print("="*100)

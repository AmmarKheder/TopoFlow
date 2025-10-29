#!/usr/bin/env python3
"""
Test avec la normalisation FIXÉE (calcul dynamique comme en septembre)
"""
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule
from src.datamodule_fixed import AQNetDataModule

print("="*100)
print("TEST AVEC NORMALISATION FIXÉE (calcul dynamique)")
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
train_loader = data_module.train_dataloader()

# Test on 10 batches
losses = []
print("\n🧪 Testing on 10 batches:")
with torch.no_grad():
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break

        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        result = model.training_step(batch, i)
        loss = result['loss'].item() if isinstance(result, dict) else result.item()
        losses.append(loss)
        print(f"  Batch {i+1}/10: loss={loss:.4f}")

avg_loss = sum(losses) / len(losses)

print("\n" + "="*100)
print("📊 RÉSULTATS FINAUX")
print("="*100)
print(f"Loss moyenne (10 batches):  {avg_loss:.4f}")
print(f"Loss attendue (checkpoint):  0.3557")
print(f"Ratio:                       {avg_loss / 0.3557:.2f}x")

if avg_loss < 0.6:
    print("\n" + "🎉"*50)
    print("✅✅✅ SUCCÈS !!! LE CHECKPOINT FONCTIONNE !!!")
    print("🎉"*50)
    print("\nLa loss est CORRECTE ! Le problème était les NORM_STATS hardcodées !")
    print("Le checkpoint utilisait un calcul dynamique des stats.")
    print("\n✅ Vous pouvez maintenant REPRENDRE L'ENTRAÎNEMENT !")
else:
    print(f"\n❌ Toujours un problème - Loss {avg_loss / 0.3557:.1f}x trop haute")

print("="*100)

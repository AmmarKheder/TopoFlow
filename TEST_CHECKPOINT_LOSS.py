#!/usr/bin/env python3
"""
Test rapide pour voir quelle loss le checkpoint produit vraiment.
"""
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule
from src.datamodule_fixed import AQNetDataModule

print("="*100)
print("TEST CHECKPOINT LOSS")
print("="*100)

# Config
config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config
config['data']['num_workers'] = 0  # Avoid multiprocessing

ckpt_path = config['model']['checkpoint_path']
print(f"\nCheckpoint: {ckpt_path}")

# Load model
print("\n📥 Loading model from checkpoint...")
model = MultiPollutantLightningModule.load_from_checkpoint(
    ckpt_path,
    config=config,
    strict=False
)

# Check if we are on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n📍 Device: {device}")
model = model.to(device)
model.eval()

# Create data
print("\n📊 Creating datamodule...")
data_module = AQNetDataModule(config)
data_module.setup('fit')
train_loader = data_module.train_dataloader()
print(f"✅ Datamodule ready: {len(train_loader)} batches")

# Test sur 10 batches
print("\n🧪 Testing on 10 batches...")
losses = []

with torch.no_grad():
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break

        # Move batch to device
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

        # Forward through training_step (which computes loss)
        result = model.training_step(batch, i)

        if isinstance(result, dict):
            loss = result['loss'].item()
        else:
            loss = result.item()

        losses.append(loss)
        print(f"  Batch {i+1}/10: loss={loss:.4f}")

print("\n" + "="*100)
print("📊 RÉSULTATS")
print("="*100)
avg_loss = sum(losses) / len(losses)
print(f"Loss moyenne sur 10 batches: {avg_loss:.4f}")
print(f"Loss du checkpoint (val):     0.3557")
print(f"Ratio:                        {avg_loss / 0.3557:.2f}x")

if avg_loss < 0.6:
    print("\n✅✅✅ CHECKPOINT VALIDE!")
    print("     Le modèle produit des résultats cohérents avec la validation loss.")
    print("     Le problème n'est PAS le chargement du checkpoint.")
    print("     Le 'train_loss=3.840' dans les logs était probablement un artefact d'affichage.")
else:
    print(f"\n❌❌❌ PROBLÈME!")
    print(f"     Loss {avg_loss / 0.3557:.1f}x trop élevée.")
    print("     Le checkpoint ne se charge PAS correctement.")
    print("\n🔍 CAUSES POSSIBLES:")
    print("   1. Les poids du modèle ne sont pas chargés")
    print("   2. Problème de normalisation des données")
    print("   3. Mismatch dans l'architecture du modèle")

print("="*100)

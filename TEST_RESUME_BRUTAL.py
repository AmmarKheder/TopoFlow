#!/usr/bin/env python3
"""
TEST BRUTAL - On charge le checkpoint et on calcule EXACTEMENT la loss sur 1 batch
pour voir si le problème vient du modèle ou du training loop
"""

import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule
from src.datamodule_fixed import AQNetDataModule

if __name__ == '__main__':
    print("="*100)
    print("🔥 TEST BRUTAL - CALCUL DE LA LOSS DIRECTE SUR 1 BATCH")
    print("="*100)

    # Config
    config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
    cfg_mgr = ConfigManager(config_path)
    config = cfg_mgr.config

ckpt_path = config['model']['checkpoint_path']

# 1. Charger le checkpoint
print(f"\n1️⃣ Chargement checkpoint: {ckpt_path}")
model = MultiPollutantLightningModule.load_from_checkpoint(
    ckpt_path,
    config=config,
    strict=False
)
model.eval()

# 2. Charger les données
print("\n2️⃣ Chargement des données...")
data_module = AQNetDataModule(config)
data_module.setup('fit')

# 3. Prendre 1 batch
print("\n3️⃣ Extraction d'un batch...")
train_loader = data_module.train_dataloader()
batch = next(iter(train_loader))

print(f"   Batch type: {type(batch)}")
print(f"   Batch length: {len(batch)}")

# 4. Forward pass
print("\n4️⃣ Forward pass...")
model.eval()

with torch.no_grad():
    if len(batch) == 3:
        x, y, lead_times = batch
        variables = model.model.variables
    elif len(batch) == 4:
        x, y, lead_times, variables = batch
    else:
        x, y, lead_times, variables, _ = batch

    print(f"   Input shape: {x.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Lead times: {lead_times.shape if hasattr(lead_times, 'shape') else lead_times}")

    # Forward
    preds = model(x, lead_times, variables)
    print(f"   Predictions shape: {preds.shape}")

    # Calculate loss manually
    china_mask = model.china_mask.to(x.device)

    # Weighted MSE loss like in training_step
    mse = (preds - y) ** 2

    # Apply China mask
    mse_masked = mse * china_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # Average over masked region
    n_valid = china_mask.sum()
    loss = mse_masked.sum() / (n_valid * y.shape[0] * y.shape[1] * y.shape[2])

    print(f"\n📊 RÉSULTATS:")
    print(f"   Loss calculée: {loss.item():.6f}")
    print(f"   Preds mean: {preds.mean().item():.6f}, std: {preds.std().item():.6f}")
    print(f"   Target mean: {y.mean().item():.6f}, std: {y.std().item():.6f}")
    print(f"   Preds min/max: {preds.min().item():.6f} / {preds.max().item():.6f}")
    print(f"   Target min/max: {y.min().item():.6f} / {y.max().item():.6f}")

# 5. Maintenant tester en mode training (comme dans le vrai training)
print("\n5️⃣ Test en mode TRAINING...")
model.train()

# Refaire le forward
if len(batch) == 3:
    x, y, lead_times = batch
    variables = model.model.variables
elif len(batch) == 4:
    x, y, lead_times, variables = batch
else:
    x, y, lead_times, variables, _ = batch

# Simuler training_step
outputs = model.training_step(batch, 0)

print(f"\n📊 LOSS EN MODE TRAINING:")
print(f"   Loss: {outputs['loss'].item():.6f}")

print("\n" + "="*100)
print("💡 ANALYSE:")
print("="*100)

expected_loss = 0.3557  # val_loss du checkpoint

if abs(outputs['loss'].item() - expected_loss) < 0.1:
    print("✅ LOSS NORMALE - Le modèle est OK!")
    print(f"   Différence: {abs(outputs['loss'].item() - expected_loss):.6f}")
elif outputs['loss'].item() > 1.0:
    print(f"❌ LOSS TROP ÉLEVÉE - PROBLÈME DÉTECTÉ!")
    print(f"   Loss actuelle: {outputs['loss'].item():.6f}")
    print(f"   Loss attendue: ~{expected_loss:.6f}")
    print(f"   Ratio: {outputs['loss'].item() / expected_loss:.2f}x trop élevé")
    print("\n🔍 CAUSES POSSIBLES:")
    print("   1. Les poids ne se sont pas chargés correctement")
    print("   2. Les données sont différentes")
    print("   3. La loss function a changé")
    print("   4. Le masque China n'est pas le même")
else:
    print(f"⚠️  LOSS UN PEU ÉLEVÉE mais pas catastrophique")
    print(f"   Loss actuelle: {outputs['loss'].item():.6f}")
    print(f"   Loss attendue: ~{expected_loss:.6f}")

print("="*100)

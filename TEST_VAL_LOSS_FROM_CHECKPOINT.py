"""
CRITICAL TEST: Verify that checkpoint gives val_loss ≈ 0.356 when evaluated
This will tell us if the problem is with:
1. Model/checkpoint loading (if val_loss is also high)
2. Training loop/data (if val_loss is correct but train_loss is high)
"""
import torch
import yaml
import sys
sys.path.insert(0, 'src')

from model_multipollutants import MultiPollutantModel, MultiPollutantLightningModule
from datamodule_fixed import AQNetDataModule

print("="*80)
print("TEST: VALIDATION LOSS FROM CHECKPOINT")
print("="*80)

# Load config
with open('configs/config_all_pollutants.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['data']['num_workers'] = 0  # Single worker for debugging

# Create data module
print("\n1. Creating data module...")
data_module = AQNetDataModule(config)
data_module.setup('fit')

# Get validation loader
val_loader = data_module.val_dataloader()
print(f"   Val loader: {len(val_loader)} batches")

# Create Lightning module (this will load checkpoint in setup())
print("\n2. Creating Lightning module...")
lightning_module = MultiPollutantLightningModule(config=config)

# Simulate what happens in training: setup() loads checkpoint
ckpt_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"
lightning_module._checkpoint_path_to_load = ckpt_path

print(f"\n3. Loading checkpoint via setup()...")
lightning_module.setup(stage='fit')

# Move to GPU (use cuda:0 on AMD GPUs)
device = torch.device('cuda:0')
lightning_module = lightning_module.to(device)
lightning_module.eval()

print(f"\n4. Computing validation loss on {min(10, len(val_loader))} batches...")
print(f"   Expected: ~0.356 (from checkpoint)")
print(f"   Device: {device}")
print(f"   GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")

total_loss = 0.0
num_batches = 0

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if i >= 10:  # Only test on 10 batches
            break

        x, y, lead_times = batch
        x = x.to(device)
        y = y.to(device)
        lead_times = lead_times.to(device)

        # Forward pass
        variables = tuple(config['data']['variables'])
        y_pred = lightning_module.model(x, lead_times, variables)

        # Compute loss (same as in validation_step)
        china_mask = lightning_module.china_mask.to(dtype=torch.bool, device=y.device).unsqueeze(0).unsqueeze(0)
        loss = lightning_module._masked_mse(y_pred, y, china_mask)

        total_loss += loss.item()
        num_batches += 1

        if i == 0:
            print(f"\n   Batch 0:")
            print(f"     x shape: {x.shape}")
            print(f"     y_pred shape: {y_pred.shape}")
            print(f"     y shape: {y.shape}")
            print(f"     loss: {loss.item():.4f}")

avg_val_loss = total_loss / num_batches

print(f"\n" + "="*80)
print(f"RESULTS:")
print(f"="*80)
print(f"Average validation loss: {avg_val_loss:.4f}")
print(f"Expected (from checkpoint): 0.3557")
print(f"Difference: {abs(avg_val_loss - 0.3557):.4f}")
print(f"")

if abs(avg_val_loss - 0.3557) < 0.05:
    print("✅✅✅ SUCCESS! Val loss matches checkpoint!")
    print("✅ This means the model loaded correctly")
    print("❌ BUT train_loss is still 2.6-3.8 in job 13631845")
    print("")
    print("🔍 ROOT CAUSE ANALYSIS:")
    print("Since val_loss is correct but train_loss is high, the problem is:")
    print("1. Training data is different from validation data")
    print("2. OR: Training uses different settings (dropout, drop_path, etc.)")
    print("3. OR: There's a bug in the training loop")
    print("")
    print("NEXT STEP: Check if drop_path is enabled during training")
    print("           (drop_path=0.1 would add stochastic noise)")
elif avg_val_loss > 0.5:
    print("❌❌❌ FAILED! Val loss is too high!")
    print(f"❌ Expected: 0.3557, Got: {avg_val_loss:.4f}")
    print("")
    print("🔍 ROOT CAUSE ANALYSIS:")
    print("The checkpoint did NOT load correctly, or there's a bug in:")
    print("1. Checkpoint loading (prefix fix, strict=False)")
    print("2. Model architecture mismatch")
    print("3. Data preprocessing (normalization different)")
    print("4. Forward pass (TopoFlow bug)")
else:
    print(f"⚠️  Val loss is close but not exact: {avg_val_loss:.4f}")
    print("   This could be due to:")
    print("   1. Small differences in data batches")
    print("   2. Random initialization of TopoFlow params")
    print("   3. GPU precision differences")

print("="*80)

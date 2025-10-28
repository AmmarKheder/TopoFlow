"""
TEST FINAL: Vérifier la train_loss en mode RESUME avec le nouveau hook
"""
import torch
import yaml
import sys
sys.path.insert(0, 'src')

from model_multipollutants import MultiPollutantLightningModule
import pytorch_lightning as pl
from datamodule import AQNetDataModule

print("="*80)
print("TEST: TRAIN_LOSS avec RESUME (nouveau hook on_load_checkpoint)")
print("="*80)

# Load config
with open('configs/config_all_pollutants.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Simplify
config['train']['devices'] = 1
config['train']['num_nodes'] = 1
config['data']['num_workers'] = 2

# Checkpoint path
ckpt_path = 'logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt'

# Create model
print("\n1. Creating model...")
model = MultiPollutantLightningModule(config)

# Create datamodule
print("\n2. Creating datamodule...")
data_module = AQNetDataModule(config)

# Create trainer
print("\n3. Creating trainer...")
trainer = pl.Trainer(
    devices=1,
    accelerator='cpu',
    max_epochs=10,
    max_steps=50,  # 50 steps to see if loss stays low
    enable_checkpointing=False,
    logger=False,
    enable_model_summary=False,
    num_sanity_val_steps=0
)

# Train with resume
print("\n4. Training with RESUME...")
print("   We expect:")
print("   - on_load_checkpoint hook to be called")
print("   - LRs to match checkpoint (8e-6, 1.6e-4, 4e-5, 8e-5)")
print("   - train_loss to start LOW (~0.35-0.50) and stay low\n")

try:
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    # Check final LRs
    print("\nFinal optimizer LRs:")
    optimizers = trainer.optimizers
    if optimizers:
        opt = optimizers[0]
        for i, pg in enumerate(opt.param_groups):
            print(f"  Group {i}: lr={pg['lr']:.6e}, name={pg.get('name', 'unknown')}")

    print("\n" + "="*80)
    print("CHECK THE TRAIN_LOSS VALUES ABOVE")
    print("="*80)
    print("If train_loss starts at ~0.35-0.50 and stays low:")
    print("  ✅ RESUME works correctly!")
    print("")
    print("If train_loss starts at ~3-4:")
    print("  ❌ RESUME doesn't work - optimizer state not loaded")
    print("="*80)

except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()

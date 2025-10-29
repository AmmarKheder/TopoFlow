"""
TEST: Vérifier que le hook on_load_checkpoint est appelé
"""
import torch
import yaml
import sys
sys.path.insert(0, 'src')

from model_multipollutants import MultiPollutantLightningModule
import pytorch_lightning as pl
from datamodule import AQNetDataModule

print("="*80)
print("TEST: on_load_checkpoint HOOK")
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
    max_steps=10,  # Only 10 steps
    enable_checkpointing=False,
    logger=False,
    enable_model_summary=False,
    num_sanity_val_steps=0
)

# Train with resume (this should trigger on_load_checkpoint hook)
print("\n4. Starting training with RESUME (ckpt_path)...")
print("   This should trigger the on_load_checkpoint hook.\n")

try:
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

    print("\n" + "="*80)
    print("TRAINING STARTED - Check output above for hook messages")
    print("="*80)

    # Check final LRs
    print("\n5. Checking final LRs...")
    optimizers = trainer.optimizers
    if optimizers:
        opt = optimizers[0]
        print(f"   Number of param_groups: {len(opt.param_groups)}")
        for i, pg in enumerate(opt.param_groups):
            print(f"     Group {i}: lr={pg['lr']:.6e}, name={pg.get('name', 'unknown')}")

except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)

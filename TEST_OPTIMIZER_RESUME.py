"""
TEST: Vérifier si l'optimizer state est correctement chargé en mode RESUME
"""
import torch
import yaml
import sys
sys.path.insert(0, 'src')

from model_multipollutants import MultiPollutantModel
import pytorch_lightning as pl

print("="*80)
print("TEST: RESUME - VÉRIFICATION DE L'OPTIMIZER STATE")
print("="*80)

# Load config
with open('configs/config_all_pollutants.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Simplify for testing
config['train']['devices'] = 1
config['train']['num_nodes'] = 1
config['data']['num_workers'] = 2

# Checkpoint path
ckpt_path = 'logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt'

# Load checkpoint to see what's in it
print("\n1. Loading checkpoint to inspect...")
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print(f"   Checkpoint epoch: {checkpoint['epoch']}")
print(f"   Checkpoint global_step: {checkpoint['global_step']}")
print(f"\n   Optimizer state in checkpoint:")
opt_state = checkpoint['optimizer_states'][0]
print(f"     Number of param_groups: {len(opt_state['param_groups'])}")
for i, pg in enumerate(opt_state['param_groups']):
    print(f"       Group {i}: lr={pg['lr']:.6e}, {len(pg['params'])} params")

print(f"\n   LR Scheduler state in checkpoint:")
sch_state = checkpoint['lr_schedulers'][0]
print(f"     last_epoch: {sch_state['last_epoch']}")
print(f"     base_lrs: {sch_state['base_lrs']}")
print(f"     _last_lr: {sch_state['_last_lr']}")

# Create model
print("\n2. Creating NEW model (like in training)...")
model = MultiPollutantModel(config)

# Create a minimal trainer and try to restore from checkpoint
print("\n3. Creating Trainer and RESUMING from checkpoint...")
print("   (This is what trainer.fit(model, data_module, ckpt_path=...) does)")

# Create a minimal trainer
trainer = pl.Trainer(
    devices=1,
    accelerator='cpu',
    max_epochs=10,
    enable_checkpointing=False,
    logger=False,
    enable_model_summary=False
)

# CRITICAL: Setup the model first (this calls configure_optimizers)
# This is what Lightning does internally before loading checkpoint
print("\n4. Setting up model (calls configure_optimizers)...")
trainer.strategy.connect(model)
trainer.strategy.setup(trainer)

# Check what LRs are configured BEFORE loading checkpoint
print("\n5. Optimizer BEFORE loading checkpoint:")
optimizers = trainer.optimizers
if optimizers:
    opt = optimizers[0]
    print(f"   Number of param_groups: {len(opt.param_groups)}")
    for i, pg in enumerate(opt.param_groups):
        print(f"     Group {i}: lr={pg['lr']:.6e}, name={pg.get('name', 'unknown')}")

# Now load checkpoint (this is what trainer.fit(ckpt_path=...) does)
print("\n6. Loading checkpoint state...")
trainer._checkpoint_connector.resume_start(ckpt_path)
trainer._checkpoint_connector.restore_model()
trainer._checkpoint_connector.restore_datamodule()

# CRITICAL: restore_training_state is what loads optimizer + scheduler
trainer._checkpoint_connector.restore_training_state()

# Check LRs AFTER loading checkpoint
print("\n7. Optimizer AFTER loading checkpoint:")
optimizers = trainer.optimizers
if optimizers:
    opt = optimizers[0]
    print(f"   Number of param_groups: {len(opt.param_groups)}")
    for i, pg in enumerate(opt.param_groups):
        print(f"     Group {i}: lr={pg['lr']:.6e}, name={pg.get('name', 'unknown')}")

# Check scheduler state
schedulers = trainer.lr_scheduler_configs
if schedulers:
    sch = schedulers[0].scheduler
    print(f"\n8. LR Scheduler AFTER loading checkpoint:")
    print(f"   last_epoch: {sch.last_epoch}")
    print(f"   base_lrs: {sch.base_lrs}")
    print(f"   _last_lr: {sch._last_lr}")

print("\n" + "="*80)
print("DIAGNOSTIC:")
print("="*80)

# Compare LRs
checkpoint_lrs = [pg['lr'] for pg in opt_state['param_groups']]
current_lrs = [pg['lr'] for pg in opt.param_groups]

print(f"\nLRs in checkpoint: {[f'{lr:.6e}' for lr in checkpoint_lrs]}")
print(f"LRs after loading: {[f'{lr:.6e}' for lr in current_lrs]}")

if checkpoint_lrs == current_lrs:
    print("\n✅ SUCCESS! Optimizer state loaded correctly!")
    print("   LRs match the checkpoint.")
else:
    print("\n❌ FAILED! Optimizer state NOT loaded!")
    print("   LRs don't match - optimizer was reset.")
    print("\n   This explains why train_loss starts high!")
    print("   The optimizer has no momentum/history from the checkpoint.")

print("="*80)

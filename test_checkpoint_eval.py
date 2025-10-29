"""Quick eval of checkpoint WITHOUT physics mask to verify it works"""
import torch
import yaml
from src.model_multipollutants import MultiPollutantLightningModule
from src.datamodule_fixed import AQNetDataModule
import pytorch_lightning as pl

# Load config WITHOUT physics mask
with open('configs/config_baseline_no_physics.yaml') as f:
    config = yaml.safe_load(f)

# Disable physics mask
config['model']['use_physics_mask'] = False

# Create model
model = MultiPollutantLightningModule(config=config)

# Load checkpoint weights
ckpt = torch.load('checkpoint_best_val_loss_0.3552.ckpt', map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=False)

# Create datamodule
dm = AQNetDataModule(config)
dm.setup('fit')

# Create trainer (1 GPU only for quick test)
trainer = pl.Trainer(
    devices=1,
    accelerator='gpu',
    precision=32,
    logger=False
)

# Validate
print("Evaluating checkpoint WITHOUT physics mask...")
result = trainer.validate(model, dm.val_dataloader())
print(f"Val loss: {result[0]['val_loss']:.4f}")

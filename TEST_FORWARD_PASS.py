"""
Test forward pass avec checkpoint version_47
pour vérifier que la loss est correcte
"""
import torch
import yaml
import sys
sys.path.insert(0, 'src')
from model_multipollutants import MultiPollutantModel
from dataloader import MultiPollutantDataModule

print("="*80)
print("TEST: FORWARD PASS AVEC CHECKPOINT")
print("="*80)

# Load config
with open('configs/config_all_pollutants.yaml') as f:
    config = yaml.safe_load(f)

# Create model
print("\n1. Creating model...")
model = MultiPollutantModel(config)

# Load checkpoint
print("\n2. Loading checkpoint...")
ckpt_path = 'logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt'
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Fix prefix and load
state_dict = {}
for key, value in checkpoint['state_dict'].items():
    if key.startswith('model.'):
        state_dict[key[6:]] = value
    else:
        state_dict[key] = value

model.load_state_dict(state_dict, strict=False)
model.eval()

print("\n3. Creating validation dataloader...")
# Override to use single GPU for testing
config['train']['devices'] = 1
config['train']['num_nodes'] = 1
data_module = MultiPollutantDataModule(config)
data_module.setup('fit')

print("\n4. Testing forward pass on validation data...")
val_loader = data_module.val_dataloader()

# Get first batch
batch = next(iter(val_loader))
if len(batch) == 4:
    x, y, lead_times, variables = batch
else:
    x, y, lead_times = batch
    variables = config["data"]["variables"]

print(f"   Batch shapes: x={x.shape}, y={y.shape}")
print(f"   Lead times: {lead_times.unique().tolist()}")

# Forward pass
with torch.no_grad():
    y_pred = model(x, lead_times, variables)

print(f"   Prediction shape: {y_pred.shape}")

# Compute loss (same as in training)
if y.dim() == 3:
    y = y.unsqueeze(1)

china_mask = model.china_mask.to(dtype=torch.bool, device=y.device).unsqueeze(0).unsqueeze(0)
loss = model._masked_mse(y_pred, y, china_mask)

print(f"\n5. Results:")
print(f"   Loss on first validation batch: {loss.item():.4f}")
print(f"   Expected (from checkpoint): ~0.3557")
print(f"   Difference: {abs(loss.item() - 0.3557):.4f}")

print("\n" + "="*80)
if abs(loss.item() - 0.3557) < 0.1:
    print("✅ Loss is close to expected! Model works correctly.")
elif abs(loss.item() - 0.3557) < 1.0:
    print("⚠️ Loss is somewhat close. May need more investigation.")
else:
    print("❌ Loss is FAR from expected! There's a problem.")
    print("   Possible issues:")
    print("   - Data normalization different")
    print("   - Wrong variables order")
    print("   - Model architecture mismatch")
print("="*80)

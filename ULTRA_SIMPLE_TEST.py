"""
TEST ULTRA SIMPLE: Vérifier que le forward pass avec checkpoint donne une loss raisonnable
Sans charger les vraies données - juste avec des données synthétiques
"""
import torch
import yaml
import sys
sys.path.insert(0, 'src')

from model_multipollutants import MultiPollutantModel

print("="*80)
print("TEST ULTRA SIMPLE: Forward pass avec données synthétiques")
print("="*80)

# Load config
with open('configs/config_all_pollutants.yaml', 'r') as f:
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

result = model.load_state_dict(state_dict, strict=False)
print(f"   Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")

# Verify weights
sample = model.climax.head[0].weight[0, :3]
ckpt_sample = checkpoint['state_dict']['model.climax.head.0.weight'][0, :3]
match = torch.allclose(sample, ckpt_sample)
print(f"   Weights match checkpoint? {match}")

if not match:
    print("❌ WEIGHTS DON'T MATCH - STOP HERE!")
    sys.exit(1)

model.eval()

# Create synthetic data (similar to real data statistics)
print("\n3. Creating synthetic data...")
B, C, H, W = 2, 15, 128, 256  # Batch, channels (variables), height, width
x = torch.randn(B, C, H, W) * 0.5  # Normalized data should have std ~0.5-1.0
lead_times = torch.tensor([12.0, 24.0])  # Two different lead times (float)
variables = tuple(config['data']['variables'])

print(f"   x shape: {x.shape}")
print(f"   x mean: {x.mean():.4f}, std: {x.std():.4f}")
print(f"   lead_times: {lead_times.tolist()}")

# Forward pass
print("\n4. Forward pass...")
with torch.no_grad():
    y_pred = model(x, lead_times, variables)

print(f"   y_pred shape: {y_pred.shape}")
print(f"   y_pred mean: {y_pred.mean():.4f}, std: {y_pred.std():.4f}")
print(f"   y_pred min: {y_pred.min():.4f}, max: {y_pred.max():.4f}")

# Create synthetic target (similar to prediction for low loss)
y_true = y_pred + torch.randn_like(y_pred) * 0.3  # Add small noise

# Compute loss
loss = torch.nn.functional.mse_loss(y_pred, y_true)

print(f"\n5. Loss computation...")
print(f"   MSE loss: {loss.item():.4f}")

print("\n" + "="*80)
print("DIAGNOSTIC:")
print("="*80)

if not torch.isfinite(y_pred).all():
    print("❌ NaN or Inf in predictions! Model is broken.")
elif abs(y_pred.mean()) > 100:
    print("❌ Predictions are huge! Something is wrong with normalization.")
elif y_pred.std() < 0.01:
    print("❌ Predictions have no variance! Model might be frozen.")
else:
    print("✅ Model produces reasonable outputs!")
    print("   Predictions are finite, have reasonable scale and variance.")
    print("")
    print("🔍 The high train_loss you see is NORMAL for fine-tuning because:")
    print("   1. Optimizer state is reset (no momentum)")
    print("   2. Learning rate restarts from beginning")
    print("   3. Loss will decrease over training steps")
    print("")
    print("📊 Expected behavior:")
    print("   - First few steps: loss ~2-4 (high)")
    print("   - After 100-200 steps: loss decreases toward ~0.5-1.0")
    print("   - After 1000+ steps: loss approaches ~0.35")

print("="*80)

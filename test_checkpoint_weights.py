"""Test if checkpoint weights are actually loaded into the model."""
import torch
import yaml
import sys
sys.path.insert(0, 'src')

from model_multipollutants import MultiPollutantModel

print("="*70)
print("TEST: Are checkpoint weights actually loaded?")
print("="*70)

# Load config
with open('configs/config_all_pollutants.yaml') as f:
    config = yaml.safe_load(f)

# Create model
print("\n1. Creating model...")
model = MultiPollutantModel(config)

# Get some initial weights BEFORE checkpoint load
print("\n2. Sampling weights BEFORE checkpoint load...")
weights_before = {}
for name, param in list(model.named_parameters())[:5]:
    if 'elevation_alpha' not in name and 'H_scale' not in name:
        weights_before[name] = param.data.clone()
        print(f"   {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")

# Simulate checkpoint loading (what setup() does)
print("\n3. Loading checkpoint...")
checkpoint_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']

result = model.load_state_dict(state_dict, strict=False)
print(f"   Missing keys: {len(result.missing_keys)}")
print(f"   Unexpected keys: {len(result.unexpected_keys)}")

# Get weights AFTER checkpoint load
print("\n4. Sampling weights AFTER checkpoint load...")
weights_after = {}
for name, param in list(model.named_parameters())[:5]:
    if 'elevation_alpha' not in name and 'H_scale' not in name:
        weights_after[name] = param.data.clone()
        print(f"   {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")

# Compare
print("\n" + "="*70)
print("COMPARISON:")
print("="*70)
weights_changed = False
for name in weights_before.keys():
    if name in weights_after:
        diff = (weights_before[name] - weights_after[name]).abs().max().item()
        if diff > 1e-6:
            print(f"✅ {name}: CHANGED (max diff = {diff:.6f})")
            weights_changed = True
        else:
            print(f"❌ {name}: NOT CHANGED")

print("\n" + "="*70)
if weights_changed:
    print("✅ WEIGHTS WERE LOADED FROM CHECKPOINT")
else:
    print("❌ WEIGHTS WERE NOT LOADED!")
print("="*70)

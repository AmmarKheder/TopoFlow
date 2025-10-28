"""Test complet du chargement du checkpoint avec tous les fixes."""
import torch
import yaml
import sys
sys.path.insert(0, 'src')
from model_multipollutants import MultiPollutantModel

print("="*70)
print("TEST 1: CHARGEMENT COMPLET DU CHECKPOINT")
print("="*70)

# Load config
with open('configs/config_all_pollutants.yaml') as f:
    config = yaml.safe_load(f)

# Create model
print("\n1. Creating model...")
model = MultiPollutantModel(config)

# Sample weights BEFORE
print("\n2. Weights BEFORE checkpoint load:")
weights_before = {}
for name, param in list(model.named_parameters())[:10]:
    if 'elevation_alpha' not in name and 'H_scale' not in name:
        weights_before[name] = param.data.clone()
        print(f"   {name}: mean={param.data.mean().item():.6f}")

# Simulate setup() - Load checkpoint with fix
print("\n3. Loading checkpoint with prefix fix...")
checkpoint_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']

# Apply prefix fix
fixed_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('model.'):
        new_key = key[6:]
        fixed_state_dict[new_key] = value
    else:
        fixed_state_dict[key] = value

print(f"   Original keys: {len(state_dict)}")
print(f"   Fixed keys with 'model.' prefix removed: {sum(1 for k in state_dict if k.startswith('model.'))}")

result = model.load_state_dict(fixed_state_dict, strict=False)
print(f"   Missing keys: {len(result.missing_keys)}")
print(f"   Unexpected keys: {len(result.unexpected_keys)}")

if result.missing_keys:
    print(f"   Missing: {result.missing_keys}")

# Apply elevation_alpha fix
for name, param in model.named_parameters():
    if 'elevation_alpha' in name:
        param.data.fill_(0.0)
        print(f"\n   ✅ FIXED: {name} = {param.data.item()}")

# Sample weights AFTER
print("\n4. Weights AFTER checkpoint load:")
weights_changed = 0
weights_unchanged = 0
for name, param in list(model.named_parameters())[:10]:
    if name in weights_before:
        diff = (weights_before[name] - param.data).abs().max().item()
        if diff > 1e-6:
            print(f"   ✅ {name}: CHANGED (diff={diff:.6f})")
            weights_changed += 1
        else:
            print(f"   ❌ {name}: NOT CHANGED")
            weights_unchanged += 1

print("\n" + "="*70)
print("RESULTS:")
print("="*70)
print(f"Weights changed: {weights_changed}")
print(f"Weights unchanged: {weights_unchanged}")
print(f"Missing keys: {len(result.missing_keys)}")
print(f"Expected missing: elevation_alpha, H_scale (2 keys)")

if weights_changed > 0 and len(result.missing_keys) == 2:
    print("\n✅✅✅ CHECKPOINT LOADED SUCCESSFULLY! ✅✅✅")
else:
    print("\n❌❌❌ CHECKPOINT LOADING FAILED! ❌❌❌")
print("="*70)

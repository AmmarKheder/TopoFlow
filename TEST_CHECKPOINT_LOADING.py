"""
TEST FINAL: Vérifie que le checkpoint se charge CORRECTEMENT avec le nouveau setup()
"""
import torch
import sys
sys.path.insert(0, 'src')

print("="*80)
print("TEST: CHECKPOINT LOADING AVEC setup()")
print("="*80)

# Simulate what happens in the real training
ckpt_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"

print(f"\n1. Loading checkpoint from: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']

print(f"   Total keys in checkpoint: {len(state_dict)}")
print(f"   First 5 keys:")
for i, key in enumerate(list(state_dict.keys())[:5]):
    print(f"     {key}")

# Check prefix
has_model_prefix = any(k.startswith('model.') for k in state_dict.keys())
print(f"\n2. Keys have 'model.' prefix? {has_model_prefix}")

# Fix prefix (like in setup())
print(f"\n3. Fixing prefix...")
fixed_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('model.'):
        fixed_state_dict[key[6:]] = value  # Remove 'model.' prefix
    else:
        fixed_state_dict[key] = value

print(f"   Total keys after fix: {len(fixed_state_dict)}")
print(f"   First 5 keys after fix:")
for i, key in enumerate(list(fixed_state_dict.keys())[:5]):
    print(f"     {key}")

# Load into model
print(f"\n4. Creating model and loading state_dict...")
from model_multipollutants import MultiPollutantModel
import yaml

with open('configs/config_all_pollutants.yaml') as f:
    config = yaml.safe_load(f)

model = MultiPollutantModel(config)

# Load with strict=False
missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)

print(f"\n5. RESULTS:")
print(f"   ✅ Loaded: {len(fixed_state_dict) - len(missing_keys)} parameters")
print(f"   ⚠️  Missing: {len(missing_keys)} keys")
if len(missing_keys) <= 10:
    for k in missing_keys:
        print(f"       - {k}")
else:
    for k in missing_keys[:5]:
        print(f"       - {k}")
    print(f"       ... and {len(missing_keys) - 5} more")

print(f"   ⚠️  Unexpected: {len(unexpected_keys)} keys")
if len(unexpected_keys) <= 10:
    for k in unexpected_keys:
        print(f"       - {k}")
else:
    for k in unexpected_keys[:5]:
        print(f"       - {k}")
    print(f"       ... and {len(unexpected_keys) - 5} more")

# Check TopoFlow params
print(f"\n6. Checking TopoFlow parameters...")
for name, param in model.named_parameters():
    if 'elevation_alpha' in name or 'H_scale' in name:
        print(f"   {name}: {param.data.item() if param.numel() == 1 else param.shape}")

print("\n" + "="*80)
if len(missing_keys) == 2 and len(unexpected_keys) == 0:
    print("✅✅✅ SUCCESS! Checkpoint loads correctly!")
    print("✅✅✅ Missing keys are ONLY the 2 TopoFlow params (expected)")
elif len(missing_keys) > 10:
    print(f"❌❌❌ FAILED! {len(missing_keys)} missing keys (should be 2!)")
    print("❌❌❌ Prefix fix did NOT work!")
else:
    print(f"⚠️ Partial success: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")
print("="*80)

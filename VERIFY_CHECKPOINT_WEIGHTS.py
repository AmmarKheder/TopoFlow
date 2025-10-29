"""
Vérifie que les poids du checkpoint sont bien chargés en comparant
les valeurs exactes des paramètres
"""
import torch
import yaml
import sys
sys.path.insert(0, 'src')
from model_multipollutants import MultiPollutantModel

print("="*80)
print("TEST: VÉRIFICATION DES POIDS DU CHECKPOINT")
print("="*80)

# Load config
with open('configs/config_all_pollutants.yaml') as f:
    config = yaml.safe_load(f)

# Create model
print("\n1. Creating model...")
model = MultiPollutantModel(config)

# Get a sample weight before loading checkpoint
print("\n2. Sample weights BEFORE loading checkpoint:")
sample_weight_before = model.climax.head[0].weight[0, :5].clone()
print(f"   climax.head.0.weight[0, :5] = {sample_weight_before}")

# Load checkpoint
print("\n3. Loading checkpoint...")
ckpt_path = 'logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt'
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Fix prefix
state_dict = {}
for key, value in checkpoint['state_dict'].items():
    if key.startswith('model.'):
        state_dict[key[6:]] = value
    else:
        state_dict[key] = value

# Load state dict
result = model.load_state_dict(state_dict, strict=False)
print(f"   Missing keys: {len(result.missing_keys)}")
print(f"   Unexpected keys: {len(result.unexpected_keys)}")

# Get the same weight AFTER loading checkpoint
print("\n4. Sample weights AFTER loading checkpoint:")
sample_weight_after = model.climax.head[0].weight[0, :5].clone()
print(f"   climax.head.0.weight[0, :5] = {sample_weight_after}")

# Compare
print("\n5. Comparison:")
weights_changed = not torch.allclose(sample_weight_before, sample_weight_after)
print(f"   Weights changed? {weights_changed}")
if weights_changed:
    print(f"   Difference: {torch.abs(sample_weight_after - sample_weight_before).mean().item():.6f}")
    print("   ✅ Checkpoint weights ARE loaded!")
else:
    print("   ❌ Checkpoint weights NOT loaded - weights are identical!")

# Also check what's in checkpoint for this specific weight
print("\n6. What's in checkpoint for this weight:")
ckpt_weight = checkpoint['state_dict']['model.climax.head.0.weight'][0, :5]
print(f"   checkpoint head.0.weight[0, :5] = {ckpt_weight}")
print(f"   Model matches checkpoint? {torch.allclose(sample_weight_after, ckpt_weight)}")

print("\n" + "="*80)
if weights_changed and torch.allclose(sample_weight_after, ckpt_weight):
    print("✅ SUCCESS: Checkpoint weights are correctly loaded!")
else:
    print("❌ PROBLEM: Checkpoint weights are NOT being loaded correctly!")
print("="*80)

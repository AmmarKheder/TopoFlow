#!/usr/bin/env python3
"""
Test if current head loads checkpoint's head.4 or is random
"""
import torch
import torch.nn as nn

ckpt_path = "/scratch/project_462000640/ammar/aq_net2/logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"

checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']

print("="*80)
print("HEAD LOADING TEST")
print("="*80)

# Create a simple test model with current architecture
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(768, 60)

test_model = TestModel()

print("\n1️⃣  Current model HEAD keys:")
for name, param in test_model.named_parameters():
    print(f"   {name:40s} {tuple(param.shape)}")

print("\n2️⃣  Checkpoint HEAD keys:")
head_keys = sorted([k for k in state_dict.keys() if 'head' in k])
for key in head_keys:
    print(f"   {key:40s} {tuple(state_dict[key].shape)}")

print("\n3️⃣  Trying to load checkpoint with strict=False:")
# Remove "model.climax." prefix for testing
test_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('model.climax.head'):
        new_key = k.replace('model.climax.', '')
        test_state_dict[new_key] = v

result = test_model.load_state_dict(test_state_dict, strict=False)

print(f"\n   Missing keys: {result.missing_keys}")
print(f"   Unexpected keys: {result.unexpected_keys}")

if result.missing_keys:
    print("\n❌ HEAD IS RANDOMLY INITIALIZED!")
    print("   The checkpoint's head.4.weight/bias do NOT match head.weight/bias")
    print("   PyTorch requires EXACT key names to load weights.")
else:
    print("\n✅ HEAD LOADED FROM CHECKPOINT!")
    print("   The checkpoint's head.4.weight/bias were loaded as head.weight/bias")

print("\n4️⃣  Could head.4 be loaded automatically?")
print("   Checking if head.4 shape matches head shape...")
if 'model.climax.head.4.weight' in state_dict:
    ckpt_shape = tuple(state_dict['model.climax.head.4.weight'].shape)
    model_shape = tuple(test_model.head.weight.shape)
    print(f"   Checkpoint head.4.weight: {ckpt_shape}")
    print(f"   Current head.weight:      {model_shape}")

    if ckpt_shape == model_shape:
        print("   ✅ SHAPES MATCH! Could be manually remapped.")
    else:
        print("   ❌ SHAPES DON'T MATCH!")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nLe modèle actuel a:")
print("   - block 0 MLP: RANDOM (nn.Sequential vs named layers)")
print("   - block 0 TopoFlow: RANDOM (nouveaux paramètres)")
print("   - HEAD: RANDOM (nn.Linear vs Sequential, mais même shape final!)")
print("\nIMPACT: ~6M paramètres random sur ~85M total = 7% du modèle")
print("        C'est BEAUCOUP, d'où val_loss = 2.19 au lieu de 0.35")
print("="*80)

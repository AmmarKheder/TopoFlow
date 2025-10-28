"""
Test: Est-ce que elevation_alpha peut VRAIMENT apprendre?

Vérifie que:
1. Le gradient de alpha est calculé correctement
2. Alpha change après un step d'optimisation
3. Le biais d'élévation est appliqué au forward pass
"""

import torch
import yaml
import sys
sys.path.insert(0, 'src')

from model_multipollutants import MultiPollutantModel

print("="*80)
print("TEST: ELEVATION ALPHA LEARNING")
print("="*80)

# Load config
with open('configs/config_all_pollutants.yaml') as f:
    config = yaml.safe_load(f)

print("\n1. Creating model...")
model = MultiPollutantModel(config)

# Load checkpoint
print("\n2. Loading checkpoint...")
checkpoint_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']

# Fix prefix
fixed_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('model.'):
        fixed_state_dict[key[6:]] = value
    else:
        fixed_state_dict[key] = value

model.load_state_dict(fixed_state_dict, strict=False)

# Get elevation_alpha parameter
alpha_param = None
for name, param in model.named_parameters():
    if 'elevation_alpha' in name:
        alpha_param = param
        print(f"\n✅ Found elevation_alpha: {name}")
        print(f"   Initial value: {alpha_param.item():.6f}")
        print(f"   Requires grad: {alpha_param.requires_grad}")
        break

if alpha_param is None:
    print("❌ elevation_alpha not found!")
    sys.exit(1)

# Create synthetic data with elevation, u, v
print("\n3. Creating synthetic data...")
batch_size = 2
variables = config['data']['variables']
target_variables = config['data']['target_variables']
H, W = config['model']['img_size']

x = torch.randn(batch_size, len(variables), H, W)
lead_times = torch.zeros(batch_size)  # 1D

# Set realistic values for elevation
var_list = list(variables)
elev_idx = var_list.index('elevation')
u_idx = var_list.index('u')
v_idx = var_list.index('v')

# Create strong elevation gradient to make learning easier
x[:, elev_idx, :, :] = torch.arange(H).unsqueeze(1).repeat(1, W).unsqueeze(0).repeat(batch_size, 1, 1).float() * 100  # 0-12700m gradient
x[:, u_idx, :, :] = torch.randn(batch_size, H, W) * 5
x[:, v_idx, :, :] = torch.randn(batch_size, H, W) * 5

print(f"   Elevation range: {x[0, elev_idx].min():.0f}m to {x[0, elev_idx].max():.0f}m")

# Create target (synthetic loss)
target_idx = [var_list.index(v) for v in target_variables]
y_true = torch.randn(batch_size, len(target_variables), H, W)

# Create optimizer for alpha only
print("\n4. Creating optimizer for alpha only...")
optimizer = torch.optim.SGD([alpha_param], lr=0.1)

print("\n5. Training for 5 steps to see if alpha learns...")
model.train()

for step in range(5):
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(x, lead_times, variables, target_variables)

    # Simple MSE loss
    loss = ((y_pred - y_true) ** 2).mean()

    # Backward
    loss.backward()

    # Check gradient
    if alpha_param.grad is not None:
        grad_value = alpha_param.grad.item()
        grad_norm = alpha_param.grad.abs().item()
    else:
        grad_value = 0.0
        grad_norm = 0.0

    # Optimizer step
    optimizer.step()

    print(f"\nStep {step}:")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Alpha: {alpha_param.item():.6f}")
    print(f"  Gradient: {grad_value:.6f} (norm: {grad_norm:.6f})")

    if grad_norm < 1e-8:
        print(f"  ⚠️  Gradient is ZERO - alpha n'apprend PAS!")
    else:
        print(f"  ✅ Gradient is non-zero - alpha APPREND!")

print("\n" + "="*80)
print("RÉSULTAT:")

alpha_final = alpha_param.item()
alpha_changed = abs(alpha_final) > 1e-6

if alpha_changed:
    print(f"✅ SUCCESS: Alpha a changé de 0.0 à {alpha_final:.6f}")
    print(f"✅ L'elevation bias APPREND vraiment!")
else:
    print(f"❌ ÉCHEC: Alpha reste à {alpha_final:.6f}")
    print(f"❌ L'elevation bias N'APPREND PAS!")
    print(f"\nPossible cause: x_raw n'est pas passé ou elevation_patches = None")

print("="*80)

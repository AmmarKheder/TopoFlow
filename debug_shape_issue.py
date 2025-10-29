"""Minimal script to debug the shape issue."""
import torch
import yaml
import sys
sys.path.insert(0, 'src')

from model_multipollutants import MultiPollutantModel

print("="*70)
print("DEBUG: MINIMAL SHAPE TRACING")
print("="*70)

# Load config
with open('configs/config_all_pollutants.yaml') as f:
    config = yaml.safe_load(f)

# Create model
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

# Fix elevation_alpha
for name, param in model.named_parameters():
    if 'elevation_alpha' in name:
        param.data.fill_(0.0)

# Create minimal input
print("\n3. Creating minimal input...")
batch_size = 2
variables = config['data']['variables']
target_variables = config['data']['target_variables']
H, W = config['model']['img_size']

x = torch.randn(batch_size, len(variables), H, W)
lead_times = torch.zeros(batch_size)  # NOTE: 1D like DataLoader

print(f"   x: {x.shape}")
print(f"   lead_times: {lead_times.shape}")

# Monkey-patch to add debug prints
original_topoflow_forward = model.climax.blocks[0].forward

def debug_topoflow_forward(x, elevation_patches=None, u_wind=None, v_wind=None):
    print(f"\n{'='*70}")
    print(f"TopoFlowBlock.forward() called")
    print(f"{'='*70}")
    print(f"  x.shape = {x.shape}")
    print(f"  elevation_patches = {elevation_patches.shape if elevation_patches is not None else None}")
    print(f"  u_wind = {u_wind.shape if u_wind is not None else None}")
    print(f"  v_wind = {v_wind.shape if v_wind is not None else None}")

    print(f"\n  Calling self.norm1(x)...")
    x_norm = model.climax.blocks[0].norm1(x)
    print(f"  x_norm.shape = {x_norm.shape}")

    print(f"\n  About to call self.attn() with:")
    print(f"    x_norm.shape = {x_norm.shape}")
    print(f"    elevation_patches = {elevation_patches.shape if elevation_patches is not None else None}")

    # This should fail
    try:
        result = model.climax.blocks[0].attn(x_norm, elevation_patches, u_wind, v_wind)
        print(f"  ✅ attn succeeded! result.shape = {result.shape}")
    except Exception as e:
        print(f"  ❌ attn FAILED: {e}")
        print(f"\n  Let's check x_norm dimensions:")
        print(f"    x_norm.dim() = {x_norm.dim()}")
        print(f"    x_norm.shape = {x_norm.shape}")
        raise

    return result

model.climax.blocks[0].forward = debug_topoflow_forward

# Try forward
print("\n4. Attempting forward pass...")
model.eval()
try:
    with torch.no_grad():
        output = model(x, lead_times, variables, target_variables)
        print(f"\n✅ SUCCESS! output.shape = {output.shape}")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)

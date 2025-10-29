"""Debug script to trace physics mask computation."""
import torch
import yaml
import sys
sys.path.insert(0, 'src')

from model_multipollutants import MultiPollutantModel

print("="*70)
print("DEBUG: PHYSICS MASK COMPUTATION")
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

# Create input with elevation, u, v
print("\n3. Creating input with elevation, u, v...")
batch_size = 2
variables = config['data']['variables']
target_variables = config['data']['target_variables']
H, W = config['model']['img_size']

x = torch.randn(batch_size, len(variables), H, W)
lead_times = torch.zeros(batch_size)  # 1D like DataLoader

# Set realistic values for elevation, u, v
var_list = list(variables)
elev_idx = var_list.index('elevation')
u_idx = var_list.index('u')
v_idx = var_list.index('v')

x[:, elev_idx, :, :] = torch.randn(batch_size, H, W) * 1000 + 500  # elevation ~ 500m
x[:, u_idx, :, :] = torch.randn(batch_size, H, W) * 5  # u wind ~ 5 m/s
x[:, v_idx, :, :] = torch.randn(batch_size, H, W) * 5  # v wind ~ 5 m/s

print(f"   x: {x.shape}")
print(f"   elevation in x: index {elev_idx}, mean={x[0, elev_idx].mean():.1f}")
print(f"   u wind in x: index {u_idx}, mean={x[0, u_idx].mean():.1f}")
print(f"   v wind in x: index {v_idx}, mean={x[0, v_idx].mean():.1f}")

# Monkey-patch forward_encoder to trace physics extraction
original_forward_encoder = model.climax.forward_encoder

def debug_forward_encoder(x, lead_times, variables, x_raw=None):
    print(f"\n{'='*70}")
    print(f"forward_encoder() called")
    print(f"{'='*70}")
    print(f"  x.shape = {x.shape}")
    print(f"  x_raw.shape = {x_raw.shape if x_raw is not None else None}")
    print(f"  use_physics_mask = {model.climax.use_physics_mask}")

    # Call original but intercept at physics extraction
    result = original_forward_encoder(x, lead_times, variables, x_raw)

    return result

# Monkey-patch TopoFlowBlock.forward to trace inputs
original_topoflow_forward = model.climax.blocks[0].forward

def debug_topoflow_forward(x, elevation_patches=None, u_wind=None, v_wind=None):
    print(f"\n{'='*70}")
    print(f"TopoFlowBlock.forward() called")
    print(f"{'='*70}")
    print(f"  x.shape = {x.shape}")
    print(f"  x.dim() = {x.dim()}")

    if elevation_patches is not None:
        print(f"  elevation_patches.shape = {elevation_patches.shape}")
        print(f"  elevation_patches.dim() = {elevation_patches.dim()}")
    else:
        print(f"  elevation_patches = None")

    if u_wind is not None:
        print(f"  u_wind.shape = {u_wind.shape}")
        print(f"  u_wind.dim() = {u_wind.dim()}")
    else:
        print(f"  u_wind = None")

    if v_wind is not None:
        print(f"  v_wind.shape = {v_wind.shape}")
        print(f"  v_wind.dim() = {v_wind.dim()}")
    else:
        print(f"  v_wind = None")

    # Call norm1
    print(f"\n  Calling self.norm1(x)...")
    x_norm = model.climax.blocks[0].norm1(x)
    print(f"  x_norm.shape = {x_norm.shape}")
    print(f"  x_norm.dim() = {x_norm.dim()}")

    # This should reveal the issue
    print(f"\n  ❌ STOPPING BEFORE attn CALL")
    print(f"  Expected: x_norm.dim() == 3 ([B, N, C])")
    print(f"  Actual: x_norm.dim() == {x_norm.dim()}")

    raise RuntimeError("STOP HERE TO DEBUG")

model.climax.forward_encoder = debug_forward_encoder
model.climax.blocks[0].forward = debug_topoflow_forward

# Try forward
print("\n4. Attempting forward pass...")
model.eval()
try:
    with torch.no_grad():
        output = model(x, lead_times, variables, target_variables)
        print(f"\n✅ SUCCESS! output.shape = {output.shape}")
except RuntimeError as e:
    if "STOP HERE" in str(e):
        print(f"\n✅ Debug stop reached successfully")
    else:
        raise
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)

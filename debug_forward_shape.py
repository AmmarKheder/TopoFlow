"""Debug script to trace shapes through forward pass."""
import torch
import yaml
import sys
sys.path.insert(0, 'src')

from model_multipollutants import MultiPollutantModel

print("="*70)
print("DEBUG: TRACING SHAPES THROUGH FORWARD PASS")
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

# Apply prefix fix
fixed_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('model.'):
        new_key = key[6:]
        fixed_state_dict[new_key] = value
    else:
        fixed_state_dict[key] = value

model.load_state_dict(fixed_state_dict, strict=False)

# Fix elevation_alpha
for name, param in model.named_parameters():
    if 'elevation_alpha' in name:
        param.data.fill_(0.0)

# Create synthetic data
print("\n3. Creating synthetic data...")
batch_size = 2
variables = config['data']['variables']
target_variables = config['data']['target_variables']
H, W = config['model']['img_size']

x = torch.randn(batch_size, len(variables), H, W)
lead_times = torch.zeros(batch_size, 1)

print(f"   Input x: {x.shape}")
print(f"   Lead times: {lead_times.shape}")
print(f"   Variables: {len(variables)}")
print(f"   Target variables: {len(target_variables)}")

# Patch the forward_encoder to add debug prints
original_forward_encoder = model.climax.forward_encoder

def debug_forward_encoder(x, lead_times, variables, out_variables=None):
    print(f"\n{'='*70}")
    print(f"ENTERING forward_encoder")
    print(f"{'='*70}")
    print(f"  x shape: {x.shape}")
    print(f"  lead_times shape: {lead_times.shape}")
    print(f"  variables: {variables}")

    # Call get_var_ids
    var_ids = model.climax.get_var_ids(variables, x.device)
    print(f"\n  var_ids: {var_ids}")

    # Tokenize
    print(f"\n  Calling token_embeds (parallel_patch_embed={model.climax.parallel_patch_embed})...")
    x_tokens = model.climax.token_embeds(x, var_ids)
    print(f"  x_tokens after token_embeds: {x_tokens.shape}")

    # Add variable embedding
    var_embed = model.climax.get_var_emb(model.climax.var_embed, variables)
    print(f"  var_embed shape: {var_embed.shape}")
    x_tokens = x_tokens + var_embed.unsqueeze(2)
    print(f"  x_tokens after var_embed: {x_tokens.shape}")

    # Variable aggregation
    print(f"\n  Calling aggregate_variables...")
    x_tokens = model.climax.aggregate_variables(x_tokens)
    print(f"  x_tokens after aggregate_variables: {x_tokens.shape}")

    # Add pos embedding
    print(f"\n  pos_embed shape: {model.climax.pos_embed.shape}")
    x_tokens = x_tokens + model.climax.pos_embed
    print(f"  x_tokens after pos_embed: {x_tokens.shape}")

    # Add lead time embedding
    lead_time_emb = model.climax.lead_time_embed(lead_times.unsqueeze(-1))
    print(f"  lead_time_emb shape: {lead_time_emb.shape}")
    lead_time_emb = lead_time_emb.unsqueeze(1)
    print(f"  lead_time_emb after unsqueeze: {lead_time_emb.shape}")
    x_tokens = x_tokens + lead_time_emb
    print(f"  x_tokens after lead_time_emb: {x_tokens.shape}")

    x_tokens = model.climax.pos_drop(x_tokens)
    print(f"  x_tokens after pos_drop: {x_tokens.shape}")

    # Check if we're using physics mask
    print(f"\n  use_physics_mask: {model.climax.use_physics_mask}")

    # Try to call the first block
    print(f"\n  Calling first block (index 0)...")
    print(f"  Block type: {type(model.climax.blocks[0])}")

    # Stop here to see the shape before the block
    print(f"\n  ⚠️  STOPPING BEFORE BLOCK CALL")
    print(f"  x_tokens shape ready for block: {x_tokens.shape}")
    print(f"  Expected shape: [B, N, C] = [batch, num_patches, embed_dim]")

    return x_tokens

# Monkey patch
model.climax.forward_encoder = debug_forward_encoder

# Try forward pass
print("\n4. Attempting forward pass with debug...")
model.eval()
try:
    with torch.no_grad():
        output = model(x, lead_times, variables, target_variables)
        print(f"\n✅ Forward pass succeeded!")
        print(f"   Output shape: {output.shape}")
except Exception as e:
    print(f"\n❌ Forward pass failed:")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)

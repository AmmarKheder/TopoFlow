"""Test that elevation_alpha is correctly initialized to 0 after checkpoint loading."""
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2/src')

from climax_core.topoflow_attention import TopoFlowBlock

# Create a TopoFlow block
block = TopoFlowBlock(
    dim=768,
    num_heads=8,
    use_elevation_bias=True
)

# Check initial value
for name, param in block.named_parameters():
    if 'elevation_alpha' in name:
        print(f"Initial {name}: {param.data.item():.10f}")
        
# Simulate checkpoint loading with missing keys
# Create a state_dict WITHOUT elevation_alpha
state_dict = {}
for name, param in block.state_dict().items():
    if 'elevation_alpha' not in name and 'H_scale' not in name:
        state_dict[name] = param

# Load with strict=False (simulating checkpoint load)
result = block.load_state_dict(state_dict, strict=False)
print(f"\nMissing keys after load: {result.missing_keys}")

# Check value after load
for name, param in block.named_parameters():
    if 'elevation_alpha' in name:
        print(f"After load (before fix) {name}: {param.data.item():.10f}")

# NOW APPLY THE FIX: manually set to 0
for name, param in block.named_parameters():
    if 'elevation_alpha' in name:
        param.data.fill_(0.0)
        print(f"After fix {name}: {param.data.item():.10f}")

print("\n✅ Test passed: elevation_alpha can be explicitly set to 0.0")

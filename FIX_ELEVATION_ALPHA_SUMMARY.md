# Fix: Validation Loss Starting High (0.964 instead of 0.356)

## Problem Identified

**Symptom**: When fine-tuning from checkpoint, the validation loss starts at 0.964 instead of the expected 0.356.

**Root Cause**: 
1. New TopoFlow parameters (`elevation_alpha`, `H_scale`) don't exist in the checkpoint
2. When loading checkpoint with `strict=False`, these parameters are marked as "missing keys"
3. They keep their initialization values from `__init__`, but in DDP multi-GPU context (400 GPUs), 
   the initialization `nn.Parameter(torch.tensor(0.0))` may have synchronization issues across ranks
4. This causes random/inconsistent values instead of the intended 0.0

**Evidence from logs** (Job 13605927):
```
⚠️  Missing keys (randomly initialized): 2
   Keys: ['model.climax.blocks.0.attn.elevation_alpha', 'model.climax.blocks.0.attn.H_scale']
```

First validation: `val_loss=0.964` (should be ~0.356)

## Solution Applied

Modified `src/model_multipollutants.py:236-266` to **explicitly set these parameters after checkpoint load**:

```python
# CRITICAL FIX: Explicitly initialize NEW TopoFlow parameters to 0
if result.missing_keys:
    for name, param in self.named_parameters():
        if 'elevation_alpha' in name:
            param.data.fill_(0.0)  # Zero perturbation at start
            if global_rank == 0:
                print(f"# # # #  FIXED: {name} = 0.0 (zero perturbation)")
        elif 'H_scale' in name:
            param.data.fill_(1000.0)  # Height scale in meters
            if global_rank == 0:
                print(f"# # # #  FIXED: {name} = 1000.0 (height scale)")
```

## Expected Result

With this fix, the model should:
1. Load checkpoint successfully
2. Print "FIXED: elevation_alpha = 0.0" messages
3. Start with val_loss ~0.356 (same as checkpoint baseline)
4. Learn the optimal `elevation_alpha` value during training via gradient descent

## Jobs

- **Old job (with bug)**: 13605927 (CANCELLED)
- **New job (with fix)**: 13616451 (PENDING/RUNNING)

Monitor progress:
```bash
/scratch/project_462000640/ammar/aq_net2/WATCH_13616451.sh
```

## Technical Details

**Why elevation_alpha must be 0 at start:**
- The checkpoint was trained WITHOUT elevation bias
- Starting with elevation_alpha=0 ensures ZERO perturbation to the model at step 0
- This gives fair comparison: baseline performance should match checkpoint (0.356)
- Then the model learns optimal elevation_alpha value through training

**Why explicit initialization is needed:**
- PyTorch's `load_state_dict(strict=False)` doesn't re-initialize missing keys
- In DDP, parameter initialization can be inconsistent across ranks
- Explicit `param.data.fill_(0.0)` ensures ALL 400 GPUs have the same value

Date: 2025-10-16

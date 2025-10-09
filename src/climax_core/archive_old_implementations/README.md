# Archive - Old Implementations

This directory contains deprecated implementations that are no longer used.

## Files Archived

### Old Physics Mask Implementations (INCORRECT - Multiplicative approach):
- `physics_attention_patch_level.py` - Multiplicative mask AFTER softmax (wrong!)
- `simple_adaptive_mask.py` - Only 1 learnable parameter (too simple)
- `adaptive_physics_mask.py` - Complex version with MLP
- `adaptive_physics_mask_v2.py` - Another variant
- `physics_mask_fixed.py` - Fixed physics only

### Old Wrappers:
- `physics_block_wrapper.py` - Old wrapper for applying bias
- `physics_helpers.py` - Helper functions
- `learnable_adjacency.py` - Adjacency experiments
- `adjacency_block_wrapper.py` - Adjacency wrapper

### Backups:
- `physics_attention_corrected.py.backup_before_multiplicative` - Old backup

## Why Archived?

These files used the WRONG approach:
- Applied mask AFTER softmax (multiplicative)
- Required manual renormalization
- Not following standard masked attention

**NEW CORRECT IMPLEMENTATIONS:**
- `physics_attention_corrected.py` - Additive bias BEFORE softmax ✅
- `relative_position_bias_3d.py` - 3D learnable MLP ✅

## Date Archived
2025-10-09

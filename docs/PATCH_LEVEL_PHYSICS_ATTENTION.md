# Physics-Guided Attention: Migration from Regional to Patch-Level

## 📋 Summary

This document describes the migration from **region-based physics attention** (1024 regions, 8 patches per region) to **patch-level physics attention** (8192 individual patches) for more granular elevation-based masking.

## 🔄 Changes Made

### 1. New Implementation File
- **Created**: `src/climax_core/physics_attention_patch_level.py`
- **Purpose**: Patch-level physics-guided attention implementation
- **Key classes**:
  - `PhysicsGuidedAttentionPatchLevel`: Direct patch-to-patch elevation masking
  - `PhysicsGuidedBlockPatchLevel`: Transformer block wrapper  
  - `ElevationPatchProcessor`: Computes elevation per patch (not per region)

### 2. Modified Core Architecture
- **File**: `src/climax_core/arch.py`
- **Backup**: `src/climax_core/arch_backup_before_patch_level.py`
- **Changes**:
  - Import switched from `physics_attention_multiplicative` to `physics_attention_patch_level`
  - Method renamed: `_compute_elevation_regions` → `_compute_elevation_patches`
  - Fixed variable mapping to support all 15 variables with proper token embeddings

### 3. Technical Specifications

#### Regional (Old) vs Patch-Level (New)
| Aspect | Regional (Old) | Patch-Level (New) |
|--------|---------------|-------------------|
| **Granularity** | 1024 regions | 8192 patches |
| **Region size** | 4×8 pixels | 2×2 pixels (patch) |
| **Attention matrix** | [B, N, N] via region lookup | [B, N, N] direct patch |
| **Memory** | O(R) = O(1024) | O(N) = O(8192) |
| **Physics precision** | Averaged per region | Per-patch precision |

#### Data Flow Comparison
```
REGIONAL (OLD):
elevation_field [B,H,W] → regional_avg [B,1024] → patch_lookup [B,8192] → attention_mask [B,N,N]

PATCH-LEVEL (NEW):
elevation_field [B,H,W] → patch_elevation [B,8192] → attention_mask [B,N,N]
```

### 4. Physics Formulation (Unchanged)
- **Method**: Multiplicative masking with sigmoid normalization
- **Formula**: `attention_final = attention_std * sigmoid(-barrier_strength * elevation_diff)`
- **Interpretation**: 
  - Uphill transport → mask ≈ 0 (attention blocked)
  - Downhill transport → mask ≈ 1 (attention preserved)

## 🧪 Testing Results

### Functionality Tests
✅ **Model creation**: Successful with all 15 variables  
✅ **Forward pass**: Clean execution with 8192 patches  
✅ **Elevation masking**: Active (0.248 difference flat vs mountain terrain)  
✅ **Integration**: Compatible with MultiPollutantModel  

### Performance Characteristics
- **Input**: `[1, 15, 128, 256]` → **Output**: `[1, 6, 128, 256]`
- **Patches**: 64×128 = 8192 patches (2×2 pixels each)
- **Attention matrix**: 8192×8192 per head
- **Memory increase**: ~8x more granular than regional approach

## 🔧 Configuration Requirements

### Model Architecture
```yaml
model:
  img_size: [128, 256]    # Image dimensions
  patch_size: 2           # Patch size (2×2 pixels)
  embed_dim: 768          # Embedding dimension
  num_heads: 8           # Attention heads
```

### Variable Requirements
- **Elevation variable**: Must be present in input variables list
- **Position**: Index 13 in standard 15-variable setup
- **Format**: `[B, H, W]` elevation field in meters

## 🚀 Usage

### Basic Usage
```python
from src.climax_core.arch import ClimaX

# Create model with patch-level physics attention
model = ClimaX(
    default_vars=['u', 'v', 'temp', 'rh', 'psfc', 'pm25', 'elevation'],
    img_size=[128, 256],
    patch_size=2,
    embed_dim=768,
    depth=8,
    num_heads=8
)

# Forward pass with elevation-aware attention
x = torch.randn(B, V, 128, 256)  # Include elevation at index 13
output = model.forward_encoder(x, lead_times, variables)
```

### Integration with MultiPollutant Model
```python
from src.model_multipollutants import MultiPollutantModel

model = MultiPollutantModel(config)
predictions = model(x, lead_times, variables)
# Patch-level physics attention applied automatically in first transformer block
```

## 📈 Benefits

1. **Higher Resolution**: Each 2×2 pixel patch has its own elevation
2. **Better Physics**: More accurate representation of terrain effects
3. **Improved Transport**: Finer-grained pollutant dispersion modeling
4. **Seamless Integration**: Compatible with existing wind scanning and multi-pollutant framework

## ⚠️ Considerations

1. **Memory Usage**: 8x increase in attention granularity
2. **Compute Cost**: More attention computations per forward pass
3. **Training Time**: Potentially longer due to increased resolution

## 🔮 Future Enhancements

- **Adaptive Masking**: Dynamic barrier strength per location
- **Multi-Scale**: Combine patch and regional attention
- **Terrain Features**: Include slope, aspect, and roughness
- **Validation**: Comprehensive comparison with meteorological models

---
**Status**: ✅ Implemented and tested  
**Date**: 2025-01-27  
**Version**: Patch-level Physics Attention v1.0

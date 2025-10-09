# Elevation-Based Attention - Implementation Summary

## For Supervisor: Zhi-Song

---

## 🎯 What Was Fixed

### ❌ PREVIOUS APPROACH (INCORRECT)
```python
# OLD: Multiplicative mask AFTER softmax
attn_weights = softmax(Q @ K^T)
mask = sigmoid(-strength × elevation_diff)  # Compressed to [0,1]
attn_weights = attn_weights * mask          # Multiply after softmax
attn_weights = attn_weights / sum()         # Manual renormalization
```

**Problems:**
- Applied mask AFTER softmax (not standard masked attention)
- Used sigmoid compression [0,1] instead of real values
- Required manual renormalization
- Not following standard Transformer masking practices

---

### ✅ NEW APPROACH (CORRECTED)
```python
# NEW: Additive bias BEFORE softmax
scores = Q @ K^T
bias = compute_elevation_bias(elevation)    # Real values (ℝ)
scores = scores + bias                      # Add BEFORE softmax
attn = softmax(scores)                      # Automatic normalization
```

**Advantages:**
- Standard masked attention approach (like causal masking)
- Real-valued bias (can be strongly negative)
- Automatic normalization via softmax
- Follows best practices from literature

---

## 📁 Implementation Files

### 1. **Simple Physics-Based Bias** (Approach 1 - Recommended)
**File:** `src/climax_core/physics_attention_corrected.py`

**Key Features:**
- Additive bias BEFORE softmax ✅
- Real-valued elevation bias (not [0,1])
- 2 learnable parameters:
  - `elevation_alpha`: strength of elevation barrier
  - `wind_beta`: wind modulation strength
- Optional wind modulation: strong winds reduce elevation barriers

**Key Code Section (lines 70-82):**
```python
# Compute raw attention scores
attn_scores = (q @ k.transpose(-2, -1)) * self.scale

# Compute elevation bias [B, N, N]
elevation_bias = self.compute_elevation_bias(elevation_patches, u_wind, v_wind)

# Expand for heads: [B, N, N] → [B, H, N, N]
elevation_bias_expanded = elevation_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

# ✅ ADD bias BEFORE softmax
attn_scores = attn_scores + elevation_bias_expanded

# Softmax (automatic normalization)
attn_weights = F.softmax(attn_scores, dim=-1)
```

**Physics Formula:**
```python
Δh = elevation[j] - elevation[i]  # Elevation difference
bias[i,j] = -α × max(0, Δh / H_scale)  # Negative for uphill

# With wind modulation:
wind_factor = sigmoid(wind_strength - threshold)
modulation = 1 - β × wind_factor
bias[i,j] = bias[i,j] × modulation
```

**Effect:**
- Uphill (Δh > 0): Negative bias → reduces attention i→j
- Flat (Δh = 0): Zero bias → neutral attention
- Downhill (Δh < 0): Zero bias → normal attention
- Strong wind: Reduces elevation barrier effect

---

### 2. **3D Learnable Positional Bias** (Approach 2 - Your Suggestion)
**File:** `src/climax_core/relative_position_bias_3d.py`

**Key Features:**
- Learnable MLP that maps (dx, dy, dz) → bias
- Per-head biases (different heads can use elevation differently)
- Fully data-driven (learns optimal elevation effect)
- Based on continuous positional encoding literature

**Key Code Section (lines 34-50):**
```python
class RelativePositionBias3D(nn.Module):
    def __init__(self, num_heads, hidden_dim=64):
        # MLP: (dx, dy, dz) → bias per head
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads)
        )

    def forward(self, coords):
        # coords: [B, N, 3] (x, y, elevation)
        # Compute pairwise relative positions
        rel_pos = coords[:, :, None, :] - coords[:, None, :, :]  # [B, N, N, 3]

        # Normalize
        rel_pos = rel_pos / (rel_pos.abs().amax(dim=(1,2), keepdim=True) + 1e-6)

        # MLP maps (dx, dy, dz) to bias
        rel_bias = self.mlp(rel_pos)  # [B, N, N, num_heads]

        # Transpose for attention
        rel_bias = rel_bias.permute(0, 3, 1, 2)  # [B, num_heads, N, N]
        return rel_bias
```

**Usage in Attention3D (lines 121-129):**
```python
# Attention scores
attn = (q @ k.transpose(-2, -1)) * self.scale

# Add 3D bias BEFORE softmax
if coords_3d is not None:
    rel_bias = self.rel_pos_bias_3d(coords_3d)  # [B, num_heads, N, N]
    attn = attn + rel_bias  # ✅ Additive before softmax

# Softmax
attn = attn.softmax(dim=-1)
```

**Advantages over Approach 1:**
- Fully learnable (no hand-crafted physics formula)
- Can learn complex elevation effects
- Per-head specialization
- Learns from data what elevation patterns matter

---

## 🔬 Comparison: Approach 1 vs Approach 2

| Aspect | **Approach 1: Physics Bias** | **Approach 2: 3D Learnable** |
|--------|------------------------------|------------------------------|
| **File** | `physics_attention_corrected.py` | `relative_position_bias_3d.py` |
| **Bias computation** | Hand-crafted physics formula | Learnable MLP |
| **Parameters** | 2 scalars (α, β) | ~5k params (MLP) |
| **Interpretability** | High (physics-based) | Low (black box) |
| **Data efficiency** | Good (strong inductive bias) | Needs more data |
| **Flexibility** | Limited (fixed formula) | High (learns from data) |
| **Recommended for** | Small datasets, physics-first | Large datasets, data-driven |

---

## 📊 Integration into ClimaX Architecture

### Current Architecture (arch.py)
```python
# Line 17: Import (OUTDATED - using simple_adaptive_mask)
from src.climax_core.simple_adaptive_mask import SimpleAdaptivePhysicsMask

# Line 110: Initialize
self.physics_mask = SimpleAdaptivePhysicsMask(grid_size=(grid_h, grid_w))

# Lines 232-249: Compute physics bias
elevation_patches = downsample_elevation(elevation_field)
physics_bias = self.physics_mask(elevation_patches, u_wind, v_wind)

# Lines 255-261: Apply to first block only
if i == 0 and self.use_physics_mask:
    x_tokens = blk(x_tokens, physics_bias=physics_bias)
```

### Recommended Change

**Option A: Use Physics-Based Bias (Simpler, Recommended)**
```python
# Change line 17:
from src.climax_core.physics_attention_corrected import PhysicsGuidedBlockCorrected

# Change block initialization (around line 100):
if i == 0 and self.use_physics_mask:
    self.blocks.append(PhysicsGuidedBlockCorrected(
        dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop_path=dpr[i]
    ))
else:
    self.blocks.append(Block(...))  # Standard block

# Forward pass (lines 255-261):
for i, blk in enumerate(self.blocks):
    if i == 0 and self.use_physics_mask:
        x_tokens = blk(x_tokens, elevation_patches, u_wind, v_wind)
    else:
        x_tokens = blk(x_tokens)
```

**Option B: Use 3D Learnable Bias (More Advanced)**
```python
# Change line 17:
from src.climax_core.relative_position_bias_3d import Attention3D, compute_patch_coords_3d

# Compute 3D coordinates before blocks:
coords_3d = compute_patch_coords_3d(elevation_field, img_size=(H,W), patch_size=2)

# Forward pass:
for i, blk in enumerate(self.blocks):
    if i == 0 and self.use_physics_mask:
        x_tokens = blk.attn(x_tokens, coords_3d=coords_3d)  # Use 3D bias
    else:
        x_tokens = blk(x_tokens)
```

---

## 🚀 Next Steps

### Immediate Actions:
1. ✅ **Code implemented** - Both approaches ready
2. ⏳ **Choose approach** - Recommend starting with Approach 1 (simpler)
3. ⏳ **Integrate into arch.py** - Modify main architecture
4. ⏳ **Create config** - New YAML for elevation attention training
5. ⏳ **Launch training job** - Test on LUMI with 100+ GPUs

### Recommended Training Plan:
```yaml
# Config: config_elevation_corrected.yaml
model:
  use_physics_mask: true
  parallel_patch_embed: false  # Row-major (not wind scanning yet)

train:
  batch_size: 2
  devices: 8
  num_nodes: 50  # 400 GPUs
  max_steps: 20000
  learning_rate: 0.0001
```

**Baseline to beat:**
- Row-major: val_loss = 0.264 (6 epochs)
- Wind scanning 32×32: val_loss = 0.3552

**Goal:** Elevation attention should improve over row-major baseline (< 0.264)

---

## 📝 For the Paper / Thesis

### Method Description:
"We incorporate topographic elevation into the attention mechanism through an additive bias applied before the softmax operation. Given elevation values e_i and e_j for patches i and j, we compute:

bias(i,j) = -α × max(0, (e_j - e_i) / H_scale)

where α is a learnable parameter and H_scale = 1000m is a normalization constant. This bias penalizes uphill transport (e_j > e_i) while leaving downhill transport unaffected, reflecting the physical constraint that pollutant dispersion is hindered by topographic barriers. The bias is added to attention logits before softmax:

Attention(Q,K,V) = softmax((Q K^T / √d) + bias) V

Optionally, we modulate the bias by local wind strength, reducing the elevation barrier effect when winds are strong enough to overcome topographic obstacles."

---

## ✅ Success Criteria

Implementation is correct if:
- ✅ Bias applied BEFORE softmax (not after)
- ✅ Uses addition (not multiplication)
- ✅ Real-valued bias (not sigmoid-compressed)
- ✅ Automatic normalization (no manual renorm)
- ✅ Gradient flows correctly through learnable parameters
- ✅ Integrates with batched inputs [B, N, C]

---

## 📧 Contact

If you have questions about the implementation:
- Approach 1 (physics-based): See `physics_attention_corrected.py` lines 65-150
- Approach 2 (3D learnable): See `relative_position_bias_3d.py` lines 20-137
- Integration: See this document section "Integration into ClimaX Architecture"

Both implementations follow your feedback and use **additive bias before softmax**.

# TopoFlow Architecture

## Model Overview

TopoFlow is a Vision Transformer-based model for air quality forecasting that incorporates two key physics-informed innovations:

1. **Wind-Following Patch Scanning**: Reorders input patches based on wind direction
2. **Elevation-Aware Attention**: Modulates attention based on terrain elevation differences

## Components

### 1. Wind Patch Embedding (`wind_embed.py`)

The input tensor `[B, C, H, W]` is processed as follows:

1. **Patch Extraction**: Each variable is independently projected to patch embeddings
2. **Wind Direction Detection**: Mean wind vector computed from u, v components
3. **Patch Reordering**: Patches are reordered from upwind to downwind using band-Hilbert curves

### 2. TopoFlow Attention Block (`topoflow.py`)

The first transformer block uses TopoFlow attention with:

**Relative Position Bias (2D Bucketed)**:
- Uses T5-style bucketing for memory efficiency
- Buckets relative positions in x and y separately
- Learnable bias table indexed by bucket pairs

**Elevation Bias**:
```
elevation_bias[i,j] = -α × ReLU((elev[j] - elev[i]) / scale)
```
- Penalizes attention to higher-elevation patches
- Learnable α parameter controls barrier strength

### 3. Standard Transformer Blocks

Blocks 1-5 use standard Vision Transformer blocks:
- Multi-head self-attention
- Layer normalization
- MLP with GELU activation
- Stochastic depth regularization

### 4. Decoder Head

Two-layer MLP projecting from `[B, N, D]` to `[B, N, V×P²]`:
- Linear → GELU → Linear
- Unpatchify to reconstruct spatial output

## Data Flow

```
Input [B, 15, 128, 256]
         ↓
Wind Patch Embed (with wind-based reordering)
         ↓
Variable Aggregation (cross-attention)
         ↓
Add Position + Lead Time Embeddings
         ↓
TopoFlow Block (elevation + relative pos bias)
         ↓
5× Standard ViT Blocks
         ↓
Layer Norm
         ↓
Decoder Head
         ↓
Unpatchify
         ↓
Output [B, 6, 128, 256]
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| embed_dim | 768 | Transformer hidden dimension |
| depth | 6 | Number of transformer blocks |
| num_heads | 8 | Attention heads per block |
| patch_size | 2 | Spatial patch size |
| mlp_ratio | 4.0 | MLP expansion ratio |
| α (elevation) | 2.0 | Elevation barrier strength |

## Memory Considerations

- Relative position bias uses bucketing (not full N×N computation)
- Wind scanning orders are precomputed for 16 sectors
- Gradient checkpointing can be enabled for large batch training

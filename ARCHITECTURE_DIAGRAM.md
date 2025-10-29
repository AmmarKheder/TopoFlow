# TopoFlow: Wind-Guided Transformer with Topographic Attention
## Architecture Diagrams for Publication

---

## Figure 1: Overview Architecture (High-Level)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TopoFlow Architecture                                 │
│                   Air Quality Forecasting Transformer                        │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: Multi-variate Spatiotemporal Data [B, 15, H, W]
├─ Meteorology: u, v, temp, rh, psfc
├─ Pollutants: PM2.5, PM10, SO2, NO2, CO, O3
└─ Geographic: lat, lon, elevation, population

         │
         ├──────────────────────────────────────────┐
         ↓                                          │
┌────────────────────────────────────┐              │
│   Wind-Following Patch Embedding   │              │
│   ─────────────────────────────    │              │
│                                    │              │
│   ┌──────────────────────────┐    │              │
│   │  1. Parallel ViT Patches │    │              │
│   │     [B,15,H,W] → [B,N,D] │    │              │
│   └──────────────────────────┘    │              │
│              │                     │              │
│   ┌──────────────────────────┐    │              │
│   │  2. Wind Direction       │    │              │
│   │     u,v → θ_wind         │    │              │
│   └──────────────────────────┘    │              │
│              │                     │              │
│   ┌──────────────────────────┐    │              │
│   │  3. Adaptive Scanning    │    │              │
│   │     Reorder patches      │    │              │
│   │     following wind       │    │              │
│   └──────────────────────────┘    │              │
└────────────────────────────────────┘              │
         │                                          │
         ↓                                          │
┌────────────────────────────────────┐              │
│      TopoFlow Block 0 (Novel)      │              │
│      ═══════════════════════       │              │
│                                    │              │
│  ┌──────────────────────────────┐  │              │
│  │  Multi-Head Attention with:  │  │              │
│  │                              │  │              │
│  │  ① Relative Position Bias   │◄─┼──────────────┤
│  │     (x,y) - Bucketed T5      │  │              │
│  │     32×32 learnable table    │  │     coords_2d [B,N,2]
│  │                              │  │     Spatial positions
│  │  ② Alpha Elevation Mask     │◄─┼──────────────┤
│  │     (z) - Physics-informed   │  │              │
│  │     -α·ReLU(Δz/1000)        │  │     elevation [B,N]
│  │                              │  │     Topographic data
│  │  bias = rel_pos + elev      │  │              │
│  │                              │  │              │
│  └──────────────────────────────┘  │              │
│              │                     │              │
│  ┌──────────────────────────────┐  │              │
│  │       Feed-Forward MLP       │  │              │
│  └──────────────────────────────┘  │              │
└────────────────────────────────────┘              │
         │                                          │
         ↓                                          │
┌────────────────────────────────────┐              │
│   Standard ViT Blocks (1-7)        │              │
│   ─────────────────────────        │              │
│   • Multi-Head Self-Attention      │              │
│   • Feed-Forward MLP               │              │
│   • Layer Norm, Residuals          │              │
└────────────────────────────────────┘              │
         │                                          │
         ↓                                          │
┌────────────────────────────────────┐              │
│      Decoder Head (2-layer)        │              │
│      ─────────────────             │              │
│      [B,N,D] → [B,N,V·p²]         │              │
└────────────────────────────────────┘              │
         │                                          │
         ↓                                          │
OUTPUT: Multi-pollutant Forecast [B, 6, H, W]
       PM2.5, PM10, SO2, NO2, CO, O3
```

---

## Figure 2: TopoFlow Block 0 - Detailed Attention Mechanism

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    TopoFlow Attention (Block 0 Only)                          │
│                Multi-Scale 3D Positional Encoding                             │
└───────────────────────────────────────────────────────────────────────────────┘

INPUT: x [B, N, D]  (N=8192 patches, D=768 embedding dim)
       coords_2d [B, N, 2]  (spatial coordinates x,y ∈ [0,1])
       elevation [B, N]  (topographic height in meters)

┌─────────────────────────────────────────────────────────────────────┐
│                         Q, K, V Projections                         │
│  x → Linear(D, 3D) → [Q, K, V] each [B, num_heads, N, head_dim]   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 1: Raw Attention Scores                     │
│                                                                     │
│              attn_raw = (Q @ K^T) / √d_k                           │
│                    [B, heads, N, N]                                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                ↓                                ↓
┌───────────────────────────────┐  ┌─────────────────────────────────┐
│  STEP 2: Relative Position    │  │  STEP 3: Elevation Bias         │
│  Bias (Spatial x,y)            │  │  (Topographic z)                │
│  ────────────────────────      │  │  ────────────────────           │
│                                │  │                                 │
│  Bucketed T5-Style:            │  │  Physics-Informed:              │
│                                │  │                                 │
│  ┌──────────────────────────┐ │  │  ┌───────────────────────────┐ │
│  │ dx = x_j - x_i           │ │  │  │ Δz = z_j - z_i            │ │
│  │ dy = y_j - y_i           │ │  │  │                           │ │
│  │                          │ │  │  │ Uphill penalty:           │ │
│  │ Convert to grid coords:  │ │  │  │                           │ │
│  │ dx_int = dx × 128        │ │  │  │ bias = -α·ReLU(Δz/1000)  │ │
│  │ dy_int = dy × 128        │ │  │  │                           │ │
│  │                          │ │  │  │ α: learnable parameter    │ │
│  │ Bucketize (log-spaced):  │ │  │  │ (initialized to 1.0)      │ │
│  │ bucket_x ∈ [0, 31]      │ │  │  │                           │ │
│  │ bucket_y ∈ [0, 31]      │ │  │  │ Clamp: [-10, 0]          │ │
│  │                          │ │  │  └───────────────────────────┘ │
│  │ Combined bucket:         │ │  │            │                   │
│  │ idx = bucket_x×32 + y    │ │  │            ↓                   │
│  │                          │ │  │  [B, N, N] → broadcast         │
│  │ Lookup bias table:       │ │  │  [B, heads, N, N]              │
│  │ bias = table[idx]        │ │  │                                 │
│  │      [1024, heads]       │ │  │  Memory: O(N²) compute         │
│  │                          │ │  │  but no learnable params!       │
│  │ Memory: 32×32×8 = 8K!   │ │  │                                 │
│  └──────────────────────────┘ │  │                                 │
│            │                   │  │                                 │
│            ↓                   │  │                                 │
│  [B, heads, N, N]             │  │                                 │
└───────────────────────────────┘  └─────────────────────────────────┘
                │                                │
                └────────────┬───────────────────┘
                             ↓
            ┌────────────────────────────────────────┐
            │    COMBINE: 3D Positional Encoding     │
            │                                        │
            │  attn_scores = attn_raw                │
            │              + rel_pos_bias (x,y)      │
            │              + elev_bias (z)           │
            │                                        │
            │  [B, heads, N, N]                     │
            └────────────────────────────────────────┘
                             │
                             ↓
            ┌────────────────────────────────────────┐
            │       STEP 4: Softmax & Apply          │
            │                                        │
            │  attn_weights = softmax(attn_scores)   │
            │  output = attn_weights @ V             │
            │                                        │
            │  [B, heads, N, head_dim]              │
            └────────────────────────────────────────┘
                             │
                             ↓
                   OUTPUT: [B, N, D]

┌────────────────────────────────────────────────────────────────┐
│                        KEY INNOVATIONS                          │
├────────────────────────────────────────────────────────────────┤
│ ① Relative Position (x,y): Bucketed for memory efficiency     │
│    • Wind-scanning compatible (distance-based, not order)      │
│    • 32² buckets vs 8192² pairs = 1000× memory reduction      │
│    • T5-style log-spacing: precise near, coarse far           │
│                                                                 │
│ ② Alpha Elevation (z): Physics-informed bias                  │
│    • YOUR INNOVATION: Learnable uphill transport penalty       │
│    • Captures topographic barriers to pollutant flow           │
│    • Interpretable: α quantifies terrain effect strength       │
│                                                                 │
│ ③ Combined 3D Encoding: rel_pos(x,y) + alpha(z)              │
│    • Spatial locality + Physical constraints                   │
│    • Additive before softmax = automatic normalization         │
│    • Complementary: learned spatial + physics-based vertical   │
└────────────────────────────────────────────────────────────────┘
```

---

## Figure 3: Wind-Following Patch Scanning

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Adaptive Patch Ordering Based on Wind Direction             │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: Compute Regional Wind Field
────────────────────────────────────

Original Grid: 128×256 pixels
├─ u (wind x-component) [128, 256]
└─ v (wind y-component) [128, 256]

Regional Aggregation (16×16 patch regions):
┌────────────────────────────────────────┐
│  Divide into 8×16 regions              │
│  Each region: 16×16 pixels             │
│                                        │
│  ┌────┬────┬────┬─────┬────┐         │
│  │ R₀ │ R₁ │ R₂ │ ... │R₁₂₇│         │
│  ├────┼────┼────┼─────┼────┤         │
│  │R₁₂₈│    │    │     │    │         │
│  └────┴────┴────┴─────┴────┘         │
│                                        │
│  For each region k:                    │
│    u_k = mean(u in region k)          │
│    v_k = mean(v in region k)          │
│    θ_k = atan2(v_k, u_k)  # direction │
└────────────────────────────────────────┘

STEP 2: Determine Scanning Order
─────────────────────────────────

Wind Direction → Scan Order Mapping:

     N (θ=90°)
      ↑
      │
W ────┼──── E
      │
      ↓
     S (θ=270°)

┌──────────────────────────────────────────────────────────────┐
│  θ ∈ [0°, 45°]:    East wind     → Scan West to East        │
│  θ ∈ [45°, 135°]:  North wind    → Scan South to North      │
│  θ ∈ [135°, 225°]: West wind     → Scan East to West        │
│  θ ∈ [225°, 315°]: South wind    → Scan North to South      │
│  θ ∈ [315°, 360°]: East wind     → Scan West to East        │
└──────────────────────────────────────────────────────────────┘

STEP 3: Reorder Patches
────────────────────────

Example: North-East Wind (θ = 45°)

Row-Major Order (Default):
┌────────────────────────────────┐
│  0   1   2   3  ...  127       │  ← Start top-left
│ 128 129 130 ...                │
│ 256 257 ...                    │
│ ...                            │
│ 7937 ... 8191                  │  ← End bottom-right
└────────────────────────────────┘

Wind-Following Order (Adaptive):
┌────────────────────────────────┐
│                    ... 8190 8191│  ↖ End top-right
│              ... 8063           │   ↖ Scan diagonally
│        ... 7936                 │    ↖ Following wind
│  ...                            │
│ 0 1 ...                         │  ← Start bottom-left
└────────────────────────────────┘

Reordering Function:
┌──────────────────────────────────────────────────────────┐
│  patches_ordered = scan_by_wind(patches, θ_wind)         │
│                                                          │
│  • Upwind patches processed first                       │
│  • Downwind patches processed last                      │
│  • Causal information flow along wind direction         │
│  • Matches physical pollutant transport!                │
└──────────────────────────────────────────────────────────┘

STEP 4: Cache for Efficiency
─────────────────────────────

Pre-compute all possible orderings:
┌────────────────────────────────────┐
│  wind_scanner_cache.pkl:           │
│  ├─ θ = 0°   → order_0             │
│  ├─ θ = 5°   → order_1             │
│  ├─ θ = 10°  → order_2             │
│  ...                                │
│  └─ θ = 355° → order_71            │
│                                     │
│  72 pre-computed orderings          │
│  (5° resolution)                    │
└────────────────────────────────────┘

Lookup at runtime: O(1)
```

---

## Figure 4: Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TopoFlow End-to-End Pipeline                     │
└─────────────────────────────────────────────────────────────────────────┘

INPUT DATA [Batch=2, Time=t]
├─ Meteorology: [2, 5, 128, 256]
│  ├─ u (m/s)
│  ├─ v (m/s)
│  ├─ temp (K)
│  ├─ rh (%)
│  └─ psfc (Pa)
│
├─ Current Pollutants: [2, 5, 128, 256]
│  ├─ PM10, SO2, NO2, CO, O3
│  └─ (PM2.5 is prediction target)
│
└─ Geography: [2, 5, 128, 256]
   ├─ lat2d, lon2d
   ├─ elevation (m)
   └─ population

         │
         ↓
┌─────────────────────────────────────────┐
│  Parallel Patch Embedding               │
│  ─────────────────────────               │
│  • 15 variables → 15 Conv2d             │
│  • Patch size: 2×2                      │
│  • Output: [2, 15, 4096, 768]          │
│           (4096 = 64×64 patches)        │
└─────────────────────────────────────────┘
         │
         ├──────────────────┬─────────────────┐
         ↓                  ↓                 ↓
┌──────────────────┐ ┌─────────────┐ ┌──────────────────┐
│ Wind Direction   │ │ Coordinates │ │ Elevation Patches│
│ ───────────────  │ │ ──────────  │ │ ──────────────── │
│                  │ │             │ │                  │
│ u,v → θ_wind     │ │ Grid (x,y)  │ │ AvgPool(2×2)     │
│ Regional avg     │ │ Normalized  │ │ elevation        │
│                  │ │ [0,1]       │ │                  │
│ [2]              │ │ [1,4096,2]  │ │ [2, 4096]        │
└──────────────────┘ └─────────────┘ └──────────────────┘
         │                  │                 │
         ↓                  │                 │
┌──────────────────┐        │                 │
│ Wind Scanning    │        │                 │
│ ───────────────  │        │                 │
│                  │        │                 │
│ Reorder patches  │        │                 │
│ based on θ_wind  │        │                 │
│                  │        │                 │
│ [2,15,4096,768] │        │                 │
│     ↓            │        │                 │
│ [2,4096,768]    │        │                 │
│  (aggregated)    │        │                 │
└──────────────────┘        │                 │
         │                  │                 │
         ↓                  ↓                 ↓
┌─────────────────────────────────────────────────┐
│             Variable Embedding                   │
│             + Positional Embedding               │
│             + Lead Time Embedding                │
│             [2, 4096, 768]                      │
└─────────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────┐
│        TopoFlow Block 0                          │
│        ═══════════════════                       │
│        • Relative Position Bias (x,y)            │
│        • Alpha Elevation Mask (z)                │
│        • Multi-Head Attention                    │
│        • Feed-Forward MLP                        │
│        [2, 4096, 768]                           │
└─────────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────┐
│        Standard Transformer Blocks 1-7           │
│        ─────────────────────────────             │
│        • Standard Self-Attention                 │
│        • Feed-Forward MLP                        │
│        • Layer Norm, Residuals                   │
│        [2, 4096, 768]                           │
└─────────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────┐
│        Decoder Head                              │
│        ────────────                              │
│        • Linear(768, 768)                        │
│        • GELU                                    │
│        • Linear(768, 6×4)  (6 pollutants×2²)    │
│        [2, 4096, 24]                            │
└─────────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────┐
│        Unpatchify                                │
│        ──────────                                │
│        [2, 4096, 24] → [2, 6, 128, 256]         │
└─────────────────────────────────────────────────┘
         │
         ↓
OUTPUT: Multi-Pollutant Forecast [2, 6, 128, 256]
       PM2.5, PM10, SO2, NO2, CO, O3
       at t + Δt (6, 12, 24, 48, 96 hours)

┌─────────────────────────────────────────────────┐
│              Loss Function                       │
│              ─────────────                       │
│  L = Σ MSE(pred_pollutant, target_pollutant)   │
│      over China region (masked)                 │
└─────────────────────────────────────────────────┘
```

---

## Table: Model Specifications

| Component | Configuration | Parameters |
|-----------|---------------|------------|
| **Input Resolution** | 128 × 256 pixels | - |
| **Patch Size** | 2 × 2 pixels | - |
| **Number of Patches** | 64 × 128 = 8,192 | - |
| **Embedding Dimension** | 768 | - |
| **Number of Blocks** | 8 total (1 TopoFlow + 7 Standard) | - |
| **Attention Heads** | 8 per block | - |
| **MLP Ratio** | 4.0 | - |
| **Total Parameters** | ~110M | - |
| **TopoFlow Block 0** | | |
| ├─ Relative Position Buckets | 32 × 32 per dimension | 8,192 |
| ├─ Alpha Elevation | Learnable scalar | 1 |
| └─ Total Additional | | ~8K |
| **Training** | | |
| ├─ Optimizer | AdamW | - |
| ├─ Learning Rate (ViT blocks) | 1e-5 | - |
| ├─ Learning Rate (Wind embed) | 2e-4 | - |
| ├─ Batch Size | 2 per GPU × 128 GPUs | 256 |
| └─ Epochs | 5 | - |

---

## Key Design Choices

### 1. Wind-Following Scanning
- **Motivation**: Pollutants transport along wind direction
- **Implementation**: Adaptive patch reordering based on regional wind field
- **Benefit**: Causal information flow matches physical process

### 2. Bucketed Relative Position Bias
- **Motivation**: Absolute positions contradict wind-based reordering
- **Implementation**: T5-style logarithmic bucketing (32² bins)
- **Benefit**: 1000× memory reduction, wind-scanning compatible

### 3. Alpha Elevation Mask
- **Motivation**: Topography creates barriers to pollutant transport
- **Implementation**: Learnable penalty for uphill flow (-α·ReLU(Δz))
- **Benefit**: Physics-informed, interpretable, complements spatial bias

### 4. Hybrid 3D Encoding
- **Spatial (x,y)**: Learned bucketed relative position
- **Vertical (z)**: Physics-based elevation penalty
- **Combination**: Additive biases before softmax

---

## References

1. **T5 Bucketing**: Raffel et al., "Exploring the Limits of Transfer Learning" (JMLR 2020)
2. **ClimaX Base**: Nguyen et al., "ClimaX: A foundation model for weather and climate" (ICML 2023)
3. **Relative Position**: Shaw et al., "Self-Attention with Relative Position Representations" (NAACL 2018)
4. **Swin Transformer**: Liu et al., "Swin Transformer: Hierarchical Vision Transformer" (ICCV 2021)

---

Created: October 2025
Model: TopoFlow v2 (Wind-Guided + Topographic Attention)
Author: Ammar Kheder

# TopoFlow

A physics-informed deep learning model for multi-pollutant air quality forecasting.

**Project page:** [https://ammarkheder.github.io/TopoFlow/](https://ammarkheder.github.io/TopoFlow/)

## Overview

TopoFlow combines Vision Transformer architecture with physics-based inductive biases for atmospheric transport modeling:

- **Wind-following patch scanning**: Reorders spatial patches according to wind direction (upwind → downwind) to capture atmospheric transport
- **Elevation-aware attention**: Learnable topographic barrier modeling in transformer attention
- **Multi-horizon prediction**: Forecasts at 12h, 24h, 48h, and 96h lead times
- **Multi-pollutant support**: PM2.5, PM10, SO2, NO2, CO, O3

## Installation

```bash
pip install -e .
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- PyTorch Lightning >= 2.0
- timm >= 0.9.0
- xarray
- zarr

## Quick Start

### Training

```bash
python scripts/train.py --config configs/default.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint path/to/checkpoint.ckpt
```

## Model Architecture

```
Input: [B, 15, 128, 256]  (Batch, Variables, Height, Width)

┌─────────────────────────────────────────────────────────────┐
│  Wind-Guided Patch Embedding                                │
│  • Variable tokenization (15 → 768D per patch)              │
│  • Wind-following patch reordering                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  TopoFlow Block (Block 0)                                   │
│  • Relative 2D positional bias                              │
│  • Elevation-aware attention (α parameter)                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Standard ViT Blocks (Blocks 1-5)                           │
│  • Multi-head self-attention                                │
│  • MLP layers                                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Decoder Head                                               │
│  • 6 pollutant outputs × 4 forecast horizons                │
└─────────────────────────────────────────────────────────────┘

Output: [B, 24, 128, 256]  (6 pollutants × 4 horizons)
```

## Data Format

TopoFlow expects Zarr files with the following structure:

```
data_YYYY_china_masked.zarr/
├── u          # Horizontal wind (m/s)
├── v          # Vertical wind (m/s)
├── temp       # Temperature (K)
├── rh         # Relative humidity (%)
├── psfc       # Surface pressure (Pa)
├── pm25       # PM2.5 concentration (μg/m³)
├── pm10       # PM10 concentration (μg/m³)
├── so2        # SO2 concentration (μg/m³)
├── no2        # NO2 concentration (μg/m³)
├── co         # CO concentration (μg/m³)
├── o3         # O3 concentration (μg/m³)
├── elevation  # Terrain elevation (m)
└── population # Population density
```

## Configuration

See `configs/default.yaml` for all available options:

```yaml
model:
  img_size: [128, 256]
  patch_size: 2
  embed_dim: 768
  depth: 6
  num_heads: 8
  parallel_patch_embed: true   # Enable wind-following scanning
  use_physics_mask: true       # Enable TopoFlow elevation attention

data:
  train_years: [2013, 2014, 2015, 2016]
  val_years: [2017]
  test_years: [2018]
  forecast_hours: [12, 24, 48, 96]
```

## Key Innovations

### 1. Wind-Following Patch Scanning

Instead of standard raster-scan ordering, patches are reordered based on the dominant wind direction. This creates an inductive bias aligned with atmospheric transport physics.

### 2. TopoFlow Attention Block

The first transformer block includes:
- **Relative 2D positional bias**: Learnable spatial locality bias using T5-style bucketing
- **Elevation mask**: Penalizes attention from high-elevation to low-elevation patches (topographic barriers)

```python
# Elevation bias computation
elev_diff = elevation[j] - elevation[i]
bias = -α × ReLU(elev_diff / scale)  # Penalize uphill attention
```

## Distributed Training

For multi-GPU training on HPC clusters:

```bash
srun python scripts/train.py \
    --config configs/default.yaml
```

## Citation

```bibtex
@article{topoflow2026,
  title={TopoFlow: Physics-Informed Deep Learning for Air Quality Forecasting},
  author={Kheder, Ammar},
  year={2026}
}
```

## License

MIT License

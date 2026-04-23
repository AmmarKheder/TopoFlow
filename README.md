# TopoFlow

<div align="center">

# 🌬️ TopoFlow

### *Topography-aware Pollutant Flow Learning for High-Resolution Air Quality Prediction*

[![Paper](https://img.shields.io/badge/Published-npj%20Climate%20%26%20Atmospheric%20Science-0B6E99?style=for-the-badge&logo=nature&logoColor=white)](https://doi.org/10.1038/s41612-026-01417-5)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41612--026--01417--5-005C99?style=for-the-badge)](https://doi.org/10.1038/s41612-026-01417-5)
[![arXiv](https://img.shields.io/badge/arXiv-2602.16821-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.16821)
[![Project Page](https://img.shields.io/badge/🌐_Project_Page-TopoFlow-4A5FD8?style=for-the-badge)](https://ammarkheder.github.io/TopoFlow/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

> 🎉 **Published in *npj Climate and Atmospheric Science* (Nature Portfolio)**
> **DOI:** [10.1038/s41612-026-01417-5](https://doi.org/10.1038/s41612-026-01417-5)

</div>

## ✨ Overview

**TopoFlow** fuses **Vision Transformer** architectures with **physics-based inductive biases** to model atmospheric transport of pollutants — turning raw meteorological and terrain data into precise, high-resolution air quality forecasts.

Where pure data-driven models ignore the physics of how air actually flows over mountains and valleys, TopoFlow *listens to the landscape*.

---

## 🧠 Core Innovations

### 🌪️ Wind-Following Patch Scanning
Reorders spatial patches along the wind direction (*upwind → downwind*), letting the transformer naturally capture the flow of pollutants across a region.

### 🏔️ Elevation-Aware Attention
A learnable **topographic barrier mechanism** inside the attention layers — mountains block flow, valleys channel it, and the model knows the difference.

### ⏱️ Multi-Horizon Forecasting
One unified model, four forecast horizons:
`12h` · `24h` · `48h` · `96h`

### 🧪 Multi-Pollutant Coverage
`PM₂.₅` · `PM₁₀` · `SO₂` · `NO₂` · `CO` · `O₃`

---

## 📊 At a Glance

| | |
|---|---|
| **Architecture** | Vision Transformer + physics priors |
| **Inputs** | Meteorology, elevation, emissions |
| **Outputs** | 6 pollutants × 4 horizons |
| **Domain** | High-resolution regional air quality |

---

## 📚 Citation

```bibtex
@article{kheder2026topoflow,
  title   = {TopoFlow: Topography-aware Pollutant Flow Learning for High-Resolution Air Quality Prediction},
  author  = {Kheder, Ammar and Toropainen, Helmi and Peng, Wenqing and Ant{\~a}o, Samuel and Chen, Jia and Boy, Michael and Liu, Zhi-Song},
  journal = {npj Climate and Atmospheric Science},
  year    = {2026},
  doi     = {10.1038/s41612-026-01417-5},
  url     = {https://doi.org/10.1038/s41612-026-01417-5}
}
```

---

## 🔗 Links

- 📄 **Paper (npj Climate & Atmospheric Science):** https://doi.org/10.1038/s41612-026-01417-5
- 📝 **arXiv preprint:** https://arxiv.org/abs/2602.16821
- 🌐 **Project page:** https://ammarkheder.github.io/TopoFlow/

---

<div align="center">

*Bringing physics back into transformers — one wind vector at a time.*

</div>

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


}
```

## License

MIT License

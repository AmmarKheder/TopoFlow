# TopoFlow - Physics-Informed Deep Learning for Multi-Pollutant Air Quality Forecasting

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)

> **Physics-guided transformer architecture with topographic awareness and wind-following attention for accurate multi-horizon air quality prediction across China**

Advanced deep learning model for multi-pollutant air quality forecasting over China using physics-guided attention, wind-following patch scanning, and chemical interaction modeling.

## Key Innovations

### 1. Wind-Guided Patch Scanning
Reorders spatial patches according to wind direction (upwind to downwind) to capture atmospheric transport patterns.

**Implementation:** `src/climax_core/parallelpatchembed_wind.py`

### 2. Terrain-Aware Attention
Applies elevation-based attention masking to model topographic barriers affecting pollutant dispersion.

**Implementation:** `src/climax_core/physics_attention_patch_level.py`

## Model Architecture

```
Input: Meteorological + Pollutant Data [B, 15, 128, 256]
  ├─ Variables: u, v, temp, rh, psfc, pm10, so2, no2, co, o3, 
  │             lat2d, lon2d, pm25, elevation, population
  │
  ├─ Wind-Guided Patch Embedding (innovation #1)
  │   └─ Reorders patches based on wind direction
  │
  ├─ 6 Transformer Blocks
  │   ├─ Block 0: Physics-Guided Attention (innovation #2)
  │   └─ Blocks 1-5: Standard attention
  │
  └─ Decoder Head
      └─ Output: 6 Pollutants [PM2.5, PM10, SO2, NO2, CO, O3]
                 4 Horizons [12h, 24h, 48h, 96h]
```

## Quick Start

### Training (800 GPUs on LUMI)

```bash
sbatch submit_multipollutants_from_scratch_6pollutants.sh
```

### Configuration

Edit `configs/config_all_pollutants.yaml`:
- Training parameters (batch size, learning rate, etc.)
- Model architecture (depth, embed_dim, etc.)
- Data paths and variables

### Monitoring

```bash
# View logs
tail -f logs/aq_net2_multipoll_from_scratch_<JOB_ID>.out

# TensorBoard (create SSH tunnel first)
ssh -L 6006:<NODE>:<PORT> user@lumi.csc.fi
# Then open: http://localhost:6006
```

## Project Structure

```
aq_net2/
├── main_multipollutants.py     # Training entry point
├── configs/                     # Configuration files
├── src/                         # Source code
│   ├── dataloader.py           # Data loading
│   ├── model_multipollutants.py # Lightning module
│   └── climax_core/            # Core architecture
│       ├── arch.py             # Main model
│       ├── parallelpatchembed_wind.py    # Wind scanning
│       └── physics_attention_patch_level.py # Terrain attention
├── scripts/                    # Execution scripts
├── data_processed/            # Zarr datasets
└── logs/                      # Training outputs
```

See `FILE_STRUCTURE.txt` for detailed organization.

## Hardware Requirements

- Designed for LUMI supercomputer (AMD MI250X GPUs)
- Scales to 800 GPUs (100 nodes × 8 GPUs)
- Uses PyTorch Lightning DDP for distributed training

## Data Format

- Input: Zarr files with consolidated metadata
- Resolution: 128×256 (China + Taiwan region)
- Temporal: Hourly data (2013-2018)
- Train: 2013-2016, Val: 2017, Test: 2018

## Key Features

- Physics-informed inductive biases
- Multi-pollutant forecasting (6 species)
- Multi-horizon prediction (12h to 96h)
- Terrain and wind-aware processing
- Large-scale distributed training support

## Performance

Best validation loss: 0.3552 (checkpoint available)
Training time: ~48h on 800 GPUs

## Citation

If you use this code, please cite:
[Your PhD thesis / paper when published]

## License

[Specify license]

## Contact

Ammar Kheddar - PhD Student in AI
Institution: [Your institution]
Email: khederam@lumi.csc.fi

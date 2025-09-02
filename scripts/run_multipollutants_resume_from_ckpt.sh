#!/bin/bash
# AQ_Net2 Project - Multi-Pollutant execution script
echo "DEBUG: Multi-Pollutant Script started at $(date)"

echo "AQ_Net2 Project - Starting multi-pollutant pipeline..."
echo "DEBUG: Initializing multi-pollutant model..."

# ACTIVATION DU BON ENVIRONNEMENT VIRTUEL
echo "DEBUG: Activating virtual environment..."
source venv_pytorch_rocm/bin/activate
echo "DEBUG: Virtual environment activated: $VIRTUAL_ENV"

# CONFIGURATION PYTHON
export PYTHONUNBUFFERED=1
export PYTHONPATH=/scratch/project_462000640/ammar/aq_net2/src:$PYTHONPATH
export PYTHONPATH=/scratch/project_462000640/ammar/aq_net2:$PYTHONPATH
export TIMM_FUSED_ATTN=0

# ROCm/HIP configuration for LUMI
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:False
export HSA_FORCE_FINE_GRAIN_PCIE=1

# RCCL/NCCL settings for multi-node
export NCCL_IB_DISABLE=1
export RCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_BLOCKING_WAIT=0

# MIOpen cache
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# VÉRIFICATIONS ENVIRONNEMENT
echo "✅ Python version: $(python --version)"
echo "✅ PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Error importing torch')"
echo "✅ Virtual env: $VIRTUAL_ENV"

# Check GPU availability
echo "Checking GPU availability..."
rocm-smi --showid || echo "Warning: rocm-smi failed"

# Check for concurrent Python processes
echo "Checking for concurrent Python processes..."
ps aux | grep $USER | grep python | grep -v grep || echo "No concurrent Python processes found"

# Debug environment variables
echo "DEBUG: MASTER_ADDR=${MASTER_ADDR}"
echo "DEBUG: MASTER_PORT=${MASTER_PORT}"
echo "DEBUG: RANK=${RANK}"
echo "DEBUG: LOCAL_RANK=${LOCAL_RANK}"
echo "DEBUG: WORLD_SIZE=${WORLD_SIZE}"

# Launch multi-pollutant training with PyTorch Lightning
echo "📋 Starting Multi-Pollutant PyTorch Lightning training..."
echo "🎯 Target pollutants: PM2.5, PM10, SO2, NO2, CO, O3"
echo "🚀 Forecast horizons: 1, 3, 5, 7 days"
python main_multipollutants_resume_from_ckpt.py --config ${1:-configs/config_all_pollutants.yaml}

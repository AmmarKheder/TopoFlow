#!/bin/bash
# AQ_Net2 Project - Multi-Pollutant execution script - FIXED for 8 GPUs
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

# ROCm/HIP configuration for LUMI - FIXED for 8 GPUs
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # ✅ FIXED: 8 GPUs
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:256,expandable_segments:False
export HSA_FORCE_FINE_GRAIN_PCIE=1
# Test: Disable SDMA for memory access fault troubleshooting
export HSA_ENABLE_SDMA=0
export MIOPEN_DISABLE_CACHE=1
export HIP_FORCE_DEV_KERNARG=1

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
echo "🚀 Forecast horizons: 12, 24, 48, 96 hours"
echo "⚙️  Elevation-based scanning order enabled"
echo "🔧 Using 8 GPUs per node, 128 nodes = 1024 GPUs total"
torchrun --nproc_per_node=8 --nnodes=$SLURM_JOB_NUM_NODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main_multipollutants.py --config ${1:-configs/config_all_pollutants.yaml}

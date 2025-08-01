#!/bin/bash
# AQ_Net2 Project - Main execution script
echo "DEBUG: Script started at $(date)"

echo "AQ_Net2 Project - Starting training..."

# Activate virtual environment
echo "DEBUG: After venv activation"
source venv_pytorch_rocm/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH=/scratch/project_462000640/ammar/aq_net2:$PYTHONPATH
export TIMM_FUSED_ATTN=0

# ROCm/HIP configuration for LUMI
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:False
export HSA_FORCE_FINE_GRAIN_PCIE=1

# FORCE DISABLE RCCL/NCCL COMPLETELY
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export RCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_DISTRIBUTED_BACKEND=gloo
export USE_GLOO=1
export USE_NCCL=0
export USE_RCCL=0
export NCCL_DEBUG=OFF
export RCCL_DEBUG=OFF

# Disable any distributed initialization
export RANK=-1
export WORLD_SIZE=1
export MASTER_ADDR=""
export MASTER_PORT=""
# # Force GLOO backend instead of NCCL for DataParallel
# export TORCH_DISTRIBUTED_BACKEND=gloo
# export USE_GLOO=1
# 
# # Disable NCCL
# export USE_NCCL=0
# export NCCL_DEBUG=WARN

# MIOpen cache
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# NetCDF protection configuration
echo "Configuring NetCDF protection..."
export HDF5_USE_FILE_LOCKING=FALSE
export NETCDF4_THREAD_SAFE=1
export HDF5_DISABLE_VERSION_CHECK=2

# Protect NetCDF files (read-only)
echo "Protecting NetCDF files..."
find /scratch/project_462000640/ammar/data_aq_net2/ -name "*.nc" -exec chmod 444 {} \;
echo "NetCDF files protected"

# Check GPU availability
echo "Checking GPU availability..."
rocm-smi --showid || echo "Warning: rocm-smi failed"

# Check for concurrent Python processes
echo "Checking for concurrent Python processes..."
ps aux | grep $USER | grep python | grep -v grep || echo "No concurrent Python processes found"

# Launch main training script
echo "Starting main training..."
venv_pytorch_rocm/bin/python src/train.py "$@"

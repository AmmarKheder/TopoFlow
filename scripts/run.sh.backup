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
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# NetCDF protection configuration
echo "Configuring NetCDF protection..."
export HDF5_USE_FILE_LOCKING=FALSE
export NETCDF4_THREAD_SAFE=1
export HDF5_DISABLE_VERSION_CHECK=2

# Protect NetCDF files (read-only)
echo "Protecting NetCDF files..."
find /scratch/project_462000640/ammar/data_aq_net2/ -name "*.nc" -exec chmod 444 {} \;
echo "NetCDF files protected"

# Check for concurrent Python processes
echo "Checking for concurrent Python processes..."
ps aux | grep $USER | grep python | grep -v grep || echo "No concurrent Python processes found"

# Launch main training script
echo "Starting main training..."
venv_pytorch_rocm/bin/python src/train.py

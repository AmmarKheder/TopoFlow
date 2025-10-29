#!/bin/bash
#SBATCH --job-name=wind_400gpu
#SBATCH --account=project_462001079
#SBATCH --partition=standard-g
#SBATCH --nodes=50
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/topoflow_wind_400gpu_%j.out
#SBATCH --error=logs/topoflow_wind_400gpu_%j.err

# TopoFlow Wind Scanning Baseline - 400 GPUs PRODUCTION
# - Wind scanning 32×32 with pre-computed cache (DDP-safe!)
# - Elevation bias active
# - Target: val_loss < 0.260
# - 30 epochs, 400 GPUs (50 nodes × 8 GPUs)

# ROCm/HIP configuration
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:256
export HSA_FORCE_FINE_GRAIN_PCIE=1

# MIOpen cache - per-node cache to avoid DB locking
export MIOPEN_USER_DB_PATH=/tmp/miopen_cache_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH
export MIOPEN_FIND_MODE=1
export MIOPEN_LOG_LEVEL=3

# PYTHONPATH
export PYTHONPATH=/scratch/project_462000640/ammar/aq_net2/src:$PYTHONPATH
export PYTHONPATH=/scratch/project_462000640/ammar/aq_net2:$PYTHONPATH

# CD to project directory
cd /scratch/project_462000640/ammar/aq_net2

# Activate venv
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

echo "========================================="
echo "TopoFlow 400 GPU Training Starting"
echo "Wind scanning: ENABLED (pre-computed cache)"
echo "Elevation bias: ENABLED"
echo "Target: val_loss < 0.260"
echo "========================================="

srun python main_multipollutants.py --config configs/config_wind_400gpu.yaml

echo "Wind Scanning 400 GPU training completed!"

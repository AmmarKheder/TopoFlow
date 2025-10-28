#!/bin/bash
#SBATCH --job-name=physics_ft
#SBATCH --account=project_462001079
#SBATCH --partition=standard-g
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/physics_finetune_%j.out
#SBATCH --error=logs/physics_finetune_%j.err

# Physics Mask Fine-Tuning Experiment
# ====================================
# Load checkpoint 0.3557 (wind scanning only)
# Add physics mask (elevation + Richardson) to first block
# Fine-tune to prove physics contribution
# Target: val_loss < 0.32

echo "========================================"
echo "PHYSICS MASK FINE-TUNING EXPERIMENT"
echo "========================================"
echo "Checkpoint: 0.3557 (wind scanning only)"
echo "Addition: Physics mask (elevation + Richardson)"
echo "Strategy: First block only"
echo "Target: val_loss < 0.32"
echo "========================================"

# ROCm/HIP configuration
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:256
export HSA_FORCE_FINE_GRAIN_PCIE=1

# MIOpen cache - per-node to avoid DB locking
export MIOPEN_USER_DB_PATH=/tmp/miopen_cache_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH
export MIOPEN_FIND_MODE=1
export MIOPEN_LOG_LEVEL=3

# PYTHONPATH
export PYTHONPATH=/scratch/project_462000640/ammar/aq_net2/src:$PYTHONPATH
export PYTHONPATH=/scratch/project_462000640/ammar/aq_net2:$PYTHONPATH

# CD to project
cd /scratch/project_462000640/ammar/aq_net2

# Activate venv
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

echo "Starting physics mask fine-tuning..."
echo "GPUs: 256 (32 nodes × 8)"
echo "Batch accumulation: 16 (optimal!)"
echo "Workers: 4 (reduced for 256 GPUs)"
echo "Epochs: 5"
echo "LR: 5e-5"
echo "Effective batch: 16384"

# LUMI-optimized CPU binding for best performance
# GPU visibility set in main_multipollutants.py based on SLURM_LOCALID
srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 python main_multipollutants.py --config configs/config_physics_finetune.yaml

echo "========================================"
echo "Physics mask fine-tuning completed!"
echo "Check results in logs/physics_finetune_*.out"
echo "========================================"

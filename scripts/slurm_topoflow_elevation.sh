#!/bin/bash
#SBATCH --job-name=topoflow_elev
#SBATCH --account=project_462001079
#SBATCH --time=24:00:00
#SBATCH --nodes=50
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --partition=standard-g
#SBATCH --exclusive
#SBATCH --output=logs/topoflow_elevation_%j.out
#SBATCH --error=logs/topoflow_elevation_%j.err

# TopoFlow: Elevation-Based Attention
# 400 GPUs (50 nodes × 8 GPUs)
# 20k steps, cosine scheduler
# From scratch - no checkpoint

echo "========================================"
echo "TopoFlow Elevation Training - 400 GPUs"
echo "$(date)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 8"
echo "Total GPUs: $(($SLURM_JOB_NUM_NODES * 8))"
echo ""

# Load modules
module purge
module load LUMI/23.09
module load partition/G
module load rocm/5.6.1
module load cray-python/3.10.10

# Activate environment
cd /scratch/project_462000640/ammar/aq_net2
source venv_pytorch_rocm/bin/activate

# Environment variables for ROCm
export MIOPEN_USER_DB_PATH=/scratch/project_462000640/ammar/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

# PyTorch distributed settings
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * 8))
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

echo "Master address: $MASTER_ADDR"
echo "World size: $WORLD_SIZE"
echo ""

# Launch training
echo "Starting TopoFlow elevation training..."
echo "Config: configs/config_topoflow_elevation.yaml"
echo ""

srun python main_multipollutants.py \
    --config configs/config_topoflow_elevation.yaml

EXIT_CODE=$?

echo ""
echo "========================================"
echo "TopoFlow elevation training completed!"
echo "$(date)"
echo "Exit code: $EXIT_CODE"
echo "Check results in logs/topoflow_elevation_$SLURM_JOB_ID.out"
echo "========================================"

exit $EXIT_CODE

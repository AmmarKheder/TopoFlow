#!/bin/bash
#SBATCH --job-name=full_topoflow
#SBATCH --account=project_462001079
#SBATCH --time=24:00:00
#SBATCH --nodes=50
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --partition=standard-g
#SBATCH --exclusive
#SBATCH --output=logs/full_topoflow_%j.out
#SBATCH --error=logs/full_topoflow_%j.err

# FULL TopoFlow: Wind Scanning + Elevation 3D MLP
# 400 GPUs (50 nodes × 8 GPUs)
# From scratch

echo "========================================"
echo "FULL TopoFlow Training - 400 GPUs"
echo "Wind Scanning + Elevation 3D MLP"
echo "$(date)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
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

# Environment variables
export MIOPEN_USER_DB_PATH=/scratch/project_462000640/ammar/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

# PyTorch distributed
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * 8))
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

echo "Master: $MASTER_ADDR"
echo "World size: $WORLD_SIZE"
echo ""

# Launch training
echo "Config: configs/config_full_topoflow.yaml"
echo ""

srun python main_multipollutants.py \
    --config configs/config_full_topoflow.yaml

EXIT_CODE=$?

echo ""
echo "========================================"
echo "FULL TopoFlow completed!"
echo "$(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE

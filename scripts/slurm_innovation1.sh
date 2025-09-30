#!/bin/bash
#SBATCH --job-name=topoflow_innov1
#SBATCH --account=project_462001079
#SBATCH --partition=standard-g
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/topoflow_innovation1_%j.out
#SBATCH --error=logs/topoflow_innovation1_%j.err

# TopoFlow Innovation #1: Pollutant Cross-Attention
# Quick 1-epoch test

module purge
module load LUMI/23.09
module load partition/G
module load rocm/5.6.1
module load PyTorch/2.0.1-rocm-5.6.1-python-3.10-singularity-20231110

export MIOPEN_USER_DB_PATH=/scratch/project_462000640/ammar/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

source venv_pytorch_rocm/bin/activate

srun python main_multipollutants.py --config configs/config_innovation1.yaml

echo "Innovation #1 training completed!"
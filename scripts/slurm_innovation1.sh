#!/bin/bash
#SBATCH --job-name=topoflow_wind_innov1
#SBATCH --account=project_462001079
#SBATCH --partition=standard-g
#SBATCH --nodes=50
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=18:00:00
#SBATCH --output=logs/topoflow_wind_innov1_%j.out
#SBATCH --error=logs/topoflow_wind_innov1_%j.err

# TopoFlow Wind + INNOVATION #1: Pollutant Cross-Attention
# From scratch, 6 epochs





# NO MODULE LOADING - Use venv PyTorch directly
# (Loading Singularity module conflicts with venv)

# ROCm/HIP configuration
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:256
export HSA_FORCE_FINE_GRAIN_PCIE=1
export MIOPEN_DISABLE_CACHE=1

# MIOpen cache
export MIOPEN_USER_DB_PATH=/scratch/project_462000640/ammar/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

# PYTHONPATH
export PYTHONPATH=/scratch/project_462000640/ammar/aq_net2/src:$PYTHONPATH
export PYTHONPATH=/scratch/project_462000640/ammar/aq_net2:$PYTHONPATH

# CD to project directory
cd /scratch/project_462000640/ammar/aq_net2

# Activate venv (PyTorch is IN the venv, not from module)
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

srun python main_multipollutants.py --config configs/config_innovation1.yaml

echo "Wind + Innovation #1 training completed!"

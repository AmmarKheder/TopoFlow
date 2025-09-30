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

module purge
module load LUMI/23.09
module load partition/G
module load rocm/5.6.1
module load PyTorch/2.0.1-rocm-5.6.1-python-3.10-singularity-20231110

export MIOPEN_USER_DB_PATH=/scratch/project_462000640/ammar/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# CD to project directory FIRST
cd /scratch/project_462000640/ammar/aq_net2

# Activate venv with absolute path
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

srun python main_multipollutants.py --config configs/config_innovation1.yaml

echo "Wind + Innovation #1 training completed!"

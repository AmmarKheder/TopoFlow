#!/bin/bash
#SBATCH --job-name=topoflow_wind_baseline
#SBATCH --account=project_462001079
#SBATCH --partition=standard-g
#SBATCH --nodes=50
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=18:00:00
#SBATCH --output=logs/topoflow_wind_baseline_%j.out
#SBATCH --error=logs/topoflow_wind_baseline_%j.err

# TopoFlow Wind Scanning Baseline (from scratch)
# - Wind scanning 32×32 ONLY
# - NO innovations
# - 6 epochs from scratch

module purge
module load LUMI/23.09
module load partition/G
module load rocm/5.6.1
module load PyTorch/2.0.1-rocm-5.6.1-python-3.10-singularity-20231110

export MIOPEN_USER_DB_PATH=/scratch/project_462000640/ammar/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

source venv_pytorch_rocm/bin/activate

srun python main_multipollutants.py --config configs/config_wind_baseline.yaml

echo "Wind Scanning Baseline training completed!"
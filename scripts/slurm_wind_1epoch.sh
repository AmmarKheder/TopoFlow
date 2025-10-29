#!/bin/bash
#SBATCH --job-name=wind_1epoch
#SBATCH --account=project_462001079
#SBATCH --partition=standard-g
#SBATCH --time=02:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --output=/scratch/project_462000640/ammar/aq_net2/logs/wind_1epoch_%j.out
#SBATCH --error=/scratch/project_462000640/ammar/aq_net2/logs/wind_1epoch_%j.err

echo "========================================"
echo "WIND SCANNING BASELINE - 1 EPOCH TEST"
echo "========================================"
echo "Architecture: ClimaX standard + wind scanning 32x32"
echo "From scratch (no checkpoint)"
echo "Purpose: Baseline for physics mask comparison"
echo "========================================"
echo "GPUs: 128 (16 nodes × 8)"
echo "Batch size: 2 per GPU"
echo "Accumulation: 4"
echo "Effective batch: 1024"
echo "Epochs: 1"
echo "LR: 1.5e-4"
echo ""

# Load modules
module load LUMI/23.09
module load partition/G
module load rocm/5.6.1

# Activate environment
cd /scratch/project_462000640/ammar/aq_net2
source venv_pytorch_rocm/bin/activate

# Set environment variables
export MIOPEN_USER_DB_PATH=/tmp/miopen_cache_${SLURM_JOBID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_cache_${SLURM_JOBID}
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
mkdir -p ${MIOPEN_USER_DB_PATH}

echo "Starting wind baseline training (1 epoch)..."
date

# Launch training
srun python main_multipollutants.py --config configs/config_wind_1epoch.yaml

echo ""
echo "========================================"
echo "Wind baseline 1 epoch completed!"
date
echo "Check results in logs/wind_1epoch_*.out"
echo "========================================"

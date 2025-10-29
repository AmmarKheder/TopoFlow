#!/bin/bash
#SBATCH --job-name=continue_baseline
#SBATCH --account=project_462001079
#SBATCH --partition=standard-g
#SBATCH --time=06:00:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --output=/scratch/project_462000640/ammar/aq_net2/logs/continue_baseline_%j.out
#SBATCH --error=/scratch/project_462000640/ammar/aq_net2/logs/continue_baseline_%j.err

echo "========================================"
echo "CONTINUE BASELINE FROM 0.3557"
echo "========================================"
echo "Checkpoint: version_47 (0.3557)"
echo "Architecture: ClimaX standard + wind scanning 32x32"
echo "Strategy: Continue training for 5 epochs"
echo "Target: val_loss < 0.35"
echo "========================================"
echo "Starting continuation training..."
echo "GPUs: 256 (32 nodes × 8)"
echo "Batch size: 2"
echo "Accumulation: 4"
echo "Effective batch: 2048"
echo "Workers: 4"
echo "Epochs: 5"
echo "LR: 1e-4"
echo ""

# Load modules
module load LUMI/23.09
module load partition/G
module load rocm/5.6.1

# Activate environment
cd /scratch/project_462000640/ammar/aq_net2
source venv_pytorch_rocm/bin/activate

# Set environment variables - use local /tmp for MIOpen cache to avoid contention
export MIOPEN_USER_DB_PATH=/tmp/miopen_cache_${SLURM_JOBID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_cache_${SLURM_JOBID}
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
mkdir -p ${MIOPEN_USER_DB_PATH}

# Launch training
srun python main_multipollutants.py --config configs/config_continue_baseline.yaml

echo "========================================"
echo "Baseline continuation completed!"
echo "Check results in logs/continue_baseline_*.out"
echo "========================================"

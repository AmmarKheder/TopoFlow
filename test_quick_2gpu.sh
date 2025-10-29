#!/bin/bash
#SBATCH --job-name=quick_test_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --time=00:10:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001079
#SBATCH --output=logs/quick_test_2gpu_%j.out
#SBATCH --error=logs/quick_test_2gpu_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2

echo "=========================================="
echo "🚀 QUICK TEST: Bash wrapper fix with 2 GPUs"
echo "=========================================="
echo "Testing if Python starts immediately with bash wrapper"
echo ""

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export PL_TORCH_DISTRIBUTED_BACKEND=nccl
export NCCL_TIMEOUT=1800

# THE FIX: Explicit venv activation in srun bash wrapper
srun bash -c "source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate && python /scratch/project_462000640/ammar/aq_net2/main_multipollutants.py --config /scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"

echo ""
echo "=========================================="
echo "✅ Test completed"
echo "=========================================="

#!/bin/bash
#SBATCH --job-name=test_8gpu_checkpoint
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:15:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001079
#SBATCH --output=logs/test_8gpu_%j.out
#SBATCH --error=logs/test_8gpu_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export PL_TORCH_DISTRIBUTED_BACKEND=nccl
export NCCL_TIMEOUT=1800

echo "=========================================="
echo "🧪 TEST: 8 GPUs (1 node) - Checkpoint loading"
echo "=========================================="
echo "Testing full pipeline: venv + checkpoint + training start"
echo ""

# THE FIX: Bash wrapper with explicit venv activation
srun bash -c "source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate && python /scratch/project_462000640/ammar/aq_net2/main_multipollutants.py --config /scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"

echo ""
echo "=========================================="
echo "✅ Test completed"
echo "=========================================="

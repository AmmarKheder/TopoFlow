#!/bin/bash
#SBATCH --job-name=TEST_RESUME_16GPU
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001079
#SBATCH --output=logs/TEST_RESUME_16GPU_%j.out
#SBATCH --error=logs/TEST_RESUME_16GPU_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

export PL_TORCH_DISTRIBUTED_BACKEND=nccl
export PL_DISABLE_FORK_DETECTION=1

export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_IFNAME=hsn0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

source venv_pytorch_rocm/bin/activate

echo "=========================================="
echo "TEST RESUME - 16 GPUs (2 nodes)"
echo "Checkpoint: version_47 val_loss=0.3557"
echo "=========================================="

# Launch training
srun bash -c "source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate && python /scratch/project_462000640/ammar/aq_net2/main_multipollutants.py --config /scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"

echo "=========================================="
echo "✅ TEST COMPLETED"
echo "=========================================="

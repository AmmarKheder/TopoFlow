#!/bin/bash
#SBATCH --job-name=test_main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --time=00:10:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001079
#SBATCH --output=logs/test_main_%j.out
#SBATCH --error=logs/test_main_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2
source venv_pytorch_rocm/bin/activate

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

echo "=========================================="
echo "Testing main_multipollutants.py with 2 GPUs"
echo "=========================================="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo ""

echo "Starting srun python..."
srun python main_multipollutants.py --config configs/config_all_pollutants.yaml

echo ""
echo "=========================================="
echo "Test completed"
echo "=========================================="

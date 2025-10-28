#!/bin/bash
#SBATCH --job-name=test_val_loss_gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:10:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001079
#SBATCH --output=logs/test_val_loss_gpu_%j.out
#SBATCH --error=logs/test_val_loss_gpu_%j.err

# LUMI setup
module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2

# Activate venv
source venv_pytorch_rocm/bin/activate

echo "========================================"
echo "TEST: VALIDATION LOSS FROM CHECKPOINT (1 GPU)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $ROCR_VISIBLE_DEVICES"
echo "========================================"

# Run test
python TEST_VAL_LOSS_FROM_CHECKPOINT.py

echo ""
echo "========================================"
echo "TEST COMPLETED"
echo "========================================"

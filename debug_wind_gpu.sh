#!/bin/bash
#SBATCH --job-name=debug_wind_gpu
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00
#SBATCH --account=project_462001078
#SBATCH --output=logs/debug_wind_gpu_%j.out
#SBATCH --error=logs/debug_wind_gpu_%j.err

echo "🚀 DEBUG WIND SCANNING ON GPU"
echo "Node: $SLURM_NODEID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=" * 50

# Load environment
source venv_pytorch_rocm/bin/activate

# Set CUDA device (même si pas DDP, pour tester)
export LOCAL_RANK=0

# Run debug
cd /scratch/project_462001078/ammar/aq_net2
python debug_wind.py

echo "🏁 GPU DEBUG FINISHED"

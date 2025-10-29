#!/bin/bash
#SBATCH --job-name=test_topoflow_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=small-g
#SBATCH --account=project_462001079
#SBATCH --output=logs/test_topoflow_1gpu_%j.out
#SBATCH --error=logs/test_topoflow_1gpu_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2

export MIOPEN_USER_DB_PATH=/scratch/project_462000640/ammar/miopen_cache

source venv_pytorch_rocm/bin/activate

echo "=========================================="
echo "🧪 QUICK TEST: TopoFlow Block 0 (1 GPU)"
echo "=========================================="
echo "Testing with checkpoint: version_144 val_loss=0.2931"
echo "Just a few steps to check train_loss and val_loss"
echo "=========================================="

# Run just a few steps with quick test config
python main_multipollutants.py --config configs/config_quick_test.yaml

echo "=========================================="
echo "✅ Test completed - check output above"
echo "=========================================="

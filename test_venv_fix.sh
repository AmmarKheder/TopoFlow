#!/bin/bash
#SBATCH --job-name=test_venv_fix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001079
#SBATCH --output=logs/test_venv_fix_%j.out
#SBATCH --error=logs/test_venv_fix_%j.err

echo "=========================================="
echo "Testing fixed venv on compute node"
echo "=========================================="

# Load modules
module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2

echo "Activating venv..."
source venv_pytorch_rocm/bin/activate

echo "Python version:"
python --version

echo ""
echo "Testing PyTorch import:"
python -c "import torch; print('SUCCESS! PyTorch:', torch.__version__); print('ROCm:', torch.version.hip); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

echo ""
echo "Testing PyTorch Lightning:"
python -c "import pytorch_lightning as pl; print('PyTorch Lightning:', pl.__version__)"

echo ""
echo "=========================================="
echo "All tests PASSED!"
echo "=========================================="

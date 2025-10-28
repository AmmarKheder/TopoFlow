#!/bin/bash
#SBATCH --job-name=test_venv
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --time=00:05:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001079
#SBATCH --output=logs/test_venv_%j.out
#SBATCH --error=logs/test_venv_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2

echo "=========================================="
echo "🧪 TEST: Venv Propagation with srun"
echo "=========================================="
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo ""

# Test 1: Without explicit venv in srun (current approach)
echo "--- Test 1: Venv activated before srun ---"
source venv_pytorch_rocm/bin/activate
srun python test_venv_distributed.py

echo ""
echo "--- Test 2: Explicit venv in srun bash wrapper ---"
srun bash -c "source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate && python /scratch/project_462000640/ammar/aq_net2/test_venv_distributed.py"

echo ""
echo "=========================================="
echo "✅ Tests completed - check logs above"
echo "=========================================="

#!/bin/bash
#SBATCH --job-name=test_srun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001079
#SBATCH --output=logs/test_srun_%j.out
#SBATCH --error=logs/test_srun_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2
source venv_pytorch_rocm/bin/activate

echo "=== Test 1: Which python ==="
which python
python --version

echo ""
echo "=== Test 2: srun which python ==="
srun which python
srun python --version

echo ""
echo "=== Test 3: srun bash -c ==="
srun bash -c "which python && python --version"

echo ""
echo "=== Test 4: Direct python script ==="
srun python -c "print('Hello from srun python!')"

echo ""
echo "=== DONE ==="

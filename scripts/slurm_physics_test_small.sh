#!/bin/bash
#SBATCH --job-name=physics_test
#SBATCH --account=project_462000640
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --partition=standard-g
#SBATCH --output=logs/physics_test_%j.out
#SBATCH --error=logs/physics_test_%j.err

module load LUMI/23.09
module load cray-python
module load rocm/5.6.1

cd /scratch/project_462000640/ammar/aq_net2
source venv_pytorch_rocm/bin/activate

export MIOPEN_USER_DB_PATH=/tmp/miopen_cache_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH
export MIOPEN_FIND_MODE=1
export MIOPEN_LOG_LEVEL=3

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "TEST: Physics mask device fix (1 node, 8 GPUs)"
echo "Config: configs/config_physics_test_small.yaml"
echo "Target: Sanity check + 10 training steps without device error"
echo "=========================================="

srun python3 main_multipollutants.py --config configs/config_physics_test_small.yaml

echo ""
echo "=========================================="
echo "Physics test completed!"
echo "=========================================="

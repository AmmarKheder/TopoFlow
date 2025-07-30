#!/bin/bash
#SBATCH --job-name=test_dp
#SBATCH --account=project_462000640
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=test_dp_%j.out
#SBATCH --error=test_dp_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3

cd /scratch/project_462000640/ammar/aq_net2
source venv_pytorch_rocm/bin/activate

export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export NCCL_DEBUG=INFO

python test_dataparallel.py

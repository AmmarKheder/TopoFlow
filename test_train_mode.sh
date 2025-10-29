#!/bin/bash
#SBATCH --job-name=TEST_TRAIN_MODE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001079
#SBATCH --output=logs/TEST_TRAIN_MODE_%j.out
#SBATCH --error=logs/TEST_TRAIN_MODE_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2
source venv_pytorch_rocm/bin/activate

python TEST_TRAIN_MODE.py

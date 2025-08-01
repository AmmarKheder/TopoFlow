#!/bin/bash
#SBATCH --job-name=aq_net2_train
#SBATCH --account=project_462000640
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=4
#SBATCH --mem=180G
#SBATCH --time=72:00:00
#SBATCH --output=logs/aq_net2_train_%j.out
#SBATCH --error=logs/aq_net2_train_%j.err
#SBATCH --exclude=nid[007878,007904-007905]

module purge
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3

cd /scratch/project_462000640/ammar/aq_net2

# Execute main run script
srun scripts/run.sh

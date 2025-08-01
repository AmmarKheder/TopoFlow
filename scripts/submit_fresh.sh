#!/bin/bash
#SBATCH --job-name=aq_net2_fresh
#SBATCH --account=project_462000640
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=4
#SBATCH --mem=180G
#SBATCH --time=72:00:00
#SBATCH --output=logs/aq_net2_fresh_%j.out
#SBATCH --error=logs/aq_net2_fresh_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3

cd /scratch/project_462000640/ammar/aq_net2

# Execute main run script
srun scripts/run.sh --fresh-start

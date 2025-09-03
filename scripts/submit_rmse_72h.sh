#!/bin/bash
#SBATCH --job-name=rmse_baseline
#SBATCH --account=project_462000640
#SBATCH --partition=small-g
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rmse_72h_2018_%j.out
#SBATCH --error=logs/rmse_72h_2018_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3

echo "=========================================="
echo "🚀 ÉVALUATION RMSE - MAX POWER!"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: 4"
echo "Nodes: 4"
echo "Partition: small-g"
echo "💪 MAX POWER SMALL-G"
echo "Date: $(date)"
echo "=========================================="

cd /scratch/project_462000640/ammar/aq_net2

source venv_pytorch_rocm/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH=/scratch/project_462000640/ammar/aq_net2:/scratch/project_462000640/ammar/aq_net2/src:$PYTHONPATH
export TIMM_FUSED_ATTN=0

export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
mkdir -p ${MIOPEN_USER_DB_PATH}

mkdir -p results

echo "🔥 Lancement évaluation RMSE MAX POWER (4 GPU, 4K échantillons)..."

/scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/python evaluation/eval_baseline_complete.py

echo "✅ TERMINÉ avec MAX POWER: $(date)"

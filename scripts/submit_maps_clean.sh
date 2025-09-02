#!/bin/bash
#SBATCH --job-name=maps_clean
#SBATCH --account=project_462000640
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/maps_clean_%j.out
#SBATCH --error=logs/maps_clean_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3

echo "=========================================="
echo "🗺️ GÉNÉRATION DES 4 MAPS VISUELLES"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: 1"
echo "Nodes: 1"
echo "Partition: small-g"
echo "Date: $(date)"
echo "=========================================="

cd /scratch/project_462000640/ammar/aq_net2

source venv_pytorch_rocm/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH=/scratch/project_462000640/ammar/aq_net2:$PYTHONPATH
export TIMM_FUSED_ATTN=0

export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:False

export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
mkdir -p ${MIOPEN_USER_DB_PATH}

echo "🎨 Lancement génération des cartes..."

/scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/python evaluation/generate_maps_only_clean.py

echo "✅ TERMINÉ: $(date)"

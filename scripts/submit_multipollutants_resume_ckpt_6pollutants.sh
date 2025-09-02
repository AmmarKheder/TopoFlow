#!/bin/bash
#SBATCH --job-name=aq_net2_multipoll
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4          # Une tâche par GPU
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --partition=small-g
#SBATCH --account=project_462000640
#SBATCH --output=logs/aq_net2_multipoll_%j.out
#SBATCH --error=logs/aq_net2_multipoll_%j.err

# CONFIGURATION PROPRE SELON BONNES PRATIQUES LUMI
module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2

# Set env for PyTorch DDP
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(ssh $MASTER_NODE "ip addr show hsn0 | grep -oP \"inet \K[^/]+\"" 2>/dev/null || echo $MASTER_NODE)
echo "DEBUG: MASTER_ADDR set to: $MASTER_ADDR"
echo "DEBUG: Nodes in job: $SLURM_JOB_NODELIST"
export MASTER_PORT=29500
export SLURM_PROCID=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID

export PL_TORCH_DISTRIBUTED_BACKEND=nccl
export PL_DISABLE_FORK_DETECTION=1

# ============================================
# 🔥 OPTIMISATIONS NCCL POUR ÉVITER LES TIMEOUTS
# ============================================
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID

# TIMEOUT AUGMENTÉ: 2 heures au lieu de 30 minutes
export NCCL_TIMEOUT=7200

# Gestion asynchrone des erreurs
export NCCL_ASYNC_ERROR_HANDLING=1

# Optimisations supplémentaires
export NCCL_TREE_THRESHOLD=0         # Désactiver l'algo tree pour grands modèles
export NCCL_IB_TIMEOUT=22            # Augmenter timeout InfiniBand
export NCCL_P2P_DISABLE=0            # Activer P2P si possible
export NCCL_COMM_ID_REUSE=0          # Éviter réutilisation des IDs de comm

# Variables PyTorch
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS

echo ""
echo "=============================================="
echo "⚙️  NCCL OPTIMIZATIONS ENABLED"
echo "=============================================="
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT seconds (2 hours)"
echo "NCCL_ASYNC_ERROR_HANDLING: $NCCL_ASYNC_ERROR_HANDLING"
echo "=============================================="

# ============================================
# TENSORBOARD LAUNCH (only on master node)
# ============================================
if [ "$SLURM_NODEID" == "0" ]; then
    source venv_pytorch_rocm/bin/activate
    
    TENSORBOARD_PORT=$((6006 + $RANDOM % 1000))
    
    echo ""
    echo "=============================================="
    echo "🚀 TENSORBOARD SETUP"
    echo "=============================================="
    echo "📊 Starting TensorBoard on node: $MASTER_NODE"
    echo "📊 Port: $TENSORBOARD_PORT"
    echo "📊 Logdir: logs/multipollutants_climax_ddp"
    
    nohup tensorboard --logdir=logs/multipollutants_climax_ddp --port=$TENSORBOARD_PORT --bind_all > tensorboard_${SLURM_JOB_ID}.log 2>&1 &
    TB_PID=$!
    
    echo "=============================================="  | tee tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "🔗 TENSORBOARD CONNECTION INFO"                   | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "=============================================="  | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "Job ID: $SLURM_JOB_ID"                           | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "Node: $MASTER_NODE"                              | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "Port: $TENSORBOARD_PORT"                         | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo ""                                                 | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "To connect from your local machine:"             | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo ""                                                 | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "1. Open a new terminal locally"                  | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "2. Create SSH tunnel:"                           | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "   ssh -L 6006:$MASTER_NODE:$TENSORBOARD_PORT $USER@lumi.csc.fi" | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo ""                                                 | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "3. Open browser: http://localhost:6006"          | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "=============================================="  | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    
    sleep 5
    
    if ps -p $TB_PID > /dev/null; then
        echo "✅ TensorBoard started successfully (PID: $TB_PID)"
    else
        echo "⚠️ TensorBoard failed to start. Check tensorboard_${SLURM_JOB_ID}.log"
    fi
fi

# ============================================
# LAUNCH DISTRIBUTED TRAINING
# ============================================
echo ""
echo "=============================================="
echo "🏃 LAUNCHING DISTRIBUTED TRAINING (16 GPUs)"
echo "=============================================="
echo "Resuming multi-pollutant training from checkpoint epoch=02, val_loss=0.0639 (6 pollutants: PM2.5, PM10, SO2, NO2, CO, O3)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "GPUs per node: 4"
echo "=============================================="

# Lancer l'entraînement distribué
srun --export=ALL scripts/run_multipollutants_resume_from_ckpt.sh configs/config_all_pollutants.yaml

# ============================================
# CLEANUP
# ============================================
if [ "$SLURM_NODEID" == "0" ] && [ ! -z "$TB_PID" ]; then
    echo ""
    echo "🛑 Stopping TensorBoard (PID: $TB_PID)..."
    kill $TB_PID 2>/dev/null || true
fi

echo ""
echo "=============================================="
echo "✅ JOB COMPLETED"
echo "=============================================="
echo "Check tensorboard_connection_${SLURM_JOB_ID}.txt for connection info"

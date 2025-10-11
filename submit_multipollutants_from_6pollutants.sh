#!/bin/bash
#SBATCH --job-name=topoflow_256gpus
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001079
#SBATCH --output=logs/topoflow_full_finetune_%j.out
#SBATCH --error=logs/topoflow_full_finetune_%j.err

# Clean setup according to LUMI best practices
module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2

# Multi-node distributed variables - let PyTorch Lightning handle
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# Set env for PyTorch DDP
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(ssh $MASTER_NODE "ip addr show hsn0 | grep -oP \"inet \K[^/]+\"" 2>/dev/null || echo $MASTER_NODE)
echo "DEBUG: MASTER_ADDR set to: $MASTER_ADDR"
echo "DEBUG: Nodes in job: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_PROCID: $SLURM_PROCID"
echo "DEBUG: SLURM_LOCALID: $SLURM_LOCALID"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"

export PL_TORCH_DISTRIBUTED_BACKEND=nccl
export PL_DISABLE_FORK_DETECTION=1

# ============================================
# 🔥 NCCL OPTIMIZATIONS TO AVOID TIMEOUTS
# ============================================
export NCCL_DEBUG=INFO  # Changed to INFO for better debugging
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID

# Timeout increased: 2 hours instead of 30 minutes
export NCCL_TIMEOUT=7200

# ============================================
# 🚀 ENABLE SLINGSHOT (LUMI's Infiniband) FOR 400 GPUs
# ============================================
# CRITICAL FIX: Re-enable Infiniband for 50-node multi-GPU training
# Disabling IB forces TCP/IP which is unstable at this scale
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export RCCL_MSCCL_ENABLE=0

# COMMENTED OUT: These were causing the pthread/segfault issues
# export NCCL_IB_DISABLE=1
# export RCCL_IB_DISABLE=1
# export NCCL_NET_PLUGIN=none

export NCCL_TREE_THRESHOLD=0         # Disable tree algo for large models
export NCCL_COMM_ID_REUSE=0          # Avoid reuse of comm IDs

# Additional NCCL optimizations for AMD/RCCL
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=8

# PyTorch variables - let SLURM/Lightning manage
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo ""
echo "=============================================="
echo "⚙️  NCCL OPTIMIZATIONS ENABLED (TOPOFLOW FINETUNE)"
echo "=============================================="
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT seconds (2 hours)"
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
    echo ""                                               | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "To connect from your local machine:"             | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo ""                                               | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "1. Open a new terminal locally"                  | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "2. Create SSH tunnel:"                           | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo "   ssh -L 6006:$MASTER_NODE:$TENSORBOARD_PORT $USER@lumi.csc.fi" | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
    echo ""                                               | tee -a tensorboard_connection_${SLURM_JOB_ID}.txt
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
echo "🔥🚀 LAUNCHING TOPOFLOW FULL FINETUNE (256 GPUs) 🚀🔥"
echo "=============================================="
echo "TopoFlow: Wind Scanning + Elevation Mask (6 pollutants: PM2.5, PM10, SO2, NO2, CO, O3)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "GPUs per node: 8 × 32 NODES = 256 GPUs TOTAL"
echo "Fine-tuning from: best-val_loss=0.3557-step=311 (MIGRATED)"
echo "=============================================="

# Launch distributed training (Lightning DDP handles distribution via srun)
srun bash -c "source venv_pytorch_rocm/bin/activate && python main_multipollutants.py --config configs/config_all_pollutants.yaml"

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
echo "✅ TOPOFLOW FULL FINETUNE COMPLETED"
echo "=============================================="
echo "Check tensorboard_connection_${SLURM_JOB_ID}.txt for connection info"

#!/bin/bash
#SBATCH --job-name=TEST_RESUME_16NODES
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001079
#SBATCH --output=logs/TEST_RESUME_16NODES_%j.out
#SBATCH --error=logs/TEST_RESUME_16NODES_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load cray-python/3.11.7

cd /scratch/project_462000640/ammar/aq_net2

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

export PL_TORCH_DISTRIBUTED_BACKEND=nccl
export PL_DISABLE_FORK_DETECTION=1

export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_IFNAME=hsn0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

source venv_pytorch_rocm/bin/activate

echo "=========================================="
echo "TEST RESUME - 128 GPUs (16 nodes)"
echo "Checkpoint: version_47 val_loss=0.3557"
echo "=========================================="

# Create temporary config for 16 nodes
cp configs/config_all_pollutants.yaml configs/config_16nodes_test.yaml

# Update num_nodes in the temporary config
python3 << 'EOF'
import yaml
with open('configs/config_16nodes_test.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['train']['num_nodes'] = 16
with open('configs/config_16nodes_test.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print("✅ Updated config to num_nodes=16")
EOF

# Launch training with 16 nodes config
srun bash -c "source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate && python /scratch/project_462000640/ammar/aq_net2/main_multipollutants.py --config /scratch/project_462000640/ammar/aq_net2/configs/config_16nodes_test.yaml"

echo "=========================================="
echo "✅ TEST COMPLETED"
echo "=========================================="

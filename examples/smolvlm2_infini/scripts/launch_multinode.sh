#!/bin/bash

# SmolVLM2-Infini Multi-Node Distributed Training Launch Script
# Template for scaling across multiple nodes

# IMPORTANT: This script needs to be adapted for your cluster setup
# You need to run this script on each node with the appropriate NODE_RANK

set -e

# ==== CONFIGURATION - MODIFY THESE ====
# Total number of nodes in the cluster
NNODES=${NNODES:-2}

# Number of GPUs per node
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# Current node rank (0, 1, 2, ..., NNODES-1)
# This should be different on each node!
NODE_RANK=${NODE_RANK:-0}

# Master node address (same across all nodes)
MASTER_ADDR=${MASTER_ADDR:-"192.168.1.100"}

# Master port (same across all nodes)
MASTER_PORT=${MASTER_PORT:-29500}

# Rendezvous ID (same across all nodes)
RDZV_ID=${RDZV_ID:-"smolvlm2_multinode"}
# ======================================

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

echo "Starting SmolVLM2-Infini multi-node distributed training..."
echo "Configuration:"
echo "  - Nodes: $NNODES"
echo "  - GPUs per node: $NPROC_PER_NODE" 
echo "  - Current node rank: $NODE_RANK"
echo "  - Master address: $MASTER_ADDR:$MASTER_PORT"
echo "  - Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "Script directory: $SCRIPT_DIR"

# Environment setup for optimal multi-node performance
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO  # Use INFO for debugging, WARN for production
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
export OMP_NUM_THREADS=1

# Network optimization for multi-node
export NCCL_NET_GDR_LEVEL=3  # Enable GPU Direct RDMA if supported
export NCCL_NET_GDR_READ=1

# Optional: Specify network interface (uncomment and modify if needed)
# export NCCL_SOCKET_IFNAME=eth0

# Validate CUDA setup on this node
echo "Available GPUs on node $NODE_RANK:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Calculate total parallelism for config validation
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))
echo "Total GPUs across all nodes: $TOTAL_GPUS"

# Determine config file based on total GPU count
if [ "$TOTAL_GPUS" -eq 16 ]; then
    CONFIG_FILE="$SCRIPT_DIR/../configs/smolvlm2_training_distributed.yaml"
    echo "Using 16-GPU distributed config (update dp to 16)"
elif [ "$TOTAL_GPUS" -eq 32 ]; then
    CONFIG_FILE="$SCRIPT_DIR/../configs/smolvlm2_training_distributed.yaml" 
    echo "Using 32-GPU distributed config (update dp to 32)"
else
    CONFIG_FILE="$SCRIPT_DIR/../configs/smolvlm2_training_distributed.yaml"
    echo "Using default distributed config (update dp to $TOTAL_GPUS)"
    echo "WARNING: Make sure to update parallelism.dp in the config to $TOTAL_GPUS"
fi

# Launch distributed training
echo "Launching torchrun on node $NODE_RANK..."

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$RDZV_ID \
    --max_restarts=3 \
    --tee=3 \
    "$SCRIPT_DIR/train_smolvlm2_infini_distributed.py" \
    --config-file "$CONFIG_FILE"

echo "Training completed on node $NODE_RANK!"

# ==== USAGE EXAMPLES ====
# 
# 1. Two nodes, 8 GPUs each (16 total GPUs):
#    On node 0: NODE_RANK=0 MASTER_ADDR=node0_ip ./launch_multinode.sh
#    On node 1: NODE_RANK=1 MASTER_ADDR=node0_ip ./launch_multinode.sh
#
# 2. Four nodes, 8 GPUs each (32 total GPUs):
#    On node 0: NODE_RANK=0 NNODES=4 MASTER_ADDR=node0_ip ./launch_multinode.sh  
#    On node 1: NODE_RANK=1 NNODES=4 MASTER_ADDR=node0_ip ./launch_multinode.sh
#    On node 2: NODE_RANK=2 NNODES=4 MASTER_ADDR=node0_ip ./launch_multinode.sh
#    On node 3: NODE_RANK=3 NNODES=4 MASTER_ADDR=node0_ip ./launch_multinode.sh
#
# 3. With SLURM (example):
#    #!/bin/bash
#    #SBATCH --nodes=2
#    #SBATCH --ntasks-per-node=1
#    #SBATCH --gres=gpu:8
#    
#    export MASTER_ADDR=$(hostname)
#    srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
#         bash -c "NODE_RANK=\$SLURM_PROCID NNODES=\$SLURM_NNODES ./launch_multinode.sh"
#
# ==== NETWORK REQUIREMENTS ====
# - All nodes must be able to communicate with each other
# - Firewall must allow traffic on the master port (default 29500)
# - High-bandwidth interconnect (InfiniBand, 100GbE) recommended for optimal performance
# - Shared filesystem for checkpoints (set checkpoints_path_is_shared_file_system: true in config)
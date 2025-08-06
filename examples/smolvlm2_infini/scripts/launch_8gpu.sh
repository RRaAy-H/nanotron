#!/bin/bash

# SmolVLM2-Infini 8-GPU Distributed Training Launch Script
# For single node with 8 GPUs

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

echo "Starting SmolVLM2-Infini 8-GPU distributed training..."
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

# Environment setup for optimal performance
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Recommended for distributed ops
export NCCL_DEBUG=INFO  # Enable NCCL debugging (remove in production)
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_NTHREADS=1  # Optimize socket performance
export NCCL_NSOCKS_PERTHREAD=1
export OMP_NUM_THREADS=1  # Prevent CPU threading conflicts

# Optional: Specify GPUs explicitly (uncomment if needed)
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Validate CUDA setup
echo "Available GPUs:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Launch distributed training
echo "Launching torchrun with 8 processes..."

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    --rdzv_id=smolvlm2_8gpu \
    --max_restarts=3 \
    --tee=3 \
    "$SCRIPT_DIR/train_smolvlm2_infini_distributed.py" \
    --config-file "$SCRIPT_DIR/../configs/smolvlm2_training_distributed.yaml"

echo "Training completed!"
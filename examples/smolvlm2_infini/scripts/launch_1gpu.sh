#!/bin/bash

# SmolVLM2-Infini Single GPU Training Launch Script
# For development, testing, or resource-constrained environments

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

echo "Starting SmolVLM2-Infini single-GPU training..."
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

# Environment setup
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

# Optional: Specify GPU (uncomment if needed)
# export CUDA_VISIBLE_DEVICES=0

# Validate CUDA setup
echo "Available GPUs:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Launch distributed training (even for single GPU, we use torchrun for consistency)
echo "Launching torchrun with 1 process..."

torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29503 \
    --rdzv_id=smolvlm2_1gpu \
    --max_restarts=3 \
    --tee=3 \
    "$SCRIPT_DIR/train_smolvlm2_infini_distributed.py" \
    --config-file "$SCRIPT_DIR/../configs/smolvlm2_training_1gpu.yaml"

echo "Training completed!"
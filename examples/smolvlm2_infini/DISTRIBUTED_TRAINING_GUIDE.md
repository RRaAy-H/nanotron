# SmolVLM2-Infini Distributed Training Guide

This guide explains how to use the new distributed training infrastructure for SmolVLM2 with Infini-Attention, built on top of Nanotron's 3D parallelism architecture.

## 🚀 Quick Start

### Prerequisites
```bash
# Install required packages
pip install torch torchvision transformers datasets accelerate
pip install opencv-python pillow

# Verify CUDA setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Launch Training

Choose the appropriate script for your GPU configuration:

```bash
# Single GPU (development/testing)
./scripts/launch_1gpu.sh

# 2 GPUs
./scripts/launch_2gpu.sh  

# 4 GPUs
./scripts/launch_4gpu.sh

# 8 GPUs  
./scripts/launch_8gpu.sh

# Multi-node (advanced)
./scripts/launch_multinode.sh
```

## 📊 Distributed Training Architecture

### 3D Parallelism Explained

SmolVLM2-Infini now supports **3D Parallelism** for scalable training:

1. **Data Parallelism (DP)**: Splits the dataset across GPUs
2. **Tensor Parallelism (TP)**: Splits model layers across GPUs (for very large models)
3. **Pipeline Parallelism (PP)**: Splits model depth across GPUs (for very large models)

### Configuration Examples

| Setup | dp | tp | pp | Total GPUs | Use Case |
|-------|----|----|----|-----------:|----------|
| Single GPU | 1 | 1 | 1 | 1 | Development/testing |
| 4 GPUs | 4 | 1 | 1 | 4 | Small-scale training |
| 8 GPUs | 8 | 1 | 1 | 8 | Standard training |
| 8 GPUs (large model) | 4 | 2 | 1 | 8 | Large model with TP |
| 16 GPUs (2 nodes) | 16 | 1 | 1 | 16 | Multi-node data parallel |

## 🔧 Configuration Files

### Available Configurations

- `configs/smolvlm2_training_1gpu.yaml` - Single GPU setup
- `configs/smolvlm2_training_2gpu.yaml` - 2 GPU setup  
- `configs/smolvlm2_training_4gpu.yaml` - 4 GPU setup
- `configs/smolvlm2_training_distributed.yaml` - 8+ GPU setup

### Key Configuration Sections

#### Parallelism Configuration
```yaml
parallelism:
  dp: 8    # Data parallel size (number of data replicas)
  tp: 1    # Tensor parallel size (splits layers)
  pp: 1    # Pipeline parallel size (splits depth)
  expert_parallel_size: 1  # For MoE models
```

#### Batch Size Configuration
```yaml
tokens:
  micro_batch_size: 2        # Per-GPU batch size
  batch_accumulation_per_replica: 8  # Gradient accumulation steps
  # Effective batch size = micro_batch_size × batch_accumulation_per_replica × dp
  # Example: 2 × 8 × 8 = 128
```

#### Data Configuration
```yaml
data_stages:
  - name: "smolvlm2_multimodal_training"
    start_training_step: 1
    data:
      dataset:
        dataset_overwrite_cache: false
        dataset_processing_num_proc_per_process: 1
        hf_dataset_or_datasets: "data/datasets_nanotron/*_nanotron.json"  # Use multiple nanotron files directly for better distribution
        hf_dataset_splits: "train"
        text_column_name: "text"
```

## 📁 Data Format

### Expected Data Structure

**After Data Preparation:**
```
data/
├── datasets_nanotron/       # Processed training data (nanotron format)
│   ├── dataset1_nanotron.json  # Converted dataset files (used directly)
│   ├── dataset2_nanotron.json  # Converted dataset files (used directly)
│   └── *_nanotron.json         # Multiple sharded nanotron files for efficient distributed loading
└── datasets/                   # Raw datasets (before conversion)
    ├── *.json                  # Raw dataset files
    └── smolvlm2_256m_mixture.yaml  # Dataset mixture configuration
```

**Media Files Location:**
```
/data1/yihao/                # Base path for all media files
├── LLaVA-OneVision-Data/    # Image/text datasets (parquet format)
├── M4-Instruct-Data/        # Multi-image datasets (ZIP format)  
├── MammoTH-VL_Instruct-12M/ # Multi-image datasets (TAR.GZ)
├── VISTA-400K/              # Video datasets (TAR)
├── ShareGPTVideo/           # Video datasets
└── ...                     # Other dataset directories
```

### Data Preparation Workflow

1. **Download and prepare datasets:**
   ```bash
   python scripts/prepare_training_data.py \
       --output_dir data/datasets \
       --base_path /data1/yihao \
       --seed 42
   ```

2. **Convert to nanotron format:**
   ```bash
   python scripts/convert_smolvlm2_data.py \
       --input_dir data/datasets \
       --output_dir data/datasets_nanotron
   ```

   **Note:** The multiple `*_nanotron.json` files are used directly by nanotron for optimal distributed training performance. No need to merge them into a single file.

### Nanotron Data Format
The converted `*_nanotron.json` files contain:
```json
[
  {
    "input_ids": [128000, 9906, 1917, ...],  # Tokenized text
    "text": "<|im_start|>user\nWhat do you see?<|im_end|>\n<|im_start|>assistant\nI see...<|im_end|>",
    "image_path": "LLaVA-OneVision-Data/images/image1.jpg",  # Relative to /data1/yihao
    "video_path": "",
    "id": "sample_001"
  }
]
```

## 🔧 Advanced Usage

### GPU Selection
```bash
# Use specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
./scripts/launch_4gpu.sh

# Or edit the launch script directly
```

### Memory Optimization
For limited GPU memory, adjust these settings:

```yaml
# In your config file
tokens:
  micro_batch_size: 1  # Reduce if OOM
  batch_accumulation_per_replica: 16  # Increase to maintain effective batch size

model:
  dtype: bfloat16  # Use mixed precision
  
optimizer:
  zero_stage: 1  # Enable ZeRO optimization
```

### Multi-Node Training

For clusters with multiple machines:

1. **On all nodes**, ensure:
   - Same codebase and environment
   - Network connectivity between nodes
   - Shared filesystem for checkpoints (recommended)

2. **Update config** for multi-node:
   ```yaml
   parallelism:
     dp: 16  # Total GPUs across all nodes
   
   checkpoints:
     checkpoints_path_is_shared_file_system: true
   ```

3. **Launch on each node**:
   ```bash
   # Node 0 (master)
   NODE_RANK=0 MASTER_ADDR=192.168.1.100 ./scripts/launch_multinode.sh
   
   # Node 1
   NODE_RANK=1 MASTER_ADDR=192.168.1.100 ./scripts/launch_multinode.sh
   ```

## 📈 Performance Guidelines

### Batch Size Guidelines

| GPU Memory | micro_batch_size | Recommended Setting |
|------------|------------------|---------------------|
| 12GB (RTX 3080 Ti) | 1 | Conservative |
| 24GB (RTX 3090/4090) | 2 | Standard |
| 40GB (A100) | 4 | Optimal |
| 80GB (A100/H100) | 8 | High performance |

### Scaling Efficiency

- **Data Parallel**: Near-linear scaling up to 8-16 GPUs
- **Multi-node**: Requires high-bandwidth interconnect (InfiniBand recommended)
- **Tensor Parallel**: Use only for models that don't fit on single GPU

## 🐛 Troubleshooting

### Common Issues

#### 1. NCCL Timeout Errors
```bash
# Enable debugging
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600  # Increase timeout

# Check network connectivity
# All nodes must be able to reach each other on the master port
```

#### 2. Out of Memory (OOM)
```yaml
# Reduce memory usage
tokens:
  micro_batch_size: 1  # Reduce batch size
optimizer:
  zero_stage: 1  # Enable ZeRO
```

#### 3. Process Group Initialization Failure
```bash
# Ensure environment variables are set correctly
echo $RANK $WORLD_SIZE $LOCAL_RANK $MASTER_ADDR $MASTER_PORT

# Check if torchrun is setting these automatically
```

#### 4. Slow Training
```bash
# Enable optimizations
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_DISABLE=0  # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=3  # Enable GPU Direct RDMA
```

### Debugging Commands

```bash
# Check GPU utilization
nvidia-smi

# Monitor NCCL communication
export NCCL_DEBUG=INFO

# Profile training
# Add profiler config to your YAML file
```

## 📝 Migration from Old Script

### Key Changes from Original `train_smolvlm2_infini.py`:

1. **No hardcoded environment variables** - Now properly uses torchrun's environment
2. **Nanotron integration** - Uses DistributedTrainer for proper 3D parallelism  
3. **Better error handling** - Proper distributed error recovery
4. **Scalable architecture** - Supports 1-1000+ GPUs
5. **Production-ready** - Includes checkpointing, monitoring, fault tolerance

### Migration Steps:

1. Replace old training script with new distributed version
2. Update config files to use new format
3. Use launch scripts instead of direct Python execution
4. Update data loading to use new format

## 🎯 Best Practices

### For Development
- Use single GPU config for development and debugging
- Enable verbose logging (`NCCL_DEBUG=INFO`)
- Use small datasets for quick iteration

### For Production  
- Use appropriate parallelism based on your hardware
- Enable checkpointing and monitoring
- Set up proper logging and error handling
- Use shared filesystem for multi-node setups

### For Performance
- Choose optimal batch sizes for your hardware
- Use high-bandwidth interconnects for multi-node
- Enable mixed precision training (bfloat16)
- Consider ZeRO optimization for large models

## 📚 References

- [Nanotron Documentation](https://github.com/huggingface/nanotron)
- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
- [NCCL Performance Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/performance.html)

## 🤝 Support

If you encounter issues:

1. Check this troubleshooting guide
2. Enable debug logging (`NCCL_DEBUG=INFO`)
3. Validate your environment setup
4. Check hardware compatibility
5. Review configuration files for typos

Happy training! 🚀
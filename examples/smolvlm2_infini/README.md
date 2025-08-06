# SmolVLM2-Infini Training Scripts

This directory contains comprehensive training scripts and tools for SmolVLM2 with Infini-Attention integration. The scripts support multimodal training with extended context processing capabilities.

## Quick Start

### 0. Install Dependencies
```bash
# From nanotron root directory
pip install -e .
pip install transformers>=4.35.0 accelerate>=0.24.0 datasets>=2.14.0
pip install pillow>=10.0.0 opencv-python-headless>=4.8.0
```

### 1. Data Preparation
```bash
# Prepare datasets from local storage
python prepare_training_data.py \
    --output_dir ../data/datasets \
    --base_path /data1/yihao \
    --seed 42

# Validate data pipeline
./validate_pipeline.sh
```

### 2. Training

#### Single GPU Training
```bash
# Basic single GPU training
python train_smolvlm2_infini.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
    --data_mixture ../data/smolvlm2_256m_mixture.yaml \
    --output_dir ../checkpoints/smolvlm2_infini \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --bf16 \
    --gradient_checkpointing \
    --do_train

# Use launch script (recommended)
./launch_1gpu.sh
```

#### Distributed Training (Multi-GPU)
```bash
# 2 GPUs
./launch_2gpu.sh

# 4 GPUs  
./launch_4gpu.sh

# 8 GPUs
./launch_8gpu.sh

# Multi-node (advanced)
./launch_multinode.sh

# Or manually with torchrun
torchrun --nproc_per_node=2 train_smolvlm2_infini_distributed.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
    --data_mixture ../data/smolvlm2_256m_mixture.yaml \
    --output_dir ../checkpoints/smolvlm2_infini \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --bf16
```

### 3. Evaluation
```bash
# Test on single image
python evaluate_smolvlm2.py \
    --model_path ../checkpoints/smolvlm2_infini \
    --image_path test.jpg

# Test long context processing
python evaluate_smolvlm2.py \
    --model_path ../checkpoints/smolvlm2_infini \
    --test_long_context
```

## Dataset Composition & Distribution

### Overview: Multimodal Data Split

| Modality | Percentage | Total Samples |
|----------|------------|---------------|
| Image | 34.4% | ~344,000 |
| Video | 33.0% | ~330,000 |
| Text | 20.2% | ~202,000 |
| Multi-image | 12.3% | ~123,000 |
| **Total** | **100%** | **~1,000,000** |

### Text Datasets (20.2%)
All text datasets are in parquet format, located at `/data1/yihao/LLaVA-OneVision-Data`.

| Dataset | Percentage | Samples |
|---------|------------|---------|
| magpie_pro(l3_80b_mt) | 6.8% | ~68,000 |
| magpie_pro(l3_80b_st) | 6.8% | ~68,000 |
| magpie_pro(qwen2_72b_st) | 5.8% | ~58,000 |
| mathqa | 0.9% | ~9,000 |

### Image Datasets (34.4%)
All image datasets are in parquet format, located at `/data1/yihao/LLaVA-OneVision-Data`.

| Dataset | Percentage | Sampling Strategy |
|---------|------------|-------------------|
| other | 17.4% | **Composite**: 70% figureqa+raven, 30% remaining |
| vision_flan(filtered) | 3.9% | Direct sampling |
| mavis_math_metagen | 2.6% | Direct sampling |
| mavis_math_rule_geo | 2.5% | Direct sampling |
| sharegpt4o | 1.7% | Direct sampling |
| sharegpt4v(coco) | 1.5% | Direct sampling |
| image_textualization | 1.3% | Direct sampling |
| sharegpt4v(llava) | 0.9% | Direct sampling |
| MAPQA(MathV360K) | 0.9% | Direct sampling |
| qa | 0.8% | **Alternative**: figureqa or MAPQA substitute |
| textocr(gpt4v) | 0.8% | Direct sampling |

### Video Datasets (33.0%)

| Dataset | Percentage | Format | Path | Strategy |
|---------|------------|--------|------|----------|
| llava-video/1-2m | 7.3% | MP4/MKV+ | `/data1/yihao/llava-video/` | Direct |
| llava-video/2-3m | 7.0% | MP4/MKV+ | `/data1/yihao/llava-video/` | Direct |
| other-video/combined | 5.7% | Video | Various | **Alternative** |
| llava-video/hound | 4.4% | MP4/MKV+ | `/data1/yihao/llava-video/` | Direct |
| llava-video/0-30s | 2.4% | MP4/MKV+ | `/data1/yihao/llava-video/` | Direct |
| video-star/starb | 2.2% | Video | Various | **Alternative** |
| vista-400k/combined | 2.2% | TAR | `/data1/yihao/VISTA-400K/` | Direct |
| vript/long | 1.0% | Video | Various | **Alternative** |
| ShareGPT4Video/all | 0.8% | Video | Various | **Alternative** |

### Multi-image Datasets (12.3%)

| Dataset | Percentage | Format | Path |
|---------|------------|--------|------|
| m4-instruct-data | 10.4% | ZIP | `/data1/yihao/M4-Instruct-Data` |
| mammoth/multi_image | 1.9% | TAR.GZ | `/data1/yihao/MAmmoTH-VL-Instruct-12M/` |

### Sampling Strategy Notes
- **Direct sampling**: Use original dataset as specified
- **Alternative sampling**: Substitute with specified alternatives when unavailable
- **Composite sampling**: Combine multiple sources with defined proportions
- **Deduplication**: All strategies avoid duplicate samples across datasets

### Important Path Corrections & Dataset Structure

**Critical Path Notes:**
- **MAmmoTH dataset**: Located at `/data1/yihao/MAmmoTH-VL-Instruct-12M` (note the capitalization)
- **Video data**: LLaVA-video-178k is located at `/data1/yihao/llava-video/` (separate from LLaVA-OneVision-Data)
- **Video subdirectories**: Follow naming pattern `{duration}_{source}` (e.g., `1_2_m_academic_v0_1`)

**Video Directory Structure:**
```
/data1/yihao/llava-video/
├── 0_30_s_academic_v0_1/
├── 0_30_s_activitynetqa/
├── 1_2_m_academic_v0_1/
├── 2_3_m_academic_v0_1/
└── ... (other time-based subdirectories)
```

**Verified Dataset Locations:**
- ✅ **LLaVA-OneVision-Data**: `/data1/yihao/LLaVA-OneVision-Data`
- ✅ **M4-Instruct-Data**: `/data1/yihao/M4-Instruct-Data`  
- ✅ **VISTA-400K**: `/data1/yihao/VISTA-400K/two_needle_niah_qa`
- ✅ **MAmmoTH-VL-Instruct-12M**: `/data1/yihao/MAmmoTH-VL-Instruct-12M/multi_image_data`
- ✅ **llava-video**: `/data1/yihao/llava-video/` (various time-based subdirs)

## Data Pipeline Testing & Validation

### Comprehensive Pipeline Validation

The data pipeline testing suite ensures all components work correctly before training.

#### Quick Validation Workflow
```bash
# Complete validation pipeline
./validate_pipeline.sh

# Show dataset statistics
python debug_data_pipeline.py --data_dir ../data/datasets --stats

# Validate mixture configuration  
python debug_data_pipeline.py --mixture ../data/smolvlm2_256m_mixture.yaml --validate_mixture
```

#### Comprehensive Pipeline Test
```bash
# Full pipeline validation
python test_data_pipeline.py \
    --data_dir ../data/datasets \
    --mixture_path ../data/smolvlm2_256m_mixture.yaml \
    --num_samples 10 \
    --verbose
```

**What it validates:**
- ✅ Prepared JSON data files structure and format
- ✅ Data mixture YAML configuration validity
- ✅ SmolVLM2 dataset loading with actual processor
- ✅ Data collation and batching
- ✅ Multi-modal data handling (text, image, video, multi-image)
- ✅ Sample validation and format compliance

**Output files:**
- `data_pipeline_validation_report.json` - Detailed validation report
- `test_data_pipeline.log` - Full test execution log

#### Advanced Debugging Tools

**Inspect specific dataset:**
```bash
python debug_data_pipeline.py \
    --dataset ../data/datasets/magpie_pro_l3_80b_mt.json \
    --inspect
```

**Debug individual samples:**
```bash
python debug_data_pipeline.py \
    --dataset ../data/datasets/llava_video_1_2m.json \
    --sample_id 0
```

**Dataset statistics dashboard:**
```bash
python debug_data_pipeline.py --data_dir ../data/datasets --stats
```

Expected output example:
```
Dataset Statistics for: ../data/datasets
================================================================================

┌─────────────────────────┬─────────┬───────────┬──────┬───────┬─────────────┬───────┐
│ Dataset                 │ Samples │ Size (MB) │ Text │ Image │ Multi-Image │ Video │
├─────────────────────────┼─────────┼───────────┼──────┼───────┼─────────────┼───────┤
│ magpie_pro_l3_80b_mt    │  68,000 │    45.23  │68,000│   0   │      0      │   0   │
│ llava_video_1_2m        │  73,000 │    89.45  │   0  │   0   │      0      │73,000 │
│ ...                     │   ...   │    ...    │ ...  │  ...  │     ...     │  ...  │
└─────────────────────────┴─────────┴───────────┴──────┴───────┴─────────────┴───────┘

📊 Summary: Total datasets: 32, Total samples: 1,000,000
```

### Expected Dataset Structure

#### Valid Sample Format
```json
{
  "id": "sample_001",
  "conversations": [
    {
      "from": "human", 
      "value": "What do you see in this image?"
    },
    {
      "from": "gpt",
      "value": "I see a beautiful landscape with mountains."
    }
  ],
  "image": "path/to/image.jpg"
}
```

#### Modality-Specific Fields
- **Text-only**: `id` + `conversations`
- **Image**: Add `image` field (string path or list)
- **Video**: Add `video` field (string path)
- **Multi-image**: Add `image` field (list of paths)

## Distributed Training Architecture

### 3D Parallelism Support

SmolVLM2-Infini supports **3D Parallelism** for scalable training:

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

### Performance Guidelines

#### Batch Size Guidelines

| GPU Memory | micro_batch_size | Recommended Setting |
|------------|------------------|---------------------|
| 12GB (RTX 3080 Ti) | 1 | Conservative |
| 24GB (RTX 3090/4090) | 2 | Standard |
| 40GB (A100) | 4 | Optimal |
| 80GB (A100/H100) | 8 | High performance |

#### Memory Optimization
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

## Complete Training Arguments Reference

### Required Arguments
| Argument | Type | Description |
|----------|------|-------------|
| `--model_name_or_path` | str | Path to pretrained model or HuggingFace model ID |
| `--data_mixture` | str | Path to data mixture YAML file |
| `--output_dir` | str | Directory where model checkpoints will be saved |

### Model Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--trust_remote_code` | True | Allow loading custom model code from HuggingFace |
| `--use_infini_attention` | True | Replace standard attention with infini-attention layers |
| `--segment_length` | 512 | Segment size for infini-attention memory mechanism |

### Data Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--max_seq_length` | 2048 | Maximum token sequence length |
| `--train_data_path` | "data/datasets_nanotron/*_nanotron.json" | Path to training data JSON files (uses glob pattern for multiple nanotron files) |
| `--eval_data_path` | None | Path to evaluation data JSON file for validation during training |
| `--image_dir` | "./images" | Base directory for images (NOT USED in actual pipeline - paths are absolute) |

### Training Configuration
| Argument | Default | Description | Implementation Status |
|----------|---------|-------------|----------------------|
| `--per_device_train_batch_size` | 8 | Training batch size per GPU | ✅ Fully implemented |
| `--per_device_eval_batch_size` | 8 | Evaluation batch size per GPU | ✅ Fully implemented |
| `--num_train_epochs` | 3.0 | Number of training epochs | ✅ Fully implemented |
| `--max_steps` | -1 | Maximum training steps (overrides epochs if > 0) | ✅ Fully implemented |
| `--learning_rate` | 5e-5 | Initial learning rate | ✅ Fully implemented |
| `--weight_decay` | 0.01 | L2 regularization weight | ✅ Fully implemented |
| `--warmup_steps` | 0 | Number of warmup steps for learning rate scheduler | ✅ Fully implemented |
| `--lr_scheduler_type` | "linear" | Type of learning rate scheduler ("linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup") | ✅ Fully implemented |
| `--max_grad_norm` | 1.0 | Maximum gradient norm for clipping (0 = no clipping) | ✅ Fully implemented |
| `--gradient_accumulation_steps` | 1 | Number of steps to accumulate gradients | ❌ NOT implemented |

### Checkpointing & Evaluation
| Argument | Default | Description | Implementation Status |
|----------|---------|-------------|----------------------|
| `--save_steps` | 500 | Save checkpoint every N steps | ✅ Fully implemented |
| `--save_strategy` | "steps" | When to save checkpoints ("steps", "epoch", "no") | ✅ Fully implemented |
| `--eval_steps` | 500 | Run evaluation every N steps during training | ✅ Fully implemented |
| `--do_train` | False | Whether to run training | ✅ Fully implemented |
| `--do_eval` | False | Whether to run evaluation after training | ✅ Fully implemented |
| `--resume_from_checkpoint` | None | Path to checkpoint or "auto" for latest | ✅ Fully implemented |
| `--max_checkpoints_to_keep` | 3 | Maximum checkpoints to retain | ✅ Fully implemented |

### Performance & Precision
| Argument | Default | Description | Implementation Status |
|----------|---------|-------------|----------------------|
| `--bf16` | False | Use bfloat16 precision (saves memory, faster training) | ✅ Fully implemented |
| `--fp16` | False | Use float16 precision | ❌ NOT implemented |
| `--gradient_checkpointing` | False | Trade compute for memory by recomputing activations | ✅ Fully implemented |

### Advanced Checkpointing Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--enable_checkpoint_validation` | True | Enable SHA256 validation and integrity checks |
| `--use_nanotron_checkpointing` | False | Use nanotron's serialize module for distributed training |
| `--incremental_checkpoint_interval` | 10 | Full checkpoint every N saves (others are incremental) |

## Script Reference

### Core Scripts

#### `prepare_training_data.py`
Prepares multimodal datasets from local storage according to the distribution strategy.

```bash
python prepare_training_data.py \
    --output_dir ../data/datasets \
    --base_path /data1/yihao \
    --seed 42 \
    --workers 4 \
    --cache_dir .cache
```

**Arguments:**
- `--output_dir`: Output directory for processed datasets
- `--base_path`: Base path for source data (`/data1/yihao`)
- `--seed`: Random seed for reproducible sampling
- `--workers`: Number of parallel workers
- `--cache_dir`: Caching directory
- `--no_cache`: Disable caching

#### `train_smolvlm2_infini.py`
Main training script with Infini-Attention integration.

```bash
python train_smolvlm2_infini.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
    --data_mixture ../data/smolvlm2_256m_mixture.yaml \
    --output_dir ../checkpoints/smolvlm2_infini \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --bf16 \
    --gradient_checkpointing
```

**Key Features:**
- Distributed training support
- Advanced checkpointing
- TensorBoard integration
- Memory-efficient training
- Infini-Attention mechanism

#### `evaluate_smolvlm2.py` 
Comprehensive evaluation tools for trained models.

```bash
# Single image evaluation
python evaluate_smolvlm2.py \
    --model_path ../checkpoints/smolvlm2_infini \
    --image_path test.jpg \
    --output_dir results

# Long context testing
python evaluate_smolvlm2.py \
    --model_path ../checkpoints/smolvlm2_infini \
    --test_long_context \
    --benchmark_path benchmarks/
```

#### `validate_setup.py`
Environment and setup validation.

```bash
python validate_setup.py
```

**Validates:**
- CUDA setup and GPU availability
- Python dependencies
- Distributed training configuration
- Memory requirements

### Utility Scripts

#### `convert_smolvlm2_data.py`
Converts datasets to Nanotron format.

#### `count_training_tokens.py`
Analyzes token distributions in training data.

#### `create_test_data.py`
Creates test datasets for validation.

#### Launch Scripts
- `launch_1gpu.sh` - Single GPU training
- `launch_2gpu.sh` - 2-GPU distributed training  
- `launch_4gpu.sh` - 4-GPU distributed training
- `launch_8gpu.sh` - 8-GPU distributed training
- `launch_multinode.sh` - Multi-node distributed training

### Training Command Examples

#### Minimal Training (Required Arguments Only)
```bash
python train_smolvlm2_infini.py \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-1.7B-Instruct" \
    --data_mixture "../data/smolvlm2_256m_mixture.yaml" \
    --output_dir "../checkpoints"
```

#### Standard Training with BF16
```bash
python train_smolvlm2_infini.py \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-1.7B-Instruct" \
    --data_mixture "../data/smolvlm2_256m_mixture.yaml" \
    --output_dir "../checkpoints" \
    --train_data_path "../data/datasets_nanotron/*_nanotron.json" \
    --per_device_train_batch_size 4 \
    --bf16 \
    --do_train
```

#### Full Training Configuration
```bash
python train_smolvlm2_infini.py \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-1.7B-Instruct" \
    --data_mixture "../data/smolvlm2_256m_mixture.yaml" \
    --output_dir "../checkpoints" \
    --train_data_path "../data/datasets_nanotron/*_nanotron.json" \
    --eval_data_path "../data/datasets_nanotron/eval_nanotron.json" \
    --use_infini_attention \
    --segment_length 512 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --max_steps 5000 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --lr_scheduler_type "cosine" \
    --max_grad_norm 1.0 \
    --save_steps 500 \
    --save_strategy "steps" \
    --eval_steps 250 \
    --bf16 \
    --gradient_checkpointing \
    --do_train \
    --do_eval
```

#### GPU Selection
```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 ./launch_4gpu.sh

# Or set in environment
export CUDA_VISIBLE_DEVICES=0,1
./launch_2gpu.sh
```

## Advanced Checkpointing System

### Key Features
- **Automatic Checkpoint Saving**: Configurable intervals with atomic writes
- **Complete State Preservation**: Model, optimizer, scheduler, RNG states
- **Automatic Resume**: Auto-detects latest checkpoint
- **Checkpoint Management**: Automatic rotation and validation
- **Graceful Interruption**: Saves on SIGINT/SIGTERM
- **Advanced Features**: Validation, incremental checkpoints, distributed coordination

### Checkpoint Structure
Each checkpoint contains:
```
checkpoint-1000/
├── training_state.pt       # Complete training state
├── pytorch_model.bin       # Model weights only
├── checkpoint_metadata.json # Metadata and step info
├── preprocessor_config.json # Processor configuration
└── tokenizer_config.json   # Tokenizer configuration
```

### Basic Checkpoint Usage
```bash
# Training with checkpoints
python train_smolvlm2_infini.py \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-256M-Video-Instruct" \
    --output_dir ../checkpoints/my_model \
    --save_steps 1000 \
    --max_checkpoints_to_keep 3 \
    --do_train

# Resume from latest
python train_smolvlm2_infini.py \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-256M-Video-Instruct" \
    --output_dir ../checkpoints/my_model \
    --resume_from_checkpoint auto \
    --do_train

# Resume from specific checkpoint
python train_smolvlm2_infini.py \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-256M-Video-Instruct" \
    --output_dir ../checkpoints/my_model \
    --resume_from_checkpoint ../checkpoints/my_model/checkpoint-5000 \
    --do_train
```

### Advanced Checkpoint Features
```bash
# Enable all advanced features
python train_smolvlm2_infini.py \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-256M-Video-Instruct" \
    --output_dir ../checkpoints/my_model \
    --enable_checkpoint_validation \
    --incremental_checkpoint_interval 5 \
    --use_nanotron_checkpointing \
    --save_steps 1000 \
    --max_checkpoints_to_keep 3 \
    --do_train
```

### Testing Checkpoint Functionality
```bash
# Basic checkpoint tests
python test_checkpoint_resume.py

# Advanced checkpoint tests (validation, incremental, distributed)
python test_advanced_checkpoints.py
```

## Multi-Node Distributed Training

### Prerequisites
- Same codebase and environment on all nodes
- Network connectivity between nodes  
- Shared filesystem for checkpoints (recommended)

### Configuration
```yaml
# Update config for multi-node
parallelism:
  dp: 16  # Total GPUs across all nodes

checkpoints:
  checkpoints_path_is_shared_file_system: true
```

### Launch Commands
```bash
# Node 0 (master)
NODE_RANK=0 MASTER_ADDR=192.168.1.100 ./launch_multinode.sh

# Node 1
NODE_RANK=1 MASTER_ADDR=192.168.1.100 ./launch_multinode.sh
```

## Troubleshooting

### Common Issues

#### Missing Dataset Files
```
❌ Dataset xyz.json not found
```
**Solution:** Re-run `prepare_training_data.py` or check file paths

#### Sample Format Issues  
```
⚠️ Found 5 problematic samples in dataset xyz
```
**Solution:** Check conversations have 'from' and 'value' fields

#### Mixture Config Errors
```
❌ JSON file not found for dataset xyz  
```
**Solution:** Update paths in `smolvlm2_256m_mixture.yaml`

#### Import Errors
```bash
# Install dependencies
pip install transformers>=4.35.0 accelerate>=0.24.0 datasets>=2.14.0
pip install pillow>=10.0.0 opencv-python-headless>=4.8.0

# Add nanotron to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/nanotron/src"
```

#### Memory Issues
- Test with smaller `--num_samples` first
- Ensure sufficient RAM for large datasets
- Monitor disk space for temporary files
- Use `--bf16` to reduce memory usage by ~50%
- Enable `--gradient_checkpointing` for memory efficiency

#### NCCL Timeout Errors (Distributed Training)
```bash
# Enable debugging
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600  # Increase timeout

# Check network connectivity
# All nodes must be able to reach each other on the master port
```

#### Out of Memory (OOM)
```yaml
# Reduce memory usage
tokens:
  micro_batch_size: 1  # Reduce batch size
optimizer:
  zero_stage: 1  # Enable ZeRO
```

#### Process Group Initialization Failure
```bash
# Ensure environment variables are set correctly
echo $RANK $WORLD_SIZE $LOCAL_RANK $MASTER_ADDR $MASTER_PORT

# Check if torchrun is setting these automatically
```

#### Checkpoint Issues
```bash
# Checkpoint not found
# Solution: Verify checkpoint path exists and contains training_state.pt

# Checkpoint corruption
# Solution: Use an earlier checkpoint; enable atomic writes

# Out of disk space
# Solution: Reduce max_checkpoints_to_keep or increase storage
```

### Path Configuration

- Mixture config paths are relative to config file location
- Image/video paths are relative to base data path (`/data1/yihao`)
- Use forward slashes (/) on all platforms

## Integration Workflow

### Step-by-Step Training Process

1. **Environment Setup**
```bash
python validate_setup.py
```

2. **Data Preparation**  
```bash
python prepare_training_data.py --output_dir ../data/datasets --base_path /data1/yihao
```

3. **Pipeline Validation**
```bash
./validate_pipeline.sh
```

4. **Training**
```bash
python train_smolvlm2_infini.py [args...]
```

5. **Evaluation**
```bash
python evaluate_smolvlm2.py --model_path ../checkpoints/smolvlm2_infini
```

## Architecture

The SmolVLM2-Infini implementation combines:
- **SmolVLM2's vision-language fusion approach**
- **Nanotron's efficient infini-attention mechanism** 
- **Custom dataset pipeline for multimodal training**

### Key Components:
- **Vision Encoder**: SigLIP-based image processing
- **Text Model**: LLaMA with infini-attention layers
- **Connector**: Vision-language projection layer
- **Memory System**: Compressed memory for extended contexts

### Core Model Implementation
The main model is implemented in:
```
../../src/nanotron/models/smolvlm2_nanotron.py
```

## Directory Structure

```
scripts/
├── Core Scripts
│   ├── prepare_training_data.py     # Dataset preparation from local storage
│   ├── train_smolvlm2_infini.py     # Training script with Infini-Attention
│   ├── evaluate_smolvlm2.py         # Evaluation tools
│   ├── validate_setup.py            # Environment validation
│   └── validate_pipeline.sh         # Complete validation workflow
├── Testing & Debugging
│   ├── test_data_pipeline.py        # Comprehensive pipeline tests
│   ├── debug_data_pipeline.py       # Data debugging tools
│   ├── create_test_data.py          # Test dataset creation
│   └── count_training_tokens.py     # Token analysis
├── Format Conversion
│   └── convert_smolvlm2_data.py     # Nanotron format converter
└── Launch Scripts
    ├── launch_1gpu.sh               # Single GPU training
    ├── launch_2gpu.sh               # 2-GPU distributed training
    ├── launch_4gpu.sh               # 4-GPU distributed training
    ├── launch_8gpu.sh               # 8-GPU distributed training
    └── launch_multinode.sh          # Multi-node distributed training
```

## Performance Features

- **Infini-Attention**: Process sequences up to 16K tokens with 512-token segments
- **Memory Compression**: Efficient long-context processing
- **Distributed Training**: Multi-GPU and multi-node support
- **Advanced Checkpointing**: Resume from any point
- **Gradient Checkpointing**: Reduce memory usage during training

## Requirements

- **GPU**: 8GB+ VRAM (A100/V100 recommended)
- **RAM**: 32GB+ system memory  
- **Storage**: 100GB+ for datasets and checkpoints
- **Python**: 3.8-3.12
- **CUDA**: 11.8+

## Files Generated

- `test_data_pipeline.log` - Complete test execution log
- `data_pipeline_validation_report.json` - Detailed validation results
- Various checkpoint and log files during training

## Citation

If you use this implementation, please cite:
- SmolVLM2 paper and repository
- Nanotron framework  
- Infini-Attention paper

## Support

For detailed instructions, see the parent directory's `SMOLVLM2_USAGE_GUIDE.md`.
For issues, refer to the Nanotron documentation or repository issues.
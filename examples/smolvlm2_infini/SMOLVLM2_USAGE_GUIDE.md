# SmolVLM2 with Infini-Attention: Comprehensive Usage Guide

This guide provides step-by-step instructions for training SmolVLM2 with Nanotron's Infini-Attention mechanism. This implementation enables extended context processing for multimodal understanding tasks.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Configuration](#configuration)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Inference](#inference)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (A100/V100 recommended for faster training)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ free disk space for datasets and checkpoints
- **OS**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.8-3.12

### Software Requirements
- CUDA 11.8+ (for GPU training)
- Git
- wget or curl for downloading datasets

## Installation

### Step 1: Setup Nanotron Environment

```bash
# Navigate to nanotron directory
cd nanotron

# Create and activate virtual environment
python3 -m venv venv_smolvlm2
source venv_smolvlm2/bin/activate  # On Windows: venv_smolvlm2\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 2: Install Dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch>=2.1.0 torchvision>=0.16.0 torchaudio>=2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install Nanotron
pip install -e .

# Install additional dependencies for SmolVLM2
pip install transformers>=4.35.0 accelerate>=0.24.0 datasets>=2.14.0
pip install pillow>=10.0.0 opencv-python-headless>=4.8.0 decord>=0.6.0
pip install timm>=0.9.0 einops>=0.7.0

# Install monitoring tools (optional but recommended)
pip install wandb tensorboard
```

### Step 3: Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check Nanotron
python -c "import nanotron; print('Nanotron installed successfully')"

# Check Transformers
python -c "from transformers import AutoProcessor; print('Transformers ready')"
```

## Dataset Preparation

### Step 1: Download Datasets

```bash
# Navigate to SmolVLM2 directory
cd examples/smolvlm2_infini

# Create data directory
mkdir -p data/datasets

# Download datasets using our script
python scripts/download_datasets.py --output_dir data/datasets --seed 42
```

This will download samples according to the new multimodal dataset distribution:
- **Image datasets**: 34.4% of total (located at `/data1/yihao/LLaVA-OneVision-Data`)
- **Video datasets**: 33.0% of total (multiple sources and formats)
- **Text datasets**: 20.2% of total (parquet format at `/data1/yihao/LLaVA-OneVision-Data`)
- **Multi-image datasets**: 12.3% of total (ZIP and TAR.GZ formats)

**Note**: The download process may take several hours depending on your internet connection.

### Step 2: Convert to Nanotron Format

```bash
# Convert all downloaded datasets to Nanotron format
python scripts/convert_smolvlm2_data.py \
    --input_dir data/datasets \
    --output_dir data/datasets_nanotron
```

### Step 3: Dataset Mixture Configuration

The dataset mixture is configured in `data/smolvlm2_256m_mixture.yaml` with the updated dataset distribution:
- Uses actual data paths (e.g., `/data1/yihao/LLaVA-OneVision-Data`)
- Includes sampling strategies: direct, composite, and alternative sampling
- Supports deduplication across datasets
- Covers all four modalities: text, image, video, and multi-image

#### Dataset Breakdown by Modality:

**Text Datasets (20.2%)** - All in parquet format at `/data1/yihao/LLaVA-OneVision-Data`:
- magpie_pro(l3_80b_mt): 6.8%
- magpie_pro(l3_80b_st): 6.8% 
- magpie_pro(qwen2_72b_st): 5.8%
- mathqa: 0.9%

**Multi-image Datasets (12.3%)**:
- m4-instruct-data: 10.4% (ZIP format at `/data1/yihao/M4-Instruct-Data`)
- mammoth/multi_image_data: 1.9% (TAR.GZ at `/data1/yihao/MammoTH-VL_Instruct-12M/multi_image_data`)

**Image Datasets (34.4%)** - All in parquet format at `/data1/yihao/LLaVA-OneVision-Data`:
- llava-onevision/other: 17.4% (composite sampling: 70% figureqa+raven, 30% remaining)
- vision_flan(filtered): 3.9%
- mavis_math_metagen: 2.6%
- mavis_math_rule_geo: 2.5%
- sharegpt4o: 1.7%
- sharegpt4v(coco): 1.5%
- image_textualization: 1.3%
- sharegpt4v(llava): 0.9%
- MAPQA(MathV360K): 0.9%
- qa: 0.8% (alternative sampling)
- textocr(gpt4v): 0.8%

**Video Datasets (33.0%)** - Multiple sources and formats:
- llava-video-178k/1-2m: 7.3% (MP4 at `/data1/yihao/LLaVA-OneVision-Data`)
- llava-video-178k/2-3m: 7.0% (MP4 at `/data1/yihao/LLaVA-OneVision-Data`)
- other-video/combined: 5.7% (alternative sampling)
- llava-video-178k/hound: 4.4% (MP4 at `/data1/yihao/LLaVA-OneVision-Data`)
- llava-video-178k/0-30s: 2.4% (MP4 at `/data1/yihao/LLaVA-OneVision-Data`)
- video-star/starb: 2.2% (alternative sampling)
- vista-400k/combined: 2.2% (TAR at `/data1/yihao/VISTA-400K/two_needle_niah_qa`)
- vript/long: 1.0% (alternative sampling)
- ShareGPT4Video/all: 0.8% (at `/data1/yihao/ShareGPTVideo/train_300k`)

## Configuration

### Model Configuration

The model configuration is defined in `configs/smolvlm2_config.py`. Key parameters:

```python
# Model size (256M parameters)
hidden_size: 512
num_hidden_layers: 12
num_attention_heads: 8

# Infini-attention settings
use_infini_attention: True
segment_length: 512  # Segment size for infinite attention

# Context length
max_position_embeddings: 16384  # Extended context support
```

### Training Configuration

Edit `configs/smolvlm2_training.yaml` for your setup:

```yaml
# Adjust batch size based on GPU memory
tokens:
  micro_batch_size: 2  # Reduce if OOM
  batch_accumulation_per_replica: 8
  sequence_length: 2048

# Training steps
train_steps: 10000  # Adjust based on actual dataset size from new distribution

# Multi-GPU settings (if applicable)
parallelism:
  dp: 2  # Data parallel GPUs
  pp: 1  # Pipeline parallel
  tp: 1  # Tensor parallel
```

## Training

### Step 1: Single GPU Training

```bash
# Basic training command
python scripts/train_smolvlm2_infini.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Instruct \
    --data_mixture data/smolvlm2_256m_mixture.yaml \
    --output_dir checkpoints/smolvlm2_infini \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_steps 1000 \
    --report_to wandb
```

### Step 2: Multi-GPU Training

```bash
# For 2 GPUs
torchrun --nproc_per_node=2 scripts/train_smolvlm2_infini.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Instruct \
    --data_mixture data/smolvlm2_256m_mixture.yaml \
    --output_dir checkpoints/smolvlm2_infini \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --bf16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters false \
    --report_to wandb
```

### Step 3: Resume Training

```bash
# Resume from checkpoint
python scripts/train_smolvlm2_infini.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Instruct \
    --data_mixture data/smolvlm2_256m_mixture.yaml \
    --output_dir checkpoints/smolvlm2_infini \
    --resume_from_checkpoint checkpoints/smolvlm2_infini/checkpoint-5000 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --bf16 \
    --gradient_checkpointing
```

## Evaluation

### Step 1: Basic Evaluation

```bash
# Test on a single image
python scripts/evaluate_smolvlm2.py \
    --model_path checkpoints/smolvlm2_infini \
    --image_path test_images/example.jpg \
    --output_dir results/

# Test long context handling
python scripts/evaluate_smolvlm2.py \
    --model_path checkpoints/smolvlm2_infini \
    --test_long_context \
    --output_dir results/
```

### Step 2: Benchmark Evaluation

```bash
# Install VLMEvalKit
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .

# Run standard benchmarks
python run.py \
    --data MMBench_DEV_EN \
    --model checkpoints/smolvlm2_infini \
    --work-dir results/mmbench
```

## Inference

### Python Script Inference

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# Load model and processor
model = AutoModelForVision2Seq.from_pretrained(
    "checkpoints/smolvlm2_infini",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "checkpoints/smolvlm2_infini",
    trust_remote_code=True
)

# Load and process image
image = Image.open("example.jpg")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

# Generate response
inputs = processor.apply_chat_template(messages, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )

response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Errors

```bash
# Reduce batch size
--per_device_train_batch_size 1

# Enable gradient checkpointing
--gradient_checkpointing

# Use mixed precision
--bf16

# Reduce sequence length in config
sequence_length: 1024  # Instead of 2048
```

#### 2. Import Errors

Make sure you're in the correct directory:
```bash
# Always run from nanotron/examples/smolvlm2_infini/
cd nanotron/examples/smolvlm2_infini
```

#### 3. Dataset Loading Issues

```bash
# Clear cache
rm -rf ~/.cache/huggingface/datasets/

# Re-download specific dataset
python scripts/download_datasets.py \
    --output_dir data/datasets
```

## File Structure

```
nanotron/examples/smolvlm2_infini/
├── configs/
│   ├── smolvlm2_config.py          # Model configuration
│   └── smolvlm2_training.yaml      # Training configuration
├── scripts/
│   ├── download_datasets.py        # Dataset downloader
│   ├── convert_smolvlm2_data.py    # Data format converter
│   ├── train_smolvlm2_infini.py    # Main training script
│   └── evaluate_smolvlm2.py        # Evaluation script
├── data/
│   └── smolvlm2_256m_mixture.yaml  # Dataset mixture config
└── SMOLVLM2_USAGE_GUIDE.md         # This guide
```

Plus the model implementation in:
```
nanotron/src/nanotron/models/smolvlm2_nanotron.py
```

## Summary

This implementation provides a complete pipeline for training SmolVLM2 with Infini-Attention:

1. **Model Integration**: SmolVLM2 adapted for Nanotron with infini-attention
2. **Data Pipeline**: Multimodal dataset with optimized sampling strategies for training
3. **Training Scripts**: Single and multi-GPU training support
4. **Evaluation Tools**: Comprehensive testing and benchmarking
5. **Extended Context**: Process sequences up to 16K tokens efficiently

The infini-attention mechanism enables processing of contexts beyond typical transformer limitations while maintaining computational efficiency.

For issues or questions, refer to the Nanotron documentation or open an issue on the repository.

Happy training! 🚀
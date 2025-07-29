# SmolVLM2 with Infini-Attention Implementation

This directory contains a complete implementation of SmolVLM2 integrated with Nanotron's Infini-Attention mechanism, enabling extended context processing for multimodal understanding tasks.

## Quick Start

### 1. Install Dependencies
```bash
# From nanotron root directory
pip install -e .
pip install transformers>=4.35.0 accelerate>=0.24.0 datasets>=2.14.0
pip install pillow>=10.0.0 opencv-python-headless>=4.8.0
```

### 2. Download and Prepare Data
```bash
# Navigate to this directory
cd examples/smolvlm2_infini

# Download datasets (800K samples)
python scripts/download_datasets.py --output_dir data/datasets

# Convert to Nanotron format
python scripts/convert_smolvlm2_data.py
```

### 3. Train Model
```bash
# Single GPU training
python scripts/train_smolvlm2_infini.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Instruct \
    --data_mixture data/smolvlm2_256m_mixture.yaml \
    --output_dir checkpoints/smolvlm2_infini \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --bf16 \
    --gradient_checkpointing

# Multi-GPU training
torchrun --nproc_per_node=2 scripts/train_smolvlm2_infini.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Instruct \
    --data_mixture data/smolvlv2_256m_mixture.yaml \
    --output_dir checkpoints/smolvlm2_infini \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --bf16
```

### 4. Evaluate
```bash
# Test on single image
python scripts/evaluate_smolvlm2.py \
    --model_path checkpoints/smolvlm2_infini \
    --image_path test.jpg

# Test long context processing
python scripts/evaluate_smolvlm2.py \
    --model_path checkpoints/smolvlm2_infini \
    --test_long_context
```

## Features

- **Infini-Attention Integration**: Process sequences up to 16K tokens with 512-token segments
- **Multimodal Support**: Handles images, videos, and text inputs
- **Optimized Dataset**: 800K samples carefully balanced for 256M parameter models
- **Extended Context**: Efficient long-context processing with memory compression
- **Comprehensive Evaluation**: Tools for testing model performance

## Architecture

The implementation combines:
- SmolVLM2's vision-language fusion approach
- Nanotron's efficient infini-attention mechanism
- Custom dataset pipeline for multimodal training

Key components:
- **Vision Encoder**: SigLIP-based image processing
- **Text Model**: LLaMA with infini-attention layers
- **Connector**: Vision-language projection layer
- **Memory System**: Compressed memory for extended contexts

## Files

```
├── configs/
│   ├── smolvlm2_config.py          # Model configuration
│   └── smolvlm2_training.yaml      # Training hyperparameters
├── scripts/
│   ├── download_datasets.py        # Dataset downloader
│   ├── convert_smolvlm2_data.py    # Format converter
│   ├── train_smolvlm2_infini.py    # Training script
│   └── evaluate_smolvlm2.py        # Evaluation tools
├── data/
│   └── smolvlm2_256m_mixture.yaml  # Dataset mixture
├── SMOLVLM2_USAGE_GUIDE.md         # Detailed guide
└── README.md                       # This file
```

## Model Implementation

The core model is implemented in:
```
../../src/nanotron/models/smolvlm2_nanotron.py
```

## Requirements

- **GPU**: 8GB+ VRAM (A100/V100 recommended)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ for datasets and checkpoints
- **Python**: 3.8-3.12
- **CUDA**: 11.8+

## Dataset Composition

Optimized 800K sample mixture:
- **Image datasets (80%)**: LLaVA-OneVision, ShareGPT4V, AI2D, ChartQA, DVQA, etc.
- **Video datasets (12.5%)**: LLaVA-Video, VideoChat, Video-ChatGPT
- **Text datasets (7.5%)**: Alpaca, ShareGPT for instruction following

## Performance

The infini-attention mechanism enables:
- Processing of sequences beyond standard transformer limits
- Memory-efficient attention computation through segmentation
- Maintained performance on both short and long contexts
- Scalable to even longer sequences with minimal overhead

## Citation

If you use this implementation, please cite:
- SmolVLM2 paper and repository
- Nanotron framework
- Infini-Attention paper

## Support

For detailed instructions, see `SMOLVLM2_USAGE_GUIDE.md`.
For issues, refer to the Nanotron documentation or repository issues.
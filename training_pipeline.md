# SmolVLM2 with Infinite Attention Traininig

This tutorial shows how to train SmolVLM2 models using Nanotron with integrated infinite attention mechanism for extended context processing.

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Model Architecture](#2-model-architecture) 
3. [Dataset Preparation](#3-dataset-preparation)
4. [Training Configuration](#4-training-configuration)
5. [Training Process](#5-training-process)
6. [Evaluation](#6-evaluation)

## 1. Environment Setup

### Requirements
- NVIDIA GPU with 8GB+ VRAM
- Python 3.8-3.12
- CUDA 11.8+
- 32GB+ RAM recommended

### Installation

```bash
# Clone nanotron repository
git clone https://github.com/huggingface/nanotron.git
cd nanotron

# Create virtual environment
python3 -m venv venv_smolvlm2
source venv_smolvlm2/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install torch>=2.1.0 torchvision>=0.16.0 torchaudio>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install transformers>=4.35.0 accelerate>=0.24.0 datasets>=2.14.0
pip install pillow>=10.0.0 opencv-python-headless>=4.8.0 decord>=0.6.0
pip install timm>=0.9.0 einops>=0.7.0
pip install wandb tensorboard
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import nanotron; print('Nanotron installed successfully')"
```

## 2. Model Architecture

SmolVLM2 is based on the Idefics3 architecture with custom vision-language fusion:

- **Vision Encoder**: SigLIP model for image processing
- **Language Model**: Small transformer with infini-attention
- **Connector**: Vision-language projection layer

### Key Components from Original SmolVLM2

The original implementation uses:
- `SmolVLMModel` extending `Idefics3Model`
- Custom `inputs_merger` for vision-text fusion
- Multimodal training with image/video support

### Integration with Nanotron

Create `src/nanotron/models/smolvlm2_nanotron.py`:

```python
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union
from transformers import Idefics3Config, Idefics3Model, Idefics3ForConditionalGeneration

from nanotron.models.base import NanotronModel
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.tensor_parallel.nn import TensorParallelColumnLinear, TensorParallelRowLinear
from nanotron.models.llama import LlamaModel  # For infini-attention integration

class SmolVLM2NanotronModel(NanotronModel):
    """SmolVLM2 model adapted for Nanotron with infini-attention support"""
    
    def __init__(self, config, parallel_context: ParallelContext):
        super().__init__()
        self.config = config
        self.parallel_context = parallel_context
        
        # Initialize base Idefics3 model structure
        self.vision_model = self._build_vision_model(config)
        self.connector = self._build_connector(config)
        
        # Use Nanotron's LlamaModel with infini-attention for text processing
        self.text_model = LlamaModel(
            config=config.text_config,
            parallel_context=parallel_context
        )
        
        # Language modeling head
        self.lm_head = TensorParallelColumnLinear(
            in_features=config.text_config.hidden_size,
            out_features=config.text_config.vocab_size,
            pg=parallel_context.tp_pg,
            bias=False
        )
        
        # Image token for multimodal fusion
        self.image_token_id = getattr(config, 'image_token_id', 32000)
    
    def _build_vision_model(self, config):
        """Build vision encoder based on original SmolVLM2"""
        # Use the vision config from Idefics3
        vision_config = config.vision_config
        
        # Simple vision model implementation
        # In practice, load from HuggingFace SigLIP
        from transformers import SiglipVisionModel
        return SiglipVisionModel(vision_config)
    
    def _build_connector(self, config):
        """Build vision-language connector"""
        return nn.Sequential(
            nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)
        )
    
    def inputs_merger(self, input_ids, inputs_embeds, image_hidden_states):
        """Merge vision and text inputs (adapted from original SmolVLM2)"""
        if image_hidden_states is None:
            return inputs_embeds
        
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        num_images, num_patches, vision_hidden_size = image_hidden_states.shape
        
        # Find image token positions
        image_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)
        
        if len(image_positions[0]) == 0:
            return inputs_embeds
        
        # Simple replacement strategy - replace image tokens with vision features
        merged_embeds = inputs_embeds.clone()
        
        for batch_idx in range(batch_size):
            batch_positions = image_positions[1][image_positions[0] == batch_idx]
            if len(batch_positions) > 0:
                # Take first num_patches positions and replace with vision features
                for i, pos in enumerate(batch_positions[:num_patches]):
                    if i < num_patches:
                        merged_embeds[batch_idx, pos] = image_hidden_states[batch_idx, i]
        
        return merged_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with multimodal processing"""
        
        # Process vision inputs if provided
        image_hidden_states = None
        if pixel_values is not None:
            # Process images through vision encoder
            vision_outputs = self.vision_model(pixel_values=pixel_values)
            vision_features = vision_outputs.last_hidden_state
            
            # Project to language model dimension
            image_hidden_states = self.connector(vision_features)
        
        # Get text embeddings
        inputs_embeds = self.text_model.token_position_embeddings.pp_block.token_embedding(input_ids)
        
        # Merge vision and text features
        if image_hidden_states is not None:
            inputs_embeds = self.inputs_merger(input_ids, inputs_embeds, image_hidden_states)
        
        # Forward through language model with infini-attention
        hidden_states = self.text_model(
            input_ids=None,  # Use inputs_embeds instead
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state
        
        # Generate logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states
        }
    
    def get_tied_parameters(self):
        """Return tied parameters for nanotron"""
        return []
```

### Configuration

Create `configs/smolvlm2_config.py`:

```python
from dataclasses import dataclass
from typing import Optional
from nanotron.config.models_config import NanotronConfigs

@dataclass 
class SmolVLM2Config:
    """Configuration for SmolVLM2 with Nanotron integration"""
    
    # Text model config (based on SmolLM)
    vocab_size: int = 49152
    hidden_size: int = 512
    intermediate_size: int = 1536
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 16384
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    
    # Vision config
    vision_config: Optional[dict] = None
    
    # Multimodal config
    image_token_id: int = 32000
    
    # Infini-attention config
    use_infini_attention: bool = True
    segment_length: int = 512
    
    def __post_init__(self):
        if self.vision_config is None:
            self.vision_config = {
                "hidden_size": 768,
                "image_size": 224,
                "patch_size": 16,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
            }
        
        # Ensure compatibility with Nanotron
        self.is_llama_config = True  # For Nanotron compatibility
```

## 3. Dataset Preparation

### Training Dataset Mixture for SmolVLM2-256M with Infini-Attention

For optimal training of a 256M parameter model, we use a carefully balanced dataset mixture totaling approximately **800K samples** (reduced from the original 3.3M to prevent overfitting on smaller models):

#### Core Dataset Composition

| Dataset | Type | Samples | Percentage | Description |
|---------|------|---------|------------|-------------|
| **Image Datasets** | | **640K** | **80%** | |
| LLaVA-Instruct-150K | Image | 150K | 18.75% | Core instruction following |
| ShareGPT4V | Image | 150K | 18.75% | High-quality conversations |
| AI2D | Image | 50K | 6.25% | Diagram understanding |
| ChartQA | Image | 40K | 5% | Chart and graph analysis |
| VQAv2 | Image | 40K | 5% | Visual question answering |
| TallyQA | Image | 30K | 3.75% | Counting tasks |
| ScienceQA | Image | 25K | 3.125% | Scientific reasoning |
| TextVQA | Image | 25K | 3.125% | OCR and text reading |
| VizWiz | Image | 20K | 2.5% | Real-world accessibility |
| COCO Captions | Image | 60K | 7.5% | Dense captioning |
| **Video Datasets** | | **100K** | **12.5%** | |
| LLaVA-Video | Video | 70K | 8.75% | Video understanding |
| VideoInstruct-100K | Video | 20K | 2.5% | Video instruction following |
| OpenOrca | Text | 10K | 1.25% | Text instruction following |
| **Text Datasets** | | **60K** | **7.5%** | |
| Alpaca | Text | 40K | 5% | Instruction following |
| ShareGPT | Text | 20K | 2.5% | Conversational AI |

**Total: 800K samples optimized for 256M parameter training**

### Dataset Configuration

Create `scripts/mixtures/smolvlm2_256m_mixture.yaml`:

```yaml
# Image datasets (80% of total - 640K samples)
- json_path: /path/to/llava_instruct_150k.json
  sampling_strategy: random:18.75%
  name: llava-instruct-subset
  path: llava-instruct
  modality: image
  source: llava-instruct
  _comment: 'Core multimodal instruction following - 150K samples'

- json_path: /path/to/sharegpt4v_150k.json
  sampling_strategy: random:18.75%
  name: sharegpt4v-quality
  path: sharegpt4v
  modality: image  
  source: sharegpt4v
  _comment: 'High-quality image conversations - 150K samples'

- json_path: /path/to/ai2d_50k.json
  sampling_strategy: random:6.25%
  name: ai2d-diagrams
  path: ai2d
  modality: image
  source: ai2d
  _comment: 'Scientific diagram understanding - 50K samples'

- json_path: /path/to/chartqa_40k.json
  sampling_strategy: random:5%
  name: chartqa-reasoning
  path: chartqa
  modality: image
  source: chartqa
  _comment: 'Chart and graph analysis - 40K samples'

- json_path: /path/to/vqav2_40k.json
  sampling_strategy: random:5%
  name: vqav2-questions
  path: vqav2
  modality: image
  source: vqav2
  _comment: 'Visual question answering - 40K samples'

# Video datasets (12.5% of total - 100K samples)
- json_path: /path/to/llava_video_70k.json
  sampling_strategy: random:8.75%
  name: llava-video-understanding
  path: llava-video
  modality: video
  source: llava-video
  _comment: 'Video understanding and reasoning - 70K samples'

- json_path: /path/to/videoinstruct_20k.json
  sampling_strategy: random:2.5%
  name: videoinstruct-conversations
  path: videoinstruct
  modality: video
  source: videoinstruct
  _comment: 'Video instruction following - 20K samples'

- json_path: /path/to/openorca_10k.json
  sampling_strategy: random:1.25%
  name: openorca-instructions
  path: openorca
  modality: text
  source: openorca
  _comment: 'Text instruction following - 10K samples'

# Text datasets (7.5% of total - 60K samples)  
- json_path: /path/to/alpaca_40k.json
  sampling_strategy: random:5%
  name: alpaca-instructions
  path: alpaca
  modality: text
  source: alpaca
  _comment: 'Pure text instruction following - 40K samples'

- json_path: /path/to/sharegpt_20k.json
  sampling_strategy: random:2.5%
  name: sharegpt-conversations
  path: sharegpt
  modality: text
  source: sharegpt
  _comment: 'Conversational AI training - 20K samples'
```

### Dataset Download and Preparation

Create `scripts/download_datasets.py`:

```python
#!/usr/bin/env python3
"""Download and prepare datasets for SmolVLM2-256M training"""

import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm
import argparse

def download_and_sample_dataset(
    dataset_name: str,
    dataset_config: str,
    split: str,
    num_samples: int,
    output_path: str,
    modality: str = "image"
):
    """Download and sample a dataset to specified size"""
    
    print(f"Downloading {dataset_name} ({num_samples:,} samples)...")
    
    try:
        # Load dataset
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True)
        
        # Sample specified number of examples
        samples = []
        for i, item in enumerate(tqdm(dataset, desc=f"Sampling {dataset_name}")):
            if i >= num_samples:
                break
            
            # Convert to SmolVLM2 format
            if modality == "image":
                sample = {
                    "conversations": item.get("conversations", []),
                    "image": item.get("image", ""),
                    "id": item.get("id", f"{dataset_name}_{i}")
                }
            elif modality == "video":
                sample = {
                    "conversations": item.get("conversations", []),
                    "video": item.get("video", ""),
                    "id": item.get("id", f"{dataset_name}_{i}")
                }
            else:  # text
                sample = {
                    "conversations": item.get("conversations", []),
                    "id": item.get("id", f"{dataset_name}_{i}")
                }
            
            samples.append(sample)
        
        # Shuffle samples
        random.shuffle(samples)
        
        # Save to JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"Saved {len(samples):,} samples to {output_path}")
        return len(samples)
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Download datasets for SmolVLM2 training")
    parser.add_argument("--output_dir", default="data/datasets", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Dataset download configuration
    datasets_config = [
        # Image datasets
        {
            "name": "liuhaotian/LLaVA-Instruct-150K", 
            "config": None,
            "split": "train",
            "samples": 150000,
            "output": f"{args.output_dir}/llava_instruct_150k.json",
            "modality": "image"
        },
        {
            "name": "Lin-Chen/ShareGPT4V",
            "config": "ShareGPT4V", 
            "split": "train",
            "samples": 150000,
            "output": f"{args.output_dir}/sharegpt4v_150k.json",
            "modality": "image"
        },
        {
            "name": "lmms-lab/ai2d",
            "config": None,
            "split": "test", 
            "samples": 50000,
            "output": f"{args.output_dir}/ai2d_50k.json",
            "modality": "image"
        },
        {
            "name": "lmms-lab/ChartQA",
            "config": None,
            "split": "test",
            "samples": 40000, 
            "output": f"{args.output_dir}/chartqa_40k.json",
            "modality": "image"
        },
        {
            "name": "HuggingFaceM4/VQAv2",
            "config": None,
            "split": "train",
            "samples": 40000,
            "output": f"{args.output_dir}/vqav2_40k.json", 
            "modality": "image"
        },
        
        # Video datasets
        {
            "name": "lmms-lab/LLaVA-Video-178K",
            "config": None,
            "split": "train",
            "samples": 70000,
            "output": f"{args.output_dir}/llava_video_70k.json",
            "modality": "video"
        },
        {
            "name": "microsoft/VideoInstruct-100K",  
            "config": None,
            "split": "train",
            "samples": 20000,
            "output": f"{args.output_dir}/videoinstruct_20k.json",
            "modality": "video"
        },
        {
            "name": "Open-Orca/OpenOrca",
            "config": None,
            "split": "train", 
            "samples": 10000,
            "output": f"{args.output_dir}/openorca_10k.json",
            "modality": "text"
        },
        
        # Text datasets
        {
            "name": "tatsu-lab/alpaca",
            "config": None,
            "split": "train",
            "samples": 40000,
            "output": f"{args.output_dir}/alpaca_40k.json",
            "modality": "text"
        },
        {
            "name": "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "config": None,
            "split": "train",
            "samples": 20000, 
            "output": f"{args.output_dir}/sharegpt_20k.json",
            "modality": "text"
        }
    ]
    
    total_samples = 0
    successful_downloads = 0
    
    for config in datasets_config:
        samples = download_and_sample_dataset(
            dataset_name=config["name"],
            dataset_config=config["config"], 
            split=config["split"],
            num_samples=config["samples"],
            output_path=config["output"],
            modality=config["modality"]
        )
        
        if samples > 0:
            total_samples += samples
            successful_downloads += 1
    
    print(f"\n=== Dataset Download Summary ===")
    print(f"Successfully downloaded: {successful_downloads}/{len(datasets_config)} datasets")
    print(f"Total samples: {total_samples:,}")
    print(f"Target for 256M model: 800K samples")
    
    if total_samples >= 700000:  # Allow some tolerance
        print("✅ Sufficient data for 256M model training")
    else:
        print("⚠️  May need additional data for optimal training")

if __name__ == "__main__":
    main()
```

### Usage Instructions

```bash
# Download datasets
python scripts/download_datasets.py --output_dir data/datasets --seed 42

# Update mixture config with actual paths
# Edit scripts/mixtures/smolvlm2_256m_mixture.yaml to use correct paths

# Use in training
python scripts/train_smolvlm2_infini.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Instruct \
    --data_mixture scripts/mixtures/smolvlm2_256m_mixture.yaml \
    --output_dir checkpoints/smolvlm2_infini \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --bf16 \
    --tune_language_model \
    --tune_mm_connector \
    --gradient_checkpointing
```

### Dataset Format

All datasets follow the SmolVLM2 conversational format:

**Image samples:**
```json
{
  "conversations": [
    {
      "from": "human", 
      "value": "<image>\nWhat do you see in this image?"
    },
    {
      "from": "gpt",
      "value": "I can see a landscape with mountains and trees."
    }
  ],
  "image": "path/to/image.jpg",
  "id": "sample_001"
}
```

**Video samples:**
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<video>\nDescribe what happens in this video."
    },
    {
      "from": "gpt", 
      "value": "The video shows a person walking through a park."
    }
  ],
  "video": "path/to/video.mp4",
  "id": "video_001"
}
```

**Text samples:**
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Explain quantum computing in simple terms."
    },
    {
      "from": "gpt",
      "value": "Quantum computing uses quantum mechanics principles..."
    }
  ],
  "id": "text_001"
}
```

### Adapt for Nanotron

Create `scripts/convert_smolvlm2_data.py`:

```python
import json
import torch
from datasets import Dataset
from transformers import AutoProcessor

def convert_smolvlm2_to_nanotron(input_file: str, output_file: str):
    """Convert SmolVLM2 dataset format to Nanotron format"""
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM2-256M-Instruct",
        trust_remote_code=True
    )
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    converted_data = []
    for item in data:
        # Format conversation
        text = ""
        for turn in item["conversations"]:
            if turn["from"] == "human":
                text += f"<|im_start|>user\n{turn['value']}<|im_end|>\n"
            elif turn["from"] == "gpt":
                text += f"<|im_start|>assistant\n{turn['value']}<|im_end|>\n"
        
        # Tokenize
        tokens = processor.tokenizer(
            text,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        converted_item = {
            "input_ids": tokens["input_ids"].squeeze().tolist(),
            "text": text,
            "image_path": item.get("image", "")
        }
        converted_data.append(converted_item)
    
    # Save in Nanotron format
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)

if __name__ == "__main__":
    convert_smolvlm2_to_nanotron("train_data.json", "train_nanotron.json")
```

## 4. Training Configuration

Create `configs/smolvlm2_training.yaml`:

```yaml
checkpoints:
  checkpoints_path: checkpoints/smolvlm2_infini
  checkpoint_interval: 1000
  resume_checkpoint_path: null

data_stages:
  - name: "multimodal_training"
    start_training_step: 1
    data:
      dataset:
        dataset_overwrite_cache: false
        dataset_processing_num_proc_per_process: 1
        hf_dataset_or_datasets: "train_nanotron.json"
        hf_dataset_splits: "train"
        text_column_name: "text"
      num_loading_workers: 4
      seed: 42

general:
  project: "smolvlm2_infini_attention"
  run: "smolvlm2_training"
  seed: 42

logging:
  iteration_step_info_interval: 10
  log_level: "info"

model:
  dtype: bfloat16
  make_vocab_size_divisible_by: 1
  model_config:
    # SmolVLM2 config
    vocab_size: 49152
    hidden_size: 512
    intermediate_size: 1536
    num_hidden_layers: 12
    num_attention_heads: 8
    num_key_value_heads: 4
    max_position_embeddings: 16384
    rope_theta: 10000.0
    rms_norm_eps: 1e-5
    
    # Infini-attention config
    use_infini_attention: true
    segment_length: 512
    
    # Vision config
    image_token_id: 32000

optimizer:
  accumulate_grad_in_fp32: true
  clip_grad: 1.0
  learning_rate_scheduler:
    learning_rate: 3.0e-4
    lr_decay_steps: 10000
    lr_decay_style: cosine
    lr_warmup_steps: 1000
    lr_warmup_style: linear
    min_decay_lr: 1.0e-5
  optimizer_factory:
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    name: adamW
    weight_decay: 0.01

parallelism:
  dp: 1
  pp: 1
  tp: 1

tokenizer:
  tokenizer_name_or_path: "HuggingFaceTB/SmolVLM2-256M-Instruct"

tokens:
  batch_accumulation_per_replica: 8
  micro_batch_size: 2
  sequence_length: 2048
  train_steps: 10000
  val_check_interval: 500
```

## 5. Training Process

### Hybrid Approach: Combine Original SmolVLM2 with Nanotron Infini-Attention

Since the original SmolVLM2 has well-tested multimodal training, we can modify it to use Nanotron's infini-attention:

1. **Start with Original SmolVLM2**: Use the existing training infrastructure
2. **Replace Attention Layers**: Swap standard attention with infini-attention
3. **Maintain Compatibility**: Keep the same data loading and training loop

### Modified Training Script

Create `scripts/train_smolvlm2_infini.py`:

```python
import sys
import os
sys.path.append('path/to/nanotron')

from smolvlm.train.train import *
from nanotron.models.llama import LlamaDecoderLayer

def replace_attention_with_infini(model):
    """Replace standard attention layers with infini-attention"""
    
    # Access the language model layers
    if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
        text_model = model.model.text_model
        
        # Replace each decoder layer
        for i, layer in enumerate(text_model.layers):
            # Create infini-attention layer with same config
            infini_layer = LlamaDecoderLayer(
                config=text_model.config,
                parallel_context=None,  # Single GPU for now
                layer_idx=i
            )
            
            # Copy weights from original layer
            with torch.no_grad():
                # Copy attention weights
                infini_layer.self_attn.q_proj.weight.copy_(layer.self_attn.q_proj.weight)
                infini_layer.self_attn.k_proj.weight.copy_(layer.self_attn.k_proj.weight)
                infini_layer.self_attn.v_proj.weight.copy_(layer.self_attn.v_proj.weight)
                infini_layer.self_attn.o_proj.weight.copy_(layer.self_attn.o_proj.weight)
                
                # Copy MLP weights
                infini_layer.mlp.gate_proj.weight.copy_(layer.mlp.gate_proj.weight)
                infini_layer.mlp.up_proj.weight.copy_(layer.mlp.up_proj.weight)
                infini_layer.mlp.down_proj.weight.copy_(layer.mlp.down_proj.weight)
                
                # Copy layer norms
                infini_layer.input_layernorm.weight.copy_(layer.input_layernorm.weight)
                infini_layer.post_attention_layernorm.weight.copy_(layer.post_attention_layernorm.weight)
            
            # Replace the layer
            text_model.layers[i] = infini_layer
    
    return model

def train_with_infini_attention():
    """Modified training function with infini-attention"""
    
    # Use original argument parsing
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load original model
    model = prepare_model(model_args, training_args)
    
    # Replace attention layers with infini-attention
    model = replace_attention_with_infini(model)
    
    # Continue with original training pipeline
    set_trainable_params(model, training_args)
    
    if training_args.gradient_checkpointing:
        enable_gradient_checkpointing(model, training_args)
    
    # Load processor and data
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    data_module = make_supervised_data_module(processor, data_args, training_args, model_args)
    
    # Initialize trainer
    trainer = SmolVLMTrainer(
        model=model,
        args=training_args,
        **data_module
    )
    
    # Train
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    train_with_infini_attention()
```

### Launch Training

```bash
# Basic training
python scripts/train_smolvlm2_infini.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Instruct \
    --data_mixture scripts/mixtures/smolvlm2_256m_mixture.yaml \
    --output_dir checkpoints/smolvlm2_infini \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --bf16 \
    --tune_language_model \
    --tune_mm_connector \
    --gradient_checkpointing \
    --report_to wandb

# Multi-GPU training
torchrun --nproc_per_node=2 scripts/train_smolvlm2_infini.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Instruct \
    --data_mixture scripts/mixtures/smolvlm2_256m_mixture.yaml \
    --output_dir checkpoints/smolvlm2_infini \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --bf16 \
    --tune_language_model \
    --tune_mm_connector \
    --gradient_checkpointing \
    --report_to wandb
```

## 6. Evaluation

### Basic Inference Test

```python
import torch
from transformers import AutoProcessor
from smolvlm.model.modeling_smolvlm import SmolVLMForConditionalGeneration

# Load trained model
model = SmolVLMForConditionalGeneration.from_pretrained(
    "checkpoints/smolvlm2_infini"
)
processor = AutoProcessor.from_pretrained(
    "checkpoints/smolvlm2_infini"
)

# Test with image
from PIL import Image
image = Image.open("test_image.jpg")

# Prepare inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What do you see in this image?"}
        ]
    }
]

inputs = processor.apply_chat_template(messages, return_tensors="pt")

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Extended Context Test

```python
# Test infini-attention with long sequences
def test_long_context():
    # Create a long sequence (longer than segment_length=512)
    long_text = "Tell me about " + "the history of artificial intelligence. " * 100
    
    inputs = processor(
        text=long_text,
        return_tensors="pt",
        max_length=2048,
        truncation=True
    )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    print(f"Long context response: {response}")

test_long_context()
```

### Benchmarking

Use the original SmolVLM2 evaluation scripts with your trained model:

```bash
# Install VLMEvalKit
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .

# Run evaluation
python run.py --data MMBench_DEV_EN --model SmolVLM2_Infini --work-dir results/
```

## Summary

This tutorial shows how to integrate SmolVLM2 with Nanotron's infini-attention mechanism:

1. **Leverage Existing Infrastructure**: Use the well-tested SmolVLM2 training pipeline
2. **Selective Integration**: Replace only the attention layers with infini-attention
3. **Maintain Compatibility**: Keep the same data format and training procedures
4. **Extended Context**: Enable processing of longer sequences through segmentation

The hybrid approach allows you to benefit from both the mature multimodal training of SmolVLM2 and the extended context capabilities of infini-attention, providing a practical path for training multimodal models with infinite attention.
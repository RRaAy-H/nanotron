#!/usr/bin/env python3
"""
Distributed SmolVLM2-Infini training script using Nanotron's infrastructure.

This script properly integrates with Nanotron's 3D parallelism (DP/TP/PP) architecture
for scalable multi-GPU training of SmolVLM2 models with Infini-Attention.

Usage:
    # Single node, multi-GPU
    torchrun --nproc_per_node=8 train_smolvlm2_infini_distributed.py --config-file configs/smolvlm2_training_distributed.yaml
    
    # Multi-node
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_smolvlm2_infini_distributed.py --config-file configs/smolvlm2_training_distributed.yaml
    
    # With specific GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_smolvlm2_infini_distributed.py --config-file configs/smolvlm2_training_distributed.yaml
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Optional, cast
import torch
import json
from PIL import Image
import cv2
import logging

# Add nanotron to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from nanotron import logging as nanotron_logging
from nanotron.trainer import DistributedTrainer
from nanotron.config import (
    Config,
    DataArgs,
    DatasetStageArgs,
    PretrainDatasetsArgs,
)
from nanotron.dataloader import (
    clm_process,
    dummy_infinite_data_generator,
    get_datasets,
    get_train_dataloader,
)
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.utils import main_rank_first
from nanotron.models.smolvlm2_nanotron import SmolVLM2NanotronModel
from nanotron.parallel.context import ParallelContext

from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoProcessor, AutoTokenizer
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import __version__ as tf_version
except ImportError:
    AutoProcessor = None
    AutoTokenizer = None
    hf_hub_version = None
    tf_version = None

logger = nanotron_logging.get_logger(__name__)

class SmolVLM2VisionLanguageDataset(Dataset):
    """Dataset for SmolVLM2 vision-language training compatible with nanotron"""
    
    def __init__(self, data_path: str, image_dir: str, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        self.image_dir = image_dir
        
        # Load data
        if os.path.isfile(data_path):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        elif os.path.isdir(data_path):
            # Load from multiple files in directory
            self.data = []
            for filename in sorted(os.listdir(data_path)):
                if filename.endswith('.json'):
                    file_path = os.path.join(data_path, filename)
                    with open(file_path, 'r') as f:
                        file_data = json.load(f)
                        self.data.extend(file_data)
                        log_rank(f"Loaded {len(file_data)} samples from {filename}", logger=logger, level=logging.INFO, rank=0)
        else:
            raise FileNotFoundError(f"Data path {data_path} not found")
            
        log_rank(f"Total dataset size: {len(self.data)} samples", logger=logger, level=logging.INFO, rank=0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # Load image (video or static image)
            if 'video' in item:
                video_path = os.path.join(self.image_dir, item['video'])
                image = self._extract_frame_from_video(video_path)
            else:
                image_data = item['image']
                image = self._load_image(image_data)
            
            # Get text
            if 'conversations' in item:
                text = self._format_conversations(item['conversations'])
            else:
                text = item.get('text', 'Sample text')
            
            # Process inputs
            if self.processor is not None:
                inputs = self.processor(
                    images=image,
                    text=text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length"
                )
                
                # Remove batch dimension for consistency with nanotron
                for key in inputs:
                    if inputs[key] is not None and inputs[key].dim() > 1:
                        inputs[key] = inputs[key].squeeze(0)
                
                return inputs
            else:
                # Fallback: return tokenized text only
                return {"input_ids": torch.randint(0, 1000, (self.max_length,))}
                
        except Exception as e:
            log_rank(f"Error loading sample {idx}: {e}", logger=logger, level=logging.WARNING, rank=0)
            # Return dummy sample
            if self.processor is not None:
                dummy_image = Image.new('RGB', (224, 224), color='white')
                dummy_text = "Error loading sample"
                inputs = self.processor(
                    images=dummy_image,
                    text=dummy_text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length"
                )
                for key in inputs:
                    if inputs[key] is not None and inputs[key].dim() > 1:
                        inputs[key] = inputs[key].squeeze(0)
                return inputs
            else:
                return {"input_ids": torch.randint(0, 1000, (self.max_length,))}
    
    def _extract_frame_from_video(self, video_path: str) -> Image.Image:
        """Extract first frame from video"""
        if not os.path.exists(video_path):
            log_rank(f"Video not found: {video_path}", logger=logger, level=logging.WARNING, rank=0)
            return Image.new('RGB', (224, 224), color='gray')
        
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                return image.resize((224, 224), Image.Resampling.LANCZOS)
            else:
                return Image.new('RGB', (224, 224), color='gray')
        except Exception as e:
            log_rank(f"Error extracting frame from {video_path}: {e}", logger=logger, level=logging.WARNING, rank=0)
            return Image.new('RGB', (224, 224), color='gray')
    
    def _load_image(self, image_data) -> Image.Image:
        """Load image from various formats"""
        try:
            if isinstance(image_data, dict) and "bytes" in image_data:
                # Base64 encoded image
                import base64
                import io
                bytes_data = image_data["bytes"]
                if isinstance(bytes_data, dict) and "data" in bytes_data:
                    base64_data = bytes_data["data"]
                    image_bytes = base64.b64decode(base64_data)
                    return Image.open(io.BytesIO(image_bytes)).convert('RGB')
            elif isinstance(image_data, str):
                # File path
                image_path = os.path.join(self.image_dir, image_data)
                if os.path.exists(image_path):
                    return Image.open(image_path).convert('RGB')
        except Exception as e:
            log_rank(f"Error loading image: {e}", logger=logger, level=logging.WARNING, rank=0)
        
        return Image.new('RGB', (224, 224), color='white')
    
    def _format_conversations(self, conversations):
        """Format conversations into text"""
        text_parts = []
        for conv in conversations:
            if conv['from'] == 'human':
                text_parts.append(f"User: {conv['value']}")
            elif conv['from'] == 'gpt':
                text_parts.append(f"Assistant: {conv['value']}")
        return "\n".join(text_parts)


class SmolVLM2DistributedTrainer(DistributedTrainer):
    """Extended DistributedTrainer for SmolVLM2-Infini training"""
    
    def __init__(self, config_file: str):
        # Initialize processor before parent init if needed
        self.processor = None
        if AutoProcessor is not None:
            try:
                # Try to load processor from config if available
                config = Config.from_yaml(config_file)
                if hasattr(config, 'tokenizer') and hasattr(config.tokenizer, 'tokenizer_name_or_path'):
                    processor_path = config.tokenizer.tokenizer_name_or_path
                    self.processor = AutoProcessor.from_pretrained(
                        processor_path,
                        trust_remote_code=True
                    )
                    log_rank(f"Loaded processor from {processor_path}", logger=logger, level=logging.INFO, rank=0)
            except Exception as e:
                log_rank(f"Failed to load processor: {e}", logger=logger, level=logging.WARNING, rank=0)
        
        super().__init__(config_file)
    
    def init_model(self):
        """Initialize SmolVLM2 model with infini-attention"""
        # Use nanotron's model building infrastructure
        from nanotron.models import build_model
        
        # Build model using nanotron's infrastructure
        model = build_model(
            model_builder=lambda: SmolVLM2NanotronModel(
                config=self.model_config,
                parallel_context=self.parallel_context,
                parallel_config=self.config.parallelism,
            ),
            parallel_context=self.parallel_context,
            dtype=self.config.model.dtype,
            device=torch.device("cuda"),
        )
        
        log_rank("SmolVLM2-Infini model initialized with nanotron infrastructure", 
                logger=logger, level=logging.INFO, rank=0)
        
        return self._load_model_checkpoint(model)


def get_smolvlm2_dataloader_from_data_stage(trainer: SmolVLM2DistributedTrainer, data: DataArgs):
    """Returns a dataloader for SmolVLM2 training."""
    
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)
    
    # Check if we have SmolVLM2 specific data configuration
    if hasattr(data, 'smolvlm2_data_path'):
        log_rank("Using SmolVLM2 vision-language dataset", logger=logger, level=logging.INFO, rank=0)
        
        # Load SmolVLM2 dataset
        with main_rank_first(trainer.parallel_context.world_pg):
            dataset = SmolVLM2VisionLanguageDataset(
                data_path=data.smolvlm2_data_path,
                image_dir=getattr(data, 'smolvlm2_image_dir', './images'),
                processor=trainer.processor,
                max_length=trainer.sequence_length
            )
            
            # Create distributed sampler
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=trainer.parallel_context.dp_pg.size(),
                rank=trainer.parallel_context.dp_pg.rank(),
                shuffle=True,
                seed=data.seed
            )
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=trainer.micro_batch_size,
                sampler=sampler,
                num_workers=getattr(data, 'num_loading_workers', 4),
                pin_memory=True,
                drop_last=True
            )
            
    elif data.dataset is None:
        # Dummy data generator fallback
        log_rank("Using dummy data generator", logger=logger, level=logging.INFO, rank=0)
        dataloader = dummy_infinite_data_generator(
            micro_batch_size=trainer.micro_batch_size,
            sequence_length=trainer.sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            vocab_size=trainer.model_config.vocab_size,
            seed=data.seed,
            parallel_context=trainer.parallel_context,
        )()
        
    elif isinstance(data.dataset, PretrainDatasetsArgs):
        # Standard HuggingFace datasets
        log_rank("Using standard HuggingFace datasets", logger=logger, level=logging.INFO, rank=0)
        
        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
        log_rank(
            f"Loading tokenizer from {tokenizer_path}",
            logger=logger, level=logging.INFO, rank=0
        )

        with main_rank_first(trainer.parallel_context.world_pg):
            raw_dataset = get_datasets(
                hf_dataset_or_datasets=data.dataset.hf_dataset_or_datasets,
                hf_dataset_config_name=data.dataset.hf_dataset_config_name,
                splits=data.dataset.hf_dataset_splits,
            )["train"]

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            train_dataset = clm_process(
                raw_dataset=raw_dataset,
                tokenizer=tokenizer,
                text_column_name=data.dataset.text_column_name,
                dataset_processing_num_proc_per_process=data.dataset.dataset_processing_num_proc_per_process,
                dataset_overwrite_cache=data.dataset.dataset_overwrite_cache,
                sequence_length=trainer.sequence_length,
            )

            dataloader = get_train_dataloader(
                train_dataset=train_dataset,
                sequence_length=trainer.sequence_length,
                parallel_context=trainer.parallel_context,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=trainer.consumed_train_samples,
                dataloader_num_workers=data.num_loading_workers,
                seed_worker=data.seed,
                dataloader_drop_last=True,
            )
    else:
        raise ValueError(f"Unhandled case of `data.dataset`. Got: {data.dataset}")
    
    return dataloader


def get_smolvlm2_dataloader(trainer: SmolVLM2DistributedTrainer) -> Dict[str, DataLoader]:
    """Get dataloaders for all training stages"""
    sorted_stages = sorted(trainer.config.data_stages, key=lambda stage: stage.start_training_step)
    dataloaders = {}
    
    for idx, stage in enumerate(sorted_stages):
        stage = cast(DatasetStageArgs, stage)
        dataloader = (
            get_smolvlm2_dataloader_from_data_stage(trainer, stage.data)
            if idx == 0
            else lambda stage=stage: get_smolvlm2_dataloader_from_data_stage(trainer, stage.data)
        )
        dataloaders[stage.name] = dataloader
    
    return dataloaders


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SmolVLM2-Infini Distributed Training")
    parser.add_argument(
        "--config-file", 
        type=str, 
        required=True,
        help="Path to the YAML config file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Override data path from config"
    )
    parser.add_argument(
        "--image-dir", 
        type=str,
        help="Override image directory from config"
    )
    return parser.parse_args()


def main():
    """Main distributed training function"""
    args = get_args()
    
    # Verify torchrun environment
    required_env_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    missing_vars = [var for var in required_env_vars if var not in os.environ]
    if missing_vars:
        raise RuntimeError(
            f"Missing environment variables: {missing_vars}. "
            "This script should be launched with torchrun."
        )
    
    log_rank(
        f"Starting SmolVLM2-Infini distributed training on rank {os.environ['RANK']}/{os.environ['WORLD_SIZE']}",
        logger=logger, level=logging.INFO, rank=0
    )
    
    # Initialize trainer
    trainer = SmolVLM2DistributedTrainer(args.config_file)
    
    # Setup dataloaders
    dataloaders = get_smolvlm2_dataloader(trainer)
    
    log_rank(
        f"Training configuration:\n"
        f"  - Data Parallel: {trainer.config.parallelism.dp}\n"
        f"  - Tensor Parallel: {trainer.config.parallelism.tp}\n"
        f"  - Pipeline Parallel: {trainer.config.parallelism.pp}\n"
        f"  - Expert Parallel: {trainer.config.parallelism.expert_parallel_size}\n"
        f"  - Global batch size: {trainer.global_batch_size}\n"
        f"  - Micro batch size: {trainer.micro_batch_size}\n"
        f"  - Sequence length: {trainer.sequence_length}",
        logger=logger, level=logging.INFO, rank=0
    )
    
    # Start training
    try:
        trainer.train(dataloaders)
        log_rank("Training completed successfully!", logger=logger, level=logging.INFO, rank=0)
    except KeyboardInterrupt:
        log_rank("Training interrupted by user", logger=logger, level=logging.INFO, rank=0)
    except Exception as e:
        log_rank(f"Training failed with error: {e}", logger=logger, level=logging.ERROR, rank=0)
        raise
    finally:
        # Cleanup
        if hasattr(trainer, 'parallel_context'):
            trainer.parallel_context.destroy()


if __name__ == "__main__":
    main()
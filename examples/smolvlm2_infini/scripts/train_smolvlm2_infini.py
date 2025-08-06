#!/usr/bin/env python3
"""Train SmolVLM2 with Nanotron's Infini-Attention"""

import sys
import os
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoProcessor,
    get_scheduler
)
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import yaml
import json
from PIL import Image
import logging
import cv2
import torch.distributed as dist
import signal
import glob
import shutil
import numpy as np
import random
from pathlib import Path
from datetime import datetime
import hashlib
import threading
import time
from collections import defaultdict
from typing import Any, Union
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Add nanotron to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from nanotron.models.smolvlm2_nanotron import SmolVLM2NanotronModel
from nanotron.models.llama import LlamaDecoderLayer
from nanotron.parallel.context import ParallelContext
from nanotron.serialize import save, load
from nanotron.serialize.metadata import CheckpointMetadata
from nanotron.random import RandomStates, get_current_random_states, set_random_states

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"

# Initialize distributed training if not already initialized
def initialize_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', init_method='env://')

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

@dataclass
class TensorBoardArguments:
    """Arguments for TensorBoard logging."""
    use_tensorboard: bool = field(
        default=False,
        metadata={"help": "Enable TensorBoard logging"}
    )
    tensorboard_dir: str = field(
        default="./tensorboard_logs",
        metadata={"help": "Directory for TensorBoard logs"}
    )
    tensorboard_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "TensorBoard run name (default: auto-generated)"}
    )

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Enable trusting remote code"}
    )
    use_infini_attention: bool = field(
        default=True,
        metadata={"help": "Use infini-attention layers"}
    )
    segment_length: int = field(
        default=512,
        metadata={"help": "Segment length for infini-attention"}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint folder to resume training from"}
    )
    max_checkpoints_to_keep: int = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep (older ones will be deleted)"}
    )
    use_nanotron_checkpointing: bool = field(
        default=False,
        metadata={"help": "Use Nanotron's native serialization system for distributed training"}
    )
    enable_checkpoint_validation: bool = field(
        default=True,
        metadata={"help": "Enable checkpoint integrity validation"}
    )
    incremental_checkpoint_interval: int = field(
        default=10,
        metadata={"help": "Save full checkpoints every N saves, incremental otherwise"}
    )

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    data_mixture: str = field(
        metadata={"help": "Path to data mixture YAML file"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    image_dir: str = field(
        default="./images",
        metadata={"help": "Directory containing training images"}
    )
    train_data_path: str = field(
        default="./train_data.json",
        metadata={"help": "Path to training data JSON file"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to evaluation data JSON file"}
    )

def replace_attention_with_infini(model, segment_length=512):
    """Replace standard attention layers with infini-attention"""
    
    # Access the language model layers
    if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
        text_model = model.model.text_model
        
        # Replace each decoder layer
        for i, layer in enumerate(text_model.layers):
            # Create infini-attention layer with same config
            config = text_model.config
            config.segment_length = segment_length
            config.use_infini_attention = True
            
            infini_layer = LlamaDecoderLayer(
                config=config,
                parallel_context=None,  # Single GPU for now
                layer_idx=i
            )
            
            # Copy weights from original layer
            with torch.no_grad():
                # Copy attention weights
                if hasattr(layer.self_attn, 'q_proj'):
                    infini_layer.self_attn.q_proj.weight.copy_(layer.self_attn.q_proj.weight)
                    infini_layer.self_attn.k_proj.weight.copy_(layer.self_attn.k_proj.weight)
                    infini_layer.self_attn.v_proj.weight.copy_(layer.self_attn.v_proj.weight)
                    infini_layer.self_attn.o_proj.weight.copy_(layer.self_attn.o_proj.weight)
                
                # Copy MLP weights
                if hasattr(layer, 'mlp'):
                    infini_layer.mlp.gate_proj.weight.copy_(layer.mlp.gate_proj.weight)
                    infini_layer.mlp.up_proj.weight.copy_(layer.mlp.up_proj.weight)
                    infini_layer.mlp.down_proj.weight.copy_(layer.mlp.down_proj.weight)
                
                # Copy layer norms
                if hasattr(layer, 'input_layernorm'):
                    infini_layer.input_layernorm.weight.copy_(layer.input_layernorm.weight)
                if hasattr(layer, 'post_attention_layernorm'):
                    infini_layer.post_attention_layernorm.weight.copy_(layer.post_attention_layernorm.weight)
            
            # Replace the layer
            text_model.layers[i] = infini_layer
            
    logger.info(f"Replaced attention layers with infini-attention (segment_length={segment_length})")
    return model

class VisionLanguageDataset(Dataset):
    """Dataset for vision-language training"""
    
    def __init__(self, data_path: str, image_dir: str, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        self.image_dir = image_dir
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # Load image (from video frame if video field exists, otherwise from image field)
            if 'video' in item:
                video_path = os.path.join(self.image_dir, item['video'])
                # Extract first frame from video with improved error handling
                if os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    
                    # Set video codec and options to avoid scaling issues
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret and frame is not None:
                        # Check if frame has valid dimensions
                        if frame.shape[0] > 0 and frame.shape[1] > 0:
                            # Convert BGR to RGB and create PIL Image
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image = Image.fromarray(frame_rgb)
                            
                            # Resize to standard dimensions to avoid scaling issues
                            image = image.resize((224, 224), Image.Resampling.LANCZOS)
                        else:
                            # Invalid frame dimensions, use placeholder
                            logger.warning(f"Invalid frame dimensions for {video_path}, using placeholder")
                            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                    else:
                        # Fallback to a blank image if video reading fails
                        logger.warning(f"Failed to read frame from {video_path}, using placeholder")
                        image = Image.new('RGB', (224, 224), color='black')
                else:
                    # Video file doesn't exist, create a placeholder image
                    logger.warning(f"Video file not found: {video_path}, using placeholder image")
                    image = Image.new('RGB', (224, 224), color=(128, 128, 128))  # Gray placeholder
            else:
                # Handle both file paths and embedded base64 image data
                image_data = item['image']
                
                if isinstance(image_data, dict) and "bytes" in image_data:
                    # Handle embedded base64 image data
                    import base64
                    import io
                    
                    bytes_data = image_data["bytes"]
                    if isinstance(bytes_data, dict) and "data" in bytes_data:
                        base64_data = bytes_data["data"]
                        # Decode base64 data
                        image_bytes = base64.b64decode(base64_data)
                        # Create PIL Image from bytes
                        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    else:
                        logger.warning(f"Invalid embedded image data structure: {bytes_data}")
                        image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                        
                elif isinstance(image_data, str):
                    # Handle file path - existing logic
                    image_path = os.path.join(self.image_dir, image_data)
                    image = Image.open(image_path).convert('RGB')
                    
                else:
                    logger.warning(f"Unsupported image format: {type(image_data)}")
                    image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            
            # Get text (from conversations if available, otherwise from text field)
            if 'conversations' in item:
                # Convert conversations to text format
                text_parts = []
                for conv in item['conversations']:
                    if conv['from'] == 'human':
                        text_parts.append(f"User: {conv['value']}")
                    elif conv['from'] == 'gpt':
                        text_parts.append(f"Assistant: {conv['value']}")
                text = "\n".join(text_parts)
            else:
                text = item['text']
            
            # Process inputs
            inputs = self.processor(
                images=image,
                text=text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )
            
            # Remove batch dimension
            for key in inputs:
                if inputs[key] is not None:
                    inputs[key] = inputs[key].squeeze(0)
            
            return inputs
            
        except KeyError as e:
            if str(e) == "'image'":
                # Handle missing 'image' key - assume text-only sample
                # Reduce logging noise - only log every 1000th text-only sample
                if idx % 1000 == 0:
                    logger.info(f"Processing text-only samples (e.g., sample {idx})")
                elif idx < 10:  # Log first few for debugging
                    logger.warning(f"Sample {idx} appears to be text-only (missing 'image' key)")
                
                # Get text from the item
                if 'conversations' in item:
                    text_parts = []
                    for conv in item['conversations']:
                        if conv['from'] == 'human':
                            text_parts.append(f"User: {conv['value']}")
                        elif conv['from'] == 'gpt':
                            text_parts.append(f"Assistant: {conv['value']}")
                    text = "\n".join(text_parts)
                else:
                    text = item.get('text', 'Sample without text')
                
                # Remove all image placeholders from text-only samples since we don't have actual images
                import re
                text = re.sub(r'<image>', '', text).strip()
                
                # Clean up any extra whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                if not text:
                    text = "Text-only sample"
                
                # Process as text-only (no images)
                inputs = self.processor(
                    text=text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length"
                )
                
                # Remove batch dimension
                for key in inputs:
                    if inputs[key] is not None:
                        inputs[key] = inputs[key].squeeze(0)
                
                return inputs
            else:
                raise
        except Exception as e:
            logger.warning(f"Error loading sample {idx}: {e}")
            # Return a dummy sample in case of error
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
                if inputs[key] is not None:
                    inputs[key] = inputs[key].squeeze(0)
                    
            return inputs

class SmolVLM2InfiniTrainer:
    """Custom trainer for SmolVLM2 with Infini-Attention"""
    
    def __init__(
        self,
        model,
        args: TrainingArguments,
        train_dataset,
        eval_dataset=None,
        processor=None,
        tensorboard_args=None,
        model_args=None,
        data_args=None,
        **kwargs
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.tensorboard_args = tensorboard_args
        self.model_args = model_args
        self.data_args = data_args
        
        # Initialize TensorBoard if enabled
        self.setup_tensorboard()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Initialize training metrics tracking
        self.training_metrics = {
            'train_loss_history': [],
            'eval_loss_history': [],
            'learning_rates': [],
            'gradient_norms': [],
            'step_times': [],
            'checkpoint_history': []
        }
        
        # For incremental checkpointing
        self.previous_model_state = None
        self.checkpoint_validation_enabled = getattr(model_args, 'enable_checkpoint_validation', True)
        self.use_nanotron_checkpointing = getattr(model_args, 'use_nanotron_checkpointing', False)
        self.incremental_checkpoint_interval = getattr(model_args, 'incremental_checkpoint_interval', 10)
        
        # Initialize distributed coordination
        self.is_distributed = dist.is_initialized()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_main_process = (not self.is_distributed) or (dist.get_rank() == 0)
    
    def _setup_signal_handlers(self):
        """Setup handlers for graceful interruption"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
            self.interrupted = True
        
        # Register handlers for common interruption signals
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        logger.info("Signal handlers registered for graceful interruption")
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _validate_checkpoint_integrity(self, checkpoint_dir: str) -> bool:
        """Validate checkpoint integrity using checksums and test loading"""
        if not self.checkpoint_validation_enabled:
            return True
            
        try:
            # Check required files exist
            required_files = ["training_state.pt", "pytorch_model.bin"]
            for file_name in required_files:
                file_path = os.path.join(checkpoint_dir, file_name)
                if not os.path.exists(file_path):
                    logger.error(f"Missing checkpoint file: {file_name}")
                    return False
            
            # Validate training_state.pt by attempting to load it
            checkpoint_path = os.path.join(checkpoint_dir, "training_state.pt")
            try:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                
                # Check required keys
                required_keys = ['model_state_dict', 'optimizer_state_dict', 
                               'scheduler_state_dict', 'global_step']
                for key in required_keys:
                    if key not in checkpoint_data:
                        logger.error(f"Missing key in checkpoint: {key}")
                        return False
                        
                # Validate global_step is reasonable
                global_step = checkpoint_data.get('global_step', -1)
                if global_step < 0:
                    logger.error(f"Invalid global_step: {global_step}")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to load checkpoint for validation: {e}")
                return False
            
            # Calculate and store checksums
            checksums = {}
            for file_name in required_files:
                file_path = os.path.join(checkpoint_dir, file_name)
                checksums[file_name] = self._calculate_file_checksum(file_path)
            
            # Save validation metadata
            validation_data = {
                'validation_timestamp': datetime.now().isoformat(),
                'file_checksums': checksums,
                'file_sizes': {
                    file_name: os.path.getsize(os.path.join(checkpoint_dir, file_name))
                    for file_name in required_files
                },
                'validation_passed': True,
                'global_step': global_step
            }
            
            validation_path = os.path.join(checkpoint_dir, "checkpoint_validation.json")
            with open(validation_path, "w") as f:
                json.dump(validation_data, f, indent=2)
                
            logger.info(f"Checkpoint validation passed: {checkpoint_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False
    
    def _save_training_metrics(self, output_dir: str, global_step: int, 
                              step_loss: float = None, eval_loss: float = None):
        """Save comprehensive training metrics with the checkpoint"""
        metrics_data = {
            'global_step': global_step,
            'timestamp': datetime.now().isoformat(),
            'training_metrics': dict(self.training_metrics),
            'current_step_loss': step_loss,
            'current_eval_loss': eval_loss,
            'model_info': {
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
        
        # Add optimizer state info
        if hasattr(self.optimizer, 'state_dict'):
            optimizer_info = {}
            state_dict = self.optimizer.state_dict()
            if 'param_groups' in state_dict:
                optimizer_info['learning_rates'] = [group.get('lr', 0.0) for group in state_dict['param_groups']]
                optimizer_info['weight_decays'] = [group.get('weight_decay', 0.0) for group in state_dict['param_groups']]
            metrics_data['optimizer_info'] = optimizer_info
        
        # Add scheduler info
        if hasattr(self.scheduler, 'get_last_lr'):
            metrics_data['scheduler_info'] = {
                'last_lr': self.scheduler.get_last_lr(),
                'state_dict': self.scheduler.state_dict()
            }
        
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.debug(f"Training metrics saved to {metrics_path}")
    
    def _coordinate_distributed_checkpoint(self, operation: str):
        """Coordinate checkpoint operations across distributed processes"""
        if not self.is_distributed:
            return True
            
        try:
            # Only rank 0 performs actual checkpoint operations
            if operation == "save":
                if self.is_main_process:
                    # Main process performs save
                    result = True
                else:
                    # Other processes wait
                    result = False
                
                # Synchronize all processes
                dist.barrier()
                return result
                
            elif operation == "load":
                # All processes can load, but coordinate
                dist.barrier()  # Ensure checkpoint is fully saved
                return True
                
        except Exception as e:
            logger.error(f"Distributed coordination failed: {e}")
            return False
            
        return True
    
    def _save_with_nanotron(self, output_dir: str, global_step: int):
        """Save checkpoint using Nanotron's native serialization"""
        try:
            from nanotron.config import Config
            from nanotron.parallel.context import ParallelContext
            
            # Create minimal parallel context for single GPU
            if not hasattr(self, 'parallel_context'):
                self.parallel_context = ParallelContext(
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    data_parallel_size=self.world_size,
                )
            
            # Create minimal config for nanotron
            # Note: In production, this should be passed from main training config
            minimal_config = {
                'checkpoints': {
                    'checkpoints_path': output_dir,
                    'checkpoint_interval': self.args.save_steps,
                }
            }
            
            # Use nanotron's save function
            checkpoint_metadata = {
                'global_step': global_step,
                'timestamp': datetime.now().isoformat(),
                'training_metrics': self.training_metrics
            }
            
            save(
                config=minimal_config,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler,
                parallel_context=self.parallel_context,
                root_folder=Path(output_dir),
                checkpoint_metadata=checkpoint_metadata
            )
            
            logger.info(f"Nanotron checkpoint saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save with Nanotron serialization: {e}")
            logger.info("Falling back to custom checkpoint saving")
            return False
    
    def setup_tensorboard(self):
        """Initialize TensorBoard logging"""
        self.tensorboard_writer = None
        
        if self.tensorboard_args and self.tensorboard_args.use_tensorboard:
            if SummaryWriter is None:
                logger.warning("TensorBoard is not available. Install it with: pip install tensorboard")
                return
            
            # Create tensorboard log directory
            os.makedirs(self.tensorboard_args.tensorboard_dir, exist_ok=True)
            
            # Generate run name if not provided
            if self.tensorboard_args.tensorboard_run_name:
                run_name = self.tensorboard_args.tensorboard_run_name
            else:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"smolvlm2_infini_{timestamp}"
            
            # Create log directory for this run
            log_dir = os.path.join(self.tensorboard_args.tensorboard_dir, run_name)
            os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists
            
            try:
                self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"Initialized TensorBoard logging: {log_dir}")
                logger.info(f"View logs with: tensorboard --logdir={self.tensorboard_args.tensorboard_dir}")
                
            except Exception as e:
                logger.error(f"Failed to initialize TensorBoard writer: {e}")
                self.tensorboard_writer = None
                return
        
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Get trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=self.args.weight_decay
        )
        
        # Create scheduler
        steps_per_epoch = len(self.train_dataset) // self.args.per_device_train_batch_size
        if hasattr(self.args, 'max_steps') and self.args.max_steps > 0:
            num_training_steps = self.args.max_steps
        else:
            num_training_steps = steps_per_epoch * self.args.num_train_epochs
        
        self.scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train(self):
        """Main training loop with checkpoint resumption support"""
        logger.info("Starting training...")
        
        # Setup signal handlers for graceful interruption
        self._setup_signal_handlers()
        
        # Move model to device and ensure proper dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = torch.bfloat16 if self.args.bf16 else torch.float32
        self.model.to(device=device, dtype=target_dtype)
        self.model.train()
        
        logger.info(f"Model moved to {device} with dtype {target_dtype}")
        
        # Create data loader
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Check for checkpoint resumption
        global_step = 0
        total_loss = 0.0
        start_epoch = 0
        
        # Try to resume from checkpoint
        checkpoint_to_resume = None
        if hasattr(self.model_args, 'resume_from_checkpoint') and self.model_args.resume_from_checkpoint:
            checkpoint_to_resume = self.model_args.resume_from_checkpoint
        elif self.args.resume_from_checkpoint:
            # Check for auto-resume from latest checkpoint
            checkpoint_to_resume = self.find_latest_checkpoint()
            if checkpoint_to_resume:
                logger.info(f"Found latest checkpoint: {checkpoint_to_resume}")
        
        if checkpoint_to_resume and os.path.exists(checkpoint_to_resume):
            try:
                global_step = self.load_checkpoint(checkpoint_to_resume)
                # Calculate which epoch to start from
                steps_per_epoch = len(train_dataloader)
                start_epoch = global_step // steps_per_epoch
                # Skip already processed batches in the first resumed epoch
                batches_to_skip = global_step % steps_per_epoch
                logger.info(f"Resuming from epoch {start_epoch}, step {global_step}")
                if batches_to_skip > 0:
                    logger.info(f"Skipping {batches_to_skip} batches in current epoch")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Starting training from scratch")
        
        # Calculate total steps based on max_steps or num_train_epochs
        steps_per_epoch = len(train_dataloader)
        if hasattr(self.args, 'max_steps') and self.args.max_steps > 0:
            total_steps = self.args.max_steps
            effective_epochs = (total_steps + steps_per_epoch - 1) // steps_per_epoch  # Ceiling division
            logger.info(f"Training for {total_steps} steps (approximately {effective_epochs} epochs)")
        else:
            total_steps = steps_per_epoch * self.args.num_train_epochs
            effective_epochs = int(self.args.num_train_epochs)
            logger.info(f"Training for {self.args.num_train_epochs} epochs")
        
        logger.info(f"Total training steps: {total_steps}")
        
        # Store for signal handler
        self.global_step = global_step
        self.interrupted = False
        
        # Training loop
        for epoch in range(start_epoch, effective_epochs):
            if self.interrupted:
                logger.info("Training interrupted by signal")
                break
                
            logger.info(f"Starting epoch {epoch + 1}/{effective_epochs}")
            self.current_epoch = epoch
            
            epoch_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            
            # Skip batches if resuming within an epoch
            batches_to_skip = 0
            if epoch == start_epoch and global_step > 0:
                steps_per_epoch = len(train_dataloader)
                batches_to_skip = global_step % steps_per_epoch
            
            for step, batch in enumerate(progress_bar):
                # Skip batches if resuming
                if step < batches_to_skip:
                    continue
                    
                # Check for interruption
                if self.interrupted:
                    logger.info("Saving checkpoint before interruption...")
                    checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-interrupted-{global_step}")
                    self.save_checkpoint(checkpoint_dir, global_step, step_loss)
                    break
                
                # Check if we've reached max_steps
                if hasattr(self.args, 'max_steps') and self.args.max_steps > 0 and global_step >= self.args.max_steps:
                    logger.info(f"Reached max_steps ({self.args.max_steps}). Stopping training.")
                    break
                
                # Move batch to device and ensure dtype consistency
                model_dtype = next(self.model.parameters()).dtype
                batch = {
                    k: v.to(device=device, dtype=model_dtype) if torch.is_tensor(v) and v.dtype.is_floating_point 
                    else v.to(device) if torch.is_tensor(v) 
                    else v 
                    for k, v in batch.items()
                }
                
                # Forward pass
                try:
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    
                    # Debug: print loss shape and type
                    if step == 0:
                        logger.info(f"Loss shape: {loss.shape}, Loss type: {type(loss)}")
                    
                    # Ensure loss is a scalar
                    if loss.dim() > 0:
                        loss = loss.mean()
                    
                except Exception as e:
                    logger.warning(f"Error in forward pass: {e}")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Calculate gradient norm for metrics
                grad_norm = 0.0
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    grad_norm = clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                else:
                    # Calculate grad norm even if not clipping
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    grad_norm = total_norm ** (1. / 2)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update metrics
                global_step += 1
                self.global_step = global_step  # Update for signal handler
                step_loss = loss.item()
                total_loss += step_loss
                epoch_loss += step_loss
                
                # Track training metrics
                current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.args.learning_rate
                self.training_metrics['train_loss_history'].append({
                    'step': global_step,
                    'loss': step_loss,
                    'timestamp': datetime.now().isoformat()
                })
                self.training_metrics['learning_rates'].append({
                    'step': global_step,
                    'lr': current_lr,
                    'timestamp': datetime.now().isoformat()
                })
                self.training_metrics['gradient_norms'].append({
                    'step': global_step,
                    'grad_norm': grad_norm,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep only recent history to manage memory
                max_history = 10000
                for key in ['train_loss_history', 'learning_rates', 'gradient_norms']:
                    if len(self.training_metrics[key]) > max_history:
                        self.training_metrics[key] = self.training_metrics[key][-max_history:]
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{step_loss:.4f}',
                    'avg_loss': f'{total_loss / global_step:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'step': f'{global_step}/{total_steps}'
                })
                
                # Log to TensorBoard
                if self.tensorboard_writer is not None:
                    try:
                        self.tensorboard_writer.add_scalar('train/loss', step_loss, global_step)
                        self.tensorboard_writer.add_scalar('train/avg_loss', total_loss / global_step, global_step)
                        self.tensorboard_writer.add_scalar('train/learning_rate', self.scheduler.get_last_lr()[0], global_step)
                        self.tensorboard_writer.add_scalar('train/epoch', epoch + 1, global_step)
                        
                        # Flush every few steps to ensure data is written
                        if global_step % 5 == 0:
                            self.tensorboard_writer.flush()
                    except Exception as e:
                        logger.warning(f"Failed to log to TensorBoard at step {global_step}: {e}")
                
                # Save checkpoint
                if self.args.save_steps is not None and self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                    
                    # Intelligent checkpoint strategy:
                    # - Save full checkpoints at start, major milestones, and periodically
                    # - Use incremental checkpoints for frequent intermediate saves
                    checkpoints_saved = len([h for h in self.training_metrics.get('checkpoint_history', []) if h.get('method', '').startswith('custom')])
                    use_incremental = (
                        global_step > 100 and  # After initial period
                        checkpoints_saved > 2 and  # After first few checkpoints
                        (checkpoints_saved % self.incremental_checkpoint_interval) != 0 and  # Not a periodic full save
                        global_step % (self.args.save_steps * 5) != 0  # Not a major milestone
                    )
                    
                    self.save_checkpoint(checkpoint_dir, global_step, step_loss, save_incremental=use_incremental)
                
                # Evaluation
                if (self.args.eval_steps is not None and self.args.eval_steps > 0 and 
                    global_step % self.args.eval_steps == 0 and self.eval_dataset is not None):
                    eval_start_time = time.time()
                    eval_loss = self.evaluate()
                    eval_time = time.time() - eval_start_time
                    
                    logger.info(f"Step {global_step}: eval_loss = {eval_loss:.4f} (eval time: {eval_time:.2f}s)")
                    
                    # Track eval metrics
                    self.training_metrics['eval_loss_history'].append({
                        'step': global_step,
                        'eval_loss': eval_loss,
                        'eval_time': eval_time,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Log eval metrics to TensorBoard
                    if self.tensorboard_writer is not None:
                        try:
                            self.tensorboard_writer.add_scalar('eval/loss', eval_loss, global_step)
                            self.tensorboard_writer.add_scalar('eval/eval_time', eval_time, global_step)
                            self.tensorboard_writer.flush()
                        except Exception as e:
                            logger.warning(f"Failed to log eval metrics to TensorBoard: {e}")
                    
                    self.model.train()  # Set back to training mode
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Log epoch metrics to TensorBoard
            if self.tensorboard_writer is not None:
                try:
                    self.tensorboard_writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch + 1)
                    self.tensorboard_writer.flush()
                except Exception as e:
                    logger.warning(f"Failed to log epoch metrics to TensorBoard: {e}")
            
            # Save at end of epoch
            if self.args.save_strategy == "epoch":
                checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch + 1}")
                eval_loss = None
                if self.eval_dataset is not None:
                    eval_loss = self.evaluate()
                    self.model.train()
                self.save_checkpoint(checkpoint_dir, global_step, avg_epoch_loss, eval_loss)
            
            # Check if we've reached max_steps (for early exit from epoch loop)
            if hasattr(self.args, 'max_steps') and self.args.max_steps > 0 and global_step >= self.args.max_steps:
                logger.info(f"Reached max_steps ({self.args.max_steps}). Ending training.")
                break
        
        logger.info("Training completed!")
        logger.info(f"Final average loss: {total_loss / global_step:.4f}")
        
        # Log final metrics to TensorBoard
        if self.tensorboard_writer is not None:
            try:
                self.tensorboard_writer.add_scalar('train/final_loss', total_loss / global_step, global_step)
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"Failed to log final metrics to TensorBoard: {e}")
        
        return {
            'train_loss': total_loss / global_step,
            'global_step': global_step
        }
    
    def evaluate(self):
        """Evaluate the model"""
        if self.eval_dataset is None:
            logger.warning("No evaluation dataset provided")
            return 0.0
            
        logger.info("Starting evaluation...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        total_eval_loss = 0.0
        num_eval_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device and ensure dtype consistency
                model_dtype = next(self.model.parameters()).dtype
                batch = {
                    k: v.to(device=device, dtype=model_dtype) if torch.is_tensor(v) and v.dtype.is_floating_point 
                    else v.to(device) if torch.is_tensor(v) 
                    else v 
                    for k, v in batch.items()
                }
                
                try:
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    total_eval_loss += loss.item()
                    num_eval_steps += 1
                except Exception as e:
                    logger.warning(f"Error in evaluation step: {e}")
                    continue
        
        avg_eval_loss = total_eval_loss / num_eval_steps if num_eval_steps > 0 else 0.0
        logger.info(f"Evaluation completed. Average loss: {avg_eval_loss:.4f}")
        
        return avg_eval_loss
    
    def save_checkpoint(self, output_dir: str, global_step: int, 
                       step_loss: float = None, eval_loss: float = None,
                       save_incremental: bool = False):
        """Save training checkpoint with validation and metrics"""
        start_time = time.time()
        
        # Distributed coordination - only main process saves
        if not self._coordinate_distributed_checkpoint("save"):
            logger.debug(f"Non-main process skipping checkpoint save at step {global_step}")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Try Nanotron serialization if enabled
        if self.use_nanotron_checkpointing:
            nanotron_success = self._save_with_nanotron(output_dir, global_step)
            if nanotron_success:
                # Still save our custom metrics and validation
                self._save_training_metrics(output_dir, global_step, step_loss, eval_loss)
                validation_success = self._validate_checkpoint_integrity(output_dir)
                
                # Update tracking
                self.training_metrics['checkpoint_history'].append({
                    'step': global_step,
                    'path': output_dir,
                    'timestamp': datetime.now().isoformat(),
                    'validation_passed': validation_success,
                    'save_time': time.time() - start_time,
                    'method': 'nanotron'
                })
                
                self._rotate_checkpoints()
                logger.info(f"Nanotron checkpoint completed in {time.time() - start_time:.2f}s")
                return
        
        # Get random states for exact reproducibility
        random_states = {
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            random_states['torch_cuda_random_state'] = torch.cuda.get_rng_state()
        
        # Build checkpoint data
        current_model_state = self.model.state_dict()
        checkpoint = {
            'model_state_dict': current_model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': global_step,
            'epoch': getattr(self, 'current_epoch', 0),
            'random_states': random_states,
            'timestamp': datetime.now().isoformat(),
            'training_args': asdict(self.args) if hasattr(self.args, '__dataclass_fields__') else vars(self.args),
            'training_metrics_snapshot': dict(self.training_metrics),
        }
        
        # Incremental checkpoint logic (save only changed parameters)
        if save_incremental and self.previous_model_state is not None:
            logger.info("Saving incremental checkpoint...")
            model_delta = {}
            changed_params = 0
            total_params = 0
            
            for name, param in current_model_state.items():
                total_params += 1
                if name in self.previous_model_state:
                    # Check if parameter changed significantly
                    prev_param = self.previous_model_state[name]
                    if not torch.allclose(param, prev_param, rtol=1e-6, atol=1e-8):
                        model_delta[name] = param
                        changed_params += 1
                else:
                    # New parameter
                    model_delta[name] = param
                    changed_params += 1
            
            if changed_params > 0:
                checkpoint['model_delta'] = model_delta
                checkpoint['delta_metadata'] = {
                    'changed_parameters': changed_params,
                    'total_parameters': total_params,
                    'compression_ratio': changed_params / max(total_params, 1)
                }
                logger.info(f"Incremental checkpoint: {changed_params}/{total_params} parameters changed")
            else:
                logger.info("No parameter changes detected, skipping incremental checkpoint")
                return
        
        # Save with temporary file for atomic write
        temp_path = os.path.join(output_dir, "training_state.pt.tmp")
        final_path = os.path.join(output_dir, "training_state.pt")
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, final_path)  # Atomic operation
        
        # Save model separately for easy loading (full model always)
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(current_model_state, model_path)
        
        # Save processor
        if self.processor:
            self.processor.save_pretrained(output_dir)
        
        # Save comprehensive training metrics
        self._save_training_metrics(output_dir, global_step, step_loss, eval_loss)
        
        # Save metadata
        metadata = {
            'global_step': global_step,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_dir': output_dir,
            'save_time_seconds': time.time() - start_time,
            'checkpoint_type': 'incremental' if save_incremental else 'full',
        }
        with open(os.path.join(output_dir, "checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Validate checkpoint integrity
        validation_success = self._validate_checkpoint_integrity(output_dir)
        if not validation_success:
            logger.error(f"Checkpoint validation failed: {output_dir}")
            # Don't raise exception, but log the issue
        
        
        # Update tracking
        self.training_metrics['checkpoint_history'].append({
            'step': global_step,
            'path': output_dir,
            'timestamp': datetime.now().isoformat(),
            'validation_passed': validation_success,
            'save_time': time.time() - start_time,
            'method': 'incremental' if save_incremental else 'custom_full'
        })
        
        # Update previous model state for incremental checkpointing
        self.previous_model_state = current_model_state.copy()
        
        save_time = time.time() - start_time
        logger.info(f"Checkpoint saved to {output_dir} in {save_time:.2f}s (validation: {'✓' if validation_success else '✗'})")
        
        # Manage checkpoint rotation
        self._rotate_checkpoints()
        
    def save_model(self, output_dir=None):
        """Save the trained model"""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save processor
        if self.processor:
            self.processor.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
        
    def load_checkpoint(self, checkpoint_dir: str) -> int:
        """Load training checkpoint and resume training state"""
        # Distributed coordination
        self._coordinate_distributed_checkpoint("load")
        
        # Try Nanotron loading first if that's what we're using
        if self.use_nanotron_checkpointing and os.path.exists(os.path.join(checkpoint_dir, "config.yaml")):
            return self._load_with_nanotron(checkpoint_dir)
        
        # Standard loading
        checkpoint_path = os.path.join(checkpoint_dir, "training_state.pt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle incremental checkpoints
        if 'model_delta' in checkpoint and 'model_state_dict' not in checkpoint:
            logger.info("Loading incremental checkpoint, need base checkpoint")
            # Find the most recent full checkpoint
            base_checkpoint_path = self._find_base_checkpoint_for_incremental(checkpoint_dir)
            if base_checkpoint_path:
                logger.info(f"Loading base checkpoint: {base_checkpoint_path}")
                base_checkpoint = torch.load(base_checkpoint_path, map_location='cpu')
                model_state = base_checkpoint['model_state_dict']
                # Apply delta
                model_state.update(checkpoint['model_delta'])
                checkpoint['model_state_dict'] = model_state
            else:
                raise RuntimeError("Cannot find base checkpoint for incremental checkpoint")
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training metrics if available
        if 'training_metrics_snapshot' in checkpoint:
            self.training_metrics = checkpoint['training_metrics_snapshot']
            logger.info("Restored training metrics history")
        
        # Restore random states for exact reproducibility
        if 'random_states' in checkpoint:
            random_states = checkpoint['random_states']
            random.setstate(random_states['python_random_state'])
            np.random.set_state(random_states['numpy_random_state'])
            torch.set_rng_state(random_states['torch_random_state'])
            if torch.cuda.is_available() and 'torch_cuda_random_state' in random_states:
                torch.cuda.set_rng_state(random_states['torch_cuda_random_state'])
        
        global_step = checkpoint['global_step']
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']
        
        # Set previous model state for incremental checkpointing
        self.previous_model_state = self.model.state_dict().copy()
        
        logger.info(f"Resumed from checkpoint at step {global_step}")
        return global_step
    
    def _load_with_nanotron(self, checkpoint_dir: str) -> int:
        """Load checkpoint using Nanotron's native serialization"""
        try:
            # Load metadata to get global step
            metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    global_step = metadata.get('global_step', 0)
            else:
                global_step = 0
            
            load(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler,
                parallel_context=getattr(self, 'parallel_context', None),
                root_folder=Path(checkpoint_dir)
            )
            
            logger.info(f"Nanotron checkpoint loaded from {checkpoint_dir}")
            return global_step
            
        except Exception as e:
            logger.error(f"Failed to load with Nanotron serialization: {e}")
            raise
    
    def _find_base_checkpoint_for_incremental(self, incremental_checkpoint_dir: str) -> Optional[str]:
        """Find the base checkpoint for an incremental checkpoint"""
        # Extract step number from incremental checkpoint
        try:
            step_str = incremental_checkpoint_dir.split('-')[-1]
            current_step = int(step_str)
        except:
            return None
        
        # Look for full checkpoints with step numbers less than current
        base_dir = os.path.dirname(incremental_checkpoint_dir)
        checkpoint_dirs = glob.glob(os.path.join(base_dir, "checkpoint-*"))
        
        valid_base_checkpoints = []
        for checkpoint_dir in checkpoint_dirs:
            try:
                checkpoint_step = int(checkpoint_dir.split('-')[-1])
                if checkpoint_step < current_step:
                    checkpoint_path = os.path.join(checkpoint_dir, "training_state.pt")
                    if os.path.exists(checkpoint_path):
                        # Check if it's a full checkpoint (not incremental)
                        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                        if 'model_state_dict' in checkpoint_data and 'model_delta' not in checkpoint_data:
                            valid_base_checkpoints.append((checkpoint_step, checkpoint_path))
            except:
                continue
        
        if valid_base_checkpoints:
            # Return the most recent full checkpoint
            valid_base_checkpoints.sort(key=lambda x: x[0])
            return valid_base_checkpoints[-1][1]
        
        return None
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the output directory"""
        if not os.path.exists(self.args.output_dir):
            return None
        
        checkpoint_dirs = glob.glob(os.path.join(self.args.output_dir, "checkpoint-*"))
        if not checkpoint_dirs:
            return None
        
        # Sort by step number
        def get_step_from_checkpoint(path):
            try:
                return int(path.split('-')[-1])
            except:
                return -1
        
        checkpoint_dirs.sort(key=get_step_from_checkpoint)
        latest_checkpoint = checkpoint_dirs[-1]
        
        # Verify it's a valid checkpoint
        if os.path.exists(os.path.join(latest_checkpoint, "training_state.pt")):
            return latest_checkpoint
        return None
    
    def _rotate_checkpoints(self):
        """Keep only the N most recent checkpoints"""
        if not hasattr(self.model_args, 'max_checkpoints_to_keep'):
            return
        
        max_to_keep = self.model_args.max_checkpoints_to_keep
        if max_to_keep <= 0:
            return
        
        checkpoint_dirs = glob.glob(os.path.join(self.args.output_dir, "checkpoint-*"))
        if len(checkpoint_dirs) <= max_to_keep:
            return
        
        # Sort by step number
        def get_step_from_checkpoint(path):
            try:
                return int(path.split('-')[-1])
            except:
                return -1
        
        checkpoint_dirs.sort(key=get_step_from_checkpoint)
        
        # Remove oldest checkpoints
        checkpoints_to_delete = checkpoint_dirs[:-max_to_keep]
        for checkpoint_dir in checkpoints_to_delete:
            logger.info(f"Removing old checkpoint: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
    
    def close_tensorboard(self):
        """Close TensorBoard writer"""
        if self.tensorboard_writer is not None:
            try:
                self.tensorboard_writer.flush()
                self.tensorboard_writer.close()
                logger.info("Closed TensorBoard logging")
            except Exception as e:
                logger.warning(f"Error closing TensorBoard writer: {e}")
            finally:
                self.tensorboard_writer = None

def main():
    """Main training function"""
    
    # Initialize distributed training
    initialize_distributed()

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, TensorBoardArguments))
    model_args, data_args, training_args, tensorboard_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )
    except Exception as e:
        logger.warning(f"Could not load processor: {e}")
        processor = None

    # Load model
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        )

        if model_args.use_infini_attention:
            model = replace_attention_with_infini(model, model_args.segment_length)

    except Exception as e:
        logger.warning(f"Could not load pretrained model: {e}")
        logger.info("Creating new SmolVLM2NanotronModel instead")

        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "configs"))
        from smolvlm2_config import SmolVLM2Config
        config = SmolVLM2Config(
            use_infini_attention=model_args.use_infini_attention,
            segment_length=model_args.segment_length,
        )

        parallel_context = ParallelContext(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        )
        from nanotron.parallel.config import ParallelConfig
        parallel_config = ParallelConfig.from_args(training_args)
        model = SmolVLM2NanotronModel(config, parallel_context, parallel_config)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    logger.info("Loading datasets...")

    try:
        if os.path.isdir(data_args.train_data_path):
            logger.info(f"Loading dataset from directory: {data_args.train_data_path}")
            all_data = []
            for filename in sorted(os.listdir(data_args.train_data_path)):
                if filename.endswith(".json"):
                    path = os.path.join(data_args.train_data_path, filename)
                    try:
                        with open(path, "r") as f:
                            data = json.load(f)
                            all_data.extend(data)
                            logger.info(f"Loaded {len(data)} samples from {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to load {filename}: {e}")

            if len(all_data) == 0:
                raise ValueError("Merged training dataset is empty. Cannot proceed.")

            merged_path = "/tmp/merged_train_data.json"
            with open(merged_path, "w") as f:
                json.dump(all_data, f)
            logger.info(f"Merged dataset saved to {merged_path}")

            data_args.train_data_path = merged_path
            train_dataset = VisionLanguageDataset(
                data_path=merged_path,
                image_dir=data_args.image_dir,
                processor=processor,
                max_length=data_args.max_seq_length,
            )

        else:
            raise FileNotFoundError(f"Training data path {data_args.train_data_path} is invalid.")

        eval_dataset = None
        if data_args.eval_data_path and os.path.exists(data_args.eval_data_path):
            eval_dataset = VisionLanguageDataset(
                data_path=data_args.eval_data_path,
                image_dir=data_args.image_dir,
                processor=processor,
                max_length=data_args.max_seq_length,
            )
            logger.info(f"Loaded evaluation dataset with {len(eval_dataset)} samples")
        else:
            logger.info("No evaluation dataset provided")

    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise RuntimeError("Fatal error in dataset loading.")

    # Create trainer
    trainer = SmolVLM2InfiniTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        tensorboard_args=tensorboard_args,
        model_args=model_args,
        data_args=data_args,
    )

    try:
        # Train
        if training_args.do_train:
            if len(train_dataset) == 0:
                raise ValueError("Cannot train on empty dataset.")
            trainer.train()
            trainer.save_model()
            trainer.close_tensorboard()  # Close TensorBoard logging

        # Evaluate
        if training_args.do_eval and eval_dataset is not None:
            eval_results = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_results}")
            eval_output_path = os.path.join(training_args.output_dir, "eval_results.json")
            with open(eval_output_path, "w") as f:
                json.dump({'eval_loss': eval_results}, f, indent=2)
            logger.info(f"Evaluation results saved to {eval_output_path}")
    
    finally:
        # Clean up distributed training
        cleanup_distributed()
        logger.info("Cleaned up distributed training resources")


if __name__ == "__main__":
    main()
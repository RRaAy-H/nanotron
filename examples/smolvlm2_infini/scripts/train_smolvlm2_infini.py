#!/usr/bin/env python3
"""Train SmolVLM2 with Nanotron's Infini-Attention"""

import sys
import os
import torch
import torch.nn as nn
from typing import Dict, Optional, List
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
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Add nanotron to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from nanotron.models.smolvlm2_nanotron import SmolVLM2NanotronModel
from nanotron.models.llama import LlamaDecoderLayer
from nanotron.parallel.context import ParallelContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"

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
            self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
            
            # Log hyperparameters
            hparams = {
                'learning_rate': self.args.learning_rate,
                'batch_size': self.args.per_device_train_batch_size,
                'num_train_epochs': self.args.num_train_epochs,
                'weight_decay': self.args.weight_decay,
                'warmup_steps': self.args.warmup_steps,
                'gradient_accumulation_steps': self.args.gradient_accumulation_steps,
            }
            
            # Add model and data config if available
            if self.model_args:
                hparams.update({
                    'model_name_or_path': str(self.model_args.model_name_or_path),
                    'use_infini_attention': self.model_args.use_infini_attention,
                    'segment_length': self.model_args.segment_length,
                })
            
            if self.data_args:
                hparams.update({
                    'max_seq_length': self.data_args.max_seq_length,
                })
            
            # Log hyperparameters to tensorboard
            self.tensorboard_writer.add_hparams(
                hparam_dict=hparams,
                metric_dict={'train/loss': 0.0}  # Placeholder metric
            )
            
            logger.info(f"Initialized TensorBoard logging: {log_dir}")
            logger.info(f"View logs with: tensorboard --logdir={self.tensorboard_args.tensorboard_dir}")
        
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
        """Main training loop"""
        logger.info("Starting training...")
        
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
        
        # Training state
        global_step = 0
        total_loss = 0.0
        
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
        
        # Training loop
        for epoch in range(effective_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{effective_epochs}")
            
            epoch_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
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
                
                # Gradient clipping
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update metrics
                global_step += 1
                step_loss = loss.item()
                total_loss += step_loss
                epoch_loss += step_loss
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{step_loss:.4f}',
                    'avg_loss': f'{total_loss / global_step:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'step': f'{global_step}/{total_steps}'
                })
                
                # Log to TensorBoard
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar('train/loss', step_loss, global_step)
                    self.tensorboard_writer.add_scalar('train/avg_loss', total_loss / global_step, global_step)
                    self.tensorboard_writer.add_scalar('train/learning_rate', self.scheduler.get_last_lr()[0], global_step)
                    self.tensorboard_writer.add_scalar('train/epoch', epoch + 1, global_step)
                
                # Save checkpoint
                if self.args.save_steps is not None and self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                    self.save_checkpoint(checkpoint_dir, global_step)
                
                # Evaluation
                if (self.args.eval_steps is not None and self.args.eval_steps > 0 and 
                    global_step % self.args.eval_steps == 0 and self.eval_dataset is not None):
                    eval_loss = self.evaluate()
                    logger.info(f"Step {global_step}: eval_loss = {eval_loss:.4f}")
                    
                    # Log eval metrics to TensorBoard
                    if self.tensorboard_writer is not None:
                        self.tensorboard_writer.add_scalar('eval/loss', eval_loss, global_step)
                    
                    self.model.train()  # Set back to training mode
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Log epoch metrics to TensorBoard
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch + 1)
            
            # Save at end of epoch
            if self.args.save_strategy == "epoch":
                checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch + 1}")
                self.save_checkpoint(checkpoint_dir, global_step)
            
            # Check if we've reached max_steps (for early exit from epoch loop)
            if hasattr(self.args, 'max_steps') and self.args.max_steps > 0 and global_step >= self.args.max_steps:
                logger.info(f"Reached max_steps ({self.args.max_steps}). Ending training.")
                break
        
        logger.info("Training completed!")
        logger.info(f"Final average loss: {total_loss / global_step:.4f}")
        
        # Log final metrics to TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar('train/final_loss', total_loss / global_step, global_step)
            self.tensorboard_writer.flush()
        
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
    
    def save_checkpoint(self, output_dir: str, global_step: int):
        """Save training checkpoint"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model state
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': global_step,
        }
        
        torch.save(model_state, os.path.join(output_dir, "training_state.pt"))
        
        # Save model separately
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save processor
        if self.processor:
            self.processor.save_pretrained(output_dir)
        
        logger.info(f"Checkpoint saved to {output_dir}")
        
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
        
    def close_tensorboard(self):
        """Close TensorBoard writer"""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            logger.info("Closed TensorBoard logging")

def main():
    """Main training function"""

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


if __name__ == "__main__":
    main()
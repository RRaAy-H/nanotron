#!/usr/bin/env python3
"""Comprehensive test suite for SmolVLM2-Infini training data pipeline

This script validates:
1. Data preparation from local storage
2. Dataset format compatibility with SmolVLM2
3. Data mixture YAML configuration
4. SmolVLM2 dataset loading and processing
5. Data collation and batching
6. Multi-modal handling (text, image, video, multi-image)

Usage:
    python test_data_pipeline.py --data_dir data/datasets --mixture_path data/smolvlm2_256m_mixture.yaml
"""

import os
import sys
import json
import yaml
import argparse
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import traceback

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
# Add smolvlm package path
smolvlm_path = project_root / "smollm" / "vision" / "smolvlm2"
if smolvlm_path.exists():
    sys.path.append(str(smolvlm_path))
    print(f"✅ Added SmolVLM path: {smolvlm_path}")
else:
    print(f"⚠️  SmolVLM path not found: {smolvlm_path}")

try:
    import transformers
    from transformers import AutoProcessor
    from smolvlm.datasets.builder import build_datasets, DataCollatorForSupervisedDataset
    from smolvlm.datasets.dataset import SupervisedDataset
    from smolvlm.train.args import DataArguments, TrainingArguments, ModelArguments
    from smolvlm.utils import mprint
    print(f"✅ Using transformers version: {transformers.__version__}")
    print("✅ Successfully imported all required modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have the smolvlm package installed and in your PYTHONPATH")
    print("Also ensure you have a recent transformers version")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPipelineValidator:
    """Comprehensive validator for SmolVLM2-Infini data pipeline"""
    
    def __init__(self, data_dir: str, mixture_path: str, num_samples_to_test: int = 10):
        self.data_dir = Path(data_dir)
        self.mixture_path = Path(mixture_path)
        self.num_samples_to_test = num_samples_to_test
        self.validation_results = defaultdict(dict)
        
        # Statistics tracking
        self.stats = {
            'total_datasets': 0,
            'valid_datasets': 0,
            'invalid_datasets': 0,
            'total_samples': 0,
            'modality_counts': defaultdict(int),
            'errors': [],
            'warnings': [],
            'data_quality': {
                'null_conversations': [],
                'missing_media_files': [],
                'invalid_conversation_format': [],
                'samples_validated': 0,
                'samples_skipped': 0
            }
        }
        
    def run_all_tests(self):
        """Run complete validation pipeline"""
        logger.info("=" * 80)
        logger.info("Starting SmolVLM2-Infini Data Pipeline Validation")
        logger.info("=" * 80)
        
        # Test 1: Validate prepared data files
        logger.info("\n[TEST 1] Validating prepared data files...")
        self.test_prepared_data_files()
        
        # Test 2: Validate data mixture configuration
        logger.info("\n[TEST 2] Validating data mixture YAML...")
        self.test_data_mixture_config()
        
        # Test 3: Test SmolVLM2 dataset loading
        logger.info("\n[TEST 3] Testing SmolVLM2 dataset loading...")
        self.test_dataset_loading()
        
        # Test 4: Test data collation
        logger.info("\n[TEST 4] Testing data collation and batching...")
        self.test_data_collation()
        
        # Test 5: Validate multimodal handling
        logger.info("\n[TEST 5] Validating multimodal data handling...")
        self.test_multimodal_handling()
        
        # Generate final report
        logger.info("\n" + "=" * 80)
        self.generate_validation_report()
        
    def test_prepared_data_files(self):
        """Validate all prepared JSON data files"""
        if not self.data_dir.exists():
            self.stats['errors'].append(f"Data directory does not exist: {self.data_dir}")
            logger.error(f"❌ Data directory not found: {self.data_dir}")
            return
            
        json_files = list(self.data_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {self.data_dir}")
        
        for json_file in json_files:
            dataset_name = json_file.stem
            self.stats['total_datasets'] += 1
            
            try:
                # Load and validate JSON structure
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    raise ValueError("Data should be a list of samples")
                
                num_samples = len(data)
                self.stats['total_samples'] += num_samples
                
                # Validate sample structure
                valid_samples = 0
                sample_errors = []
                null_conversation_count = 0
                invalid_format_count = 0
                
                samples_to_test = data if self.num_samples_to_test is None else data[:self.num_samples_to_test]
                for idx, sample in enumerate(samples_to_test):
                    try:
                        self._validate_sample_structure(sample, dataset_name)
                        valid_samples += 1
                        self.stats['data_quality']['samples_validated'] += 1
                    except Exception as e:
                        error_msg = str(e)
                        sample_errors.append(f"Sample {idx}: {error_msg}")
                        self.stats['data_quality']['samples_skipped'] += 1
                        
                        # Categorize error types
                        if "conversations' field is null/missing" in error_msg:
                            null_conversation_count += 1
                            self.stats['data_quality']['null_conversations'].append(
                                f"{dataset_name}[{idx}]: {sample.get('id', 'unknown')}"
                            )
                        elif "must be a list" in error_msg or "missing 'from' field" in error_msg:
                            invalid_format_count += 1
                            self.stats['data_quality']['invalid_conversation_format'].append(
                                f"{dataset_name}[{idx}]: {error_msg}"
                            )
                
                if sample_errors:
                    self.validation_results[dataset_name]['sample_errors'] = sample_errors
                    self.validation_results[dataset_name]['null_conversations'] = null_conversation_count
                    self.validation_results[dataset_name]['invalid_format'] = invalid_format_count
                    self.stats['warnings'].append(
                        f"{dataset_name}: {len(sample_errors)} errors "
                        f"({null_conversation_count} null conversations, {invalid_format_count} invalid format)"
                    )
                    logger.warning(f"⚠️  {dataset_name}: Found {len(sample_errors)} problematic samples")
                else:
                    tested_count = len(samples_to_test)
                    logger.info(f"✅ {dataset_name}: All {tested_count} tested samples valid")
                
                self.validation_results[dataset_name]['num_samples'] = num_samples
                self.validation_results[dataset_name]['valid'] = len(sample_errors) == 0
                self.stats['valid_datasets'] += 1
                
            except Exception as e:
                self.stats['invalid_datasets'] += 1
                self.stats['errors'].append(f"{dataset_name}: {str(e)}")
                self.validation_results[dataset_name]['error'] = str(e)
                logger.error(f"❌ {dataset_name}: {str(e)}")
    
    def _validate_sample_structure(self, sample: Dict[str, Any], dataset_name: str):
        """Validate individual sample structure"""
        # Use .get() with defaults for safe access
        conversations = sample.get('conversations', None)
        sample_id = sample.get('id', 'unknown_id')
        
        # Check if conversations is None or not present
        if conversations is None:
            raise ValueError(f"Sample {sample_id}: 'conversations' field is null/missing")
        
        # Validate conversations is a list
        if not isinstance(conversations, list):
            raise ValueError(f"Sample {sample_id}: 'conversations' must be a list, got {type(conversations).__name__}")
        
        if len(conversations) < 2:
            raise ValueError(f"Sample {sample_id}: conversations must have at least 2 turns, found {len(conversations)}")
        
        # Check conversation format
        for i, turn in enumerate(conversations):
            if turn is None:
                raise ValueError(f"Sample {sample_id}: conversation turn {i} is null")
            
            if not isinstance(turn, dict):
                raise ValueError(f"Sample {sample_id}: turn {i} must be a dict, got {type(turn).__name__}")
            
            speaker = turn.get('from', None)
            value = turn.get('value', None)
            
            if speaker is None:
                raise ValueError(f"Sample {sample_id}: turn {i} missing 'from' field")
            if value is None:
                raise ValueError(f"Sample {sample_id}: turn {i} missing 'value' field")
            
            if speaker not in ['human', 'gpt', 'system', 'assistant']:
                raise ValueError(f"Sample {sample_id}: Invalid speaker '{speaker}' in turn {i}")
        
        # Determine and validate modality
        has_image = 'image' in sample
        has_video = 'video' in sample
        
        if has_image and has_video:
            raise ValueError("Sample cannot have both image and video")
        
        if has_image:
            self.stats['modality_counts']['image'] += 1
            # Check if image path exists (relative to data folder)
            if isinstance(sample['image'], str) and sample['image']:
                # Note: In production, images would be at /data1/yihao/...
                logger.debug(f"Image path: {sample['image']}")
            elif isinstance(sample['image'], list):
                self.stats['modality_counts']['multi-image'] += 1
        elif has_video:
            self.stats['modality_counts']['video'] += 1
            if isinstance(sample['video'], str) and sample['video']:
                logger.debug(f"Video path: {sample['video']}")
        else:
            self.stats['modality_counts']['text'] += 1
    
    def test_data_mixture_config(self):
        """Validate data mixture YAML configuration"""
        try:
            with open(self.mixture_path, 'r') as f:
                mixture_config = yaml.safe_load(f)
            
            logger.info(f"Loaded mixture config with {len(mixture_config)} modality groups")
            
            # Validate each dataset entry
            total_entries = 0
            for modality, datasets in mixture_config.items():
                if not isinstance(datasets, list):
                    raise ValueError(f"Modality {modality} should contain a list of datasets")
                
                logger.info(f"\nValidating {modality} datasets ({len(datasets)} entries)...")
                
                for dataset in datasets:
                    total_entries += 1
                    self._validate_mixture_entry(dataset, modality)
            
            logger.info(f"\n✅ Mixture config validation complete: {total_entries} dataset entries")
            
        except Exception as e:
            self.stats['errors'].append(f"Mixture config error: {str(e)}")
            logger.error(f"❌ Failed to validate mixture config: {str(e)}")
    
    def _validate_mixture_entry(self, entry: Dict[str, Any], modality: str):
        """Validate individual mixture entry"""
        required_fields = ['json_path', 'sampling_strategy', 'name', 'modality']
        
        for field in required_fields:
            if field not in entry:
                raise ValueError(f"Missing required field '{field}' in {entry.get('name', 'unknown')}")
        
        # Check if JSON file exists with improved path resolution
        json_path_str = entry['json_path']
        json_path = Path(json_path_str)
        
        # Try multiple path resolution strategies
        candidate_paths = []
        
        if json_path.is_absolute():
            candidate_paths.append(json_path)
        else:
            # Strategy 1: Relative to mixture file directory
            candidate_paths.append(self.mixture_path.parent / json_path)
            
            # Strategy 2: Relative to current working directory
            candidate_paths.append(Path.cwd() / json_path)
            
            # Strategy 3: Try removing potential duplicate "data/" prefix
            if str(json_path).startswith('data/') and 'data' in str(self.mixture_path.parent):
                relative_path = str(json_path)[5:]  # Remove "data/" prefix
                candidate_paths.append(self.mixture_path.parent / relative_path)
        
        # Find the first existing path
        found_path = None
        for candidate in candidate_paths:
            if candidate.exists():
                found_path = candidate
                break
        
        if found_path:
            logger.debug(f"  ✓ {entry['name']}: JSON file exists at {found_path}")
        else:
            self.stats['warnings'].append(f"{entry['name']}: JSON file not found at {json_path_str}")
            logger.warning(f"  ⚠️  {entry['name']}: JSON file not found")
            logger.debug(f"    Tried paths: {[str(p) for p in candidate_paths]}")
        
        # Validate modality matches
        if entry['modality'] != modality and modality not in ['multiimage', 'multi-image']:
            logger.warning(f"  ⚠️  {entry['name']}: Modality mismatch ({entry['modality']} vs {modality})")
    
    def test_dataset_loading(self):
        """Test loading datasets through SmolVLM2 dataset classes"""
        try:
            # Create mock processor
            processor = self._create_mock_processor()
            
            # Create data arguments
            data_args = DataArguments(
                data_mixture=str(self.mixture_path),
                data_folder="/data1/yihao",  # Set to server data path
                mask_user_tokens=False,
                mask_system_tokens=True,
                add_media_intro_outro=False,
                max_frames=25,
                video_target_size=384,
                image_target_size=1536,
            )
            
            # Create training arguments
            training_args = TrainingArguments(
                output_dir="test_output",
                model_max_length=2048,
                per_device_train_batch_size=1,
            )
            
            # Create model arguments
            model_args = ModelArguments(
                model_name_or_path="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
                fps=1.0,
                frames_per_clip=1,
            )
            
            # Test loading a few samples from each dataset
            with open(self.mixture_path, 'r') as f:
                mixture_config = yaml.safe_load(f)
            
            samples_tested = 0
            for modality, datasets in mixture_config.items():
                for dataset_config in datasets[:2]:  # Test first 2 datasets per modality
                    try:
                        dataset = SupervisedDataset(
                            dataset_args=dataset_config,
                            processor=processor,
                            data_args=data_args,
                            training_args=training_args,
                            model_args=model_args,
                        )
                        
                        # Test loading a sample
                        if len(dataset) > 0:
                            sample = dataset[0]
                            self._validate_dataset_sample(sample, dataset_config['name'])
                            samples_tested += 1
                            logger.info(f"✅ Successfully loaded sample from {dataset_config['name']}")
                        else:
                            logger.warning(f"⚠️  Dataset {dataset_config['name']} is empty")
                            
                    except FileNotFoundError as e:
                        logger.error(f"❌ Media file missing for {dataset_config['name']}: {str(e)}")
                        self.stats['data_quality']['missing_media_files'].append(
                            f"{dataset_config['name']}: {str(e)}"
                        )
                        self.stats['errors'].append(f"Media file missing for {dataset_config['name']}: {str(e)}")
                    except Exception as e:
                        error_type = type(e).__name__
                        logger.error(f"❌ Failed to load {dataset_config['name']}: {error_type}: {str(e)}")
                        self.stats['errors'].append(f"Dataset loading error for {dataset_config['name']}: {error_type}: {str(e)}")
            
            logger.info(f"\nDataset loading test complete: {samples_tested} samples tested")
            
        except Exception as e:
            logger.error(f"❌ Dataset loading test failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.stats['errors'].append(f"Dataset loading test failed: {str(e)}")
    
    def _create_mock_processor(self):
        """Create a mock processor for testing"""
        model_name = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        
        # Try with and without trust_remote_code
        for trust_remote_code in [True, False]:
            try:
                # Try to load actual processor
                processor = AutoProcessor.from_pretrained(
                    model_name, 
                    trust_remote_code=trust_remote_code
                )
                logger.info(f"✅ Loaded actual SmolVLM2 processor from {model_name} (trust_remote_code={trust_remote_code})")
                return processor
            except Exception as e:
                logger.debug(f"Failed to load processor from {model_name} with trust_remote_code={trust_remote_code}: {e}")
                continue
        
        logger.warning("⚠️  Could not load actual processor from any configuration")
        logger.warning("Creating mock processor for testing")
        
        # Create mock processor
        class MockProcessor:
            class MockTokenizer:
                model_max_length = 2048
                pad_token_id = 0
                additional_special_tokens = []
                
                def encode(self, text, add_special_tokens=False):
                    return [1, 2, 3, 4, 5]  # Mock token IDs
                
                def convert_tokens_to_ids(self, token):
                    return 1
                
                def convert_ids_to_tokens(self, ids):
                    return ['token'] * len(ids)
                
                def get_vocab(self):
                    return {}
            
            class MockImageProcessor:
                size = {"longest_edge": 384}
                do_resize = True
                do_image_splitting = False
            
            def __init__(self):
                self.tokenizer = self.MockTokenizer()
                self.image_processor = self.MockImageProcessor()
            
            def apply_chat_template(self, conversation, add_generation_prompt=False):
                return "Mock conversation text"
            
            def __call__(self, text=None, images=None, return_tensors="pt", padding=False):
                # Return mock encoded data
                return {
                    "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                    "pixel_values": torch.randn(1, 3, 384, 384) if images else None
                }
        
        return MockProcessor()
    
    def _validate_dataset_sample(self, sample: Dict[str, torch.Tensor], dataset_name: str):
        """Validate a sample loaded from dataset"""
        required_keys = ['input_ids', 'attention_mask', 'labels']
        
        for key in required_keys:
            if key not in sample:
                raise ValueError(f"Missing required key '{key}' in sample")
            
            if not isinstance(sample[key], torch.Tensor):
                raise ValueError(f"'{key}' should be a torch.Tensor")
            
            if sample[key].dim() != 1:
                raise ValueError(f"'{key}' should be 1-dimensional")
        
        # Check sequence lengths match
        seq_len = sample['input_ids'].size(0)
        if sample['attention_mask'].size(0) != seq_len or sample['labels'].size(0) != seq_len:
            raise ValueError("Sequence length mismatch between input_ids, attention_mask, and labels")
        
        # Check for pixel values if multimodal
        if 'pixel_values' in sample:
            pv = sample['pixel_values']
            if not isinstance(pv, torch.Tensor):
                raise ValueError("'pixel_values' should be a torch.Tensor")
            
            # Expected shape: (num_frames, 3, H, W) or (3, H, W) for single image
            if pv.dim() not in [3, 4]:
                raise ValueError(f"'pixel_values' has unexpected dimensions: {pv.dim()}")
    
    def test_data_collation(self):
        """Test data collation and batching"""
        try:
            # Create mock samples with proper tensor dtypes
            samples = [
                {
                    'input_ids': torch.randint(0, 1000, (10,), dtype=torch.long),
                    'attention_mask': torch.ones(10, dtype=torch.long),
                    'labels': torch.randint(-100, 1000, (10,), dtype=torch.long),
                },
                {
                    'input_ids': torch.randint(0, 1000, (15,), dtype=torch.long),
                    'attention_mask': torch.ones(15, dtype=torch.long),
                    'labels': torch.randint(-100, 1000, (15,), dtype=torch.long),
                },
                {
                    'input_ids': torch.randint(0, 1000, (8,), dtype=torch.long),
                    'attention_mask': torch.ones(8, dtype=torch.long),
                    'labels': torch.randint(-100, 1000, (8,), dtype=torch.long),
                    'pixel_values': torch.randn(1, 3, 384, 384),  # Add frames dimension
                }
            ]
            
            # Create collator
            collator = DataCollatorForSupervisedDataset(
                pad_token_id=0,
                model_max_length=2048,
                image_size=384,
            )
            
            # Test collation
            batch = collator(samples)
            
            # Validate batch structure
            if not isinstance(batch, dict):
                raise ValueError("Batch should be a dictionary")
                
            required_keys = ['input_ids', 'attention_mask', 'labels']
            for key in required_keys:
                if key not in batch:
                    raise ValueError(f"Missing required key '{key}' in batch")
                    
            logger.info("✅ Data collation test passed")
            
        except Exception as e:
            logger.error(f"❌ Data collation test failed: {str(e)}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            self.stats['errors'].append(f"Data collation test failed: {str(e)}")
    
    def _validate_batch(self, batch: Dict[str, torch.Tensor], batch_size: int):
        """Validate a collated batch"""
        required_keys = ['input_ids', 'attention_mask', 'labels']
        
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Missing required key '{key}' in batch")
            
            if not isinstance(batch[key], torch.Tensor):
                raise ValueError(f"'{key}' should be a torch.Tensor")
            
            if batch[key].size(0) != batch_size:
                raise ValueError(f"Batch size mismatch for '{key}'")
            
            if batch[key].dim() != 2:
                raise ValueError(f"'{key}' should be 2-dimensional (batch_size, seq_len)")
        
        # Check if all sequences have same length (padded)
        seq_len = batch['input_ids'].size(1)
        for key in required_keys:
            if batch[key].size(1) != seq_len:
                raise ValueError(f"Sequence length mismatch for '{key}'")
        
        # Check pixel values if present
        if 'pixel_values' in batch:
            pv = batch['pixel_values']
            if pv.size(0) != batch_size:
                raise ValueError("Batch size mismatch for pixel_values")
            
            # Expected shape: (batch_size, num_frames, 3, H, W)
            if pv.dim() != 5:
                raise ValueError(f"pixel_values has unexpected dimensions: {pv.dim()}")
    
    def test_multimodal_handling(self):
        """Test handling of different modalities"""
        modality_tests = {
            'text': self._create_text_sample(),
            'image': self._create_image_sample(),
            'video': self._create_video_sample(),
            'multi-image': self._create_multiimage_sample(),
        }
        
        for modality, sample in modality_tests.items():
            try:
                # Validate sample structure
                self._validate_sample_structure(sample, f"test_{modality}")
                logger.info(f"✅ {modality.capitalize()} sample validation passed")
            except Exception as e:
                logger.error(f"❌ {modality.capitalize()} sample validation failed: {str(e)}")
                self.stats['errors'].append(f"{modality} validation failed: {str(e)}")
    
    def _create_text_sample(self) -> Dict[str, Any]:
        """Create a valid text-only sample"""
        return {
            "id": "text_sample_001",
            "conversations": [
                {"from": "human", "value": "What is the capital of France?"},
                {"from": "gpt", "value": "The capital of France is Paris."}
            ]
        }
    
    def _create_image_sample(self) -> Dict[str, Any]:
        """Create a valid image sample"""
        return {
            "id": "image_sample_001",
            "conversations": [
                {"from": "human", "value": "What do you see in this image?"},
                {"from": "gpt", "value": "I see a beautiful landscape with mountains."}
            ],
            "image": "path/to/image.jpg"
        }
    
    def _create_video_sample(self) -> Dict[str, Any]:
        """Create a valid video sample"""
        return {
            "id": "video_sample_001",
            "conversations": [
                {"from": "human", "value": "Describe this video."},
                {"from": "gpt", "value": "This video shows a person cooking pasta."}
            ],
            "video": "path/to/video.mp4"
        }
    
    def _create_multiimage_sample(self) -> Dict[str, Any]:
        """Create a valid multi-image sample"""
        return {
            "id": "multiimage_sample_001",
            "conversations": [
                {"from": "human", "value": "Compare these images."},
                {"from": "gpt", "value": "The first image shows a cat, the second shows a dog."}
            ],
            "image": ["path/to/image1.jpg", "path/to/image2.jpg"]
        }
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        logger.info("VALIDATION REPORT")
        logger.info("=" * 80)
        
        # Overall statistics
        logger.info(f"\n📊 OVERALL STATISTICS:")
        logger.info(f"  Total datasets tested: {self.stats['total_datasets']}")
        logger.info(f"  Valid datasets: {self.stats['valid_datasets']}")
        logger.info(f"  Invalid datasets: {self.stats['invalid_datasets']}")
        logger.info(f"  Total samples: {self.stats['total_samples']:,}")
        
        # Modality distribution
        logger.info(f"\n📈 MODALITY DISTRIBUTION (from {self.stats['data_quality']['samples_validated']} validated samples):")
        for modality, count in self.stats['modality_counts'].items():
            percentage = (count / self.stats['data_quality']['samples_validated'] * 100) if self.stats['data_quality']['samples_validated'] > 0 else 0
            logger.info(f"  {modality}: {count:,} samples ({percentage:.1f}%)")
        
        # Data quality metrics
        logger.info(f"\n📊 DATA QUALITY METRICS:")
        logger.info(f"  Samples validated: {self.stats['data_quality']['samples_validated']:,}")
        logger.info(f"  Samples skipped: {self.stats['data_quality']['samples_skipped']:,}")
        logger.info(f"  Null conversations: {len(self.stats['data_quality']['null_conversations']):,}")
        logger.info(f"  Invalid conversation format: {len(self.stats['data_quality']['invalid_conversation_format']):,}")
        logger.info(f"  Missing media files: {len(self.stats['data_quality']['missing_media_files']):,}")
        
        # Errors summary
        if self.stats['errors']:
            logger.error(f"\n❌ ERRORS ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:10]:  # Show first 10 errors
                logger.error(f"  - {error}")
            if len(self.stats['errors']) > 10:
                logger.error(f"  ... and {len(self.stats['errors']) - 10} more errors")
        
        # Warnings summary
        if self.stats['warnings']:
            logger.warning(f"\n⚠️  WARNINGS ({len(self.stats['warnings'])}):")
            for warning in self.stats['warnings'][:10]:  # Show first 10 warnings
                logger.warning(f"  - {warning}")
            if len(self.stats['warnings']) > 10:
                logger.warning(f"  ... and {len(self.stats['warnings']) - 10} more warnings")
        
        # Final verdict
        logger.info("\n" + "=" * 80)
        if not self.stats['errors'] and self.stats['valid_datasets'] > 0:
            logger.info("✅ DATA PIPELINE IS READY FOR TRAINING!")
            logger.info("All critical tests passed. The data pipeline appears to be properly configured.")
        else:
            logger.error("❌ DATA PIPELINE IS NOT READY FOR TRAINING!")
            logger.error(f"Found {len(self.stats['errors'])} critical errors that must be fixed.")
        
        # Recommendations
        logger.info("\n📝 RECOMMENDATIONS:")
        if self.stats['warnings']:
            logger.info("  - Review and address warnings before training")
        logger.info("  - Verify all dataset JSON files are accessible on the training server")
        logger.info("  - Ensure image/video paths are correct for the /data1/yihao base path")
        logger.info("  - Consider running this test on the actual training server")
        
        # Save detailed report
        report_path = Path("data_pipeline_validation_report.json")
        
        # Convert defaultdict to regular dict for JSON serialization
        stats_dict = dict(self.stats)
        stats_dict['modality_counts'] = dict(self.stats['modality_counts'])
        
        report_data = {
            'stats': stats_dict,
            'validation_results': dict(self.validation_results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_samples_tested': self.num_samples_to_test if self.num_samples_to_test else 'all'
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"\n💾 Detailed report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Test SmolVLM2-Infini data pipeline")
    parser.add_argument(
        "--data_dir",
        default="data/datasets",
        help="Directory containing prepared JSON data files"
    )
    parser.add_argument(
        "--mixture_path",
        default="data/smolvlm2_256m_mixture.yaml",
        help="Path to data mixture YAML configuration"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to test per dataset"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Validate all samples in each dataset (warning: very slow)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validation
    num_samples = None if args.full_validation else args.num_samples
    if args.full_validation:
        logger.warning("Full validation mode enabled - this will check ALL samples and may take a long time!")
    
    validator = DataPipelineValidator(
        data_dir=args.data_dir,
        mixture_path=args.mixture_path,
        num_samples_to_test=num_samples
    )
    
    try:
        validator.run_all_tests()
    except Exception as e:
        logger.error(f"Fatal error during validation: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Setup Validation Script for SmolVLM2-Infini Distributed Training

This script helps validate your environment and configuration before running training.

Usage:
    python validate_setup.py --config configs/smolvlm2_training_distributed.yaml
    
    # With GPU validation
    torchrun --nproc_per_node=4 validate_setup.py --config configs/smolvlm2_training_4gpu.yaml --test-distributed
"""

import sys
import os
import argparse
import torch
import torch.distributed as dist
from pathlib import Path
import json
from typing import Dict, List
import warnings

# Add nanotron to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

try:
    from nanotron.config import Config
    from nanotron.parallel.context import ParallelContext
    from transformers import AutoProcessor, AutoTokenizer
    NANOTRON_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Nanotron not available: {e}")
    NANOTRON_AVAILABLE = False

def check_cuda_setup() -> Dict[str, bool]:
    """Validate CUDA and GPU setup"""
    results = {}
    
    print("🔍 Checking CUDA setup...")
    
    # Basic CUDA availability
    results['cuda_available'] = torch.cuda.is_available()
    print(f"  ✓ CUDA available: {results['cuda_available']}")
    
    if results['cuda_available']:
        gpu_count = torch.cuda.device_count()
        results['gpu_count'] = gpu_count
        print(f"  ✓ GPU count: {gpu_count}")
        
        # List GPU details
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"    - GPU {i}: {props.name} ({memory_gb:.1f} GB)")
            
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            results['gpu_memory_test'] = True
            print("  ✓ GPU memory allocation test passed")
        except Exception as e:
            results['gpu_memory_test'] = False
            print(f"  ✗ GPU memory allocation test failed: {e}")
    else:
        results['gpu_count'] = 0
        results['gpu_memory_test'] = False
    
    return results

def check_distributed_setup(config_path: str) -> Dict[str, bool]:
    """Validate distributed training setup"""
    results = {}
    
    print("🌐 Checking distributed setup...")
    
    # Check environment variables
    env_vars = ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']
    results['env_vars_set'] = all(var in os.environ for var in env_vars)
    
    if results['env_vars_set']:
        print("  ✓ Required environment variables set:")
        for var in env_vars:
            print(f"    - {var}={os.environ.get(var)}")
    else:
        missing = [var for var in env_vars if var not in os.environ]
        print(f"  ✗ Missing environment variables: {missing}")
        print("    (This is expected if not using torchrun)")
    
    # Test distributed initialization if environment is set
    if results['env_vars_set']:
        try:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
            
            results['dist_init'] = True
            print(f"  ✓ Distributed initialized: rank {dist.get_rank()}/{dist.get_world_size()}")
            
            # Test basic communication
            if torch.cuda.is_available() and dist.get_world_size() > 1:
                test_tensor = torch.ones(1).cuda()
                dist.all_reduce(test_tensor)
                expected_value = dist.get_world_size()
                results['communication_test'] = abs(test_tensor.item() - expected_value) < 0.1
                print(f"  ✓ Communication test: {'passed' if results['communication_test'] else 'failed'}")
            else:
                results['communication_test'] = True  # Skip for single GPU
                
        except Exception as e:
            results['dist_init'] = False
            results['communication_test'] = False
            print(f"  ✗ Distributed initialization failed: {e}")
    else:
        results['dist_init'] = False
        results['communication_test'] = False
    
    return results

def check_config_file(config_path: str) -> Dict[str, bool]:
    """Validate configuration file"""
    results = {}
    
    print(f"📄 Checking config file: {config_path}")
    
    # Check file exists
    config_file = Path(config_path)
    results['config_exists'] = config_file.exists()
    
    if not results['config_exists']:
        print(f"  ✗ Config file not found: {config_path}")
        return results
    
    print(f"  ✓ Config file exists")
    
    # Try to load config
    if NANOTRON_AVAILABLE:
        try:
            config = Config.from_yaml(config_path)
            results['config_loads'] = True
            print("  ✓ Config loads successfully")
            
            # Validate parallelism settings
            dp = config.parallelism.dp
            tp = config.parallelism.tp  
            pp = config.parallelism.pp
            total_gpus = dp * tp * pp
            
            results['parallelism_valid'] = total_gpus > 0
            print(f"  ✓ Parallelism: dp={dp}, tp={tp}, pp={pp} (total={total_gpus} GPUs)")
            
            # Check if parallelism matches available GPUs
            if torch.cuda.is_available():
                available_gpus = torch.cuda.device_count()
                world_size = int(os.environ.get('WORLD_SIZE', 1))
                expected_gpus = total_gpus * world_size // dist.get_world_size() if dist.is_initialized() else total_gpus
                
                if expected_gpus <= available_gpus:
                    results['gpu_parallelism_match'] = True
                    print(f"  ✓ GPU count matches parallelism settings")
                else:
                    results['gpu_parallelism_match'] = False
                    print(f"  ✗ Parallelism requires {expected_gpus} GPUs, but only {available_gpus} available")
            else:
                results['gpu_parallelism_match'] = False
            
            # Check batch size configuration
            micro_batch_size = config.tokens.micro_batch_size
            batch_accumulation = config.tokens.batch_accumulation_per_replica
            effective_batch_size = micro_batch_size * batch_accumulation * dp
            
            results['batch_config_valid'] = micro_batch_size > 0 and batch_accumulation > 0
            print(f"  ✓ Batch config: micro={micro_batch_size}, accum={batch_accumulation}, effective={effective_batch_size}")
            
        except Exception as e:
            results['config_loads'] = False
            print(f"  ✗ Failed to load config: {e}")
            return results
    else:
        results['config_loads'] = False
        print("  ✗ Cannot validate config (nanotron not available)")
    
    return results

def check_data_setup(config_path: str) -> Dict[str, bool]:
    """Validate data setup"""
    results = {}
    
    print("📊 Checking data setup...")
    
    if not NANOTRON_AVAILABLE:
        results['data_accessible'] = False
        print("  ✗ Cannot check data (nanotron not available)")
        return results
    
    try:
        config = Config.from_yaml(config_path)
        
        # Check if using nanotron data format  
        data_stage = config.data_stages[0]
        if isinstance(data_stage.data.dataset, PretrainDatasetsArgs):
            data_path = data_stage.data.dataset.hf_dataset_or_datasets
            image_base_path = "/data1/yihao"  # Base path for media files
            
            # Check nanotron data file
            if Path(data_path).exists():
                results['data_file_exists'] = True
                print(f"  ✓ Nanotron data file exists: {data_path}")
                
                # Try to load and validate nanotron format
                try:
                    with open(data_path, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list) and len(data) > 0:
                        results['data_format_valid'] = True
                        print(f"  ✓ Nanotron data format valid: {len(data)} samples")
                        
                        # Check nanotron format structure
                        sample = data[0]
                        has_input_ids = 'input_ids' in sample
                        has_text = 'text' in sample
                        has_paths = 'image_path' in sample or 'video_path' in sample
                        
                        if has_input_ids and has_text:
                            results['text_format_valid'] = True
                            print("  ✓ Nanotron text format valid (tokenized)")
                        else:
                            results['text_format_valid'] = False
                            print("  ✗ Invalid nanotron format (missing input_ids or text)")
                        
                        if has_paths:
                            results['image_format_valid'] = True
                            print("  ✓ Media path references found")
                        else:
                            results['image_format_valid'] = False
                            print("  ⚠ No media path references found")
                            
                    else:
                        results['data_format_valid'] = False
                        print("  ✗ Invalid data format (not a list or empty)")
                        
                except Exception as e:
                    results['data_format_valid'] = False
                    print(f"  ✗ Failed to parse nanotron data file: {e}")
            else:
                results['data_file_exists'] = False
                print(f"  ✗ Nanotron data file not found: {data_path}")
            
            # Check media base directory
            if Path(image_base_path).exists():
                results['image_dir_exists'] = True
                media_count = len(list(Path(image_base_path).rglob('*')))
                print(f"  ✓ Media base directory exists: {image_base_path} ({media_count} files/dirs)")
            else:
                results['image_dir_exists'] = False
                print(f"  ✗ Media base directory not found: {image_base_path}")
                
        else:
            # Standard dataset format
            results['data_accessible'] = True
            print("  ✓ Using standard dataset format")
            
    except Exception as e:
        results['data_accessible'] = False
        print(f"  ✗ Failed to check data setup: {e}")
    
    return results

def check_dependencies() -> Dict[str, bool]:
    """Check required dependencies"""
    results = {}
    
    print("📦 Checking dependencies...")
    
    # Core dependencies
    deps = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('datasets', 'datasets'),
        ('PIL', 'pillow'),
        ('cv2', 'opencv-python'),
        ('nanotron', 'nanotron (local)')
    ]
    
    for module, package in deps:
        try:
            if module == 'nanotron':
                results[f'{module}_available'] = NANOTRON_AVAILABLE
            else:
                __import__(module)
                results[f'{module}_available'] = True
            print(f"  ✓ {package}")
        except ImportError:
            results[f'{module}_available'] = False
            print(f"  ✗ {package} not available")
    
    # Check versions
    if results.get('torch_available', False):
        print(f"    - PyTorch version: {torch.__version__}")
        
    if results.get('transformers_available', False):
        import transformers
        print(f"    - Transformers version: {transformers.__version__}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Validate SmolVLM2-Infini distributed training setup")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--test-distributed', action='store_true', help='Test distributed setup')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("🧪 SmolVLM2-Infini Setup Validation")
    print("=" * 50)
    
    all_results = {}
    
    # Run all checks
    all_results.update(check_dependencies())
    all_results.update(check_cuda_setup())
    all_results.update(check_config_file(args.config))
    
    if args.test_distributed:
        all_results.update(check_distributed_setup(args.config))
    
    all_results.update(check_data_setup(args.config))
    
    print("\n📋 Summary")
    print("=" * 50)
    
    # Count results
    passed = sum(1 for v in all_results.values() if v is True)
    failed = sum(1 for v in all_results.values() if v is False) 
    total = len(all_results)
    
    print(f"Checks passed: {passed}/{total}")
    
    if args.verbose:
        print("\nDetailed results:")
        for key, value in all_results.items():
            status = "✓" if value else "✗" 
            print(f"  {status} {key}: {value}")
    
    # Overall assessment
    critical_checks = [
        'cuda_available', 'config_loads', 'torch_available', 
        'transformers_available', 'nanotron_available'
    ]
    
    critical_passed = all(all_results.get(check, False) for check in critical_checks if check in all_results)
    
    if critical_passed and failed == 0:
        print("\n🎉 All checks passed! Your setup is ready for training.")
        exit_code = 0
    elif critical_passed:
        print(f"\n⚠️  Setup mostly ready with {failed} minor issues. Training should work.")
        exit_code = 0
    else:
        print(f"\n❌ Setup has {failed} critical issues. Please fix before training.")
        exit_code = 1
    
    print("\n💡 Next steps:")
    if exit_code == 0:
        print("  - Run the appropriate launch script for your GPU setup")
        print("  - Monitor training with tensorboard or wandb")
        print("  - Check DISTRIBUTED_TRAINING_GUIDE.md for optimization tips")
    else:
        print("  - Fix the issues listed above")
        print("  - Check DISTRIBUTED_TRAINING_GUIDE.md for troubleshooting")
        print("  - Re-run this validation script")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
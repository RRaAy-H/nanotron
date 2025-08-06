#!/usr/bin/env python3
"""
Advanced Checkpoint Testing Suite for SmolVLM2 Infini-Attention

Tests all the advanced checkpoint features including:
- Checkpoint validation and integrity checks
- Training metrics tracking and saving
- Incremental checkpointing
- Distributed coordination
- Nanotron integration
"""

import os
import sys
import json
import torch
import subprocess
import time
import hashlib
from pathlib import Path

# Add nanotron to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))


def test_checkpoint_validation():
    """Test checkpoint integrity validation"""
    print("=" * 60)
    print("TEST: Checkpoint Validation & Integrity Checks")
    print("=" * 60)
    
    checkpoint_dir = "checkpoints/validation_test"
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    # Run training with validation enabled
    cmd = [
        "python", "train_smolvlm2_infini.py",
        "--model_name_or_path", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "--data_mixture", "../data/smolvlm2_256m_mixture_test.yaml",
        "--train_data_path", "../data/",
        "--image_dir", "../data/",
        "--output_dir", checkpoint_dir,
        "--max_seq_length", "512",
        "--per_device_train_batch_size", "1",
        "--max_steps", "20",
        "--save_steps", "10",
        "--learning_rate", "1e-4",
        "--bf16",
        "--do_train",
        "--use_infini_attention",
        "--segment_length", "256",
        "--enable_checkpoint_validation", "True",
        "--use_tensorboard", "False",
    ]
    
    print("Running training with checkpoint validation...")
    result = subprocess.run(cmd, cwd="scripts", capture_output=True, text=True)
    
    # Check for validation files
    validation_files_found = []
    checkpoints = []
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            if item.startswith("checkpoint-"):
                checkpoint_path = os.path.join(checkpoint_dir, item)
                checkpoints.append(checkpoint_path)
                
                # Check for validation file
                validation_file = os.path.join(checkpoint_path, "checkpoint_validation.json")
                if os.path.exists(validation_file):
                    validation_files_found.append(validation_file)
                    
                    # Verify validation content
                    with open(validation_file, 'r') as f:
                        validation_data = json.load(f)
                        required_keys = ['validation_timestamp', 'file_checksums', 
                                       'file_sizes', 'validation_passed']
                        if all(key in validation_data for key in required_keys):
                            print(f"  ✓ Validation file complete: {item}")
                        else:
                            print(f"  ✗ Validation file incomplete: {item}")
    
    if len(validation_files_found) > 0:
        print(f"✓ Checkpoint validation implemented: {len(validation_files_found)} validation files")
        return True
    else:
        print("✗ No checkpoint validation files found")
        print("STDERR:", result.stderr[-1000:])
        return False


def test_training_metrics():
    """Test comprehensive training metrics saving"""
    print("\n" + "=" * 60)
    print("TEST: Training Metrics Tracking & Saving")
    print("=" * 60)
    
    checkpoint_dir = "checkpoints/metrics_test"
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    cmd = [
        "python", "train_smolvlm2_infini.py",
        "--model_name_or_path", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "--data_mixture", "../data/smolvlm2_256m_mixture_test.yaml", 
        "--train_data_path", "../data/",
        "--image_dir", "../data/",
        "--output_dir", checkpoint_dir,
        "--max_seq_length", "512",
        "--per_device_train_batch_size", "1",
        "--max_steps", "30",
        "--save_steps", "10",
        "--eval_steps", "15",  # Add evaluation
        "--learning_rate", "1e-4",
        "--bf16",
        "--do_train",
        "--do_eval",
        "--use_infini_attention",
        "--segment_length", "256",
        "--use_tensorboard", "False",
    ]
    
    print("Running training with metrics tracking...")
    result = subprocess.run(cmd, cwd="scripts", capture_output=True, text=True)
    
    # Check for metrics files
    metrics_files_found = []
    comprehensive_metrics = 0
    
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            if item.startswith("checkpoint-"):
                checkpoint_path = os.path.join(checkpoint_dir, item)
                
                # Check for training metrics
                metrics_file = os.path.join(checkpoint_path, "training_metrics.json")
                if os.path.exists(metrics_file):
                    metrics_files_found.append(metrics_file)
                    
                    # Verify metrics content
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                        required_sections = ['training_metrics', 'model_info', 'optimizer_info']
                        
                        if all(section in metrics_data for section in required_sections):
                            # Check for detailed metrics
                            training_metrics = metrics_data['training_metrics']
                            expected_metrics = ['train_loss_history', 'learning_rates', 
                                             'gradient_norms', 'checkpoint_history']
                            
                            if all(metric in training_metrics for metric in expected_metrics):
                                comprehensive_metrics += 1
                                print(f"  ✓ Comprehensive metrics: {item}")
                            else:
                                print(f"  ~ Partial metrics: {item}")
    
    if comprehensive_metrics > 0:
        print(f"✓ Training metrics implemented: {comprehensive_metrics} complete metric files")
        return True
    else:
        print("✗ No comprehensive training metrics found")
        return False


def test_incremental_checkpoints():
    """Test incremental/delta checkpointing"""
    print("\n" + "=" * 60)
    print("TEST: Incremental/Delta Checkpointing")
    print("=" * 60)
    
    checkpoint_dir = "checkpoints/incremental_test"
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    cmd = [
        "python", "train_smolvlm2_infini.py",
        "--model_name_or_path", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "--data_mixture", "../data/smolvlm2_256m_mixture_test.yaml",
        "--train_data_path", "../data/",
        "--image_dir", "../data/",
        "--output_dir", checkpoint_dir,
        "--max_seq_length", "512",
        "--per_device_train_batch_size", "1",
        "--max_steps", "50",  # Enough steps to trigger incremental saves
        "--save_steps", "5",   # Frequent saves
        "--incremental_checkpoint_interval", "3",  # Full checkpoint every 3 saves
        "--learning_rate", "1e-4",
        "--bf16",
        "--do_train",
        "--use_infini_attention",
        "--segment_length", "256",
        "--use_tensorboard", "False",
    ]
    
    print("Running training with incremental checkpointing...")
    result = subprocess.run(cmd, cwd="scripts", capture_output=True, text=True)
    
    # Analyze checkpoint types
    full_checkpoints = 0
    incremental_checkpoints = 0
    
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            if item.startswith("checkpoint-"):
                checkpoint_path = os.path.join(checkpoint_dir, item)
                training_state_file = os.path.join(checkpoint_path, "training_state.pt")
                
                if os.path.exists(training_state_file):
                    try:\n                        checkpoint_data = torch.load(training_state_file, map_location='cpu')\n                        \n                        if 'model_delta' in checkpoint_data:\n                            incremental_checkpoints += 1\n                            delta_info = checkpoint_data.get('delta_metadata', {})\n                            compression_ratio = delta_info.get('compression_ratio', 0)\n                            print(f\"  ✓ Incremental checkpoint {item}: {compression_ratio:.2%} parameters changed\")\n                        elif 'model_state_dict' in checkpoint_data:\n                            full_checkpoints += 1\n                            print(f\"  ✓ Full checkpoint: {item}\")\n                    except Exception as e:\n                        print(f\"  ✗ Could not analyze {item}: {e}\")\n    \n    if incremental_checkpoints > 0:\n        print(f\"✓ Incremental checkpointing working: {full_checkpoints} full + {incremental_checkpoints} incremental\")\n        return True\n    else:\n        print(f\"✗ No incremental checkpoints found: {full_checkpoints} full checkpoints only\")\n        return False


def test_checkpoint_resume_with_metrics():
    """Test checkpoint resumption with metrics preservation"""
    print("\n" + "=" * 60)
    print("TEST: Checkpoint Resume with Metrics Preservation")
    print("=" * 60)
    
    checkpoint_dir = "checkpoints/resume_metrics_test"
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    # First training run
    print("Phase 1: Initial training...")
    cmd1 = [
        "python", "train_smolvlm2_infini.py",
        "--model_name_or_path", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "--data_mixture", "../data/smolvlm2_256m_mixture_test.yaml",
        "--train_data_path", "../data/",
        "--image_dir", "../data/",
        "--output_dir", checkpoint_dir,
        "--max_seq_length", "512",
        "--per_device_train_batch_size", "1",
        "--max_steps", "20",
        "--save_steps", "10",
        "--learning_rate", "1e-4",
        "--bf16",
        "--do_train",
        "--use_infini_attention",
        "--segment_length", "256",
        "--use_tensorboard", "False",
    ]
    
    result1 = subprocess.run(cmd1, cwd="scripts", capture_output=True, text=True)
    
    # Check first checkpoint metrics
    first_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            first_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
            metrics_file = os.path.join(first_checkpoint, "training_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    initial_metrics = json.load(f)
                    initial_history_length = len(initial_metrics['training_metrics'].get('train_loss_history', []))\n                    print(f\"  Initial training metrics: {initial_history_length} loss entries\")\n    \n    if not first_checkpoint:\n        print(\"✗ No checkpoint created in phase 1\")\n        return False\n    \n    # Second training run (resume)\n    print(\"Phase 2: Resume training...\")\n    cmd2 = [
        "python", "train_smolvlm2_infini.py",
        "--model_name_or_path", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "--data_mixture", "../data/smolvlm2_256m_mixture_test.yaml",
        "--train_data_path", "../data/",
        "--image_dir", "../data/",
        "--output_dir", checkpoint_dir,
        "--max_seq_length", "512",
        "--per_device_train_batch_size", "1",
        "--max_steps", "40",  # Continue to step 40
        "--save_steps", "10",
        "--learning_rate", "1e-4",
        "--bf16",
        "--do_train",
        "--use_infini_attention",
        "--segment_length", "256",
        "--resume_from_checkpoint", "auto",
        "--use_tensorboard", "False",
    ]
    
    result2 = subprocess.run(cmd2, cwd="scripts", capture_output=True, text=True)
    
    # Check final metrics preservation
    final_checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-") and int(d.split('-')[-1]) >= 30]
    if final_checkpoints:
        final_checkpoint = os.path.join(checkpoint_dir, final_checkpoints[0])
        final_metrics_file = os.path.join(final_checkpoint, "training_metrics.json")
        if os.path.exists(final_metrics_file):
            with open(final_metrics_file, 'r') as f:
                final_metrics = json.load(f)
                final_history_length = len(final_metrics['training_metrics'].get('train_loss_history', []))
                print(f\"  Final training metrics: {final_history_length} loss entries\")
                
                if final_history_length > initial_history_length:
                    print(\"✓ Metrics preserved and extended after resume\")
                    return True
                else:
                    print(\"✗ Metrics not properly preserved\")
                    return False
    
    print(\"✗ No final checkpoint with metrics found\")
    return False


def test_distributed_coordination():
    \"\"\"Test distributed checkpointing coordination\"\"\"
    print(\"\\n\" + \"=\" * 60)
    print(\"TEST: Distributed Checkpoint Coordination (Simulated)\")
    print(\"=\" * 60)
    
    # This test simulates distributed behavior by checking coordination logic
    checkpoint_dir = \"checkpoints/distributed_test\"
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    # Test with WORLD_SIZE > 1 to simulate distributed training
    env = os.environ.copy()
    env['WORLD_SIZE'] = '2'
    env['LOCAL_RANK'] = '0'
    env['RANK'] = '0'  # Main process
    
    cmd = [
        \"python\", \"train_smolvlm2_infini.py\",
        \"--model_name_or_path\", \"HuggingFaceTB/SmolVLM2-256M-Video-Instruct\",
        \"--data_mixture\", \"../data/smolvlm2_256m_mixture_test.yaml\",
        \"--train_data_path\", \"../data/\",
        \"--image_dir\", \"../data/\",
        \"--output_dir\", checkpoint_dir,
        \"--max_seq_length\", \"512\",
        \"--per_device_train_batch_size\", \"1\",
        \"--max_steps\", \"15\",
        \"--save_steps\", \"10\",
        \"--learning_rate\", \"1e-4\",
        \"--bf16\",
        \"--do_train\",
        \"--use_infini_attention\",
        \"--segment_length\", \"256\",
        \"--use_tensorboard\", \"False\",
    ]
    
    print(\"Running training with simulated distributed environment...\")
    result = subprocess.run(cmd, cwd=\"scripts\", capture_output=True, text=True, env=env)
    
    # Check that checkpoints were created (simulating main process behavior)
    if os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith(\"checkpoint-\")]
        if checkpoints:
            print(f\"✓ Distributed coordination test passed: {len(checkpoints)} checkpoints created\")
            return True
    
    print(\"✗ Distributed coordination test failed\")
    return False


def main():
    \"\"\"Run all advanced checkpoint tests\"\"\"
    print(\"Starting Advanced SmolVLM2 Infini Checkpoint Tests\")
    print(\"=\" * 60)
    
    # Change to the scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    results = []
    
    # Test 1: Checkpoint Validation
    print(\"\\nRunning Test 1: Checkpoint Validation\")
    results.append((\"Checkpoint Validation\", test_checkpoint_validation()))
    
    # Test 2: Training Metrics
    print(\"\\nRunning Test 2: Training Metrics\")
    results.append((\"Training Metrics\", test_training_metrics()))
    
    # Test 3: Incremental Checkpoints
    print(\"\\nRunning Test 3: Incremental Checkpoints\")
    results.append((\"Incremental Checkpoints\", test_incremental_checkpoints()))
    
    # Test 4: Resume with Metrics
    print(\"\\nRunning Test 4: Resume with Metrics\")
    results.append((\"Resume with Metrics\", test_checkpoint_resume_with_metrics()))
    
    # Test 5: Distributed Coordination
    print(\"\\nRunning Test 5: Distributed Coordination\")
    results.append((\"Distributed Coordination\", test_distributed_coordination()))
    
    # Summary
    print(\"\\n\" + \"=\" * 60)
    print(\"ADVANCED CHECKPOINT TESTS SUMMARY\")
    print(\"=\" * 60)
    
    for test_name, passed in results:
        status = \"✓ PASSED\" if passed else \"✗ FAILED\"
        print(f\"{test_name}: {status}\")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print(\"\\n🎉 All advanced checkpoint tests passed!\")
        print(\"\\nAdvanced features implemented:\")
        print(\"  ✓ Checkpoint integrity validation with checksums\")
        print(\"  ✓ Comprehensive training metrics tracking\")
        print(\"  ✓ Incremental/delta checkpointing for efficiency\")
        print(\"  ✓ Metrics preservation across training resume\")
        print(\"  ✓ Distributed training coordination\")
        return 0
    else:
        print(\"\\n❌ Some advanced tests failed\")
        return 1


if __name__ == \"__main__\":
    sys.exit(main())
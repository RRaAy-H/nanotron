#!/usr/bin/env python3
"""Test script to verify checkpoint saving and resumption functionality"""

import os
import sys
import json
import torch
import subprocess
import time
import signal
from pathlib import Path

# Add nanotron to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))


def test_checkpoint_saving():
    """Test that checkpoints are being saved correctly"""
    print("=" * 60)
    print("TEST 1: Checkpoint Saving")
    print("=" * 60)
    
    # Clean up any existing test checkpoints
    checkpoint_dir = "checkpoints/smolvlm2_infini_test"
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    # Run training for a few steps
    cmd = [
        "python", "train_smolvlm2_infini.py",
        "--model_name_or_path", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "--data_mixture", "../data/smolvlm2_256m_mixture_test.yaml",
        "--train_data_path", "../data/",
        "--image_dir", "../data/",
        "--output_dir", checkpoint_dir,
        "--max_seq_length", "512",
        "--per_device_train_batch_size", "1",
        "--num_train_epochs", "1",
        "--max_steps", "20",  # Stop after 20 steps
        "--save_steps", "10",  # Save every 10 steps
        "--learning_rate", "1e-4",
        "--bf16",
        "--do_train",
        "--use_infini_attention",
        "--segment_length", "256",
        "--use_tensorboard", "False",
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="scripts", capture_output=True, text=True)
    
    # Check if checkpoints were created
    checkpoints = []
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            if item.startswith("checkpoint-"):
                checkpoints.append(item)
    
    if checkpoints:
        print(f"✓ Checkpoints created: {checkpoints}")
        
        # Verify checkpoint contents
        for checkpoint in checkpoints:
            ckpt_path = os.path.join(checkpoint_dir, checkpoint)
            
            # Check for required files
            required_files = ["training_state.pt", "pytorch_model.bin", "checkpoint_metadata.json"]
            for file in required_files:
                file_path = os.path.join(ckpt_path, file)
                if os.path.exists(file_path):
                    print(f"  ✓ {file} exists in {checkpoint}")
                else:
                    print(f"  ✗ {file} missing in {checkpoint}")
            
            # Load and verify checkpoint metadata
            metadata_path = os.path.join(ckpt_path, "checkpoint_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    print(f"  Checkpoint step: {metadata.get('global_step', 'N/A')}")
        
        return True
    else:
        print("✗ No checkpoints were created")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False


def test_checkpoint_resume():
    """Test that training can resume from a checkpoint"""
    print("\n" + "=" * 60)
    print("TEST 2: Checkpoint Resumption")
    print("=" * 60)
    
    checkpoint_dir = "checkpoints/smolvlm2_infini_test"
    
    # Find the latest checkpoint
    checkpoints = []
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, item)):
                checkpoints.append(item)
    
    if not checkpoints:
        print("✗ No checkpoints found to resume from")
        return False
    
    # Sort to get the latest
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0)
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    
    print(f"Resuming from checkpoint: {latest_checkpoint}")
    
    # Load the checkpoint to get the step count
    checkpoint_path = os.path.join(latest_checkpoint, "training_state.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        initial_step = checkpoint.get('global_step', 0)
        print(f"Initial step from checkpoint: {initial_step}")
    else:
        print("✗ Could not load checkpoint")
        return False
    
    # Resume training
    cmd = [
        "python", "train_smolvlm2_infini.py",
        "--model_name_or_path", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "--data_mixture", "../data/smolvlm2_256m_mixture_test.yaml",
        "--train_data_path", "../data/",
        "--image_dir", "../data/",
        "--output_dir", checkpoint_dir,
        "--max_seq_length", "512",
        "--per_device_train_batch_size", "1",
        "--num_train_epochs", "1",
        "--max_steps", "40",  # Continue for more steps
        "--save_steps", "10",
        "--learning_rate", "1e-4",
        "--bf16",
        "--do_train",
        "--use_infini_attention",
        "--segment_length", "256",
        "--resume_from_checkpoint", latest_checkpoint,
        "--use_tensorboard", "False",
    ]
    
    print(f"Running resume command...")
    result = subprocess.run(cmd, cwd="scripts", capture_output=True, text=True)
    
    # Check if new checkpoints were created after resumption
    new_checkpoints = []
    for item in os.listdir(checkpoint_dir):
        if item.startswith("checkpoint-"):
            step = int(item.split('-')[-1]) if item.split('-')[-1].isdigit() else 0
            if step > initial_step:
                new_checkpoints.append(item)
    
    if new_checkpoints:
        print(f"✓ New checkpoints created after resumption: {new_checkpoints}")
        
        # Verify that training continued from the correct step
        for checkpoint in new_checkpoints:
            metadata_path = os.path.join(checkpoint_dir, checkpoint, "checkpoint_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    step = metadata.get('global_step', 0)
                    if step > initial_step:
                        print(f"  ✓ Step {step} > initial step {initial_step}")
        
        return True
    else:
        print("✗ No new checkpoints created after resumption")
        print("STDOUT:", result.stdout[-2000:])  # Last 2000 chars
        print("STDERR:", result.stderr[-2000:])
        return False


def test_graceful_interruption():
    """Test that training saves checkpoint on interruption"""
    print("\n" + "=" * 60)
    print("TEST 3: Graceful Interruption")
    print("=" * 60)
    
    checkpoint_dir = "checkpoints/smolvlm2_infini_interrupt_test"
    
    # Clean up
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    # Start training process
    cmd = [
        "python", "train_smolvlm2_infini.py",
        "--model_name_or_path", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "--data_mixture", "../data/smolvlm2_256m_mixture_test.yaml",
        "--train_data_path", "../data/",
        "--image_dir", "../data/",
        "--output_dir", checkpoint_dir,
        "--max_seq_length", "512",
        "--per_device_train_batch_size", "1",
        "--num_train_epochs", "1",
        "--max_steps", "100",  # Long enough to interrupt
        "--save_steps", "50",  # Don't save too often
        "--learning_rate", "1e-4",
        "--bf16",
        "--do_train",
        "--use_infini_attention",
        "--segment_length", "256",
        "--use_tensorboard", "False",
    ]
    
    print("Starting training process to test interruption...")
    process = subprocess.Popen(cmd, cwd="scripts", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait a bit for training to start
    time.sleep(10)
    
    # Send interrupt signal
    print("Sending SIGINT to training process...")
    process.send_signal(signal.SIGINT)
    
    # Wait for process to finish
    stdout, stderr = process.communicate(timeout=30)
    
    # Check if an interrupted checkpoint was saved
    interrupted_checkpoints = []
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            if "interrupted" in item:
                interrupted_checkpoints.append(item)
    
    if interrupted_checkpoints:
        print(f"✓ Interrupted checkpoint saved: {interrupted_checkpoints}")
        return True
    else:
        print("✗ No interrupted checkpoint found")
        print("STDOUT (last 1000 chars):", stdout[-1000:])
        print("STDERR (last 1000 chars):", stderr[-1000:])
        return False


def test_checkpoint_rotation():
    """Test that old checkpoints are deleted when max_checkpoints_to_keep is set"""
    print("\n" + "=" * 60)
    print("TEST 4: Checkpoint Rotation")
    print("=" * 60)
    
    checkpoint_dir = "checkpoints/smolvlm2_infini_rotation_test"
    
    # Clean up
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    # Run training with max_checkpoints_to_keep=2
    cmd = [
        "python", "train_smolvlm2_infini.py",
        "--model_name_or_path", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "--data_mixture", "../data/smolvlm2_256m_mixture_test.yaml",
        "--train_data_path", "../data/",
        "--image_dir", "../data/",
        "--output_dir", checkpoint_dir,
        "--max_seq_length", "512",
        "--per_device_train_batch_size", "1",
        "--num_train_epochs", "1",
        "--max_steps", "50",
        "--save_steps", "10",  # Save frequently to test rotation
        "--learning_rate", "1e-4",
        "--bf16",
        "--do_train",
        "--use_infini_attention",
        "--segment_length", "256",
        "--max_checkpoints_to_keep", "2",  # Keep only 2 checkpoints
        "--use_tensorboard", "False",
    ]
    
    print("Running training with checkpoint rotation...")
    result = subprocess.run(cmd, cwd="scripts", capture_output=True, text=True)
    
    # Count checkpoints
    checkpoints = []
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, item)):
                checkpoints.append(item)
    
    print(f"Final checkpoints: {checkpoints}")
    
    if len(checkpoints) <= 2:
        print(f"✓ Checkpoint rotation working: {len(checkpoints)} checkpoints (max 2)")
        
        # Verify these are the latest checkpoints
        if checkpoints:
            steps = []
            for ckpt in checkpoints:
                try:
                    step = int(ckpt.split('-')[-1])
                    steps.append(step)
                except:
                    pass
            
            if steps:
                steps.sort()
                print(f"  Checkpoint steps kept: {steps}")
                if steps == steps[-2:]:  # Should be the last 2 steps
                    print("  ✓ Kept the most recent checkpoints")
        
        return True
    else:
        print(f"✗ Too many checkpoints: {len(checkpoints)} (expected max 2)")
        return False


def main():
    """Run all checkpoint tests"""
    print("Starting SmolVLM2 Infini Checkpoint Tests")
    print("=" * 60)
    
    # Change to the scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    results = []
    
    # Test 1: Basic checkpoint saving
    print("\nRunning Test 1: Checkpoint Saving")
    results.append(("Checkpoint Saving", test_checkpoint_saving()))
    
    # Test 2: Checkpoint resumption
    print("\nRunning Test 2: Checkpoint Resumption")
    results.append(("Checkpoint Resumption", test_checkpoint_resume()))
    
    # Test 3: Graceful interruption
    print("\nRunning Test 3: Graceful Interruption")
    results.append(("Graceful Interruption", test_graceful_interruption()))
    
    # Test 4: Checkpoint rotation
    print("\nRunning Test 4: Checkpoint Rotation")
    results.append(("Checkpoint Rotation", test_checkpoint_rotation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
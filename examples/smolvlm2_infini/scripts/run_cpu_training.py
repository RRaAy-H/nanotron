#!/usr/bin/env python3
"""Run SmolVLM2 training on CPU"""
import os
import sys
import subprocess

# Set environment variables
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage

# Add patch for CPU-only training
import torch
if not torch.cuda.is_available():
    # Create dummy CUDA functions to prevent errors
    torch.cuda.set_device = lambda x: None
    torch.cuda.device_count = lambda: 1
    torch.cuda.current_device = lambda: 0
    torch.cuda.device = lambda x: x

# Launch the training script with all arguments passed through
cmd = [sys.executable, "scripts/train_smolvlm2_infini.py"] + sys.argv[1:]
subprocess.run(cmd)
#!/usr/bin/env python3
"""Download and prepare datasets for SmolVLM2-256M training

This script supports two download methods:
1. Git clone (default) - Bypasses SSL certificate issues by using git instead of HTTPS
2. HuggingFace datasets library - Traditional method

Usage with git clone (recommended):
    python download_datasets.py --output_dir data/datasets

Usage with HF token authentication:
    export HF_TOKEN=your_token_here
    python download_datasets.py --output_dir data/datasets

Usage without git (fallback):
    python download_datasets.py --output_dir data/datasets --no_git

Prerequisites for git method:
    - git installed
    - git-lfs installed (run: git lfs install)
"""

import os
import json
import random
import ssl
import urllib3
import subprocess
import shutil
import tempfile
from pathlib import Path
from datasets import load_dataset, Dataset
from tqdm import tqdm
import argparse
import pandas as pd

# Fix SSL certificate verification issues
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = 'true'
os.environ['DATASETS_DISABLE_SSL_VERIFY'] = 'true'
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = 'true'
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Additional SSL bypass for requests
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class SSLBypassAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = ssl.create_default_context()
        kwargs['ssl_context'].check_hostname = False
        kwargs['ssl_context'].verify_mode = ssl.CERT_NONE
        return super().init_poolmanager(*args, **kwargs)

# Apply SSL bypass to requests session
session = requests.Session()
session.mount('https://', SSLBypassAdapter())

# Monkey patch requests to use our session
original_get = requests.get
original_post = requests.post

def patched_get(*args, **kwargs):
    kwargs['verify'] = False
    return original_get(*args, **kwargs)

def patched_post(*args, **kwargs):
    kwargs['verify'] = False
    return original_post(*args, **kwargs)

requests.get = patched_get
requests.post = patched_post

def check_existing_dataset(output_path: str) -> bool:
    """Check if dataset file already exists"""
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0

def process_local_llava_video(llava_video_path: str, output_path: str, num_samples: int) -> int:
    """Process already downloaded llava-video dataset from local directory"""
    print(f"Processing local llava-video from {llava_video_path}...")
    
    if not os.path.exists(llava_video_path):
        print(f"Warning: llava-video path {llava_video_path} does not exist")
        return 0
    
    # Look for subdirectories like 0_30_s_academic_v0_1
    subdirs = [d for d in os.listdir(llava_video_path) 
               if os.path.isdir(os.path.join(llava_video_path, d))]
    
    print(f"Found {len(subdirs)} subdirectories in llava-video")
    
    samples = []
    sample_count = 0
    
    for subdir in subdirs:
        subdir_path = os.path.join(llava_video_path, subdir)
        
        # Look for JSON or video files in subdirectory
        for file in os.listdir(subdir_path):
            if sample_count >= num_samples:
                break
                
            file_path = os.path.join(subdir_path, file)
            
            # Create a sample in SmolVLM2 format
            if file.endswith(('.mp4', '.avi', '.mov')):
                sample = {
                    "conversations": [
                        {"role": "user", "content": "Describe this video."},
                        {"role": "assistant", "content": f"This is a video from {subdir}."}
                    ],
                    "video": file_path,
                    "id": f"llava_video_{sample_count}"
                }
                samples.append(sample)
                sample_count += 1
        
        if sample_count >= num_samples:
            break
    
    # Shuffle and save
    random.shuffle(samples)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Processed {len(samples)} samples from local llava-video to {output_path}")
    return len(samples)

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

def download_dataset_with_git(
    dataset_name: str,
    dataset_config: str,
    split: str,
    num_samples: int,
    output_path: str,
    modality: str = "image",
    use_git: bool = True
):
    """Download dataset using git clone (bypasses SSL issues)"""
    
    if not use_git:
        return download_and_sample_dataset(
            dataset_name, dataset_config, split, num_samples, output_path, modality
        )
    
    print(f"Downloading {dataset_name} with git clone ({num_samples:,} samples)...")
    
    # Create temporary directory for git clone
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Setup git LFS
            subprocess.run(["git", "lfs", "install"], check=True, capture_output=True)
            
            # Prepare clone URL
            repo_url = f"https://huggingface.co/datasets/{dataset_name}"
            
            # Add HF_TOKEN for authentication if available
            hf_token = os.environ.get('HF_TOKEN')
            if hf_token:
                # Parse the dataset name for URL construction
                if '/' in dataset_name:
                    username, repo_name = dataset_name.split('/', 1)
                    repo_url = f"https://{username}:{hf_token}@huggingface.co/datasets/{dataset_name}"
            
            repo_dir = os.path.join(temp_dir, dataset_name.replace('/', '_'))
            
            print(f"Cloning {repo_url} to {repo_dir}...")
            
            # Clone the repository
            result = subprocess.run([
                "git", "clone", repo_url, repo_dir
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Git clone failed: {result.stderr}")
                print("Falling back to datasets library...")
                return download_and_sample_dataset(
                    dataset_name, dataset_config, split, num_samples, output_path, modality
                )
            
            # Pull LFS files
            print("Pulling LFS files...")
            lfs_result = subprocess.run([
                "git", "lfs", "pull"
            ], cwd=repo_dir, capture_output=True, text=True, timeout=600)
            
            if lfs_result.returncode != 0:
                print(f"Warning: LFS pull failed: {lfs_result.stderr}")
            
            # Load dataset from local directory
            print("Loading dataset from local repository...")
            try:
                # Try to load using datasets library from local path
                if dataset_config:
                    dataset = load_dataset(repo_dir, dataset_config, split=split, streaming=True)
                else:
                    dataset = load_dataset(repo_dir, split=split, streaming=True)
                
            except Exception as e:
                print(f"Direct load failed: {e}")
                # Try to find and load parquet files manually
                dataset = load_dataset_from_parquet(repo_dir, split, dataset_config)
                
                if dataset is None:
                    print("Falling back to datasets library...")
                    return download_and_sample_dataset(
                        dataset_name, dataset_config, split, num_samples, output_path, modality
                    )
            
            # Sample and convert data
            samples = []
            count = 0
            
            for item in tqdm(dataset, desc=f"Processing {dataset_name}"):
                if count >= num_samples:
                    break
                
                # Convert to SmolVLM2 format
                if modality == "image":
                    sample = {
                        "conversations": item.get("conversations", []),
                        "image": item.get("image", ""),
                        "id": item.get("id", f"{dataset_name}_{count}")
                    }
                elif modality == "video":
                    sample = {
                        "conversations": item.get("conversations", []),
                        "video": item.get("video", ""),
                        "id": item.get("id", f"{dataset_name}_{count}")
                    }
                else:  # text
                    sample = {
                        "conversations": item.get("conversations", []),
                        "id": item.get("id", f"{dataset_name}_{count}")
                    }
                
                samples.append(sample)
                count += 1
            
            # Shuffle and save
            random.shuffle(samples)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(samples, f, indent=2)
            
            print(f"Saved {len(samples):,} samples to {output_path}")
            return len(samples)
            
        except subprocess.TimeoutExpired:
            print("Git operation timed out. Falling back to datasets library...")
            return download_and_sample_dataset(
                dataset_name, dataset_config, split, num_samples, output_path, modality
            )
        except Exception as e:
            print(f"Git download failed: {e}")
            print("Falling back to datasets library...")
            return download_and_sample_dataset(
                dataset_name, dataset_config, split, num_samples, output_path, modality
            )

def load_dataset_from_parquet(repo_dir: str, split: str, config: str = None):
    """Load dataset from parquet files in cloned repository"""
    
    try:
        data_dir = Path(repo_dir) / "data"
        
        if not data_dir.exists():
            print(f"No data directory found in {repo_dir}")
            return None
        
        # Look for parquet files
        parquet_files = []
        
        # Check for config-specific directory
        if config:
            config_dir = data_dir / config
            if config_dir.exists():
                data_dir = config_dir
        
        # Look for split-specific directory
        split_dir = data_dir / split
        if split_dir.exists():
            parquet_files = list(split_dir.glob("*.parquet"))
        else:
            # Look for parquet files with split prefix
            parquet_files = list(data_dir.glob(f"{split}*.parquet"))
            if not parquet_files:
                parquet_files = list(data_dir.glob("*.parquet"))
        
        if not parquet_files:
            print(f"No parquet files found for split '{split}' in {data_dir}")
            return None
        
        print(f"Found {len(parquet_files)} parquet files")
        
        # Load parquet files
        data_files = [str(f) for f in parquet_files]
        dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
        
        return dataset
        
    except Exception as e:
        print(f"Error loading parquet files: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download datasets for SmolVLM2 training")
    parser.add_argument("--output_dir", default="data/datasets", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip_existing", action="store_true", help="Skip datasets that already exist")
    parser.add_argument("--llava_video_path", default="../../llava-video", help="Path to local llava-video dataset")
    parser.add_argument("--skip_datasets", nargs="*", help="List of dataset names to skip")
    parser.add_argument("--use_git", action="store_true", default=True, help="Use git clone for downloads (bypasses SSL issues)")
    parser.add_argument("--no_git", action="store_true", help="Disable git clone, use datasets library")
    args = parser.parse_args()
    
    # Handle git usage flag
    use_git = args.use_git and not args.no_git
    
    # Check git prerequisites if using git method
    if use_git:
        try:
            # Check if git is available
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            
            # Check if git-lfs is available
            result = subprocess.run(["git", "lfs", "version"], capture_output=True)
            if result.returncode != 0:
                print("⚠️  Git LFS not found. Installing...")
                subprocess.run(["git", "lfs", "install"], check=True)
                print("✅ Git LFS installed successfully")
            
            print("✅ Git and Git LFS are available")
            
            # Check for HF_TOKEN
            if os.environ.get('HF_TOKEN'):
                print("✅ HF_TOKEN found for authentication")
            else:
                print("ℹ️  No HF_TOKEN found. Public datasets will be downloaded without authentication.")
                
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"⚠️  Git prerequisites not met: {e}")
            print("Falling back to datasets library method...")
            use_git = False
    
    if use_git:
        print("🔄 Using git clone method to bypass SSL issues")
    else:
        print("🔄 Using HuggingFace datasets library method")
    
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
    
    # Initialize skip list
    skip_datasets = args.skip_datasets or []
    
    for config in datasets_config:
        dataset_name = config["name"]
        output_path = config["output"]
        
        # Check if we should skip this dataset
        if any(skip_name in dataset_name for skip_name in skip_datasets):
            print(f"Skipping {dataset_name} (requested to skip)")
            continue
            
        # Check if dataset already exists
        if args.skip_existing and check_existing_dataset(output_path):
            print(f"Skipping {dataset_name} (already exists at {output_path})")
            # Count existing samples
            try:
                with open(output_path, 'r') as f:
                    existing_samples = len(json.load(f))
                total_samples += existing_samples
                successful_downloads += 1
            except:
                pass
            continue
        
        # Special handling for llava-video dataset
        if "LLaVA-Video" in dataset_name:
            samples = process_local_llava_video(
                llava_video_path=args.llava_video_path,
                output_path=output_path,
                num_samples=config["samples"]
            )
        else:
            samples = download_dataset_with_git(
                dataset_name=config["name"],
                dataset_config=config["config"], 
                split=config["split"],
                num_samples=config["samples"],
                output_path=config["output"],
                modality=config["modality"],
                use_git=use_git
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
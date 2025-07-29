#!/usr/bin/env python3
"""Download and prepare datasets for SmolVLM2-256M training"""

import os
import json
import random
import ssl
import urllib3
from datasets import load_dataset
from tqdm import tqdm
import argparse

# Fix SSL certificate verification issues
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = 'true'
os.environ['DATASETS_DISABLE_SSL_VERIFY'] = 'true'

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

def main():
    parser = argparse.ArgumentParser(description="Download datasets for SmolVLM2 training")
    parser.add_argument("--output_dir", default="data/datasets", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip_existing", action="store_true", help="Skip datasets that already exist")
    parser.add_argument("--llava_video_path", default="../../llava-video", help="Path to local llava-video dataset")
    parser.add_argument("--skip_datasets", nargs="*", help="List of dataset names to skip")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Dataset download configuration
    datasets_config = [
        # Image datasets
        {
            "name": "lmms-lab/LLaVA-OneVision", 
            "config": None,
            "split": "train",
            "samples": 200000,
            "output": f"{args.output_dir}/llava_onevision_200k.json",
            "modality": "image"
        },
        {
            "name": "Lin-Chen/ShareGPT4V",
            "config": None, 
            "split": "train",
            "samples": 150000,
            "output": f"{args.output_dir}/sharegpt4v_150k.json",
            "modality": "image"
        },
        {
            "name": "lmms-lab/ai2d",
            "config": None,
            "split": "train", 
            "samples": 50000,
            "output": f"{args.output_dir}/ai2d_50k.json",
            "modality": "image"
        },
        {
            "name": "lmms-lab/ChartQA",
            "config": None,
            "split": "train",
            "samples": 40000, 
            "output": f"{args.output_dir}/chartqa_40k.json",
            "modality": "image"
        },
        {
            "name": "lmms-lab/DVQA",
            "config": None,
            "split": "train",
            "samples": 40000,
            "output": f"{args.output_dir}/dvqa_40k.json", 
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
            "name": "lmms-lab/VideoChat",  
            "config": None,
            "split": "train",
            "samples": 20000,
            "output": f"{args.output_dir}/videochat_20k.json",
            "modality": "video"
        },
        {
            "name": "lmms-lab/Video-ChatGPT",
            "config": None,
            "split": "train", 
            "samples": 10000,
            "output": f"{args.output_dir}/video_chatgpt_10k.json",
            "modality": "video"
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
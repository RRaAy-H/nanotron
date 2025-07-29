#!/usr/bin/env python3
"""Download and prepare datasets for SmolVLM2-256M training"""

import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm
import argparse

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
    
    for config in datasets_config:
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
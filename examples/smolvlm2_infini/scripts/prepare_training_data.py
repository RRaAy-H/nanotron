#!/usr/bin/env python3
"""Prepare training data from locally downloaded datasets for SmolVLM2-256M training

This script processes locally stored datasets from /data1/yihao and creates
training-ready JSON files following the SmolVLM2 format.

Usage:
    python prepare_training_data.py --output_dir data/datasets
"""

import os
import json
import random
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def load_parquet_data(parquet_path: str, num_samples: int) -> List[Dict[str, Any]]:
    """Load data from parquet file and sample specified number of examples"""
    print(f"Loading {parquet_path}...")
    
    try:
        df = pd.read_parquet(parquet_path)
        total_rows = len(df)
        
        if total_rows <= num_samples:
            sampled_df = df
        else:
            sampled_df = df.sample(n=num_samples, random_state=42)
        
        # Convert to list of dicts
        samples = []
        for _, row in sampled_df.iterrows():
            sample = row.to_dict()
            
            # Ensure proper format for SmolVLM2
            if "conversations" not in sample:
                # Try to reconstruct conversations from other fields
                if "question" in sample and "answer" in sample:
                    sample["conversations"] = [
                        {"from": "human", "value": sample["question"]},
                        {"from": "gpt", "value": sample["answer"]}
                    ]
            
            # Add ID if missing
            if "id" not in sample:
                sample["id"] = f"{Path(parquet_path).stem}_{len(samples)}"
                
            samples.append(sample)
        
        print(f"Loaded {len(samples)} samples from {parquet_path}")
        return samples
        
    except Exception as e:
        print(f"Error loading parquet file {parquet_path}: {e}")
        return []


def load_video_data(video_path: str, num_samples: int, dataset_name: str) -> List[Dict[str, Any]]:
    """Process video dataset from local directory"""
    print(f"Processing video dataset {dataset_name} from {video_path}...")
    
    samples = []
    
    if os.path.isfile(video_path) and video_path.endswith('.mp4'):
        # Single video file
        sample = {
            "conversations": [
                {"from": "human", "value": "Describe this video."},
                {"from": "gpt", "value": f"This is a video from {dataset_name}."}
            ],
            "video": video_path,
            "id": f"{dataset_name}_0"
        }
        samples.append(sample)
    
    elif os.path.isdir(video_path):
        # Directory of videos
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(Path(video_path).glob(f"**/{ext}"))
        
        # Sample videos
        if len(video_files) > num_samples:
            video_files = random.sample(video_files, num_samples)
        
        for idx, video_file in enumerate(video_files[:num_samples]):
            sample = {
                "conversations": [
                    {"from": "human", "value": "Describe this video."},
                    {"from": "gpt", "value": f"This is a video from {dataset_name}."}
                ],
                "video": str(video_file.relative_to(video_path)),
                "id": f"{dataset_name}_{idx}"
            }
            samples.append(sample)
    
    print(f"Processed {len(samples)} video samples")
    return samples


def load_zip_data(zip_path: str, num_samples: int) -> List[Dict[str, Any]]:
    """Load data from ZIP archive (for M4-Instruct-Data)"""
    print(f"Loading ZIP archive {zip_path}...")
    
    samples = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Look for JSON files in the archive
            json_files = [f for f in zf.namelist() if f.endswith('.json')]
            
            for json_file in json_files[:num_samples]:
                with zf.open(json_file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples.extend(data[:num_samples - len(samples)])
                    else:
                        samples.append(data)
                    
                    if len(samples) >= num_samples:
                        break
    
    except Exception as e:
        print(f"Error loading ZIP file {zip_path}: {e}")
    
    return samples[:num_samples]


def load_tar_gz_data(tar_path: str, num_samples: int) -> List[Dict[str, Any]]:
    """Load data from TAR.GZ archive (for Mammoth multi-image data)"""
    print(f"Loading TAR.GZ archive {tar_path}...")
    
    samples = []
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tf:
            # Look for JSON files in the archive
            for member in tf.getmembers():
                if member.name.endswith('.json') and member.isfile():
                    f = tf.extractfile(member)
                    if f:
                        data = json.load(f)
                        if isinstance(data, list):
                            samples.extend(data[:num_samples - len(samples)])
                        else:
                            samples.append(data)
                        f.close()
                        
                        if len(samples) >= num_samples:
                            break
    
    except Exception as e:
        print(f"Error loading TAR.GZ file {tar_path}: {e}")
    
    return samples[:num_samples]


def apply_composite_sampling(
    datasets: Dict[str, str], 
    num_samples: int,
    primary_ratio: float = 0.7
) -> List[Dict[str, Any]]:
    """Apply composite sampling strategy for llava-onevision/other dataset"""
    
    primary_datasets = ["figureqa", "raven"]
    samples = []
    
    # Calculate sample distribution
    primary_samples = int(num_samples * primary_ratio)
    secondary_samples = num_samples - primary_samples
    
    # Load primary datasets
    primary_count = 0
    for dataset_name in primary_datasets:
        for path in datasets.values():
            if dataset_name in path.lower():
                dataset_samples = load_parquet_data(
                    path, 
                    primary_samples // len(primary_datasets)
                )
                samples.extend(dataset_samples)
                primary_count += len(dataset_samples)
    
    # Load from remaining datasets
    remaining_datasets = [p for p in datasets.values() 
                         if not any(pd in p.lower() for pd in primary_datasets)]
    
    if remaining_datasets and secondary_samples > 0:
        samples_per_dataset = secondary_samples // len(remaining_datasets)
        for path in remaining_datasets:
            dataset_samples = load_parquet_data(path, samples_per_dataset)
            samples.extend(dataset_samples)
    
    # Shuffle and trim to exact number
    random.shuffle(samples)
    return samples[:num_samples]


def apply_alternative_sampling(
    primary_path: str,
    alternative_paths: List[str],
    num_samples: int
) -> List[Dict[str, Any]]:
    """Apply alternative sampling when primary dataset is unavailable"""
    
    # Try primary path first
    if os.path.exists(primary_path):
        return load_parquet_data(primary_path, num_samples)
    
    # Use alternatives
    samples = []
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            samples = load_parquet_data(alt_path, num_samples)
            if samples:
                break
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from local datasets")
    parser.add_argument("--output_dir", default="data/datasets", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_path", default="/data1/yihao", help="Base path for datasets")
    args = parser.parse_args()
    
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dataset configurations based on dataset_distribution.md
    datasets_config = [
        # Text datasets (20.2% total)
        {
            "name": "magpie_pro_l3_80b_mt",
            "samples": 68000,  # 6.8%
            "output": f"{args.output_dir}/magpie_pro_l3_80b_mt.json",
            "modality": "text",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/magpie_pro_l3_80b_mt.parquet",
            "format": "parquet"
        },
        {
            "name": "magpie_pro_l3_80b_st",
            "samples": 68000,  # 6.8%
            "output": f"{args.output_dir}/magpie_pro_l3_80b_st.json",
            "modality": "text",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/magpie_pro_l3_80b_st.parquet",
            "format": "parquet"
        },
        {
            "name": "magpie_pro_qwen2_72b_st",
            "samples": 58000,  # 5.8%
            "output": f"{args.output_dir}/magpie_pro_qwen2_72b_st.json",
            "modality": "text",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/magpie_pro_qwen2_72b_st.parquet",
            "format": "parquet"
        },
        {
            "name": "mathqa",
            "samples": 9000,  # 0.9%
            "output": f"{args.output_dir}/mathqa.json",
            "modality": "text",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/mathqa.parquet",
            "format": "parquet"
        },
        
        # Multi-image datasets (12.3% total)
        {
            "name": "m4_instruct_data",
            "samples": 104000,  # 10.4%
            "output": f"{args.output_dir}/m4_instruct_data.json",
            "modality": "multi-image",
            "path": f"{args.base_path}/M4-Instruct-Data/m4_instruct_data.zip",
            "format": "zip"
        },
        {
            "name": "mammoth_multi_image",
            "samples": 19000,  # 1.9%
            "output": f"{args.output_dir}/mammoth_multi_image.json",
            "modality": "multi-image",
            "path": f"{args.base_path}/MammoTH-VL_Instruct-12M/multi_image_data/shard_1.tar.gz",
            "format": "tar.gz"
        },
        
        # Image datasets (34.4% total)
        {
            "name": "llava_onevision_other",
            "samples": 174000,  # 17.4%
            "output": f"{args.output_dir}/llava_onevision_other.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/other.parquet",
            "format": "parquet",
            "sampling_strategy": "composite"
        },
        {
            "name": "vision_flan_filtered",
            "samples": 39000,  # 3.9%
            "output": f"{args.output_dir}/vision_flan_filtered.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/vision_flan_filtered.parquet",
            "format": "parquet"
        },
        {
            "name": "mavis_math_metagen",
            "samples": 26000,  # 2.6%
            "output": f"{args.output_dir}/mavis_math_metagen.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/mavis_math_metagen.parquet",
            "format": "parquet"
        },
        {
            "name": "mavis_math_rule_geo",
            "samples": 25000,  # 2.5%
            "output": f"{args.output_dir}/mavis_math_rule_geo.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/mavis_math_rule_geo.parquet",
            "format": "parquet"
        },
        {
            "name": "sharegpt4o",
            "samples": 17000,  # 1.7%
            "output": f"{args.output_dir}/sharegpt4o.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/sharegpt4o.parquet",
            "format": "parquet"
        },
        {
            "name": "sharegpt4v_coco",
            "samples": 15000,  # 1.5%
            "output": f"{args.output_dir}/sharegpt4v_coco.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/sharegpt4v_coco.parquet",
            "format": "parquet"
        },
        {
            "name": "image_textualization",
            "samples": 13000,  # 1.3%
            "output": f"{args.output_dir}/image_textualization.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/image_textualization.parquet",
            "format": "parquet"
        },
        {
            "name": "sharegpt4v_llava",
            "samples": 9000,  # 0.9%
            "output": f"{args.output_dir}/sharegpt4v_llava.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/sharegpt4v_llava.parquet",
            "format": "parquet"
        },
        {
            "name": "mapqa_mathv360k",
            "samples": 9000,  # 0.9%
            "output": f"{args.output_dir}/mapqa_mathv360k.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/mapqa_mathv360k.parquet",
            "format": "parquet"
        },
        {
            "name": "qa",
            "samples": 8000,  # 0.8%
            "output": f"{args.output_dir}/qa.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/qa.parquet",
            "format": "parquet",
            "sampling_strategy": "alternative",
            "alternatives": ["figureqa.parquet", "mapqa_mathv360k.parquet"]
        },
        {
            "name": "textocr_gpt4v",
            "samples": 8000,  # 0.8%
            "output": f"{args.output_dir}/textocr_gpt4v.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-onevision/textocr_gpt4v.parquet",
            "format": "parquet"
        },
        
        # Video datasets (33.0% total)
        {
            "name": "llava_video_1_2m",
            "samples": 73000,  # 7.3%
            "output": f"{args.output_dir}/llava_video_1_2m.json",
            "modality": "video",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-video-178k/1-2m",
            "format": "video"
        },
        {
            "name": "llava_video_2_3m",
            "samples": 70000,  # 7.0%
            "output": f"{args.output_dir}/llava_video_2_3m.json",
            "modality": "video",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-video-178k/2-3m",
            "format": "video"
        },
        {
            "name": "other_video_combined",
            "samples": 57000,  # 5.7%
            "output": f"{args.output_dir}/other_video_combined.json",
            "modality": "video",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-video-178k",
            "format": "video",
            "sampling_strategy": "alternative"
        },
        {
            "name": "llava_video_hound",
            "samples": 44000,  # 4.4%
            "output": f"{args.output_dir}/llava_video_hound.json",
            "modality": "video",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-video-178k/hound",
            "format": "video"
        },
        {
            "name": "llava_video_0_30s",
            "samples": 24000,  # 2.4%
            "output": f"{args.output_dir}/llava_video_0_30s.json",
            "modality": "video",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-video-178k/0-30s",
            "format": "video"
        },
        {
            "name": "video_star_starb",
            "samples": 22000,  # 2.2%
            "output": f"{args.output_dir}/video_star_starb.json",
            "modality": "video",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/llava-video-178k",
            "format": "video",
            "sampling_strategy": "alternative"
        },
        {
            "name": "vista_400k_combined",
            "samples": 22000,  # 2.2%
            "output": f"{args.output_dir}/vista_400k_combined.json",
            "modality": "video",
            "path": f"{args.base_path}/VISTA-400K/two_needle_niah_qa",
            "format": "tar"
        },
        {
            "name": "vript_long",
            "samples": 10000,  # 1.0%
            "output": f"{args.output_dir}/vript_long.json",
            "modality": "video",
            "path": f"{args.base_path}/VISTA-400K/two_needle_niah_qa",
            "format": "video",
            "sampling_strategy": "alternative"
        },
        {
            "name": "sharegpt4video_all",
            "samples": 8000,  # 0.8%
            "output": f"{args.output_dir}/sharegpt4video_all.json",
            "modality": "video",
            "path": f"{args.base_path}/ShareGPTVideo/train_300k",
            "format": "video"
        },
    ]
    
    total_samples = 0
    successful_datasets = 0
    
    # Process each dataset
    for config in tqdm(datasets_config, desc="Processing datasets"):
        dataset_name = config["name"]
        output_path = config["output"]
        
        # Check if output already exists
        if os.path.exists(output_path):
            print(f"Skipping {dataset_name} (already exists)")
            with open(output_path, 'r') as f:
                existing_samples = len(json.load(f))
            total_samples += existing_samples
            successful_datasets += 1
            continue
        
        samples = []
        
        try:
            # Load data based on format
            if config["format"] == "parquet":
                if config.get("sampling_strategy") == "composite":
                    # Special handling for composite sampling
                    base_dir = Path(config["path"]).parent
                    available_datasets = {p.stem: str(p) for p in base_dir.glob("*.parquet")}
                    samples = apply_composite_sampling(available_datasets, config["samples"])
                elif config.get("sampling_strategy") == "alternative":
                    # Alternative sampling
                    alt_paths = [f"{Path(config['path']).parent}/{alt}" for alt in config.get("alternatives", [])]
                    samples = apply_alternative_sampling(config["path"], alt_paths, config["samples"])
                else:
                    # Direct sampling
                    samples = load_parquet_data(config["path"], config["samples"])
                    
            elif config["format"] == "video":
                samples = load_video_data(config["path"], config["samples"], dataset_name)
                
            elif config["format"] == "zip":
                samples = load_zip_data(config["path"], config["samples"])
                
            elif config["format"] == "tar.gz":
                samples = load_tar_gz_data(config["path"], config["samples"])
                
            elif config["format"] == "tar":
                # Similar to tar.gz but without gzip compression
                samples = load_tar_gz_data(config["path"], config["samples"])
            
            # Save samples
            if samples:
                with open(output_path, 'w') as f:
                    json.dump(samples, f, indent=2)
                print(f"Saved {len(samples)} samples to {output_path}")
                total_samples += len(samples)
                successful_datasets += 1
            else:
                print(f"Warning: No samples generated for {dataset_name}")
                
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
    
    # Print summary
    print(f"\n=== Data Preparation Summary ===")
    print(f"Successfully processed: {successful_datasets}/{len(datasets_config)} datasets")
    print(f"Total samples: {total_samples:,}")
    print(f"Output directory: {args.output_dir}")
    
    # Calculate and display distribution
    modality_stats = defaultdict(int)
    for config in datasets_config:
        if os.path.exists(config["output"]):
            with open(config["output"], 'r') as f:
                count = len(json.load(f))
                modality_stats[config["modality"]] += count
    
    print("\nModality Distribution:")
    for modality, count in modality_stats.items():
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  - {modality}: {count:,} samples ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Convert SmolVLM2 dataset format to Nanotron format"""

import json
import torch
from datasets import Dataset
from transformers import AutoProcessor
import argparse
import os
from tqdm import tqdm

def convert_smolvlm2_to_nanotron(input_file: str, output_file: str, processor=None):
    """Convert SmolVLM2 dataset format to Nanotron format"""
    
    # Load processor if not provided
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            trust_remote_code=True
        )
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    converted_data = []
    for item in tqdm(data, desc=f"Converting {os.path.basename(input_file)}"):
        # Format conversation
        text = ""
        for turn in item["conversations"]:
            if turn["from"] == "human":
                text += f"<|im_start|>user\n{turn['value']}<|im_end|>\n"
            elif turn["from"] == "gpt":
                text += f"<|im_start|>assistant\n{turn['value']}<|im_end|>\n"
        
        # Tokenize
        tokens = processor.tokenizer(
            text,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        converted_item = {
            "input_ids": tokens["input_ids"].squeeze().tolist(),
            "text": text,
            "image_path": item.get("image", ""),
            "video_path": item.get("video", ""),
            "id": item.get("id", "unknown")
        }
        converted_data.append(converted_item)
    
    # Save in Nanotron format
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {len(converted_data)} samples to {output_file}")

def convert_jsonl_to_nanotron(input_file: str, output_file: str, processor=None):
    """Convert JSONL dataset format to Nanotron format"""
    
    # Load processor if not provided
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            "HuggingFaceTB/SmolVLM2-256M-Instruct",
            trust_remote_code=True
        )
    
    # Load JSONL file line by line
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    converted_data = []
    for item in tqdm(data, desc=f"Converting {os.path.basename(input_file)}"):
        # Format conversation
        text = ""
        for turn in item["conversations"]:
            if turn["from"] == "human":
                text += f"<|im_start|>user\n{turn['value']}<|im_end|>\n"
            elif turn["from"] == "gpt":
                text += f"<|im_start|>assistant\n{turn['value']}<|im_end|>\n"
        
        # Tokenize
        tokens = processor.tokenizer(
            text,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        converted_item = {
            "input_ids": tokens["input_ids"].squeeze().tolist(),
            "text": text,
            "image_path": item.get("image", ""),
            "video_path": item.get("video", ""),
            "id": item.get("id", "unknown")
        }
        converted_data.append(converted_item)
    
    # Save in Nanotron format
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {len(converted_data)} samples to {output_file}")

def convert_all_datasets(input_dir: str, output_dir: str):
    """Convert all JSON files in input directory to Nanotron format"""
    
    # Load processor once for efficiency
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        trust_remote_code=True
    )
    
    # Find all JSON files
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to convert")
    
    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)
        output_path = os.path.join(output_dir, json_file.replace('.json', '_nanotron.json'))
        
        convert_smolvlm2_to_nanotron(input_path, output_path, processor)
    
    print(f"\nConversion complete! Nanotron format files saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert SmolVLM2 data to Nanotron format")
    parser.add_argument("--input", type=str, help="Input file or directory")
    parser.add_argument("--output", type=str, help="Output file or directory")
    parser.add_argument("--input_dir", type=str, default="data/datasets", 
                        help="Input directory (if --input not specified)")
    parser.add_argument("--output_dir", type=str, default="data/datasets_nanotron", 
                        help="Output directory (if --output not specified)")
    
    args = parser.parse_args()
    
    if args.input and args.output:
        # Convert single file based on extension
        if args.input.endswith('.jsonl'):
            convert_jsonl_to_nanotron(args.input, args.output)
        else:
            convert_smolvlm2_to_nanotron(args.input, args.output)
    else:
        # Convert all files in directory
        convert_all_datasets(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
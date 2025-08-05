#!/usr/bin/env python3
"""Convert SmolVLM2 dataset format to Nanotron format"""

import json
import torch
from datasets import Dataset
from transformers import AutoProcessor
import argparse
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import math
import tempfile

def process_data_chunk(chunk_data, processor_name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct", device="cpu", batch_size=32):
    """Process a chunk of data in a separate process"""
    from transformers import AutoProcessor
    
    # Initialize processor in this process
    processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
    if hasattr(processor.tokenizer, 'to'):
        processor.tokenizer.to(device)
    
    converted_data = []
    
    # Prepare text data
    text_data = []
    metadata = []
    for item in chunk_data:
        # Format conversation
        text = ""
        for turn in item["conversations"]:
            if turn["from"] == "human":
                text += f"<|im_start|>user\n{turn['value']}<|im_end|>\n"
            elif turn["from"] == "gpt":
                text += f"<|im_start|>assistant\n{turn['value']}<|im_end|>\n"
        
        text_data.append(text)
        metadata.append({
            "image_path": item.get("image", ""),
            "video_path": item.get("video", ""),
            "id": item.get("id", "unknown")
        })
    
    # Process in batches
    for i in range(0, len(text_data), batch_size):
        batch_texts = text_data[i:i+batch_size]
        batch_metadata = metadata[i:i+batch_size]
        
        # Batch tokenize
        tokens = processor.tokenizer(
            batch_texts,
            truncation=True,
            max_length=2048,
            padding=True,
            return_tensors="pt"
        )
        
        # Process each item in the batch
        for j, (text, meta) in enumerate(zip(batch_texts, batch_metadata)):
            converted_item = {
                "input_ids": tokens["input_ids"][j].tolist(),
                "text": text,
                "image_path": meta["image_path"],
                "video_path": meta["video_path"],
                "id": meta["id"]
            }
            converted_data.append(converted_item)
    
    return converted_data

def convert_smolvlm2_to_nanotron_streaming(input_file: str, output_file: str, processor=None, device="cpu", batch_size=32, chunk_size=1000):
    """Convert SmolVLM2 dataset format to Nanotron format using streaming (memory efficient)"""
    
    # Load processor if not provided
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            trust_remote_code=True
        )
    
    # Move tokenizer to specified device
    if hasattr(processor.tokenizer, 'to'):
        processor.tokenizer.to(device)
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} items in streaming mode with chunk size {chunk_size}")
    
    # Prepare output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    converted_data = []
    
    # Process in chunks to save memory
    for chunk_start in tqdm(range(0, len(data), chunk_size), desc=f"Processing chunks from {os.path.basename(input_file)}"):
        chunk_end = min(chunk_start + chunk_size, len(data))
        chunk_data = data[chunk_start:chunk_end]
        
        # Prepare text data for this chunk
        text_data = []
        metadata = []
        for item in chunk_data:
            # Format conversation
            text = ""
            for turn in item["conversations"]:
                if turn["from"] == "human":
                    text += f"<|im_start|>user\n{turn['value']}<|im_end|>\n"
                elif turn["from"] == "gpt":
                    text += f"<|im_start|>assistant\n{turn['value']}<|im_end|>\n"
            
            text_data.append(text)
            metadata.append({
                "image_path": item.get("image", ""),
                "video_path": item.get("video", ""),
                "id": item.get("id", "unknown")
            })
        
        # Process this chunk in batches
        for i in range(0, len(text_data), batch_size):
            batch_texts = text_data[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            
            # Batch tokenize
            tokens = processor.tokenizer(
                batch_texts,
                truncation=True,
                max_length=2048,
                padding=True,
                return_tensors="pt"
            )
            
            # Process each item in the batch
            for j, (text, meta) in enumerate(zip(batch_texts, batch_metadata)):
                converted_item = {
                    "input_ids": tokens["input_ids"][j].tolist(),
                    "text": text,
                    "image_path": meta["image_path"],
                    "video_path": meta["video_path"],
                    "id": meta["id"]
                }
                converted_data.append(converted_item)
    
    # Save all results
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {len(converted_data)} samples to {output_file}")

def convert_smolvlm2_to_nanotron(input_file: str, output_file: str, processor=None, device="cpu", batch_size=32, num_workers=None, stream=False):
    """Convert SmolVLM2 dataset format to Nanotron format"""
    
    # Use streaming mode if requested
    if stream:
        return convert_smolvlm2_to_nanotron_streaming(input_file, output_file, processor, device, batch_size)
    
    # Load processor if not provided
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            trust_remote_code=True
        )
    
    # Move tokenizer to specified device
    if hasattr(processor.tokenizer, 'to'):
        processor.tokenizer.to(device)
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 4)  # Limit to 4 to avoid memory issues
    
    if num_workers > 1 and len(data) > num_workers:
        # Use multiprocessing for large datasets
        chunk_size = math.ceil(len(data) / num_workers)
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        print(f"Processing {len(data)} items using {num_workers} workers in {len(chunks)} chunks")
        
        # Process chunks in parallel
        process_func = partial(process_data_chunk, 
                              processor_name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
                              device=device, 
                              batch_size=batch_size)
        
        with Pool(processes=num_workers) as pool:
            chunk_results = list(tqdm(pool.imap(process_func, chunks), 
                                    total=len(chunks), 
                                    desc=f"Converting {os.path.basename(input_file)}"))
        
        # Flatten results
        converted_data = []
        for chunk_result in chunk_results:
            converted_data.extend(chunk_result)
    else:
        # Single process fallback
        print(f"Processing {len(data)} items in single process")
        converted_data = []
        
        # Prepare text data first
        text_data = []
        metadata = []
        for item in data:
            # Format conversation
            text = ""
            for turn in item["conversations"]:
                if turn["from"] == "human":
                    text += f"<|im_start|>user\n{turn['value']}<|im_end|>\n"
                elif turn["from"] == "gpt":
                    text += f"<|im_start|>assistant\n{turn['value']}<|im_end|>\n"
            
            text_data.append(text)
            metadata.append({
                "image_path": item.get("image", ""),
                "video_path": item.get("video", ""),
                "id": item.get("id", "unknown")
            })
        
        # Process in batches
        for i in tqdm(range(0, len(text_data), batch_size), desc=f"Converting {os.path.basename(input_file)}"):
            batch_texts = text_data[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            
            # Batch tokenize
            tokens = processor.tokenizer(
                batch_texts,
                truncation=True,
                max_length=2048,
                padding=True,
                return_tensors="pt"
            )
            
            # Process each item in the batch
            for j, (text, meta) in enumerate(zip(batch_texts, batch_metadata)):
                converted_item = {
                    "input_ids": tokens["input_ids"][j].tolist(),
                    "text": text,
                    "image_path": meta["image_path"],
                    "video_path": meta["video_path"],
                    "id": meta["id"]
                }
                converted_data.append(converted_item)
    
    # Save in Nanotron format
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {len(converted_data)} samples to {output_file}")

def convert_jsonl_to_nanotron(input_file: str, output_file: str, processor=None, device="cpu", batch_size=32, num_workers=None, stream=False):
    """Convert JSONL dataset format to Nanotron format"""
    
    # Load processor if not provided
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            trust_remote_code=True
        )
    
    # Move tokenizer to specified device
    if hasattr(processor.tokenizer, 'to'):
        processor.tokenizer.to(device)
    
    # Load JSONL file line by line
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 4)  # Limit to 4 to avoid memory issues
    
    if num_workers > 1 and len(data) > num_workers:
        # Use multiprocessing for large datasets
        chunk_size = math.ceil(len(data) / num_workers)
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        print(f"Processing {len(data)} items using {num_workers} workers in {len(chunks)} chunks")
        
        # Process chunks in parallel
        process_func = partial(process_data_chunk, 
                              processor_name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
                              device=device, 
                              batch_size=batch_size)
        
        with Pool(processes=num_workers) as pool:
            chunk_results = list(tqdm(pool.imap(process_func, chunks), 
                                    total=len(chunks), 
                                    desc=f"Converting {os.path.basename(input_file)}"))
        
        # Flatten results
        converted_data = []
        for chunk_result in chunk_results:
            converted_data.extend(chunk_result)
    else:
        # Single process fallback
        print(f"Processing {len(data)} items in single process")
        converted_data = []
        
        # Prepare text data first
        text_data = []
        metadata = []
        for item in data:
            # Format conversation
            text = ""
            for turn in item["conversations"]:
                if turn["from"] == "human":
                    text += f"<|im_start|>user\n{turn['value']}<|im_end|>\n"
                elif turn["from"] == "gpt":
                    text += f"<|im_start|>assistant\n{turn['value']}<|im_end|>\n"
            
            text_data.append(text)
            metadata.append({
                "image_path": item.get("image", ""),
                "video_path": item.get("video", ""),
                "id": item.get("id", "unknown")
            })
        
        # Process in batches
        for i in tqdm(range(0, len(text_data), batch_size), desc=f"Converting {os.path.basename(input_file)}"):
            batch_texts = text_data[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            
            # Batch tokenize
            tokens = processor.tokenizer(
                batch_texts,
                truncation=True,
                max_length=2048,
                padding=True,
                return_tensors="pt"
            )
            
            # Process each item in the batch
            for j, (text, meta) in enumerate(zip(batch_texts, batch_metadata)):
                converted_item = {
                    "input_ids": tokens["input_ids"][j].tolist(),
                    "text": text,
                    "image_path": meta["image_path"],
                    "video_path": meta["video_path"],
                    "id": meta["id"]
                }
                converted_data.append(converted_item)
    
    # Save in Nanotron format
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {len(converted_data)} samples to {output_file}")

def convert_all_datasets(input_dir: str, output_dir: str, device="cpu", batch_size=32, num_workers=None, stream=False):
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
        
        convert_smolvlm2_to_nanotron(input_path, output_path, processor, device, batch_size, num_workers, stream)
    
    print(f"\nConversion complete! Nanotron format files saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert SmolVLM2 data to Nanotron format")
    parser.add_argument("--input", type=str, help="Input file or directory")
    parser.add_argument("--output", type=str, help="Output file or directory")
    parser.add_argument("--input_dir", type=str, default="data/datasets", 
                        help="Input directory (if --input not specified)")
    parser.add_argument("--output_dir", type=str, default="data/datasets_nanotron", 
                        help="Output directory (if --output not specified)")
    
    # Optimization options
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                        help="Device to use for tokenization (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for tokenization (default: 32)")
    parser.add_argument("--num_workers", type=int, default=None, 
                        help="Number of multiprocessing workers (default: auto)")
    parser.add_argument("--disable_multiprocessing", action="store_true", 
                        help="Disable multiprocessing (use single process)")
    parser.add_argument("--stream", action="store_true", 
                        help="Enable memory-efficient streaming for large datasets")
    
    args = parser.parse_args()
    
    # Handle multiprocessing settings
    num_workers = args.num_workers
    if args.disable_multiprocessing:
        num_workers = 1
    
    # GPU availability check
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers: {num_workers if num_workers else 'auto'}")
    print(f"  Streaming: {args.stream}")
    
    if args.input and args.output:
        # Convert single file based on extension
        if args.input.endswith('.jsonl'):
            convert_jsonl_to_nanotron(args.input, args.output, device=args.device, 
                                    batch_size=args.batch_size, num_workers=num_workers, stream=args.stream)
        else:
            convert_smolvlm2_to_nanotron(args.input, args.output, device=args.device, 
                                       batch_size=args.batch_size, num_workers=num_workers, stream=args.stream)
    else:
        # Convert all files in directory
        convert_all_datasets(args.input_dir, args.output_dir, device=args.device, 
                           batch_size=args.batch_size, num_workers=num_workers, stream=args.stream)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Count total tokens in training data"""

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import glob

def count_tokens_in_file(file_path):
    """Count tokens in a single nanotron JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        total_tokens = 0
        valid_samples = 0
        
        for sample in data:
            if 'input_ids' in sample:
                # Count non-padding tokens (assuming padding token is 0)
                input_ids = sample['input_ids']
                if isinstance(input_ids, list):
                    # Remove padding tokens (0) from count
                    non_padding_tokens = [token for token in input_ids if token != 0]
                    total_tokens += len(non_padding_tokens)
                    valid_samples += 1
        
        return total_tokens, valid_samples, len(data)
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, 0, 0

def count_tokens_in_directory(directory):
    """Count tokens in all nanotron JSON files in directory"""
    json_files = glob.glob(os.path.join(directory, "*_nanotron.json"))
    
    if not json_files:
        print(f"No *_nanotron.json files found in {directory}")
        return
    
    print(f"Found {len(json_files)} nanotron files to analyze")
    
    total_tokens = 0
    total_valid_samples = 0
    total_samples = 0
    file_stats = []
    
    for file_path in tqdm(json_files, desc="Counting tokens"):
        file_tokens, valid_samples, total_file_samples = count_tokens_in_file(file_path)
        
        file_name = os.path.basename(file_path)
        file_stats.append({
            'file': file_name,
            'tokens': file_tokens,
            'valid_samples': valid_samples,
            'total_samples': total_file_samples,
            'avg_tokens_per_sample': file_tokens / valid_samples if valid_samples > 0 else 0
        })
        
        total_tokens += file_tokens
        total_valid_samples += valid_samples
        total_samples += total_file_samples
    
    # Print detailed results
    print("\n" + "="*80)
    print("TOKEN COUNT ANALYSIS")
    print("="*80)
    
    print(f"\nOverall Statistics:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total valid samples: {total_valid_samples:,}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Average tokens per sample: {total_tokens / total_valid_samples:.1f}" if total_valid_samples > 0 else "  No valid samples")
    
    # Tokens in billions
    tokens_in_billions = total_tokens / 1_000_000_000
    print(f"  Total tokens: {tokens_in_billions:.2f}B tokens")
    
    print(f"\nPer-file breakdown:")
    print(f"{'File':<40} {'Tokens':<12} {'Samples':<8} {'Avg/Sample':<12}")
    print("-" * 80)
    
    for stat in sorted(file_stats, key=lambda x: x['tokens'], reverse=True):
        print(f"{stat['file']:<40} {stat['tokens']:>11,} {stat['valid_samples']:>7} {stat['avg_tokens_per_sample']:>11.1f}")
    
    # Estimate training compute
    print(f"\nTraining Estimates:")
    print(f"  1 epoch = {total_tokens:,} tokens")
    print(f"  3 epochs = {total_tokens * 3:,} tokens ({tokens_in_billions * 3:.2f}B)")
    
    return total_tokens, total_valid_samples

def count_tokens_with_epochs_and_steps(directory, epochs=1, batch_size=4, max_steps=None):
    """Count tokens considering training configuration"""
    total_tokens, total_samples = count_tokens_in_directory(directory)
    
    if total_tokens == 0:
        return
    
    # Calculate steps per epoch
    steps_per_epoch = total_samples // batch_size
    
    if max_steps and max_steps > 0:
        # Limited by max_steps
        actual_steps = max_steps
        actual_epochs = max_steps / steps_per_epoch
        tokens_trained = min(total_tokens * actual_epochs, total_tokens * epochs)
    else:
        # Limited by epochs
        actual_steps = int(steps_per_epoch * epochs)
        actual_epochs = epochs
        tokens_trained = total_tokens * epochs
    
    print(f"\nTraining Configuration Analysis:")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Max steps: {max_steps if max_steps else 'None (limited by epochs)'}")
    print(f"  Target epochs: {epochs}")
    print(f"  Actual steps: {actual_steps:,}")
    print(f"  Actual epochs: {actual_epochs:.2f}")
    print(f"  Tokens during training: {tokens_trained:,} ({tokens_trained / 1_000_000_000:.2f}B)")

def main():
    parser = argparse.ArgumentParser(description="Count tokens in training data")
    parser.add_argument("--data_dir", type=str, default="data/datasets_nanotron",
                        help="Directory containing *_nanotron.json files")
    parser.add_argument("--file", type=str, help="Count tokens in specific file")
    parser.add_argument("--epochs", type=float, default=1,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size (default: 4)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum training steps (overrides epochs)")
    
    args = parser.parse_args()
    
    if args.file:
        # Count tokens in single file
        total_tokens, valid_samples, total_samples = count_tokens_in_file(args.file)
        print(f"File: {args.file}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Valid samples: {valid_samples:,}")
        print(f"Average tokens per sample: {total_tokens / valid_samples:.1f}" if valid_samples > 0 else "No valid samples")
    else:
        # Count tokens in directory with training analysis
        count_tokens_with_epochs_and_steps(args.data_dir, args.epochs, args.batch_size, args.max_steps)

if __name__ == "__main__":
    main()
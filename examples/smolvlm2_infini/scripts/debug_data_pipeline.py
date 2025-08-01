#!/usr/bin/env python3
"""Debug tool for SmolVLM2-Infini data pipeline

This script provides quick debugging capabilities for:
- Individual dataset inspection
- Sample visualization
- Data format validation
- Quick statistics

Usage:
    # Inspect a specific dataset
    python debug_data_pipeline.py --dataset data/datasets/magpie_pro_l3_80b_mt.json --inspect
    
    # Show statistics for all datasets
    python debug_data_pipeline.py --data_dir data/datasets --stats
    
    # Debug specific sample
    python debug_data_pipeline.py --dataset data/datasets/llava_video_1_2m.json --sample_id 0
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import pandas as pd
from tabulate import tabulate

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))


class DataPipelineDebugger:
    """Quick debugging tool for data pipeline issues"""
    
    def __init__(self):
        self.stats = defaultdict(lambda: defaultdict(int))
    
    def inspect_dataset(self, dataset_path: str, num_samples: int = 5):
        """Inspect a specific dataset file"""
        print(f"\n{'='*80}")
        print(f"Inspecting dataset: {dataset_path}")
        print(f"{'='*80}\n")
        
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            print(f"📊 Dataset Statistics:")
            print(f"  - Total samples: {len(data)}")
            print(f"  - File size: {Path(dataset_path).stat().st_size / 1024 / 1024:.2f} MB")
            
            # Analyze sample structure
            if data:
                sample = data[0]
                print(f"\n📋 Sample Structure:")
                self._print_sample_structure(sample)
                
                # Show first few samples
                print(f"\n📝 First {min(num_samples, len(data))} Samples:")
                for i in range(min(num_samples, len(data))):
                    print(f"\n--- Sample {i} ---")
                    self._print_sample_summary(data[i])
            
            # Analyze modality distribution
            modalities = self._analyze_modalities(data)
            print(f"\n📈 Modality Distribution:")
            for modality, count in modalities.items():
                percentage = (count / len(data)) * 100
                print(f"  - {modality}: {count} ({percentage:.1f}%)")
            
            # Check for common issues
            issues = self._check_common_issues(data)
            if issues:
                print(f"\n⚠️  Potential Issues Found:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print(f"\n✅ No common issues detected")
                
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
    
    def _print_sample_structure(self, sample: Dict[str, Any], indent: int = 0):
        """Print the structure of a sample"""
        for key, value in sample.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}: dict")
                self._print_sample_structure(value, indent + 1)
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    print(f"{prefix}{key}: list[dict] (length: {len(value)})")
                    if key == "conversations":
                        for i, turn in enumerate(value):
                            print(f"{prefix}  [{i}] {turn.get('from', 'unknown')}: {turn.get('value', '')[:50]}...")
                else:
                    print(f"{prefix}{key}: list (length: {len(value)})")
            elif isinstance(value, str):
                if len(value) > 50:
                    print(f"{prefix}{key}: str ('{value[:50]}...')")
                else:
                    print(f"{prefix}{key}: str ('{value}')")
            else:
                print(f"{prefix}{key}: {type(value).__name__} ({value})")
    
    def _print_sample_summary(self, sample: Dict[str, Any]):
        """Print a concise summary of a sample"""
        print(f"  ID: {sample.get('id', 'N/A')}")
        
        # Determine modality
        if 'video' in sample:
            print(f"  Modality: Video ({sample['video']})")
        elif 'image' in sample:
            if isinstance(sample['image'], list):
                print(f"  Modality: Multi-image ({len(sample['image'])} images)")
            else:
                print(f"  Modality: Image ({sample['image']})")
        else:
            print(f"  Modality: Text-only")
        
        # Show conversation summary
        conversations = sample.get('conversations', [])
        print(f"  Conversation turns: {len(conversations)}")
        if conversations:
            # Show first user message
            user_turn = next((t for t in conversations if t.get('from') in ['human', 'user']), None)
            if user_turn:
                print(f"  User: {user_turn['value'][:80]}...")
            
            # Show first assistant response
            assistant_turn = next((t for t in conversations if t.get('from') in ['gpt', 'assistant']), None)
            if assistant_turn:
                print(f"  Assistant: {assistant_turn['value'][:80]}...")
    
    def _analyze_modalities(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze modality distribution in dataset"""
        modalities = defaultdict(int)
        
        for sample in data:
            if 'video' in sample:
                modalities['video'] += 1
            elif 'image' in sample:
                if isinstance(sample['image'], list):
                    modalities['multi-image'] += 1
                else:
                    modalities['image'] += 1
            else:
                modalities['text'] += 1
        
        return dict(modalities)
    
    def _check_common_issues(self, data: List[Dict[str, Any]]) -> List[str]:
        """Check for common data issues"""
        issues = []
        
        # Check for empty conversations
        empty_conversations = sum(1 for s in data if not s.get('conversations'))
        if empty_conversations > 0:
            issues.append(f"{empty_conversations} samples with empty conversations")
        
        # Check for missing IDs
        missing_ids = sum(1 for s in data if not s.get('id'))
        if missing_ids > 0:
            issues.append(f"{missing_ids} samples missing ID")
        
        # Check for invalid conversation format
        invalid_format = 0
        for sample in data:
            for turn in sample.get('conversations', []):
                if 'from' not in turn or 'value' not in turn:
                    invalid_format += 1
                    break
        if invalid_format > 0:
            issues.append(f"{invalid_format} samples with invalid conversation format")
        
        # Check for suspicious file paths
        suspicious_paths = 0
        for sample in data:
            if 'image' in sample:
                if isinstance(sample['image'], str) and sample['image'].startswith('/'):
                    suspicious_paths += 1
            if 'video' in sample:
                if isinstance(sample['video'], str) and sample['video'].startswith('/'):
                    suspicious_paths += 1
        if suspicious_paths > 0:
            issues.append(f"{suspicious_paths} samples with absolute paths (should be relative)")
        
        return issues
    
    def show_dataset_stats(self, data_dir: str):
        """Show statistics for all datasets in directory"""
        print(f"\n{'='*80}")
        print(f"Dataset Statistics for: {data_dir}")
        print(f"{'='*80}\n")
        
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"❌ Directory not found: {data_dir}")
            return
        
        json_files = list(data_path.glob("*.json"))
        if not json_files:
            print(f"❌ No JSON files found in {data_dir}")
            return
        
        # Collect statistics
        dataset_stats = []
        total_samples = 0
        modality_totals = defaultdict(int)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                num_samples = len(data)
                total_samples += num_samples
                
                # Get modality distribution
                modalities = self._analyze_modalities(data)
                
                # Get file size
                file_size_mb = json_file.stat().st_size / 1024 / 1024
                
                dataset_stats.append({
                    'Dataset': json_file.stem,
                    'Samples': num_samples,
                    'Size (MB)': f"{file_size_mb:.2f}",
                    'Text': modalities.get('text', 0),
                    'Image': modalities.get('image', 0),
                    'Multi-Image': modalities.get('multi-image', 0),
                    'Video': modalities.get('video', 0),
                })
                
                # Update totals
                for mod, count in modalities.items():
                    modality_totals[mod] += count
                    
            except Exception as e:
                dataset_stats.append({
                    'Dataset': json_file.stem,
                    'Samples': 'ERROR',
                    'Size (MB)': 'ERROR',
                    'Text': '-',
                    'Image': '-',
                    'Multi-Image': '-',
                    'Video': '-',
                })
                print(f"⚠️  Error loading {json_file.name}: {str(e)}")
        
        # Display table
        print(tabulate(dataset_stats, headers='keys', tablefmt='grid'))
        
        # Display summary
        print(f"\n📊 Summary:")
        print(f"  Total datasets: {len(json_files)}")
        print(f"  Total samples: {total_samples:,}")
        print(f"\n📈 Modality Totals:")
        for modality, count in sorted(modality_totals.items()):
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"  - {modality}: {count:,} ({percentage:.1f}%)")
    
    def debug_specific_sample(self, dataset_path: str, sample_id: int):
        """Debug a specific sample in detail"""
        print(f"\n{'='*80}")
        print(f"Debugging Sample {sample_id} from {dataset_path}")
        print(f"{'='*80}\n")
        
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            if sample_id >= len(data):
                print(f"❌ Sample ID {sample_id} out of range (dataset has {len(data)} samples)")
                return
            
            sample = data[sample_id]
            
            # Pretty print the entire sample
            print("📄 Full Sample Data:")
            print(json.dumps(sample, indent=2))
            
            # Analyze conversation flow
            print(f"\n💬 Conversation Analysis:")
            for i, turn in enumerate(sample.get('conversations', [])):
                speaker = turn.get('from', 'unknown')
                content = turn.get('value', '')
                
                print(f"\nTurn {i} ({speaker}):")
                print(f"  Length: {len(content)} characters")
                
                # Check for special tokens
                special_tokens = ['<image>', '<video>', '<|im_start|>', '<|im_end|>']
                found_tokens = [token for token in special_tokens if token in content]
                if found_tokens:
                    print(f"  Special tokens: {found_tokens}")
                
                # Show content preview
                print(f"  Content: {content[:200]}...")
                if len(content) > 200:
                    print(f"  ... (truncated, {len(content) - 200} more characters)")
            
            # Check media references
            if 'image' in sample or 'video' in sample:
                print(f"\n🖼️ Media References:")
                if 'image' in sample:
                    if isinstance(sample['image'], list):
                        print(f"  Images ({len(sample['image'])}):")
                        for img in sample['image']:
                            print(f"    - {img}")
                    else:
                        print(f"  Image: {sample['image']}")
                
                if 'video' in sample:
                    print(f"  Video: {sample['video']}")
            
        except Exception as e:
            print(f"❌ Error debugging sample: {str(e)}")
    
    def validate_mixture_config(self, mixture_path: str):
        """Validate data mixture configuration"""
        print(f"\n{'='*80}")
        print(f"Validating Mixture Config: {mixture_path}")
        print(f"{'='*80}\n")
        
        try:
            with open(mixture_path, 'r') as f:
                config = yaml.safe_load(f)
            
            total_datasets = 0
            missing_files = []
            
            print("📋 Mixture Configuration:")
            for modality, datasets in config.items():
                print(f"\n{modality.upper()} ({len(datasets)} datasets):")
                
                for dataset in datasets:
                    total_datasets += 1
                    name = dataset.get('name', 'unknown')
                    json_path = dataset.get('json_path', '')
                    
                    # Check if file exists
                    if not Path(json_path).is_absolute():
                        json_path = Path(mixture_path).parent / json_path
                    
                    exists = Path(json_path).exists()
                    status = "✅" if exists else "❌"
                    
                    print(f"  {status} {name}")
                    print(f"     Path: {dataset.get('json_path', '')}")
                    print(f"     Strategy: {dataset.get('sampling_strategy', 'unknown')}")
                    
                    if not exists:
                        missing_files.append(dataset.get('json_path', ''))
            
            print(f"\n📊 Summary:")
            print(f"  Total datasets: {total_datasets}")
            print(f"  Missing files: {len(missing_files)}")
            
            if missing_files:
                print(f"\n❌ Missing Files:")
                for file in missing_files:
                    print(f"  - {file}")
            else:
                print(f"\n✅ All dataset files found!")
                
        except Exception as e:
            print(f"❌ Error validating mixture config: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Debug SmolVLM2-Infini data pipeline")
    parser.add_argument("--dataset", help="Path to specific dataset JSON file")
    parser.add_argument("--data_dir", default="data/datasets", help="Directory containing datasets")
    parser.add_argument("--mixture", default="data/smolvlm2_256m_mixture.yaml", help="Mixture config path")
    parser.add_argument("--inspect", action="store_true", help="Inspect dataset structure")
    parser.add_argument("--stats", action="store_true", help="Show statistics for all datasets")
    parser.add_argument("--sample_id", type=int, help="Debug specific sample by ID")
    parser.add_argument("--validate_mixture", action="store_true", help="Validate mixture configuration")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to show")
    
    args = parser.parse_args()
    
    debugger = DataPipelineDebugger()
    
    if args.inspect and args.dataset:
        debugger.inspect_dataset(args.dataset, args.num_samples)
    elif args.stats:
        debugger.show_dataset_stats(args.data_dir)
    elif args.sample_id is not None and args.dataset:
        debugger.debug_specific_sample(args.dataset, args.sample_id)
    elif args.validate_mixture:
        debugger.validate_mixture_config(args.mixture)
    else:
        print("Please specify an action:")
        print("  --inspect with --dataset: Inspect a specific dataset")
        print("  --stats: Show statistics for all datasets")
        print("  --sample_id with --dataset: Debug a specific sample")
        print("  --validate_mixture: Validate mixture configuration")


if __name__ == "__main__":
    main()
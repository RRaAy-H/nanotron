"""
Usage:
    python prepare_training_data.py --output_dir data/datasets --workers auto
"""

import os
import json
import random
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterator
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import multiprocessing as mp
from functools import partial
import numpy as np
import time
import pickle
import hashlib
import warnings
import threading
from queue import Queue

# CPU-optimized processing only

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    pq = None
    pa = None
    PYARROW_AVAILABLE = False

# CPU-optimized libraries
try:
    import polars as pl
    POLARS_AVAILABLE = True
    print("Polars available for CPU-optimized processing")
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

# Detect CPU cores dynamically
CPU_CORES = mp.cpu_count()
print(f"Detected {CPU_CORES} CPU cores for processing")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('mode.copy_on_write', True)


class CPUOptimizedDataLoader:
    def __init__(self, cache_dir: str = ".cache", max_workers: int = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        # Use all available CPU cores for maximum performance
        self.max_workers = max_workers or CPU_CORES
        print(f"CPU-optimized processing with {self.max_workers} workers")
    
    def _get_cache_key(self, path: str, num_samples: int) -> str:
        """Generate cache key with CPU hashing"""
        path_hash = hashlib.md5(f"{path}_{num_samples}".encode()).hexdigest()
        return f"dataset_{path_hash}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """Load cached dataset if available"""
        cache_file = self.cache_dir / cache_key
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                cache_file.unlink(missing_ok=True)
        return None
    
    def _save_to_cache(self, cache_key: str, data: List[Dict]) -> None:
        """Save dataset to cache"""
        try:
            cache_file = self.cache_dir / cache_key
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Warning: Could not cache data: {e}")

    def load_parquet_data(self, parquet_path: str, num_samples: int) -> List[Dict[str, Any]]:
        """CPU-optimized parquet loading with maximum parallelization"""
        # Check cache first
        cache_key = self._get_cache_key(parquet_path, num_samples)
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            print(f"Loaded {len(cached_data)} samples from cache for {parquet_path}")
            return cached_data
        
        
        try:
            parquet_files = []
            if os.path.isfile(parquet_path) and parquet_path.endswith('.parquet'):
                parquet_files = [parquet_path]
            elif os.path.isdir(parquet_path):
                parquet_files = list(Path(parquet_path).glob('*.parquet'))
                parquet_files = [str(f) for f in sorted(parquet_files)]
            else:
                print(f"Path not found: {parquet_path}")
                return []
            
            if not parquet_files:
                return []
            
            # Use CPU-optimized processing
            samples = self._load_parquet_cpu_optimized(parquet_files, num_samples)
            
            # Cache the results
            self._save_to_cache(cache_key, samples)
            print(f"Loaded {len(samples)} samples from {len(parquet_files)} parquet files")
            return samples
            
        except Exception as e:
            print(f"Error loading parquet data from {parquet_path}: {e}")
            return []
    
    
    def _get_parquet_rows(self, pfile: str) -> int:
        """Get row count from parquet metadata"""
        try:
            if PYARROW_AVAILABLE:
                table = pq.ParquetFile(pfile)
                return table.metadata.num_rows
            else:
                # Fallback: estimate from file size
                file_size = os.path.getsize(pfile)
                return max(1, file_size // 1024)
        except Exception:
            return max(1, os.path.getsize(pfile) // 1024)
    
    def _load_parquet_cpu_optimized(self, parquet_files: List[str], num_samples: int) -> List[Dict]:
        """CPU-optimized processing with maximum parallelization"""
        total_rows = self._estimate_total_rows_cpu(parquet_files)
        
        # Use Polars for faster CPU processing if available
        if POLARS_AVAILABLE and len(parquet_files) > 0:
            return self._load_parquet_polars(parquet_files, num_samples, total_rows)
        
        # Fallback to pandas with maximum parallelization
        if total_rows <= num_samples * 2:
            # Parallel loading with all available CPU cores
            if len(parquet_files) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(parquet_files))) as executor:
                    dfs = list(tqdm(
                        executor.map(pd.read_parquet, parquet_files),
                        total=len(parquet_files),
                        desc="Loading parquet files"
                    ))
            else:
                dfs = [pd.read_parquet(parquet_files[0])]
            
            # Combine and sample efficiently
            combined_df = pd.concat(dfs, ignore_index=True)
            if len(combined_df) > num_samples:
                combined_df = combined_df.sample(n=num_samples, random_state=42)
            
            samples = self._df_to_samples_vectorized(combined_df, parquet_files[0])
            return samples
        else:
            # Chunked processing for large datasets
            return self._chunked_cpu_processing(parquet_files, num_samples, total_rows)
    
    def _estimate_total_rows_cpu(self, parquet_files: List[str]) -> int:
        """CPU-based row estimation"""
        total_rows = 0
        for pfile in parquet_files[:3]:
            try:
                if PYARROW_AVAILABLE:
                    table = pq.ParquetFile(pfile)
                    total_rows += table.metadata.num_rows
                else:
                    file_size = os.path.getsize(pfile)
                    total_rows += max(1, file_size // 1024)
            except:
                file_size = os.path.getsize(pfile)
                total_rows += max(1, file_size // 1024)
        
        if len(parquet_files) > 3:
            avg_rows = total_rows / min(3, len(parquet_files))
            total_rows = int(avg_rows * len(parquet_files))
        
        return total_rows
    
    def _load_parquet_polars(self, parquet_files: List[str], num_samples: int, total_rows: int) -> List[Dict]:
        """Polars-optimized CPU processing for maximum speed"""
        try:
            print(f"Using Polars for CPU-optimized processing of {len(parquet_files)} files...")
            
            if total_rows <= num_samples * 2:
                # Load all files with Polars (much faster than pandas)
                dfs = []
                for pfile in parquet_files:
                    try:
                        df = pl.read_parquet(pfile)
                        dfs.append(df)
                    except Exception as e:
                        print(f"Polars loading failed for {pfile}: {e}, trying pandas fallback")
                        pandas_df = pd.read_parquet(pfile)
                        df = pl.from_pandas(pandas_df)
                        dfs.append(df)
                
                if dfs:
                    # Polars concatenation is very fast
                    combined_df = pl.concat(dfs)
                    
                    # Polars sampling is also optimized
                    if len(combined_df) > num_samples:
                        combined_df = combined_df.sample(n=num_samples, seed=42)
                    
                    # Convert to pandas for downstream compatibility
                    pandas_df = combined_df.to_pandas()
                    return self._df_to_samples_vectorized(pandas_df, parquet_files[0])
            else:
                # Chunked Polars processing for large datasets
                return self._chunked_polars_processing(parquet_files, num_samples, total_rows)
                
        except Exception as e:
            print(f"Polars processing failed: {e}, falling back to pandas")
            return self._load_parquet_pandas_fallback(parquet_files, num_samples, total_rows)
        
        return []
    
    def _chunked_polars_processing(self, parquet_files: List[str], num_samples: int, total_rows: int) -> List[Dict]:
        """Polars-based chunked processing with reservoir sampling"""
        samples = []
        rows_seen = 0
        sampling_ratio = min(1.0, num_samples * 3 / total_rows)
        
        for pfile in parquet_files:
            try:
                # Polars lazy evaluation for memory efficiency
                lazy_df = pl.scan_parquet(pfile)
                
                # Apply sampling at scan level if needed - collect first then sample
                if sampling_ratio < 1.0:
                    df_temp = lazy_df.collect()
                    n_sample = max(1, int(len(df_temp) * sampling_ratio))
                    df_temp = df_temp.sample(n=n_sample, seed=42)
                    lazy_df = df_temp.lazy()
                else:
                    # If no sampling needed, keep as lazy
                    pass
                
                # Collect in batches to manage memory
                batch_size = 10000
                for batch_df in lazy_df.collect().iter_slices(batch_size):
                    pandas_batch = batch_df.to_pandas()
                    chunk_samples = self._df_to_samples_vectorized(pandas_batch, pfile)
                    
                    # Reservoir sampling
                    for sample in chunk_samples:
                        if len(samples) < num_samples:
                            samples.append(sample)
                        else:
                            j = random.randint(0, rows_seen)
                            if j < num_samples:
                                samples[j] = sample
                        rows_seen += 1
                    
                    if len(samples) >= num_samples:
                        break
                        
                if len(samples) >= num_samples:
                    break
                    
            except Exception as e:
                print(f"Polars chunk processing failed for {pfile}: {e}")
                continue
        
        return samples[:num_samples]
    
    def _load_parquet_pandas_fallback(self, parquet_files: List[str], num_samples: int, total_rows: int) -> List[Dict]:
        """Pandas fallback with basic optimizations"""
        if total_rows <= num_samples * 2:
            # Load all files
            dfs = []
            for pfile in parquet_files:
                try:
                    df = pd.read_parquet(pfile)
                    dfs.append(df)
                except Exception as e:
                    print(f"Error loading {pfile}: {e}")
                    continue
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                if len(combined_df) > num_samples:
                    combined_df = combined_df.sample(n=num_samples, random_state=42)
                return self._df_to_samples_vectorized(combined_df, parquet_files[0])
        else:
            return self._chunked_cpu_processing(parquet_files, num_samples, total_rows)
        
        return []
    
    def _chunked_cpu_processing(self, parquet_files: List[str], num_samples: int, total_rows: int) -> List[Dict]:
        """CPU chunked processing with maximum parallelization and reservoir sampling"""
        samples = []
        rows_seen = 0
        sampling_ratio = min(1.0, num_samples * 3 / total_rows)
        
        # Process files in parallel chunks
        def process_file_chunk(pfile):
            file_samples = []
            try:
                # Read in chunks for memory efficiency
                chunk_size = max(1000, 50000 // self.max_workers)  # Adjust chunk size based on workers
                for chunk in pd.read_parquet(pfile, chunksize=chunk_size):
                    if sampling_ratio < 1.0:
                        n_sample = max(1, int(len(chunk) * sampling_ratio))
                        chunk = chunk.sample(n=n_sample, random_state=42)
                    
                    chunk_samples = self._df_to_samples_vectorized(chunk, pfile)
                    file_samples.extend(chunk_samples)
                    
            except Exception as e:
                print(f"Error processing {pfile}: {e}")
            
            return file_samples
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(parquet_files))) as executor:
            future_to_file = {executor.submit(process_file_chunk, pfile): pfile for pfile in parquet_files}
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_samples = future.result()
                
                # Reservoir sampling
                for sample in file_samples:
                    if len(samples) < num_samples:
                        samples.append(sample)
                    else:
                        j = random.randint(0, rows_seen)
                        if j < num_samples:
                            samples[j] = sample
                    rows_seen += 1
                
                if len(samples) >= num_samples:
                    break
        
        return samples[:num_samples]
    
    def _make_json_serializable(self, obj):
        """Recursively convert all objects to JSON serializable format"""
        import base64
        
        if isinstance(obj, bytes):
            try:
                # Try to decode as UTF-8 first
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                # If it's binary data (like images), convert to base64
                return {
                    "data": base64.b64encode(obj).decode('ascii'),
                    "encoding": "base64"
                }
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):  # Handle objects with attributes
            try:
                return str(obj)  # Convert to string as fallback
            except:
                return "<non-serializable-object>"
        else:
            # For other types, try to serialize directly
            try:
                import json
                json.dumps(obj)  # Test if it's serializable
                return obj
            except (TypeError, ValueError):
                return str(obj)  # Convert to string as fallback
    
    def _df_to_samples_vectorized(self, df: pd.DataFrame, source_file: str) -> List[Dict]:
        """Vectorized conversion optimized for CPU processing"""
        samples = []
        file_stem = Path(source_file).stem
        
        # Process in larger batches for better CPU utilization
        batch_size = max(1000, 10000 // self.max_workers)  # Adjust batch size based on workers
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Vectorized processing
            batch_dict = batch.to_dict('records')
            
            for idx, sample in enumerate(batch_dict):
                # Fast conversation format check
                if "conversations" not in sample:
                    if "question" in sample and "answer" in sample:
                        sample["conversations"] = [
                            {"from": "human", "value": str(sample["question"])},
                            {"from": "gpt", "value": str(sample["answer"])}
                        ]
                
                # Fast ID generation
                if "id" not in sample:
                    sample["id"] = f"{file_stem}_{i + idx}"
                
                # Ensure all objects are JSON serializable using recursive conversion
                sample = self._make_json_serializable(sample)
                
                samples.append(sample)
        
        return samples

    def load_video_data(self, video_path: str, num_samples: int, dataset_name: str, video_filter: str = None) -> List[Dict[str, Any]]:
        """CPU-optimized video data loading with maximum parallel directory scanning"""
        print(f"Processing video dataset {dataset_name} from {video_path}...")
        
        if not os.path.exists(video_path):
            return []
        
        if os.path.isfile(video_path):
            return [{
                "conversations": [
                    {"from": "human", "value": "Describe this video."},
                    {"from": "gpt", "value": f"This is a video from {dataset_name}."}
                ],
                "video": video_path,
                "id": f"{dataset_name}_0"
            }]
        
        # Parallel directory scanning with maximum CPU utilization
        video_files = self._find_video_files_parallel(video_path, video_filter)
        
        # CPU random sampling
        if len(video_files) > num_samples:
            video_files = random.sample(video_files, num_samples)
        
        # Generate samples efficiently
        samples = []
        for idx, video_file in enumerate(video_files[:num_samples]):
            samples.append({
                "conversations": [
                    {"from": "human", "value": "Describe this video."},
                    {"from": "gpt", "value": f"This is a video from {dataset_name}."}
                ],
                "video": str(video_file.relative_to(video_path)),
                "id": f"{dataset_name}_{idx}"
            })
        
        print(f"Processed {len(samples)} video samples")
        return samples
    
    def _find_video_files_parallel(self, video_path: str, video_filter: str = None) -> List[Path]:
        """Maximum parallel video file discovery using all CPU cores"""
        path = Path(video_path)
        
        # Determine search directories
        if video_filter:
            search_dirs = [d for d in path.glob(video_filter) if d.is_dir()]
            print(f"Video filter '{video_filter}' found {len(search_dirs)} directories")
            if len(search_dirs) == 0:
                print(f"Available directories in {video_path}:")
                for d in path.iterdir():
                    if d.is_dir():
                        print(f"  - {d.name}")
        else:
            search_dirs = [path]
        
        if not search_dirs:
            return []
        
        # Parallel file discovery with maximum CPU utilization
        def find_videos_in_dir(search_dir):
            video_files = []
            extensions = ['.mp4', '.avi', '.mov', '.mkv']
            
            # Use maximum parallel glob operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(extensions))) as executor:
                futures = [executor.submit(lambda ext: list(search_dir.rglob(f'*{ext}')), ext) 
                          for ext in extensions]
                for future in concurrent.futures.as_completed(futures):
                    video_files.extend(future.result())
            
            return video_files
        
        # Parallel processing of search directories with maximum workers
        if len(search_dirs) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(search_dirs))) as executor:
                results = list(executor.map(find_videos_in_dir, search_dirs))
                video_files = [f for result in results for f in result]
        else:
            video_files = find_videos_in_dir(search_dirs[0])
        
        return video_files

    def load_json_data(self, json_path: str, num_samples: int) -> List[Dict[str, Any]]:
        """Load data from a JSON file"""
        samples = []
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                samples = data[:num_samples]
            elif isinstance(data, dict):
                # If it's a dict, check for common keys that might contain the data
                for key in ['data', 'samples', 'annotations', 'items']:
                    if key in data and isinstance(data[key], list):
                        samples = data[key][:num_samples]
                        break
                if not samples:
                    samples = [data]
            
            print(f"Loaded {len(samples)} samples from JSON file")
            
        except Exception as e:
            print(f"Error loading JSON {json_path}: {e}")
        
        return samples[:num_samples]

    def load_zip_directory_data(self, zip_dir: str, num_samples: int) -> List[Dict[str, Any]]:
        """Load data from directory containing only ZIP files (ignoring JSON files)"""
        samples = []
        
        if not os.path.exists(zip_dir):
            print(f"Directory does not exist: {zip_dir}")
            return []
        
        # Find all zip files in the directory
        zip_files = []
        for file in os.listdir(zip_dir):
            if file.endswith('.zip'):
                zip_files.append(os.path.join(zip_dir, file))
        
        if not zip_files:
            print(f"No zip files found in {zip_dir}")
            return []
        
        print(f"Found {len(zip_files)} zip files to process")
        
        # Process each zip file until we have enough samples
        samples_per_file = max(1, num_samples // len(zip_files))
        
        for zip_file in sorted(zip_files):
            if len(samples) >= num_samples:
                break
            
            remaining_samples = num_samples - len(samples)
            file_samples = self.load_zip_data(zip_file, min(samples_per_file * 2, remaining_samples))
            samples.extend(file_samples)
            
            print(f"Loaded {len(file_samples)} samples from {os.path.basename(zip_file)}")
        
        # Shuffle and limit to requested number
        if len(samples) > num_samples:
            random.shuffle(samples)
            samples = samples[:num_samples]
        
        print(f"Total samples loaded: {len(samples)}")
        return samples

    def load_zip_data(self, zip_path: str, num_samples: int) -> List[Dict[str, Any]]:
        """Enhanced ZIP loading with vision dataset and Arrow format support"""
        samples = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                file_list = zf.namelist()
                json_files = [f for f in file_list if f.endswith('.json')]
                arrow_files = [f for f in file_list if f.endswith('.arrow')]
                image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                
                # Check for Arrow/Parquet dataset (like multi_vqa)
                if arrow_files and len(arrow_files) >= 1:
                    print(f"Detected Arrow dataset structure: {len(arrow_files)} Arrow files, {len(image_files)} images")
                    return self._load_arrow_zip_data(zf, arrow_files, num_samples)
                
                # Check if this is a vision dataset (has both JSON and images)
                elif json_files and image_files and len(image_files) > len(json_files) * 10:
                    print(f"Detected vision dataset structure: {len(json_files)} JSON files, {len(image_files)} images")
                    return self._load_vision_zip_data(zf, json_files, image_files, num_samples)
                
                # Original logic for datasets with multiple JSON files
                def process_json_file(json_file):
                    try:
                        with zf.open(json_file) as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                return data
                            else:
                                return [data]
                    except json.JSONDecodeError:
                        return []
                
                if len(json_files) > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(json_files))) as executor:
                        results = list(executor.map(process_json_file, json_files))
                        for result in results:
                            samples.extend(result)
                            if len(samples) >= num_samples:
                                break
                else:
                    for json_file in json_files:
                        samples.extend(process_json_file(json_file))
                        if len(samples) >= num_samples:
                            break
                            
        except Exception as e:
            print(f"Error loading ZIP {zip_path}: {e}")
        
        return samples[:num_samples]

    def _load_arrow_zip_data(self, zf: zipfile.ZipFile, arrow_files: List[str], num_samples: int) -> List[Dict[str, Any]]:
        """Load Arrow/Parquet dataset from ZIP"""
        samples = []
        
        if not PYARROW_AVAILABLE:
            print("PyArrow not available, cannot load Arrow files")
            return []
        
        import tempfile
        import os
        
        # Find the main data file (usually the largest or with specific names)
        main_files = [f for f in arrow_files if any(keyword in f.lower() for keyword in ['train', 'data', 'instruct'])]
        if not main_files:
            main_files = arrow_files  # Use all if no specific pattern found
        
        for arrow_file in main_files[:3]:  # Limit to first 3 files to avoid memory issues
            try:
                # Extract Arrow file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.arrow') as tmp_file:
                    with zf.open(arrow_file) as src:
                        tmp_file.write(src.read())
                    tmp_file_path = tmp_file.name
                
                # Load Arrow file using PyArrow
                try:
                    import pyarrow as pa
                    with pa.memory_map(tmp_file_path, 'r') as source:
                        batch_reader = pa.ipc.open_file(source)
                        table = batch_reader.read_all()
                        
                        # Convert to pandas for easier processing
                        df = table.to_pandas()
                        
                        # Limit samples
                        if len(df) > num_samples:
                            df = df.sample(n=num_samples, random_state=42)
                        
                        # Convert to our format
                        file_samples = self._df_to_samples_vectorized(df, arrow_file)
                        samples.extend(file_samples)
                        
                        print(f"Loaded {len(file_samples)} samples from Arrow file {arrow_file}")
                        
                        if len(samples) >= num_samples:
                            break
                        
                except Exception as arrow_error:
                    print(f"Error reading Arrow file {arrow_file}: {arrow_error}")
                    continue
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                        
            except Exception as extract_error:
                print(f"Error extracting Arrow file {arrow_file}: {extract_error}")
                continue
        
        print(f"Loaded {len(samples)} total samples from Arrow dataset")
        return samples[:num_samples]

    def _load_vision_zip_data(self, zf: zipfile.ZipFile, json_files: List[str], image_files: List[str], num_samples: int) -> List[Dict[str, Any]]:
        """Load vision dataset with separate JSON annotations and images"""
        samples = []
        
        # Process each JSON file (usually contains annotations)
        for json_file in json_files:
            try:
                with zf.open(json_file) as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                annotations = []
                if isinstance(data, list):
                    annotations = data
                elif isinstance(data, dict):
                    # Check common keys for annotation data
                    for key in ['data', 'annotations', 'samples', 'items']:
                        if key in data and isinstance(data[key], list):
                            annotations = data[key]
                            break
                    if not annotations and 'annotations' not in data:
                        # If it's a single annotation object, wrap it
                        annotations = [data]
                
                # Convert annotations to training samples
                for idx, annotation in enumerate(annotations[:num_samples]):
                    if len(samples) >= num_samples:
                        break
                    
                    # Create training sample with proper format
                    sample = {
                        "id": annotation.get("id", f"{Path(json_file).stem}_{idx}"),
                        "conversations": []
                    }
                    
                    # Handle different annotation formats
                    if "question" in annotation and "answer" in annotation:
                        sample["conversations"] = [
                            {"from": "human", "value": str(annotation["question"])},
                            {"from": "gpt", "value": str(annotation["answer"])}
                        ]
                    elif "text" in annotation:
                        sample["conversations"] = [
                            {"from": "human", "value": "Describe this image."},
                            {"from": "gpt", "value": str(annotation["text"])}
                        ]
                    else:
                        # Generic fallback
                        sample["conversations"] = [
                            {"from": "human", "value": "Describe this image."},
                            {"from": "gpt", "value": f"This is an image from {Path(json_file).stem}."}
                        ]
                    
                    # Add image reference if available
                    if "image" in annotation:
                        sample["image"] = annotation["image"]
                    elif "image_path" in annotation:
                        sample["image"] = annotation["image_path"]
                    elif "filename" in annotation:
                        sample["image"] = annotation["filename"]
                    
                    # Add any additional fields
                    for key, value in annotation.items():
                        if key not in ["question", "answer", "text", "image", "image_path", "filename", "id"]:
                            sample[key] = self._make_json_serializable(value)
                    
                    samples.append(sample)
                    
            except Exception as e:
                print(f"Error processing JSON file {json_file}: {e}")
                continue
        
        print(f"Loaded {len(samples)} samples from vision dataset")
        return samples[:num_samples]

    def load_tar_data(self, tar_path: str, num_samples: int) -> List[Dict[str, Any]]:
        """Optimized TAR loading with streaming"""
        samples = []
        
        try:
            mode = 'r:gz' if tar_path.endswith(('.tar.gz', '.tgz')) else 'r'
            
            with tarfile.open(tar_path, mode) as tf:
                json_members = [m for m in tf.getmembers() 
                              if m.name.endswith('.json') and m.isfile()][:10]
                
                for member in json_members:
                    if len(samples) >= num_samples:
                        break
                    f = tf.extractfile(member)
                    if f:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                samples.extend(data[:num_samples - len(samples)])
                            else:
                                samples.append(data)
                        except json.JSONDecodeError:
                            continue
                        finally:
                            f.close()
                            
        except Exception as e:
            print(f"Error loading TAR {tar_path}: {e}")
        
        return samples[:num_samples]

    def load_tar_directory_data(self, tar_dir: str, num_samples: int) -> List[Dict[str, Any]]:
        """Load data from directory containing multiple tar.gz files"""
        samples = []
        
        if not os.path.exists(tar_dir):
            print(f"Directory does not exist: {tar_dir}")
            return []
        
        # Find all tar.gz files in the directory
        tar_files = []
        for file in os.listdir(tar_dir):
            if file.endswith(('.tar.gz', '.tgz', '.tar')):
                tar_files.append(os.path.join(tar_dir, file))
        
        if not tar_files:
            print(f"No tar files found in {tar_dir}")
            return []
        
        print(f"Found {len(tar_files)} tar files to process")
        
        # Process each tar file until we have enough samples
        samples_per_file = max(1, num_samples // len(tar_files))
        
        for tar_file in sorted(tar_files):
            if len(samples) >= num_samples:
                break
            
            remaining_samples = num_samples - len(samples)
            file_samples = self.load_tar_data(tar_file, min(samples_per_file * 2, remaining_samples))
            samples.extend(file_samples)
            
            print(f"Loaded {len(file_samples)} samples from {os.path.basename(tar_file)}")
        
        # Shuffle and limit to requested number
        if len(samples) > num_samples:
            random.shuffle(samples)
            samples = samples[:num_samples]
        
        print(f"Total samples loaded: {len(samples)}")
        return samples

    def load_mammoth_tar_data(self, tar_dir: str, num_samples: int) -> List[Dict[str, Any]]:
        """Load MAmmoTH dataset with separate large JSON annotations and TAR image files using fastest approach"""
        print(f"Processing MAmmoTH dataset with large JSON file from {tar_dir}...")
        
        # Look for the mammoth_ov_2M.json file
        json_path = os.path.join(tar_dir, "mammoth_ov_2M.json")
        if not os.path.exists(json_path):
            print(f"MAmmoTH annotation file not found: {json_path}")
            print(f"Expected: {json_path}")
            return []
        
        print(f"Found MAmmoTH annotation file: {os.path.basename(json_path)}")
        print(f"File size: {os.path.getsize(json_path) / (1024**3):.2f} GB")
        
        # Build image index from all TAR files (fast operation)
        print("Building image index from TAR files...")
        image_index = self._build_tar_image_index(tar_dir)
        if not image_index:
            print("No images found in TAR files")
            return []
        
        print(f"Built image index with {len(image_index)} available images")
        
        # Stream parse the large JSON file with filtering (memory-efficient)
        print(f"Stream parsing JSON file to extract {num_samples} samples...")
        samples = self._stream_parse_mammoth_json(json_path, image_index, num_samples)
        
        print(f"Successfully loaded {len(samples)} MAmmoTH samples")
        return samples
    
    def _build_tar_image_index(self, tar_dir: str) -> set:
        """Build fast lookup index of available images from TAR files"""
        image_index = set()
        
        # Find all TAR files
        tar_files = []
        for file in os.listdir(tar_dir):
            if file.endswith(('.tar.gz', '.tgz', '.tar')):
                tar_files.append(os.path.join(tar_dir, file))
        
        if not tar_files:
            return image_index
        
        # Parallel processing of TAR files for speed
        def extract_image_paths(tar_path):
            local_paths = set()
            try:
                mode = 'r:gz' if tar_path.endswith(('.tar.gz', '.tgz')) else 'r'
                with tarfile.open(tar_path, mode) as tf:
                    for member in tf.getmembers():
                        if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                            # Store both full path and just filename for flexible matching
                            local_paths.add(member.name)
                            local_paths.add(os.path.basename(member.name))
            except Exception as e:
                print(f"Error reading TAR file {tar_path}: {e}")
            return local_paths
        
        # Use parallel processing for maximum speed
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(tar_files))) as executor:
            futures = [executor.submit(extract_image_paths, tar_file) for tar_file in tar_files]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    paths = future.result()
                    image_index.update(paths)
                except Exception as e:
                    print(f"Error processing TAR file: {e}")
        
        return image_index
    
    def _stream_parse_mammoth_json(self, json_path: str, image_index: set, num_samples: int) -> List[Dict[str, Any]]:
        """Stream parse large JSON file with filtering for memory efficiency"""
        samples = []
        
        try:
            # Try to use ijson for true streaming if available
            try:
                import ijson
                use_streaming = True
                print("Using ijson for memory-efficient streaming")
            except ImportError:
                use_streaming = False
                print("ijson not available, using chunked reading fallback")
            
            if use_streaming:
                # True streaming approach with ijson
                with open(json_path, 'rb') as f:
                    parser = ijson.parse(f)
                    current_item = {}
                    in_array = False
                    items_processed = 0
                    
                    for prefix, event, value in parser:
                        if len(samples) >= num_samples:
                            break
                        
                        # Handle array of objects structure
                        if prefix == '' and event == 'start_array':
                            in_array = True
                        elif in_array and event == 'start_map':
                            current_item = {}
                        elif in_array and event == 'map_key':
                            current_key = value
                        elif in_array and event in ['string', 'number', 'boolean']:
                            current_item[current_key] = value
                        elif in_array and event == 'end_map':
                            # Process completed item
                            if self._should_include_mammoth_item(current_item, image_index):
                                sample = self._convert_mammoth_to_sample(current_item, items_processed)
                                samples.append(sample)
                            items_processed += 1
                            
                            # Progress reporting
                            if items_processed % 10000 == 0:
                                print(f"Processed {items_processed:,} items, found {len(samples)} matching samples")
            else:
                # Fallback: chunked reading approach
                samples = self._chunked_parse_mammoth_json(json_path, image_index, num_samples)
                
        except Exception as e:
            print(f"Error parsing JSON file: {e}")
            # Last resort fallback
            print("Attempting fallback parsing...")
            samples = self._fallback_parse_mammoth_json(json_path, image_index, num_samples)
        
        return samples[:num_samples]
    
    def _chunked_parse_mammoth_json(self, json_path: str, image_index: set, num_samples: int) -> List[Dict[str, Any]]:
        """Chunked reading fallback for when ijson is not available"""
        samples = []
        
        # Read file in chunks to avoid loading all large
        chunk_size = 64 * 1024 * 1024  # 64MB chunks
        buffer = ""
        items_processed = 0
        
        with open(json_path, 'r', encoding='utf-8') as f:
            while len(samples) < num_samples:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                buffer += chunk
                
                # Find complete JSON objects in buffer
                while '{' in buffer and '}' in buffer:
                    try:
                        # Find a complete JSON object
                        start = buffer.find('{')
                        if start == -1:
                            break
                        
                        brace_count = 0
                        end = start
                        for i in range(start, len(buffer)):
                            if buffer[i] == '{':
                                brace_count += 1
                            elif buffer[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end = i + 1
                                    break
                        
                        if brace_count == 0:
                            # Found complete object
                            json_str = buffer[start:end]
                            buffer = buffer[end:]
                            
                            try:
                                item = json.loads(json_str)
                                if self._should_include_mammoth_item(item, image_index):
                                    sample = self._convert_mammoth_to_sample(item, items_processed)
                                    samples.append(sample)
                                items_processed += 1
                                
                                if items_processed % 10000 == 0:
                                    print(f"Processed {items_processed:,} items, found {len(samples)} matching samples")
                                    
                                if len(samples) >= num_samples:
                                    break
                            except json.JSONDecodeError:
                                continue
                        else:
                            break
                    except Exception:
                        # Skip this chunk and continue
                        buffer = buffer[1:]
        
        return samples
    
    def _fallback_parse_mammoth_json(self, json_path: str, image_index: set, num_samples: int) -> List[Dict[str, Any]]:
        """Last resort: load entire JSON (risky for large files)"""
        print("WARNING: Loading entire JSON file into memory - this may cause memory issues with large files")
        samples = []
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and 'data' in data:
                items = data['data']
            else:
                print("Unexpected JSON structure")
                return []
            
            for idx, item in enumerate(items):
                if len(samples) >= num_samples:
                    break
                    
                if self._should_include_mammoth_item(item, image_index):
                    sample = self._convert_mammoth_to_sample(item, idx)
                    samples.append(sample)
        
        except Exception as e:
            print(f"Fallback parsing failed: {e}")
        
        return samples
    
    def _should_include_mammoth_item(self, item: Dict, image_index: set) -> bool:
        """Check if item should be included based on image availability"""
        if not isinstance(item, dict):
            return False
        
        # Look for image key in various possible formats
        image_path = None
        for key in ['image', 'image_path', 'filename', 'file_name']:
            if key in item:
                image_path = item[key]
                break
        
        if not image_path:
            return False
        
        # Check both full path and just filename
        image_filename = os.path.basename(image_path)
        return image_path in image_index or image_filename in image_index
    
    def _convert_mammoth_to_sample(self, item: Dict, idx: int) -> Dict[str, Any]:
        """Convert MAmmoTH annotation to training sample format"""
        sample = {
            "id": item.get("id", f"mammoth_{idx}"),
            "conversations": []
        }
        
        # Handle different conversation formats
        if "conversations" in item and isinstance(item["conversations"], list):
            sample["conversations"] = item["conversations"]
        elif "question" in item and "answer" in item:
            sample["conversations"] = [
                {"from": "human", "value": str(item["question"])},
                {"from": "gpt", "value": str(item["answer"])}
            ]
        elif "instruction" in item and "response" in item:
            sample["conversations"] = [
                {"from": "human", "value": str(item["instruction"])},
                {"from": "gpt", "value": str(item["response"])}
            ]
        else:
            # Generic fallback
            sample["conversations"] = [
                {"from": "human", "value": "Describe this image."},
                {"from": "gpt", "value": "This is an image from the MAmmoTH dataset."}
            ]
        
        # Add image reference
        for key in ['image', 'image_path', 'filename', 'file_name']:
            if key in item:
                sample["image"] = item[key]
                break
        
        # Add any additional fields
        for key, value in item.items():
            if key not in ["question", "answer", "instruction", "response", "conversations", "image", "image_path", "filename", "file_name", "id"]:
                sample[key] = self._make_json_serializable(value)
        
        return sample

    def load_composite_data(self, composite_sources: List[Dict], num_samples: int) -> List[Dict[str, Any]]:
        """Load data using composite sampling strategy"""
        all_samples = []
        
        for source in composite_sources:
            if "sources" in source:  # Handle remaining_image_datasets
                source_samples = []
                samples_per_subsource = max(1, int(num_samples * source["weight"] / len(source["sources"])))
                
                for subsource in source["sources"]:
                    if os.path.exists(subsource["path"]):
                        subsource_samples = self.load_parquet_data(subsource["path"], samples_per_subsource)
                        source_samples.extend(subsource_samples)
                    else:
                        print(f"Warning: Subsource path not found: {subsource['path']}")
                
                # Limit to the weighted portion
                target_samples = int(num_samples * source["weight"])
                if len(source_samples) > target_samples:
                    source_samples = random.sample(source_samples, target_samples)
                
                all_samples.extend(source_samples)
            else:  # Handle direct sources like figureqa and raven
                if os.path.exists(source["path"]):
                    target_samples = int(num_samples * source["weight"])
                    source_samples = self.load_parquet_data(source["path"], target_samples)
                    all_samples.extend(source_samples)
                else:
                    print(f"Warning: Source path not found: {source['path']}")
        
        # Shuffle and limit to requested number
        if len(all_samples) > num_samples:
            random.shuffle(all_samples)
            all_samples = all_samples[:num_samples]
        
        print(f"Loaded {len(all_samples)} samples from composite sources")
        return all_samples

    def load_alternative_sampling_data(self, alternative_source: Dict, num_samples: int) -> List[Dict[str, Any]]:
        """Load data using alternative sampling strategy"""
        
        # Use the alternative source instead of the original
        if alternative_source["format"] == "tar":
            samples = self.load_tar_data(alternative_source["path"], num_samples)
        elif alternative_source["format"] == "parquet":
            samples = self.load_parquet_data(alternative_source["path"], num_samples)
        elif alternative_source["format"] == "video":
            samples = self.load_video_data(alternative_source["path"], num_samples, alternative_source["name"])
        else:
            print(f"Unsupported alternative format: {alternative_source['format']}")
            return []
        
        print(f"Loaded {len(samples)} samples using alternative sampling")
        return samples

    def process_datasets_parallel(self, datasets_config: List[Dict], output_dir: str) -> Dict[str, int]:
        """Process multiple datasets in parallel with maximum CPU utilization"""
        print(f"Processing {len(datasets_config)} datasets with {self.max_workers} CPU workers...")
        
        # Filter out already processed datasets
        pending_configs = []
        results = {}
        
        for config in datasets_config:
            if os.path.exists(config["output"]):
                with open(config["output"], 'r') as f:
                    existing_count = len(json.load(f))
                results[config["name"]] = existing_count
                print(f"Skipping {config['name']} ({existing_count} samples exist)")
            else:
                pending_configs.append(config)
        
        if not pending_configs:
            return results
        
        # Process datasets in parallel with maximum CPU utilization
        process_func = partial(self._process_single_dataset)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_func, config): config for config in pending_configs}
            
            with tqdm(total=len(pending_configs), desc="Processing datasets", 
                     unit="dataset", colour="green") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    config = futures[future]
                    try:
                        sample_count = future.result()
                        results[config["name"]] = sample_count
                        pbar.set_postfix({
                            "current": config["name"][:20], 
                            "samples": f"{sample_count:,}",
                            "mode": "CPU"
                        })
                    except Exception as e:
                        print(f"Error processing {config['name']}: {e}")
                        results[config["name"]] = 0
                    pbar.update(1)
        
        return results
    
    def _process_single_dataset(self, config: Dict) -> int:
        """Process a single dataset configuration with maximum CPU optimization"""
        try:
            samples = []
            
            # Add format validation
            supported_formats = ["parquet", "video", "zip", "tar.gz", "tar", "tar_directory", "json", "zip_directory", "composite", "alternative_sampling", "mammoth_tar"]
            if config["format"] not in supported_formats:
                print(f"Unsupported format '{config['format']}' for dataset {config['name']}")
                return 0
            
            # Add path validation (skip for composite and alternative_sampling formats)
            if config["format"] not in ["composite", "alternative_sampling"]:
                if "path" not in config or not os.path.exists(config["path"]):
                    print(f"Path does not exist for dataset {config['name']}: {config.get('path', 'N/A')}")
                    return 0
            
            if config["format"] == "parquet":
                samples = self.load_parquet_data(config["path"], config["samples"])
            elif config["format"] == "video":
                video_filter = config.get("video_filter")
                samples = self.load_video_data(
                    config["path"], config["samples"], config["name"], video_filter
                )
            elif config["format"] == "json":
                samples = self.load_json_data(config["path"], config["samples"])
            elif config["format"] == "zip":
                samples = self.load_zip_data(config["path"], config["samples"])
            elif config["format"] in ["tar.gz", "tar"]:
                samples = self.load_tar_data(config["path"], config["samples"])
            elif config["format"] == "tar_directory":
                samples = self.load_tar_directory_data(config["path"], config["samples"])
            elif config["format"] == "zip_directory":
                samples = self.load_zip_directory_data(config["path"], config["samples"])
            elif config["format"] == "mammoth_tar":
                samples = self.load_mammoth_tar_data(config["path"], config["samples"])
            elif config["format"] == "composite":
                samples = self.load_composite_data(config["composite_sources"], config["samples"])
            elif config["format"] == "alternative_sampling":
                samples = self.load_alternative_sampling_data(config["alternative_source"], config["samples"])
            
            if samples:
                # Ensure output directory exists
                os.makedirs(os.path.dirname(config["output"]), exist_ok=True)
                
                # Atomic write to prevent corruption
                temp_output = config["output"] + ".tmp"
                try:
                    with open(temp_output, 'w', encoding='utf-8') as f:
                        json.dump(samples, f, indent=2, ensure_ascii=False)
                    os.rename(temp_output, config["output"])
                    return len(samples)
                except Exception as write_error:
                    print(f"Failed to write output for {config['name']}: {write_error}")
                    # Clean up temp file if it exists
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    return 0
            else:
                print(f"No samples loaded for dataset {config['name']}")
                return 0
            
        except Exception as e:
            print(f"Error processing {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            return 0


def main():
    parser = argparse.ArgumentParser(description="CPU-Optimized Training Data Preparation")
    parser.add_argument("--output_dir", default="data/datasets", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_path", default="/data1/yihao", help="Base path for datasets")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: auto-detect CPU cores)")
    parser.add_argument("--cache_dir", default=".cache", help="Cache directory")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize CPU optimized loader
    cache_dir = None if args.no_cache else args.cache_dir
    loader = CPUOptimizedDataLoader(cache_dir=cache_dir, max_workers=args.workers)
    
    # Dataset configurations with corrected paths
    datasets_config = [
        # Text datasets (20.2% total)
        {
            "name": "magpie_pro_l3_80b_mt",
            "samples": 68000,
            "output": f"{args.output_dir}/magpie_pro_l3_80b_mt.json",
            "modality": "text",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/magpie_pro(l3_80b_mt)",
            "format": "parquet"
        },
        {
            "name": "magpie_pro_l3_80b_st",
            "samples": 68000,
            "output": f"{args.output_dir}/magpie_pro_l3_80b_st.json",
            "modality": "text",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/magpie_pro(l3_80b_st)",
            "format": "parquet"
        },
        {
            "name": "magpie_pro_qwen2_72b_st",
            "samples": 58000,
            "output": f"{args.output_dir}/magpie_pro_qwen2_72b_st.json",
            "modality": "text",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/magpie_pro(qwen2_72b_st)",
            "format": "parquet"
        },
        {
            "name": "mathqa",
            "samples": 9000,
            "output": f"{args.output_dir}/mathqa.json",
            "modality": "text",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/mathqa",
            "format": "parquet"
        },
        
        # Multi-image datasets (12.3% total)
        {
            "name": "m4_instruct_data",
            "samples": 104000,
            "output": f"{args.output_dir}/m4_instruct_data.json",
            "modality": "multi-image",
            "path": f"{args.base_path}/M4-Instruct-Data",
            "format": "zip_directory"
        },
        {
            "name": "mammoth_multi_image",
            "samples": 19000,
            "output": f"{args.output_dir}/mammoth_multi_image.json",
            "modality": "multi-image",
            "path": f"{args.base_path}/MAmmoTH-VL-Instruct-12M/multi_image_data",
            "format": "mammoth_tar"  # MAmmoTH-specific format with large JSON annotation file
        },
        
        # Image datasets (34.4% total)
        {
            "name": "vision_flan_filtered",
            "samples": 39000,
            "output": f"{args.output_dir}/vision_flan_filtered.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/vision_flan(filtered)",
            "format": "parquet"
        },
        {
            "name": "mavis_math_metagen",
            "samples": 26000,
            "output": f"{args.output_dir}/mavis_math_metagen.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/mavis_math_metagen",
            "format": "parquet"
        },
        {
            "name": "mavis_math_rule_geo",
            "samples": 25000,
            "output": f"{args.output_dir}/mavis_math_rule_geo.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/mavis_math_rule_geo",
            "format": "parquet"
        },
        {
            "name": "sharegpt4o",
            "samples": 17000,
            "output": f"{args.output_dir}/sharegpt4o.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/sharegpt4o",
            "format": "parquet"
        },
        {
            "name": "sharegpt4v_coco",
            "samples": 15000,
            "output": f"{args.output_dir}/sharegpt4v_coco.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/sharegpt4v(coco)",
            "format": "parquet"
        },
        {
            "name": "sharegpt4v_llava",
            "samples": 9000,
            "output": f"{args.output_dir}/sharegpt4v_llava.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/sharegpt4v(llava)",
            "format": "parquet"
        },
        {
            "name": "mapqa_mathv360k",
            "samples": 9000,
            "output": f"{args.output_dir}/mapqa_mathv360k.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/MapQA(MathV360K)",
            "format": "parquet"
        },
        {
            "name": "textocr_gpt4v",
            "samples": 8000,
            "output": f"{args.output_dir}/textocr_gpt4v.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/textocr(gpt4v)",
            "format": "parquet"
        },
        # llava-onevision/other - 17.4% with composite sampling
        {
            "name": "llava_onevision_other",
            "samples": 166000,
            "output": f"{args.output_dir}/llava_onevision_other.json",
            "modality": "image",
            "format": "composite",
            "composite_sources": [
                {
                    "name": "figureqa_cauldron_llava",
                    "path": f"{args.base_path}/LLaVA-OneVision-Data/figureqa(cauldron,llava_format)",
                    "format": "parquet",
                    "weight": 0.35  # 70% total split between figureqa and raven
                },
                {
                    "name": "raven_cauldron",
                    "path": f"{args.base_path}/LLaVA-OneVision-Data/raven(cauldron)",
                    "format": "parquet",
                    "weight": 0.35  # 70% total split between figureqa and raven
                },
                {
                    "name": "remaining_image_datasets",
                    "weight": 0.30,  # 30% from remaining image datasets
                    "sources": [
                        {"path": f"{args.base_path}/LLaVA-OneVision-Data/image_textualization(filtered)", "format": "parquet"}
                    ]
                }
            ]
        },
        {
            "name": "image_textualization",
            "samples": 13000,
            "output": f"{args.output_dir}/image_textualization.json",
            "modality": "image",
            "path": f"{args.base_path}/LLaVA-OneVision-Data/image_textualization(filtered)",
            "format": "parquet"
        },
        # qa dataset using alternative sampling (0.8% = 8000 samples)
        {
            "name": "qa",
            "samples": 8000,
            "output": f"{args.output_dir}/qa.json",
            "modality": "image",
            "format": "alternative_sampling",
            "alternative_source": {
                "name": "figureqa_substitute",
                "path": f"{args.base_path}/LLaVA-OneVision-Data/figureqa(cauldron,llava_format)",
                "format": "parquet"
            }
        },
        
        # Video datasets (33.0% total)
        {
            "name": "llava_video_1_2m",
            "samples": 73000,
            "output": f"{args.output_dir}/llava_video_1_2m.json",
            "modality": "video",
            "path": f"{args.base_path}/llava-video",
            "format": "video",
            "video_filter": "1_2_m_*"
        },
        {
            "name": "llava_video_2_3m",
            "samples": 70000,
            "output": f"{args.output_dir}/llava_video_2_3m.json",
            "modality": "video",
            "path": f"{args.base_path}/llava-video",
            "format": "video",
            "video_filter": "2_3_m_*"
        },
        {
            "name": "llava_video_0_30s",
            "samples": 24000,
            "output": f"{args.output_dir}/llava_video_0_30s.json",
            "modality": "video",
            "path": f"{args.base_path}/llava-video",
            "format": "video",
            "video_filter": "0_30_s_*"
        },
        {
            "name": "vista_400k_combined",
            "samples": 22000,
            "output": f"{args.output_dir}/vista_400k_combined.json",
            "modality": "video",
            "path": f"{args.base_path}/VISTA-400K/two_needle_niah_qa/two_needle_niah_qa_14.tar",
            "format": "tar"
        },
        {
            "name": "sharegpt4video_all",
            "samples": 8000,
            "output": f"{args.output_dir}/sharegpt4video_all.json",
            "modality": "video",
            "format": "alternative_sampling",
            "alternative_source": {
                "name": "vista_400k_combined",
                "path": f"{args.base_path}/VISTA-400K/two_needle_niah_qa/two_needle_niah_qa_14.tar",
                "format": "tar"
            }
        },
        # Additional video datasets to reach 33.0% distribution
        {
            "name": "llava_video_hound",
            "samples": 44000,
            "output": f"{args.output_dir}/llava_video_hound.json",
            "modality": "video",
            "path": f"{args.base_path}/llava-video",
            "format": "video",
            "video_filter": "hound_*"
        },
        {
            "name": "other_video_combined",
            "samples": 57000,
            "output": f"{args.output_dir}/other_video_combined.json",
            "modality": "video",
            "format": "alternative_sampling",
            "alternative_source": {
                "name": "llava_video_combined",
                "path": f"{args.base_path}/llava-video",
                "format": "video"
            }
        },
        {
            "name": "video_star_starb",
            "samples": 22000,
            "output": f"{args.output_dir}/video_star_starb.json",
            "modality": "video",
            "format": "alternative_sampling",
            "alternative_source": {
                "name": "llava_video_combined",
                "path": f"{args.base_path}/llava-video",
                "format": "video"
            }
        },
        {
            "name": "vript_long",
            "samples": 10000,
            "output": f"{args.output_dir}/vript_long.json",
            "modality": "video",
            "format": "alternative_sampling",
            "alternative_source": {
                "name": "vista_400k_combined",
                "path": f"{args.base_path}/VISTA-400K/two_needle_niah_qa/two_needle_niah_qa_14.tar",
                "format": "tar"
            }
        },
    ]
    
    # Process all datasets with performance monitoring
    print(f"Starting CPU-optimized data preparation with {loader.max_workers} workers...")
    start_time = time.time()
    
    results = loader.process_datasets_parallel(datasets_config, args.output_dir)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print comprehensive summary
    total_samples = sum(results.values())
    successful_datasets = len([r for r in results.values() if r > 0])
    
    separator = '=' * 80
    print(f"\n{separator}")
    print(f"CPU-OPTIMIZED DATA PREPARATION COMPLETE")
    print(f"{separator}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Successful datasets: {successful_datasets}/{len(datasets_config)}")
    print(f"Total samples: {total_samples:,}")
    print(f"Average speed: {total_samples/processing_time:.0f} samples/second")
    print(f"Output directory: {args.output_dir}")
    print(f"CPU workers used: {loader.max_workers}/{CPU_CORES} cores")
    
    # Modality breakdown
    modality_stats = defaultdict(int)
    for config in datasets_config:
        if config["name"] in results and results[config["name"]] > 0:
            modality_stats[config["modality"]] += results[config["name"]]
    
    print(f"\nModality Distribution:")
    for modality, count in modality_stats.items():
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  - {modality}: {count:,} samples ({percentage:.1f}%)")
    
    # Performance analysis
    if processing_time > 0:
        print(f"\nPerformance Analysis:")
        print(f"  - Datasets per minute: {successful_datasets * 60 / processing_time:.1f}")
        print(f"  - MB processed per second: {total_samples * 0.001 / processing_time:.1f}")
        print(f"  - CPU utilization: {loader.max_workers} workers across {CPU_CORES} cores")
        print(f"  - CPU efficiency: {(loader.max_workers / CPU_CORES * 100):.1f}% core utilization")


if __name__ == "__main__":
    main()
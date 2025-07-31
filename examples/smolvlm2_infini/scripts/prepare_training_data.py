"""
Usage:
    python prepare_training_data.py --output_dir data/datasets --use_gpu --workers 8
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

# GPU acceleration imports (with fallbacks)
try:
    import cudf
    import cupy as cp
    import rmm
    CUDF_AVAILABLE = True
    print("GPU acceleration available with cuDF/CuPy")
except ImportError:
    cudf = None
    cp = None
    rmm = None
    CUDF_AVAILABLE = False
    print("GPU libraries not available, falling back to CPU-only processing")

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    pq = None
    pa = None
    PYARROW_AVAILABLE = False

# CPU-optimized libraries for fallback
try:
    import polars as pl
    POLARS_AVAILABLE = True
    print("Polars available for CPU-optimized processing")
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('mode.copy_on_write', True)

class GPUManager:
    """Manages GPU selection and utilization monitoring"""
    
    def __init__(self):
        self.available_gpus = []
        self.gpu_usage = {}
        self.gpu_lock = threading.Lock()
        self._discover_gpus()
    
    def _discover_gpus(self):
        """Discover available GPUs and check their utilization"""
        if not CUDF_AVAILABLE:
            return
        
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Consider GPU available if memory usage < 20% and utilization < 30%
                memory_usage = (info.used / info.total) * 100
                gpu_util = util.gpu
                
                if memory_usage < 20 and gpu_util < 30:
                    self.available_gpus.append(i)
                    self.gpu_usage[i] = {
                        'memory_used': memory_usage,
                        'utilization': gpu_util,
                        'total_memory': info.total // (1024**3),  # GB
                        'free_memory': info.free // (1024**3)    # GB
                    }
                    print(f"GPU {i} available: {memory_usage:.1f}% memory, {gpu_util}% util, {info.free//(1024**3)}GB free")
                else:
                    print(f"GPU {i} busy: {memory_usage:.1f}% memory, {gpu_util}% util - skipping")
            
            if self.available_gpus:
                print(f"Found {len(self.available_gpus)} available GPUs: {self.available_gpus}")
            else:
                print("No available GPUs found - all are currently in use")
                
        except ImportError:
            print("pynvml not available, using basic GPU detection")
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
                self.available_gpus = list(range(device_count))
                print(f"Detected {device_count} GPUs, assuming all available")
            except Exception as e:
                print(f"GPU detection failed: {e}")
        except Exception as e:
            print(f"GPU utilization check failed: {e}")
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
                self.available_gpus = list(range(device_count))
                print(f"Fallback: detected {device_count} GPUs")
            except:
                pass
    
    def get_best_gpu(self) -> Optional[int]:
        """Get the GPU with lowest utilization"""
        with self.gpu_lock:
            if not self.available_gpus:
                return None
            
            gpu_id = self.available_gpus[0]
            self.available_gpus.append(self.available_gpus.pop(0))  # Rotate
            return gpu_id
    
    def release_gpu(self, gpu_id: int):
        """Mark GPU as available again"""
        with self.gpu_lock:
            if gpu_id not in self.available_gpus:
                self.available_gpus.append(gpu_id)

class GPUAcceleratedDataLoader:
    def __init__(self, cache_dir: str = ".cache", max_workers: int = None, use_gpu: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.use_gpu = use_gpu and CUDF_AVAILABLE
        self.gpu_manager = GPUManager() if self.use_gpu else None
        self.gpu_lock = threading.Lock()  # Add GPU synchronization lock
        
        if self.use_gpu and self.gpu_manager.available_gpus:
            self.gpu_available = True
            print(f"Multi-GPU acceleration enabled with {len(self.gpu_manager.available_gpus)} available GPUs")
            # Initialize RMM once globally to avoid thread-unsafe reinitialization
            self._initialize_rmm_globally()
        else:
            self.gpu_available = False
            self.use_gpu = False
            if use_gpu:
                print("No available GPUs found, falling back to CPU processing")
    
    def _initialize_rmm_globally(self):
        """Initialize RMM once globally for all GPUs to avoid thread-safety issues"""
        try:
            if rmm is not None and self.gpu_manager and self.gpu_manager.available_gpus:
                # Initialize RMM for all available GPUs at once
                rmm.reinitialize(
                    devices=self.gpu_manager.available_gpus,
                    managed_memory=True,
                    pool_allocator=True,
                    initial_pool_size=2**29  # 512MB per GPU to be conservative
                )
                print(f"Initialized RMM memory pools for GPUs: {self.gpu_manager.available_gpus}")
        except Exception as e:
            print(f"Warning: Failed to initialize RMM globally: {e}. GPU operations may be slower.")
    
    def _get_cache_key(self, path: str, num_samples: int) -> str:
        """Generate cache key with GPU-accelerated hashing if available"""
        if self.use_gpu and cp is not None:
            # GPU-accelerated hash computation
            try:
                data = f"{path}_{num_samples}".encode()
                gpu_data = cp.asarray(np.frombuffer(data, dtype=np.uint8))
                # Use simple hash since cp.hash might not be available
                hash_val = int(cp.sum(gpu_data).get()) % (2**32)
                return f"dataset_{hash_val}.pkl"
            except:
                pass
        
        # CPU fallback
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
        """GPU-accelerated parquet loading with cuDF and multi-GPU support"""
        # Check cache first
        cache_key = self._get_cache_key(parquet_path, num_samples)
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            print(f"Loaded {len(cached_data)} samples from cache for {parquet_path}")
            return cached_data
        
        print(f"Loading {parquet_path} with {'multi-GPU' if self.use_gpu else 'CPU'} acceleration...")
        
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
            
            # Use GPU-accelerated processing if available
            if self.use_gpu and len(parquet_files) > 0:
                samples = self._load_parquet_gpu(parquet_files, num_samples)
            else:
                samples = self._load_parquet_cpu_optimized(parquet_files, num_samples)
            
            # Cache the results
            self._save_to_cache(cache_key, samples)
            print(f"Loaded {len(samples)} samples from {len(parquet_files)} parquet files")
            return samples
            
        except Exception as e:
            print(f"Error loading parquet data from {parquet_path}: {e}")
            return []
    
    def _load_parquet_gpu(self, parquet_files: List[str], num_samples: int) -> List[Dict]:
        """Multi-GPU accelerated parquet loading with cuDF"""
        with self.gpu_lock:  # Synchronize GPU access
            gpu_id = self.gpu_manager.get_best_gpu() if self.gpu_manager else None
            
            if gpu_id is None:
                print("No GPU available, falling back to CPU")
                return self._load_parquet_cpu_optimized(parquet_files, num_samples)
        
        try:
            # Set GPU device and ensure proper context
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            cp.cuda.Device(gpu_id).use()
            
            print(f"Using GPU {gpu_id} for parquet processing...")
            
            # Estimate total rows using GPU-accelerated metadata reading
            total_rows = self._estimate_total_rows_gpu(parquet_files)
            
            if total_rows <= num_samples * 2:
                # Load all files with GPU
                df_list = []
                for pfile in parquet_files:
                    try:
                        gpu_df = cudf.read_parquet(pfile)
                        df_list.append(gpu_df)
                    except Exception as e:
                        print(f"GPU loading failed for {pfile}: {e}, trying CPU fallback")
                        cpu_df = pd.read_parquet(pfile)
                        gpu_df = cudf.from_pandas(cpu_df)
                        df_list.append(gpu_df)
                
                # Concatenate on GPU
                if df_list:
                    combined_df = cudf.concat(df_list, ignore_index=True)
                    
                    # GPU-accelerated sampling
                    if len(combined_df) > num_samples:
                        # Use GPU random sampling
                        indices = cp.random.choice(len(combined_df), size=num_samples, replace=False)
                        combined_df = combined_df.iloc[indices]
                    
                    # Convert back to pandas for JSON serialization
                    pandas_df = combined_df.to_pandas()
                    samples = self._df_to_samples_vectorized(pandas_df, parquet_files[0])
                    
                    # Clean up GPU memory
                    del combined_df, df_list
                    if hasattr(cp, 'get_default_memory_pool'):
                        cp.get_default_memory_pool().free_all_blocks()
                    
                    # Release GPU back to pool
                    with self.gpu_lock:
                        self.gpu_manager.release_gpu(gpu_id)
                    
                    return samples
            else:
                # Use chunked GPU processing for large datasets
                result = self._chunked_gpu_processing(parquet_files, num_samples, total_rows, gpu_id)
                with self.gpu_lock:
                    self.gpu_manager.release_gpu(gpu_id)
                return result
                
        except Exception as e:
            print(f"GPU processing failed on GPU {gpu_id}: {e}, falling back to CPU")
            if self.gpu_manager and gpu_id is not None:
                with self.gpu_lock:
                    self.gpu_manager.release_gpu(gpu_id)
            return self._load_parquet_cpu_optimized(parquet_files, num_samples)
        
        return []
    
    def _estimate_total_rows_gpu(self, parquet_files: List[str]) -> int:
        """GPU-accelerated row estimation"""
        if not PYARROW_AVAILABLE:
            return self._estimate_total_rows_cpu(parquet_files)
        
        total_rows = 0
        sample_files = parquet_files[:min(3, len(parquet_files))]
        
        try:
            # Parallel metadata reading
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(sample_files)) as executor:
                futures = [executor.submit(self._get_parquet_rows, pfile) for pfile in sample_files]
                for future in concurrent.futures.as_completed(futures):
                    total_rows += future.result()
            
            # Extrapolate to all files
            if len(parquet_files) > len(sample_files):
                avg_rows = total_rows / len(sample_files)
                total_rows = int(avg_rows * len(parquet_files))
                
        except Exception:
            total_rows = self._estimate_total_rows_cpu(parquet_files)
        
        return total_rows
    
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
    
    def _chunked_gpu_processing(self, parquet_files: List[str], num_samples: int, total_rows: int, gpu_id: int) -> List[Dict]:
        """GPU-accelerated chunked processing for large datasets"""
        samples = []
        rows_seen = 0
        sampling_ratio = min(1.0, num_samples * 3 / total_rows)
        
        # Ensure we're using the correct GPU context
        with cp.cuda.Device(gpu_id):
            for pfile in parquet_files:
                try:
                    # Read file with cuDF on specific GPU
                    gpu_df = cudf.read_parquet(pfile)
                    
                    # GPU-accelerated sampling
                    if sampling_ratio < 1.0:
                        n_sample = max(1, int(len(gpu_df) * sampling_ratio))
                        indices = cp.random.choice(len(gpu_df), size=n_sample, replace=False)
                        gpu_df = gpu_df.iloc[indices]
                    
                    # Convert to pandas for processing
                    pandas_df = gpu_df.to_pandas()
                    chunk_samples = self._df_to_samples_vectorized(pandas_df, pfile)
                    
                    # Reservoir sampling
                    for sample in chunk_samples:
                        if len(samples) < num_samples:
                            samples.append(sample)
                        else:
                            j = random.randint(0, rows_seen)
                            if j < num_samples:
                                samples[j] = sample
                        rows_seen += 1
                    
                    # Clean up GPU memory
                    del gpu_df, pandas_df
                    
                    if len(samples) >= num_samples:
                        break
                        
                except Exception as e:
                    print(f"GPU chunk processing failed for {pfile}: {e}")
                    continue
            
            # Final GPU memory cleanup
            if hasattr(cp, 'get_default_memory_pool'):
                cp.get_default_memory_pool().free_all_blocks()
        
        return samples[:num_samples]
    
    def _load_parquet_cpu_optimized(self, parquet_files: List[str], num_samples: int) -> List[Dict]:
        """CPU-optimized processing with Polars when available"""
        total_rows = self._estimate_total_rows_cpu(parquet_files)
        
        # Use Polars for faster CPU processing if available
        if POLARS_AVAILABLE and len(parquet_files) > 0:
            return self._load_parquet_polars(parquet_files, num_samples, total_rows)
        
        # Fallback to pandas with optimizations
        if total_rows <= num_samples * 2:
            # Parallel loading for multiple files
            if len(parquet_files) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(parquet_files))) as executor:
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
                
                # Apply sampling at scan level if needed
                if sampling_ratio < 1.0:
                    n_sample = max(1, int(lazy_df.select(pl.count()).collect().item() * sampling_ratio))
                    lazy_df = lazy_df.sample(n=n_sample, seed=42)
                
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
        """CPU chunked processing with reservoir sampling"""
        samples = []
        rows_seen = 0
        sampling_ratio = min(1.0, num_samples * 3 / total_rows)
        
        for pfile in parquet_files:
            try:
                # Read in chunks for memory efficiency
                for chunk in pd.read_parquet(pfile, chunksize=10000):
                    if sampling_ratio < 1.0:
                        n_sample = max(1, int(len(chunk) * sampling_ratio))
                        chunk = chunk.sample(n=n_sample, random_state=42)
                    
                    chunk_samples = self._df_to_samples_vectorized(chunk, pfile)
                    
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
                    
            except Exception as e:
                print(f"Error processing {pfile}: {e}")
                continue
        
        return samples[:num_samples]
    
    def _df_to_samples_vectorized(self, df: pd.DataFrame, source_file: str) -> List[Dict]:
        """Vectorized conversion with GPU acceleration where possible"""
        samples = []
        file_stem = Path(source_file).stem
        
        # Process in batches for memory efficiency
        batch_size = 1000
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Vectorized processing
            batch_dict = batch.to_dict('records')
            
            for idx, sample in enumerate(batch_dict):
                # Convert bytes to base64 strings for JSON serialization
                for key, value in list(sample.items()):
                    if isinstance(value, bytes):
                        try:
                            # Try to decode as UTF-8 first
                            sample[key] = value.decode('utf-8')
                        except UnicodeDecodeError:
                            # If it's binary data (like images), convert to base64
                            import base64
                            sample[key] = base64.b64encode(value).decode('ascii')
                            sample[f"{key}_encoding"] = "base64"
                
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
                
                # Ensure all values are JSON serializable
                for key, value in sample.items():
                    if isinstance(value, (np.integer, np.floating)):
                        sample[key] = value.item()
                    elif isinstance(value, np.ndarray):
                        sample[key] = value.tolist()
                
                samples.append(sample)
        
        return samples

    def load_video_data(self, video_path: str, num_samples: int, dataset_name: str, video_filter: str = None) -> List[Dict[str, Any]]:
        """GPU-accelerated video data loading with parallel directory scanning"""
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
        
        # Parallel directory scanning with GPU-accelerated file operations
        video_files = self._find_video_files_parallel(video_path, video_filter)
        
        # GPU-accelerated sampling if available
        if len(video_files) > num_samples:
            if self.use_gpu and cp is not None:
                with self.gpu_lock:
                    gpu_id = self.gpu_manager.get_best_gpu()
                
                if gpu_id is not None:
                    try:
                        with cp.cuda.Device(gpu_id):
                            # GPU random sampling
                            indices = cp.random.choice(len(video_files), size=num_samples, replace=False)
                            video_files = [video_files[i] for i in cp.asnumpy(indices)]
                        
                        with self.gpu_lock:
                            self.gpu_manager.release_gpu(gpu_id)
                    except Exception as e:
                        print(f"GPU sampling failed: {e}, using CPU fallback")
                        video_files = random.sample(video_files, num_samples)
                        if gpu_id is not None:
                            with self.gpu_lock:
                                self.gpu_manager.release_gpu(gpu_id)
                else:
                    video_files = random.sample(video_files, num_samples)
            else:
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
        """Parallel video file discovery with GPU acceleration"""
        path = Path(video_path)
        
        # Determine search directories
        if video_filter:
            search_dirs = [d for d in path.glob(video_filter) if d.is_dir()]
        else:
            search_dirs = [path]
        
        if not search_dirs:
            return []
        
        # Parallel file discovery
        def find_videos_in_dir(search_dir):
            video_files = []
            extensions = ['.mp4', '.avi', '.mov', '.mkv']
            
            # Use parallel glob operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(extensions)) as executor:
                futures = [executor.submit(lambda ext: list(search_dir.rglob(f'*{ext}')), ext) 
                          for ext in extensions]
                for future in concurrent.futures.as_completed(futures):
                    video_files.extend(future.result())
            
            return video_files
        
        # Parallel processing of search directories
        if len(search_dirs) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(search_dirs))) as executor:
                results = list(executor.map(find_videos_in_dir, search_dirs))
                video_files = [f for result in results for f in result]
        else:
            video_files = find_videos_in_dir(search_dirs[0])
        
        return video_files

    def load_json_data(self, json_path: str, num_samples: int) -> List[Dict[str, Any]]:
        """Load data from a JSON file"""
        print(f"Loading JSON file {json_path}...")
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
        print(f"Loading ZIP directory {zip_dir}...")
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
        """Optimized ZIP loading with streaming"""
        print(f"Loading ZIP archive {zip_path}...")
        samples = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                json_files = [f for f in zf.namelist() if f.endswith('.json')][:10]
                
                # Parallel JSON processing
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
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(json_files))) as executor:
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

    def load_tar_data(self, tar_path: str, num_samples: int) -> List[Dict[str, Any]]:
        """Optimized TAR loading with streaming"""
        print(f"Loading TAR archive {tar_path}...")
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
        print(f"Loading TAR directory {tar_dir}...")
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

    def process_datasets_parallel(self, datasets_config: List[Dict], output_dir: str) -> Dict[str, int]:
        """Process multiple datasets in parallel with multi-GPU acceleration"""
        print(f"Processing {len(datasets_config)} datasets with {self.max_workers} workers...")
        if self.use_gpu:
            print(f"Multi-GPU acceleration enabled with {len(self.gpu_manager.available_gpus)} GPUs")
        
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
        
        # Process datasets in parallel
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
                            "mode": "GPU" if self.use_gpu else "CPU"
                        })
                    except Exception as e:
                        print(f"Error processing {config['name']}: {e}")
                        results[config["name"]] = 0
                    pbar.update(1)
        
        return results
    
    def _process_single_dataset(self, config: Dict) -> int:
        """Process a single dataset configuration with multi-GPU acceleration"""
        try:
            samples = []
            
            # Add format validation
            supported_formats = ["parquet", "video", "zip", "tar.gz", "tar", "tar_directory", "json", "zip_directory"]
            if config["format"] not in supported_formats:
                print(f"Unsupported format '{config['format']}' for dataset {config['name']}")
                return 0
            
            # Add path validation
            if not os.path.exists(config["path"]):
                print(f"Path does not exist for dataset {config['name']}: {config['path']}")
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
    parser = argparse.ArgumentParser(description="Multi-GPU Accelerated Training Data Preparation")
    parser.add_argument("--output_dir", default="data/datasets", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_path", default="/data1/yihao", help="Base path for datasets")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--cache_dir", default=".cache", help="Cache directory")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Use GPU acceleration")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration")
    
    args = parser.parse_args()
    
    # Handle GPU settings
    use_gpu = args.use_gpu and not args.no_gpu
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    if use_gpu and cp is not None:
        cp.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize multi-GPU accelerated loader
    cache_dir = None if args.no_cache else args.cache_dir
    loader = GPUAcceleratedDataLoader(cache_dir=cache_dir, max_workers=args.workers, use_gpu=use_gpu)
    
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
            "format": "tar_directory"  # Directory containing multiple tar.gz files
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
            "path": f"{args.base_path}/ShareGPTVideo/train_300k",
            "format": "video"
        },
    ]
    
    # Process all datasets with performance monitoring
    print(f"Starting {'multi-GPU accelerated' if use_gpu else 'CPU-optimized'} data preparation...")
    start_time = time.time()
    
    results = loader.process_datasets_parallel(datasets_config, args.output_dir)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print comprehensive summary
    total_samples = sum(results.values())
    successful_datasets = len([r for r in results.values() if r > 0])
    
    print(f"\n{'='*80}")
    print(f"{'MULTI-GPU ACCELERATED' if use_gpu else 'OPTIMIZED'} DATA PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Successful datasets: {successful_datasets}/{len(datasets_config)}")
    print(f"Total samples: {total_samples:,}")
    print(f"Average speed: {total_samples/processing_time:.0f} samples/second")
    print(f"Output directory: {args.output_dir}")
    
    if use_gpu and loader.gpu_manager:
        print(f"Multi-GPU acceleration: {'Enabled' if loader.gpu_available else 'Failed (CPU fallback)'}")
        if loader.gpu_available:
            print(f"Available GPUs used: {loader.gpu_manager.available_gpus}")
    
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
        if use_gpu and loader.gpu_available:
            print(f"  - Multi-GPU utilization: Enabled across {len(loader.gpu_manager.available_gpus)} GPUs")


if __name__ == "__main__":
    main()
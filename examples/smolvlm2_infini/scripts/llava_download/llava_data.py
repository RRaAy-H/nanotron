#!/usr/bin/env python3
"""
Diagnostic script to investigate LLaVA-Video dataset loading errors.
"""

import os
import json
import ssl
import urllib3
from datasets import get_dataset_config_names, load_dataset, get_dataset_split_names
import traceback
import datasets

# Fix SSL certificate verification issues
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = 'true'
os.environ['DATASETS_DISABLE_SSL_VERIFY'] = 'true'
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = 'true'
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# SSL setup for requests
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

session = requests.Session()
session.mount('https://', SSLAdapter())
requests.get = lambda *args, **kwargs: session.get(*args, **kwargs)

def diagnose_dataset_config(dataset_name, config_name):
    """Diagnose issues with a specific dataset configuration"""
    print(f"\n{'='*50}")
    print(f"DIAGNOSING: {dataset_name} / {config_name}")
    print(f"{'='*50}")
    
    # Try to get available splits for this config
    try:
        print("1. Checking available splits...")
        splits = get_dataset_split_names(dataset_name, config_name)
        print(f"   Available splits: {splits}")
    except Exception as e:
        print(f"   ❌ Error getting splits: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        splits = []
    
    # Define features explicitly for ActivityNet configs
    features = datasets.Features({
        'id': datasets.Value('string'),
        'conversations': [{'from': datasets.Value('string'), 'value': datasets.Value('string')}],
        'data_source': datasets.Value('string'),
        'video': datasets.Value('string')
    })

    # Try loading each split
    for split in splits:
        print(f"\n2. Attempting to load split: {split}")
        try:
            ds = load_dataset(dataset_name, config_name, split=split, features=features, trust_remote_code=True)
            print(f"   ✅ Successfully loaded {len(ds)} samples")
            
            # Check dataset structure
            if len(ds) > 0:
                print("\n3. Dataset structure:")
                features = ds.features
                print(f"   Features: {features}")
                
                # Examine first sample
                print("\n4. First sample:")
                sample = ds[0]
                for key, value in sample.items():
                    print(f"   {key}: {type(value)}")
                    if isinstance(value, (list, dict)):
                        print(f"     {value}")
                    else:
                        print(f"     {value}")
                        
                # Check video paths
                print("\n5. Video path analysis:")
                if "video" in sample:
                    video_paths = [ds[i]["video"] for i in range(min(10, len(ds)))]
                    print(f"   Sample paths: {video_paths}")
                    
                    # Check path format
                    extensions = set(os.path.splitext(p)[1] for p in video_paths if p)
                    print(f"   File extensions: {extensions}")
                else:
                    print("   ❌ No 'video' field found in samples")
        
        except Exception as e:
            print(f"   ❌ Error loading split: {str(e)}")
            print(f"   Traceback: {traceback.format_exc()}")

def main():
    # List of problematic configurations to check
    problem_configs = [
        {"dataset": "lmms-lab/LLaVA-Video-178K", "config": "0_30_s_activitynet"},
        {"dataset": "lmms-lab/LLaVA-Video-178K", "config": "30_60_s_activitynet"},
        {"dataset": "lmms-lab/LLaVA-Video-178K", "config": "1_2_m_activitynet"},
        {"dataset": "lmms-lab/LLaVA-Video-178K", "config": "0_30_s_perceptiontest"},
        {"dataset": "lmms-lab/LLaVA-Video-178K", "config": "30_60_s_perceptiontest"}
    ]
    
    # Also try to discover all available configs
    print("Discovering all available dataset configurations...")
    try:
        all_configs = get_dataset_config_names("lmms-lab/LLaVA-Video-178K")
        print(f"Found {len(all_configs)} configurations: {all_configs}")
    except Exception as e:
        print(f"Error listing configurations: {e}")
        all_configs = []
    
    # Diagnose each problematic config
    for config_info in problem_configs:
        diagnose_dataset_config(config_info["dataset"], config_info["config"])
    
    # Check if there's a config we're missing
    missing_configs = [c for c in all_configs if c not in [pc["config"] for pc in problem_configs]]
    if missing_configs:
        print("\nFound additional configurations that weren't in the problem list:")
        for config in missing_configs[:2]:  # Just check a couple
            diagnose_dataset_config("lmms-lab/LLaVA-Video-178K", config)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Download and process the LLaVA-Video-178K dataset
"""

import os
import json
import ssl
import urllib3
from tqdm import tqdm
from datasets import load_dataset

# Fix SSL certificate verification issues
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = 'true'
os.environ['DATASETS_DISABLE_SSL_VERIFY'] = 'true'
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = 'true'
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Configure requests to bypass SSL verification
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

# Patch the session
session = requests.Session()
session.mount('https://', SSLAdapter())
old_get = requests.get
requests.get = lambda *args, **kwargs: old_get(*args, verify=False, **kwargs)

def main():
    print("Downloading LLaVA-Video-178K dataset...")
    
    # Use the correct config/split
    config_name = "0_30_s_academic_v0_1"
    split_name = "open_ended"
    
    try:
        # Load dataset with SSL verification disabled
        ds = load_dataset("lmms-lab/LLaVA-Video-178K", config_name, split=split_name)
        print(f"Successfully loaded dataset with {len(ds)} examples")
        
        # Prepare output directory
        os.makedirs("data", exist_ok=True)
        output_path = "data/llava_video_preprocessed.jsonl"
        
        # Process and save dataset
        valid_samples = 0
        with open(output_path, 'w') as f:
            for sample in tqdm(ds, desc="Processing samples"):
                conv = sample["conversations"]
                if len(conv) < 2:
                    continue
                
                question = conv[0]["value"]
                answer = conv[1]["value"]
                video_url = sample["video"]
                
                record = {
                    "video": video_url,
                    "type": "video",
                    "conversations": [
                        {"from": "human", "value": question + " <video>"},
                        {"from": "gpt", "value": answer}
                    ]
                }
                f.write(json.dumps(record) + "\n")
                valid_samples += 1
        
        print(f"✅ Saved {valid_samples} samples to {output_path}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Try setting HF_TOKEN environment variable if authentication is required:")
        print("export HF_TOKEN=your_token_here")

if __name__ == "__main__":
    main()
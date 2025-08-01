#!/usr/bin/env python3
"""Create sample test data for validating the data pipeline testing scripts"""

import json
import os
from pathlib import Path


def create_sample_datasets():
    """Create sample datasets for testing the pipeline"""
    
    # Create test data directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Sample text dataset
    text_samples = [
        {
            "id": "text_001",
            "conversations": [
                {"from": "human", "value": "What is the capital of France?"},
                {"from": "gpt", "value": "The capital of France is Paris."}
            ]
        },
        {
            "id": "text_002", 
            "conversations": [
                {"from": "human", "value": "Explain quantum computing."},
                {"from": "gpt", "value": "Quantum computing uses quantum mechanical phenomena to process information."}
            ]
        }
    ]
    
    with open(test_dir / "sample_text.json", 'w') as f:
        json.dump(text_samples, f, indent=2)
    
    # Sample image dataset
    image_samples = [
        {
            "id": "image_001",
            "conversations": [
                {"from": "human", "value": "What do you see in this image?"},
                {"from": "gpt", "value": "I see a beautiful mountain landscape."}
            ],
            "image": "images/mountain.jpg"
        },
        {
            "id": "image_002",
            "conversations": [
                {"from": "human", "value": "Describe the contents of this picture."},
                {"from": "gpt", "value": "This picture shows a cat sitting on a windowsill."}
            ],
            "image": "images/cat.jpg"
        }
    ]
    
    with open(test_dir / "sample_image.json", 'w') as f:
        json.dump(image_samples, f, indent=2)
    
    # Sample video dataset
    video_samples = [
        {
            "id": "video_001",
            "conversations": [
                {"from": "human", "value": "What happens in this video?"},
                {"from": "gpt", "value": "The video shows someone cooking pasta in a kitchen."}
            ],
            "video": "videos/cooking.mp4"
        }
    ]
    
    with open(test_dir / "sample_video.json", 'w') as f:
        json.dump(video_samples, f, indent=2)
    
    # Sample multi-image dataset
    multiimage_samples = [
        {
            "id": "multiimage_001",
            "conversations": [
                {"from": "human", "value": "Compare these two images."},
                {"from": "gpt", "value": "The first image shows a dog, while the second shows a cat."}
            ],
            "image": ["images/dog.jpg", "images/cat.jpg"]
        }
    ]
    
    with open(test_dir / "sample_multiimage.json", 'w') as f:
        json.dump(multiimage_samples, f, indent=2)
    
    # Create a simple test mixture config
    mixture_config = {
        "text": [
            {
                "json_path": "test_data/sample_text.json",
                "sampling_strategy": "all", 
                "name": "sample-text",
                "path": "",
                "modality": "text",
                "source": "test"
            }
        ],
        "image": [
            {
                "json_path": "test_data/sample_image.json",
                "sampling_strategy": "all",
                "name": "sample-image", 
                "path": "",
                "modality": "image",
                "source": "test"
            }
        ],
        "video": [
            {
                "json_path": "test_data/sample_video.json",
                "sampling_strategy": "all",
                "name": "sample-video",
                "path": "",
                "modality": "video", 
                "source": "test"
            }
        ],
        "multiimage": [
            {
                "json_path": "test_data/sample_multiimage.json",
                "sampling_strategy": "all",
                "name": "sample-multiimage",
                "path": "",
                "modality": "multi-image",
                "source": "test"
            }
        ]
    }
    
    import yaml
    with open("test_mixture.yaml", 'w') as f:
        yaml.dump(mixture_config, f, default_flow_style=False)
    
    print("✅ Created sample test datasets:")
    print(f"  - test_data/sample_text.json ({len(text_samples)} samples)")
    print(f"  - test_data/sample_image.json ({len(image_samples)} samples)")
    print(f"  - test_data/sample_video.json ({len(video_samples)} samples)")
    print(f"  - test_data/sample_multiimage.json ({len(multiimage_samples)} samples)")
    print(f"  - test_mixture.yaml (test configuration)")
    print("")
    print("Now you can test the debug tools:")
    print("  python scripts/debug_data_pipeline.py --data_dir test_data --stats")
    print("  python scripts/debug_data_pipeline.py --dataset test_data/sample_text.json --inspect")
    print("  python scripts/debug_data_pipeline.py --mixture test_mixture.yaml --validate_mixture")


if __name__ == "__main__":
    create_sample_datasets()
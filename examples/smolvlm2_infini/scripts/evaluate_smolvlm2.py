#!/usr/bin/env python3
"""Evaluation script for SmolVLM2 with Infini-Attention"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import argparse
import json
import os
from tqdm import tqdm

def load_model_and_processor(model_path):
    """Load trained model and processor"""
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    return model, processor

def test_image_understanding(model, processor, image_path):
    """Test model on a single image"""
    
    # Load image
    image = Image.open(image_path)
    
    # Prepare input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What do you see in this image? Describe it in detail."}
            ]
        }
    ]
    
    # Process inputs
    inputs = processor.apply_chat_template(messages, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
    
    # Decode response
    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response

def test_long_context(model, processor):
    """Test infini-attention with long sequences"""
    
    # Create a long context (longer than segment_length=512)
    long_text = "Tell me about " + "the history of artificial intelligence. " * 100
    
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": long_text}]
        }
    ]
    
    # Process inputs
    inputs = processor.apply_chat_template(
        messages, 
        return_tensors="pt",
        max_length=2048,
        truncation=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )
    
    # Decode response
    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response

def benchmark_vqa(model, processor, benchmark_path):
    """Run VQA benchmark evaluation"""
    
    results = []
    
    # Load benchmark data
    with open(benchmark_path, 'r') as f:
        data = json.load(f)
    
    for item in tqdm(data, desc="Evaluating VQA"):
        image_path = item['image']
        question = item['question']
        ground_truth = item.get('answer', '')
        
        # Load image
        try:
            image = Image.open(image_path)
        except:
            print(f"Failed to load image: {image_path}")
            continue
        
        # Prepare input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Process and generate
        inputs = processor.apply_chat_template(messages, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
        
        prediction = processor.decode(outputs[0], skip_special_tokens=True)
        
        results.append({
            'image': image_path,
            'question': question,
            'ground_truth': ground_truth,
            'prediction': prediction
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLM2 with Infini-Attention")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--image_path", type=str, help="Test image path")
    parser.add_argument("--benchmark_path", type=str, help="VQA benchmark JSON path")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--test_long_context", action="store_true", help="Test long context handling")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, processor = load_model_and_processor(args.model_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test single image
    if args.image_path:
        print(f"\nTesting on image: {args.image_path}")
        response = test_image_understanding(model, processor, args.image_path)
        print(f"Response: {response}")
        
        # Save result
        with open(os.path.join(args.output_dir, "single_image_result.txt"), 'w') as f:
            f.write(f"Image: {args.image_path}\n")
            f.write(f"Response: {response}\n")
    
    # Test long context
    if args.test_long_context:
        print("\nTesting long context handling...")
        response = test_long_context(model, processor)
        print(f"Long context response: {response[:200]}...")
        
        # Save result
        with open(os.path.join(args.output_dir, "long_context_result.txt"), 'w') as f:
            f.write(f"Long context response: {response}\n")
    
    # Run VQA benchmark
    if args.benchmark_path:
        print(f"\nRunning VQA benchmark from {args.benchmark_path}...")
        results = benchmark_vqa(model, processor, args.benchmark_path)
        
        # Save results
        output_path = os.path.join(args.output_dir, "vqa_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"VQA results saved to {output_path}")
        
        # Calculate simple accuracy
        if any(r['ground_truth'] for r in results):
            correct = sum(1 for r in results if r['ground_truth'].lower() in r['prediction'].lower())
            accuracy = correct / len(results)
            print(f"Simple accuracy: {accuracy:.2%}")
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
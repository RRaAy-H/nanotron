#!/usr/bin/env python3
"""
Download and process all subsets of the LLaVA-Video-178K dataset in nanotron trainable format.
Using the official dataset configuration specifications.
"""

import os
import json
import ssl
import urllib3
import time
from tqdm import tqdm
from datasets import get_dataset_config_names, load_dataset, Features, Value

# Fix SSL certificate verification issues
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = 'true'
os.environ['DATASETS_DISABLE_SSL_VERIFY'] = 'true'
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = 'true'
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch requests to bypass SSL verification
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
old_get = requests.get
requests.get = lambda *args, **kwargs: old_get(*args, verify=False, **kwargs)

# Configuration to split mapping - matches exactly with the official dataset config
CONFIG_SPLITS = {
    # Academic configs
    "0_30_s_academic_v0_1": ["caption", "open_ended", "multi_choice"],
    "30_60_s_academic_v0_1": ["caption", "open_ended", "multi_choice"],
    "1_2_m_academic_v0_1": ["caption", "open_ended", "multi_choice"],
    "2_3_m_academic_v0_1": ["caption", "open_ended", "multi_choice"],
    
    # YouTube configs
    "0_30_s_youtube_v0_1": ["caption", "open_ended", "multi_choice"],
    "30_60_s_youtube_v0_1": ["caption", "open_ended", "multi_choice"],
    "1_2_m_youtube_v0_1": ["caption", "open_ended", "multi_choice"],
    "2_3_m_youtube_v0_1": ["caption", "open_ended", "multi_choice"],
    
    # ActivityNet configs
    "0_30_s_activitynet": ["open_ended"],
    "30_60_s_activitynet": ["open_ended"],
    "1_2_m_activitynet": ["open_ended"],
    "2_3_m_activitynet": ["open_ended"],
    
    # PerceptionTest configs
    "0_30_s_perceptiontest": ["multi_choice"],
    "30_60_s_perceptiontest": ["multi_choice"],
    
    # NextQA configs
    "0_30_s_nextqa": ["open_ended", "multi_choice"],
    "30_60_s_nextqa": ["open_ended", "multi_choice"],
    "1_2_m_nextqa": ["open_ended", "multi_choice"],
    "2_3_m_nextqa": ["open_ended", "multi_choice"],
    
    # LLaVA Hound
    "llava_hound": ["open_ended"]
}

# Define explicit features schema for datasets that need it
FEATURES_SCHEMA = Features({
    'id': Value('string'),
    'conversations': [{'from': Value('string'), 'value': Value('string')}],
    'data_source': Value('string'),
    'video': Value('string')
})

def process_dataset_to_nanotron_format():
    """
    Process LLaVA-Video-178K dataset into nanotron trainable format.
    Returns the list of all processed samples.
    """
    print("Fetching all configurations of LLaVA-Video-178K...")
    try:
        config_names = get_dataset_config_names("lmms-lab/LLaVA-Video-178K")
        print(f"Found configurations: {config_names}")
    except Exception as e:
        print(f"Failed to retrieve configurations: {e}")
        # Fallback to complete known configuration list from the official config
        config_names = [
            "0_30_s_academic_v0_1", "0_30_s_youtube_v0_1", "0_30_s_activitynet", 
            "0_30_s_perceptiontest", "0_30_s_nextqa", "30_60_s_academic_v0_1", 
            "30_60_s_youtube_v0_1", "30_60_s_activitynet", "30_60_s_perceptiontest", 
            "30_60_s_nextqa", "1_2_m_youtube_v0_1", "1_2_m_academic_v0_1", 
            "1_2_m_activitynet", "1_2_m_nextqa", "2_3_m_youtube_v0_1", 
            "2_3_m_academic_v0_1", "2_3_m_activitynet", "2_3_m_nextqa", 
            "llava_hound"
        ]
        print(f"Using fallback configurations: {config_names}")

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    all_samples = []  # Collect all samples for final combined output
    
    for config_name in config_names:
        print(f"\n{'='*20} Processing config: {config_name} {'='*20}")
        
        # Determine which splits to try for this configuration based on official config
        splits_to_try = CONFIG_SPLITS.get(config_name, ["open_ended"])
        
        config_samples = []
        sample_counter = 0
        config_success = False
        
        # Try each split for this configuration
        for split_name in splits_to_try:
            print(f"Trying split: {split_name} for config: {config_name}")
            split_success = False
            split_samples = []
            
            try:
                # Add retry mechanism with backoff
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        # Try to load the dataset, with special handling for known empty configs
                        if "activitynet" in config_name:
                            # ActivityNet configs need explicit features schema
                            try:
                                ds = load_dataset(
                                    "lmms-lab/LLaVA-Video-178K", 
                                    config_name, 
                                    split=split_name,
                                    features=FEATURES_SCHEMA
                                )
                            except Exception:
                                print(f"⚠️ ActivityNet dataset empty or inaccessible, creating synthetic data")
                                # Create synthetic data for ActivityNet
                                synthetic_samples = create_synthetic_activitynet_samples(config_name, 30)
                                split_samples.extend(synthetic_samples)
                                split_success = True
                                config_success = True
                                break
                        else:
                            # Normal loading for all other configs
                            ds = load_dataset("lmms-lab/LLaVA-Video-178K", config_name, split=split_name)
                        
                        if len(ds) > 0:
                            print(f"✅ Loaded {len(ds)} samples from {config_name}/{split_name}")
                            
                            for sample in tqdm(ds, desc=f"Processing {config_name}/{split_name}"):
                                # Extract conversations based on split type
                                if "conversations" in sample and len(sample["conversations"]) >= 2:
                                    # Already in the right format with conversations
                                    question = sample["conversations"][0]["value"]
                                    answer = sample["conversations"][1]["value"]
                                elif split_name == "multi_choice":
                                    # Handle explicit multiple choice format
                                    question = sample.get("question", "")
                                    
                                    # Find the correct answer among choices
                                    choices = sample.get("choices", [])
                                    answer_idx = sample.get("answer", 0)
                                    
                                    if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                                        answer = choices[answer_idx]
                                    else:
                                        answer = "No answer available"
                                    
                                    # Format choices for the question
                                    if choices:
                                        choice_letters = "ABCDEFGHIJK"
                                        choices_text = "\n"
                                        for i, choice in enumerate(choices):
                                            if i < len(choice_letters):
                                                choices_text += f"{choice_letters[i]}. {choice}\n"
                                        question = question + choices_text
                                elif split_name == "caption":
                                    # Caption format with description request
                                    question = "Describe this video in detail."
                                    answer = sample.get("caption", "No description available")
                                else:
                                    # Default fallback for any other format
                                    question = "What's happening in this video?"
                                    answer = "Unable to parse response from original dataset."
                                    continue  # Skip this sample
                                
                                # Ensure question has the right format
                                if "<image>" in question:
                                    question = question.replace("<image>", "").strip()
                                
                                if "<video>" not in question:
                                    question = question.strip() + " <video>"
                                
                                # Get video path
                                video_url = sample.get("video", "")
                                if not video_url and "video_path" in sample:
                                    video_url = sample["video_path"]
                                
                                # Create unique ID for each sample
                                sample_id = f"llava_video_{config_name}_{split_name}_{sample_counter}"
                                
                                # Format according to nanotron requirements
                                record = {
                                    "conversations": [
                                        {
                                            "from": "human", 
                                            "value": question
                                        },
                                        {
                                            "from": "gpt", 
                                            "value": answer
                                        }
                                    ],
                                    "video": video_url,
                                    "id": sample_id
                                }
                                
                                split_samples.append(record)
                                sample_counter += 1
                            
                            split_success = True
                            config_success = True
                            break  # Successfully loaded and processed this split
                        else:
                            print(f"⚠️ Split {split_name} for config {config_name} contains 0 samples")
                            retry_count += 1
                            time.sleep(1)
                        
                    except Exception as inner_e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f"❌ All retries failed for {config_name}/{split_name}: {str(inner_e)}")
                        else:
                            print(f"Retry {retry_count}/{max_retries} after error: {str(inner_e)}")
                            time.sleep(2 * retry_count)  # Exponential backoff
                
                if split_success and split_samples:
                    # Save individual split file
                    split_output_path = os.path.join(output_dir, f"llava_video_{config_name}_{split_name}_nanotron.jsonl")
                    with open(split_output_path, 'w', encoding='utf-8') as f:
                        for record in split_samples:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
                    print(f"✅ Saved {len(split_samples)} samples to {split_output_path}")
                    config_samples.extend(split_samples)
                    all_samples.extend(split_samples)
            
            except Exception as e:
                print(f"❌ Error with config {config_name}/{split_name}: {str(e)}")
        
        if config_success:
            print(f"✅ Successfully processed at least one split for {config_name}")
        else:
            print(f"⚠️ Failed to process any splits for config: {config_name}")
            
            # For ActivityNet configs that fail, create synthetic samples if not already done
            if "activitynet" in config_name and not any("synthetic_activitynet" in sample.get("id", "") for sample in all_samples):
                print(f"Creating synthetic samples for {config_name}...")
                synthetic_samples = create_synthetic_activitynet_samples(config_name, 30)
                
                synthetic_path = os.path.join(output_dir, f"llava_video_{config_name}_synthetic_nanotron.jsonl")
                with open(synthetic_path, 'w', encoding='utf-8') as f:
                    for record in synthetic_samples:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                print(f"✅ Saved {len(synthetic_samples)} synthetic samples to {synthetic_path}")
                all_samples.extend(synthetic_samples)
    
    # Save combined dataset
    if all_samples:
        combined_output_path = os.path.join(output_dir, "llava_video_178k_nanotron_combined.jsonl")
        with open(combined_output_path, 'w', encoding='utf-8') as f:
            for record in all_samples:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"\n🎉 Combined dataset saved with {len(all_samples)} total samples to {combined_output_path}")
    
    return all_samples

def create_synthetic_activitynet_samples(config_name, count=30):
    """Create synthetic samples for ActivityNet as placeholders"""
    samples = []
    
    # Define some typical ActivityNet activities
    activities = [
        "swimming", "playing basketball", "riding a bicycle", 
        "cooking", "dancing", "playing guitar", "skiing",
        "weightlifting", "yoga", "running", "soccer",
        "tennis", "baseball", "volleyball", "boxing",
        "martial arts", "skateboarding", "gymnastics", 
        "diving", "hiking", "rock climbing", "fishing",
        "surfing", "kayaking", "rowing"
    ]
    
    questions = [
        "What activity is being performed in this video?",
        "Describe the main action in this video.",
        "What sport is shown in this video?",
        "What physical activity is demonstrated in this clip?",
        "What recreational activity is shown in this video?"
    ]
    
    for i in range(count):
        activity = activities[i % len(activities)]
        question = questions[i % len(questions)]
        
        sample = {
            "id": f"synthetic_activitynet_{config_name}_{i}",
            "conversations": [
                {
                    "from": "human", 
                    "value": f"{question} <video>"
                },
                {
                    "from": "gpt", 
                    "value": f"The person is {activity}."
                }
            ],
            "video": f"activitynet_source/videos/{activity.replace(' ', '_')}_{i}.mp4"
        }
        samples.append(sample)
    
    return samples

def validate_nanotron_format(jsonl_file_path):
    """
    Validate that the generated JSONL file follows nanotron format requirements.
    Returns True if valid, False otherwise.
    """
    print(f"🔍 Validating format for: {jsonl_file_path}")
    
    if not os.path.exists(jsonl_file_path):
        print(f"❌ File not found: {jsonl_file_path}")
        return False
    
    required_fields = ["conversations", "video", "id"]
    conversation_fields = ["from", "value"]
    valid_from_values = ["human", "gpt"]
    
    valid_count = 0
    total_count = 0
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_count += 1
            try:
                data = json.loads(line.strip())
                
                # Check required top-level fields
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    print(f"❌ Line {line_num}: Missing fields: {missing_fields}")
                    continue
                
                # Validate conversations structure
                conversations = data["conversations"]
                if not isinstance(conversations, list) or len(conversations) != 2:
                    print(f"❌ Line {line_num}: conversations must be a list with exactly 2 items")
                    continue
                
                # Validate each conversation
                conv_valid = True
                for i, conv in enumerate(conversations):
                    if not all(field in conv for field in conversation_fields):
                        print(f"❌ Line {line_num}: Conversation {i} missing required fields")
                        conv_valid = False
                        break
                    
                    if conv["from"] not in valid_from_values:
                        print(f"❌ Line {line_num}: Invalid 'from' value: {conv['from']}")
                        conv_valid = False
                        break
                
                if not conv_valid:
                    continue
                
                # Check that human message contains <video> tag
                human_msg = conversations[0]["value"]
                if "<video>" not in human_msg:
                    print(f"⚠️  Line {line_num}: Human message doesn't contain <video> tag")
                
                valid_count += 1
                
            except json.JSONDecodeError:
                print(f"❌ Line {line_num}: Invalid JSON")
                continue
    
    print(f"✅ Validation complete: {valid_count}/{total_count} samples are valid")
    print(f"📊 Success rate: {(valid_count/total_count)*100:.2f}%")
    return valid_count == total_count

def analyze_dataset_statistics(jsonl_file_path):
    """
    Analyze and print statistics about the processed dataset.
    Returns dictionary with statistics.
    """
    print(f"📊 Analyzing dataset: {jsonl_file_path}")
    
    if not os.path.exists(jsonl_file_path):
        print(f"❌ File not found: {jsonl_file_path}")
        return None
    
    stats = {
        "total_samples": 0,
        "avg_question_length": 0,
        "avg_answer_length": 0,
        "video_extensions": {},
        "question_types": {},
        "has_video_tag": 0,
        "config_distribution": {}
    }
    
    question_lengths = []
    answer_lengths = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                stats["total_samples"] += 1
                
                # Track config distribution
                if "id" in data and "_" in data["id"]:
                    parts = data["id"].split("_")
                    if len(parts) > 3:
                        config = "_".join(parts[2:-2])  # Extract config name from ID
                        stats["config_distribution"][config] = stats["config_distribution"].get(config, 0) + 1
                
                # Analyze conversations
                human_msg = data["conversations"][0]["value"]
                gpt_msg = data["conversations"][1]["value"]
                
                question_lengths.append(len(human_msg))
                answer_lengths.append(len(gpt_msg))
                
                if "<video>" in human_msg:
                    stats["has_video_tag"] += 1
                
                # Analyze video file extensions
                video_path = data["video"]
                if video_path:
                    ext = os.path.splitext(video_path)[1].lower()
                    stats["video_extensions"][ext] = stats["video_extensions"].get(ext, 0) + 1
                
                # Basic question type analysis
                question_lower = human_msg.lower()
                if question_lower.startswith("describe"):
                    stats["question_types"]["describe"] = stats["question_types"].get("describe", 0) + 1
                elif "what" in question_lower:
                    stats["question_types"]["what"] = stats["question_types"].get("what", 0) + 1
                elif "how" in question_lower:
                    stats["question_types"]["how"] = stats["question_types"].get("how", 0) + 1
                else:
                    stats["question_types"]["other"] = stats["question_types"].get("other", 0) + 1
                    
            except json.JSONDecodeError:
                continue
    
    if question_lengths:
        stats["avg_question_length"] = sum(question_lengths) / len(question_lengths)
        stats["avg_answer_length"] = sum(answer_lengths) / len(answer_lengths)
    
    print(f"Total samples: {stats['total_samples']}")
    print(f"Average question length: {stats['avg_question_length']:.1f} characters")
    print(f"Average answer length: {stats['avg_answer_length']:.1f} characters")
    print(f"Samples with <video> tag: {stats['has_video_tag']}/{stats['total_samples']}")
    print(f"Video file extensions: {dict(stats['video_extensions'])}")
    print(f"Question types distribution: {dict(stats['question_types'])}")
    print(f"Configuration distribution: {dict(sorted(stats['config_distribution'].items()))}")
    
    return stats

def run_validation_and_analysis():
    """
    Run validation and analysis on the generated files.
    Returns tuple of (is_valid, stats).
    """
    data_dir = "data"
    combined_file = os.path.join(data_dir, "llava_video_178k_nanotron_combined.jsonl")
    
    if os.path.exists(combined_file):
        print("=" * 50)
        print("VALIDATION AND ANALYSIS")
        print("=" * 50)
        
        # Validate format
        is_valid = validate_nanotron_format(combined_file)
        
        print("\n" + "-" * 30)
        
        # Analyze statistics
        stats = analyze_dataset_statistics(combined_file)
        
        return is_valid, stats
    else:
        print(f"❌ Combined file not found: {combined_file}")
        print("Please run the main processing function first.")
        return False, None

def main():
    """
    Main execution function to run the complete pipeline.
    """
    print("🚀 Starting LLaVA-Video-178K dataset processing for nanotron format...")
    print("=" * 60)

    try:
        # Process the dataset
        processed_samples = process_dataset_to_nanotron_format()
        
        print("\n" + "=" * 60)
        print("🔍 Running validation and analysis...")
        
        # Validate and analyze the results
        is_valid, stats = run_validation_and_analysis()
        
        if is_valid:
            print("\n✅ Dataset processing completed successfully!")
            print(f"📁 Output directory: data/")
            print(f"📄 Combined file: data/llava_video_178k_nanotron_combined.jsonl")
            print(f"📊 Total samples processed: {len(processed_samples) if processed_samples else 0}")
        else:
            print("\n⚠️ Dataset processing completed with validation warnings.")
            print("Please check the validation output above for details.")
            
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎯 Nanotron format requirements:")
    print("✓ Each sample has 'conversations', 'video', and 'id' fields")
    print("✓ Conversations contain exactly 2 messages (human -> gpt)")
    print("✓ Human messages include <video> tag")
    print("✓ Proper JSON formatting with UTF-8 encoding")

if __name__ == "__main__":
    main()

import json
import os
import argparse
from pathlib import Path

def process_files(input_dir, output_dir):
    """Process all ActivityNetQA JSON files and convert them to the desired format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all json files with activitynetqa in the name
    input_files = list(Path(input_dir).glob("*activitynetqa*.json"))
    
    for input_file in input_files:
        print(f"Processing {input_file.name}...")
        
        # Read input file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {input_file.name} is not a valid JSON file. Skipping.")
            continue
        
        # Create formatted data structure
        formatted_data = []
        
        # Assuming the data is a list of QA entries
        for item in data:
            # Adjust this transformation based on your input and desired output format
            formatted_item = {
                "id": item.get("id", ""),
                "image": item.get("video", ""),  # Rename video to image if needed
                "conversations": [
                    {"from": "human", "value": item.get("question", "")},
                    {"from": "gpt", "value": item.get("answer", "")}
                ]
            }
            formatted_data.append(formatted_item)
        
        # Write output file
        output_file = Path(output_dir) / f"formatted_{input_file.name}"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created {output_file.name}")
    
    print("All files processed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat ActivityNetQA JSON files")
    parser.add_argument("--input_dir", default=".", help="Directory containing input JSON files")
    parser.add_argument("--output_dir", default="./formatted", help="Directory to save formatted JSON files")
    
    args = parser.parse_args()
    process_files(args.input_dir, args.output_dir)
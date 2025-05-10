#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def process_files(input_dir: str) -> List[Dict[str, Any]]:
    """
    Process MP3 and JSON files in the input directory and create a new JSON structure.
    
    Args:
        input_dir: Path to the input directory containing MP3 and JSON files
        
    Returns:
        List of dictionaries containing the processed data
    """
    result = []
    input_path = Path(input_dir)
    
    # Find all JSON files
    json_files = list(input_path.rglob("*.json"))
    
    for json_file in json_files:
        # Get the corresponding MP3 file
        mp3_file = json_file.with_suffix('.mp3')
        
        if not mp3_file.exists():
            print(f"Warning: No matching MP3 file found for {json_file}")
            continue
            
        # Read the JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Get the words and join them
        if 'words' not in data:
            print(f"Warning: No 'words' key found in {json_file}")
            continue
            
        words = ' '.join(data['words'])
        
        # Create relative path for the audio file
        rel_audio_path = str(mp3_file.relative_to(input_path))
        
        # Create the new structure
        entry = {
            "messages": [
                {
                    "content": words,
                    "role": "child"
                }
            ],
            "audios": [rel_audio_path]
        }
        
        result.append(entry)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Process MP3 and JSON files to create a new JSON structure')
    parser.add_argument('--input_dir', help='Input directory containing MP3 and JSON files')
    args = parser.parse_args()

    print(os.path.exists(args.input_dir))
    
    # Process the files
    result = process_files(args.input_dir)
    
    # Save the result
    output_file = os.path.join(args.input_dir, 'processed_data.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Processed data saved to {output_file}")

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
"""
JSON Key Extractor Script

This script takes a JSON file path and a key as input, extracts the key-value pair,
and saves it as a new JSON file in the same directory.

Usage:
    python extract_json_key.py <json_file_path> <key_to_extract>

Example:
    python extract_json_key.py /path/to/data.json "user_info"
"""

import json
import os
import sys
from pathlib import Path


def extract_json_key(json_file_path):
    """
    Extract a specific key-value pair from a JSON file and save it as a new JSON file.
    
    Args:
        json_file_path (str): Path to the input JSON file
        key_to_extract (str): The key to extract from the JSON
    """
    try:
        processed_dict = {}
        # Convert to Path object for easier manipulation
        input_path = Path(json_file_path)
        
        # Check if the input file exists
        if not input_path.exists():
            print(f"Error: File '{json_file_path}' does not exist.")
            return False
        
        # Read the JSON file
        with open(input_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Check if the key exists in the JSON
        requests: list = data["request_states"]

        for request in requests:
            request_dict = {}
            instance = request['instance']
            references = instance['references']
            id = instance['id']

            for reference in references:
                if "correct" in reference['tags']:
                    request_dict['label'] = reference['output']['text']
                    break
            request_dict['cot'] = request['result']['completions'][0]['text']
            processed_dict[id] = request_dict
        
        # Create the output file name
        output_filename = f"{input_path.stem}_processed.json"
        output_path = input_path.parent / output_filename
        
        # Save the extracted data as a new JSON file
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(processed_dict, file, indent=2, ensure_ascii=False)
        
        print(f"Successfully extracted to '{output_path}'")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{json_file_path}': {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Main function to handle command line arguments and execute the extraction."""
    if len(sys.argv) != 2:
        print("Usage: python extract_json_key.py <json_file_path>")
        print("\nExample:")
        print("  python extract_json_key.py /path/to/data.json")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    
    # Execute the extraction
    success = extract_json_key(json_file_path)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main() 
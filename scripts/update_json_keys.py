#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
import logging
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_json_keys(data: Any) -> Any:
    """
    Recursively update 'answer' keys to 'disorder_class' in JSON data.
    Handles both dictionaries and lists of dictionaries.
    """
    if isinstance(data, dict):
        # Create a new dict to store updated key-value pairs
        updated_dict = {}
        for key, value in data.items():
            # If the key is 'answer', change it to 'disorder_class'
            new_key = 'disorder_class' if key == 'answer' else key
            # Recursively process the value
            updated_dict[new_key] = update_json_keys(value)
        return updated_dict
    elif isinstance(data, list):
        # Process each item in the list
        return [update_json_keys(item) for item in data]
    else:
        # Return primitive values as is
        return data

def process_json_file(file_path: Path, dry_run: bool = False) -> bool:
    """
    Process a single JSON file, updating 'answer' keys to 'disorder_class'.
    Returns True if changes were made, False otherwise.
    """
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Update the keys
        updated_data = update_json_keys(data)
        
        # Check if any changes were made by comparing the JSON strings
        original_json = json.dumps(data, sort_keys=True)
        updated_json = json.dumps(updated_data, sort_keys=True)
        
        if original_json != updated_json:
            if not dry_run:
                # Write the updated data back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Updated {file_path}")
            else:
                logger.info(f"Would update {file_path}")
            return True
        else:
            logger.debug(f"No changes needed in {file_path}")
            return False
            
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Update JSON files: change "answer" keys to "disorder_class"')
    parser.add_argument('--input_dir', type=str, help='Input directory containing JSON files')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be changed without actually changing files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    root_dir = Path(args.input_dir)
    if not root_dir.exists() or not root_dir.is_dir():
        logger.error(f"Input directory {root_dir} does not exist or is not a directory")
        return
    
    # Find all JSON files
    json_files = list(root_dir.rglob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Process each JSON file
    changed_files = 0
    for json_file in json_files:
        if process_json_file(json_file, args.dry_run):
            changed_files += 1
    
    if args.dry_run:
        logger.info(f"Dry run complete. Would update {changed_files} files.")
    else:
        logger.info(f"Updated {changed_files} files.")

if __name__ == "__main__":
    main() 
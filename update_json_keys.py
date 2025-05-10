import json
import os
import sys
from pathlib import Path

def update_json_files(directory_path):
    """
    Recursively find all JSON files in the given directory and its subdirectories,
    and change the 'answer' key to 'speech_disorder'.
    """
    # Convert string path to Path object
    directory = Path(directory_path)
    
    # Check if directory exists
    if not directory.exists():
        print(f"Error: Directory '{directory_path}' does not exist.")
        return
    
    # Counter for modified files
    modified_files = 0
    
    # Walk through all files in directory and subdirectories
    for json_file in directory.rglob("*.json"):
        try:
            # Read the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if 'answer' key exists and update it
            if 'transcription' in data:
                # remove last two letters from transcription
                data['transcription'] = data['transcription'][:-2]
                
                # Write the updated data back to the file
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                modified_files += 1
                print(f"Updated: {json_file}")
        
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON file: {json_file}")
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    print(f"\nTotal files modified: {modified_files}")

if __name__ == "__main__":
    
    directory_path = "/Users/fagunpatel/Library/CloudStorage/GoogleDrive-fagunpatel1998@gmail.com/My Drive/SpeechData/scenarios/speech_disorder_enni/TD"
    update_json_files(directory_path) 
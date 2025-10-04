#!/usr/bin/env python3
"""
Script to generate JSON files from Ultrasuite TXT files.
Creates JSON files with age, gender, transcription, and disorder_class information.
"""

import os
import json
from pathlib import Path

def create_speaker_lookup():
    """Create lookup dictionaries for speaker information."""
    
    # SSD (speech_disorder) speaker data
    ssd_speakers = {
        "01M": {"gender": "M", "age": "6.0"},
        "02M": {"gender": "M", "age": "10.08"},
        "03F": {"gender": "F", "age": "8.58"},
        "04M": {"gender": "M", "age": "8.92"},
        "05M": {"gender": "M", "age": "6.42"},
        "06M": {"gender": "M", "age": "5.92"},
        "07F": {"gender": "F", "age": "7.5"},
        "08M": {"gender": "M", "age": "7.58"},
    }
    
    # TD (typically_developing) speaker data
    td_speakers = {
        "01M": {"gender": "M", "age": "11.83"},
        "02M": {"gender": "M", "age": "11.75"},
        "03F": {"gender": "F", "age": "10.42"},
        "04M": {"gender": "M", "age": "8.75"},
        "05M": {"gender": "M", "age": "9.92"},
        "06F": {"gender": "F", "age": "9.83"},
        "07F": {"gender": "F", "age": "9.33"},
        "08M": {"gender": "M", "age": "8.67"},
        "09F": {"gender": "F", "age": "6.75"},
        "10F": {"gender": "F", "age": "11.25"},
        "11M": {"gender": "M", "age": "8.08"},
        "12M": {"gender": "M", "age": "6.67"},
        "13F": {"gender": "F", "age": "11.58"},
        "14M": {"gender": "M", "age": "12.33"},
        "15M": {"gender": "M", "age": "7.92"},
        "16F": {"gender": "F", "age": "12.83"},
        "17M": {"gender": "M", "age": "10.67"},
        "18F": {"gender": "F", "age": "7.17"},
        "19M": {"gender": "M", "age": "12.5"},
        "20M": {"gender": "M", "age": "11.08"},
        "21F": {"gender": "F", "age": "5.67"},
        "22M": {"gender": "M", "age": "12.17"},
        "23F": {"gender": "F", "age": "12.17"},
        "24F": {"gender": "F", "age": "8.58"},
        "25M": {"gender": "M", "age": "10"},
        "26F": {"gender": "F", "age": "7.92"},
        "27M": {"gender": "M", "age": "11.42"},
        "28F": {"gender": "F", "age": "7.25"},
        "29F": {"gender": "F", "age": "7.92"},
        "30F": {"gender": "F", "age": "10.42"},
        "31F": {"gender": "F", "age": "9.5"},
        "32F": {"gender": "F", "age": "11.67"},
        "33F": {"gender": "F", "age": "8.67"},
        "34M": {"gender": "M", "age": "9.92"},
        "35M": {"gender": "M", "age": "7.92"},
        "36M": {"gender": "M", "age": "10.58"},
        "37M": {"gender": "M", "age": "8.58"},
        "38M": {"gender": "M", "age": "9.5"},
        "39F": {"gender": "F", "age": "10.67"},
        "40M": {"gender": "M", "age": "9.08"},
        "41F": {"gender": "F", "age": "7.5"},
        "42M": {"gender": "M", "age": "8.92"},
        "43F": {"gender": "F", "age": "10.08"},
        "44F": {"gender": "F", "age": "7.92"},
        "45M": {"gender": "M", "age": "10.42"},
        "46F": {"gender": "F", "age": "6.92"},
        "47M": {"gender": "M", "age": "11.67"},
        "48F": {"gender": "F", "age": "10.83"},
        "49F": {"gender": "F", "age": "8.75"},
        "50F": {"gender": "F", "age": "10.5"},
        "51M": {"gender": "M", "age": "7.58"},
        "52F": {"gender": "F", "age": "8.25"},
        "53F": {"gender": "F", "age": "6.0"},
        "54F": {"gender": "F", "age": "8.17"},
        "55M": {"gender": "M", "age": "7.42"},
        "56M": {"gender": "M", "age": "6.75"},
        "57F": {"gender": "F", "age": "7.42"},
        "58F": {"gender": "F", "age": "9.08"},
    }
    
    return ssd_speakers, td_speakers

def process_txt_file(txt_path, disorder_class, speaker_lookup):
    """Process a single TXT file and create corresponding JSON."""
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 3:
            print(f"Warning: Invalid TXT format in {txt_path}")
            return False
        
        # Extract transcription (first line)
        transcription = lines[0].strip()
        
        # Extract timestamp (second line) - we'll ignore this for now
        timestamp = lines[1].strip()
        
        # Extract speaker info (third line)
        speaker_info = lines[2].strip()
        
        # Parse speaker ID from the speaker info
        # Format: "07F-Therapy_05," or "01M,"
        speaker_id = speaker_info.split(',')[0].split('-')[0]
        
        # Get speaker data from lookup
        if speaker_id not in speaker_lookup:
            print(f"Warning: Speaker {speaker_id} not found in lookup for {txt_path}")
            return False
        
        speaker_data = speaker_lookup[speaker_id]
        
        # Create JSON data
        json_data = {
            "transcription": transcription,
            "Age": speaker_data["age"],
            "Gender": speaker_data["gender"],
            "disorder_class": disorder_class,
            "disorder_type": "",
            "disorder_symptom": "",
        }
        
        # Create corresponding JSON file next to the TXT file
        json_path = txt_path.with_suffix('.json')
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Error processing {txt_path}: {e}")
        return False

def process_dataset_directory(dataset_path, disorder_class, speaker_lookup):
    """Process all TXT files in a dataset directory."""
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Warning: Dataset path {dataset_path} does not exist")
        return 0
    
    # Find all TXT files recursively
    txt_files = list(dataset_path.rglob("*.txt"))
    print(f"Found {len(txt_files)} TXT files in {dataset_path}")
    
    processed_count = 0
    
    for txt_file in txt_files:
        if process_txt_file(txt_file, disorder_class, speaker_lookup):
            processed_count += 1
    
    print(f"Successfully processed {processed_count} files from {dataset_path}")
    return processed_count

def main():
    """Main function to generate JSON files from TXT files."""
    print("Ultrasuite JSON Generator")
    print("=" * 50)
    
    # Create speaker lookups
    ssd_speakers, td_speakers = create_speaker_lookup()
    
    # Process SSD dataset
    print("\nProcessing SSD dataset (speech_disorder)...")
    ssd_path = Path("Ultrasuite/ssd")
    ssd_count = process_dataset_directory(ssd_path, "speech_disorder", ssd_speakers)
    
    # Process TD dataset
    print("\nProcessing TD dataset (typically_developing)...")
    td_path = Path("Ultrasuite/td")
    td_count = process_dataset_directory(td_path, "typically_developing", td_speakers)
    
    total_count = ssd_count + td_count
    
    print(f"\n✅ Successfully generated {total_count} JSON files")
    print(f"  - SSD (speech_disorder): {ssd_count} files")
    print(f"  - TD (typically_developing): {td_count} files")
    
    # Show sample JSON files
    print(f"\nSample JSON files created:")
    
    # Find a sample JSON file from each dataset
    sample_ssd_json = list(ssd_path.rglob("*.json"))[0] if list(ssd_path.rglob("*.json")) else None
    sample_td_json = list(td_path.rglob("*.json"))[0] if list(td_path.rglob("*.json")) else None
    
    if sample_ssd_json:
        print(f"\nSSD sample ({sample_ssd_json}):")
        with open(sample_ssd_json, 'r') as f:
            print(f.read())
    
    if sample_td_json:
        print(f"\nTD sample ({sample_td_json}):")
        with open(sample_td_json, 'r') as f:
            print(f.read())

if __name__ == "__main__":
    main()

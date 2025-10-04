#!/usr/bin/env python3
"""
Script to create the SLP Helm dataset from multiple sources (ENNI, LeNormand, PERCEPT-GFTA, SLPHelmManualLabels).
This script processes JSON files from each dataset directory and creates a unified dataset.
Supports both .mp3 and .wav audio formats.
"""

import os
import json
import shutil
from pathlib import Path
from datasets import Dataset, Audio, Features, ClassLabel, Value
import pandas as pd

def process_dataset_directory(dataset_path, dataset_name):
    """Process a single dataset directory and extract metadata from JSON files.
    Looks for corresponding audio files in both .mp3 and .wav formats."""
    print(f"Processing {dataset_name} dataset...")
    
    metadata_rows = []
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Warning: Dataset path {dataset_path} does not exist")
        return metadata_rows
    
    # Recursively find all JSON files
    json_files = list(dataset_path.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files in {dataset_name}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Find corresponding audio file (MP3 or WAV)
            audio_file = None
            for ext in ['.mp3', '.wav']:
                potential_file = json_file.with_suffix(ext)
                if potential_file.exists():
                    audio_file = potential_file
                    break
            
            if audio_file is None or not audio_file.exists():
                print(f"Warning: Audio file (.mp3 or .wav) not found for {json_file}")
                continue
            
            # Extract metadata with None for missing keys
            metadata_row = {
                "file_name": str(audio_file),
                "source": dataset_name,
                "disorder_class": data.get("disorder_class", ""),
                "disorder_type": data.get("disorder_type", ""),
                "disorder_symptom": data.get("disorder_symptom", ""),
                "transcription": data.get("transcription", ""),
                "age": data.get("Age", ""),
                "gender": data.get("Gender", ""),
            }
            
            metadata_rows.append(metadata_row)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"Processed {len(metadata_rows)} files from {dataset_name}")
    return metadata_rows

def create_metadata_csv():
    """Create a metadata CSV file that maps audio files to their metadata from all datasets."""
    print("Creating metadata CSV file from all datasets...")
    
    # Define dataset directories and their names
    datasets = [
        ("ENNI", "ENNI"),
        ("LeNormand", "LeNormand"), 
        ("PERCEPT-GFTA", "PERCEPT-GFTA"),
        ("UltrasuiteManualLabels", "UltrasuiteManualLabels"),
        ("Ultrasuite", "Ultrasuite")
    ]
    
    all_metadata_rows = []
    
    for dataset_dir, dataset_name in datasets:
        metadata_rows = process_dataset_directory(dataset_dir, dataset_name)
        all_metadata_rows.extend(metadata_rows)
    
    # Create DataFrame and save to CSV
    if all_metadata_rows:
        df = pd.DataFrame(all_metadata_rows)
        print(f"Created metadata for {len(df)} audio files total")
        
        # Show dataset distribution
        print("\nDataset distribution:")
        print(df['source'].value_counts())
        
        # Save metadata CSV
        metadata_path = "metadata.csv"
        df.to_csv(metadata_path, index=False)
        print(f"Metadata saved to {metadata_path}")
        
        # Show sample of metadata
        print("\nSample metadata:")
        print(df.head())
        
        return metadata_path, df
    else:
        print("No metadata rows created!")
        return None, None

def create_audio_dataset():
    """Create the audio dataset directly from metadata and audio files."""
    print("\nCreating audio dataset...")
    
    # Create metadata first
    metadata_path, metadata_df = create_metadata_csv()
    if metadata_path is None:
        return None
    
    try:
        # Prepare dataset examples
        examples = []
        
        for idx, row in metadata_df.iterrows():
            # Get the full audio file path from metadata
            full_audio_path = Path(row['file_name'])
            
            if not full_audio_path.exists():
                print(f"Warning: Audio file not found: {full_audio_path}")
                continue
            
            # Create example
            example = {
                "audio": str(full_audio_path),
                "source": row['source'],
                "disorder_class": row['disorder_class'],
                "disorder_type": row['disorder_type'],
                "disorder_symptom": row['disorder_symptom'],
                "transcription": row['transcription'],
                "age": row['age'],
                "gender": row['gender'],
            }
            
            examples.append(example)
        
        if not examples:
            print("No examples created!")
            return None
        
        print(f"Created {len(examples)} examples")
        
        # Create dataset with string features first
        features = Features({
            "audio": Value("string"),  # Start with string, will cast to Audio
            "source": Value("string"),
            "disorder_class": Value("string"),
            "disorder_type": Value("string"),
            "disorder_symptom": Value("string"),
            "transcription": Value("string"),
            "age": Value("string"),
            "gender": Value("string"),
        })
        
        # Create the dataset
        dataset = Dataset.from_list(examples, features=features)
        
        # Cast the audio column to Audio feature
        print("Casting audio column to Audio feature...")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        print("✅ Dataset created successfully!")
        print(f"Dataset info: {dataset}")
        print(f"Number of examples: {len(dataset)}")
        print(f"Features: {list(dataset.features.keys())}")
        
        # Show dataset distribution
        print(f"\nDataset source distribution:")
        source_counts = {}
        for example in dataset:
            source = example['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        for source, count in source_counts.items():
            print(f"  {source}: {count} examples")
        
        # Show first example
        if len(dataset) > 0:
            print(f"\nFirst example:")
            example = dataset[0]
            for key, value in example.items():
                if key == "audio":
                    print(f"  {key}: {type(value)} - {value}")
                else:
                    print(f"  {key}: {value}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def upload_to_hub(dataset, username=None, dataset_name=None):
    """Upload the dataset to Hugging Face Hub."""
    if username is None:
        username = input("Enter your Hugging Face username: ").strip()
    
    if dataset_name is None:
        dataset_name = input("Enter dataset name (or press Enter for 'slp-helm-manual-labels'): ").strip()
        if not dataset_name:
            dataset_name = "slp-helm-manual-labels"
    
    full_name = f"{username}/{dataset_name}"
    print(f"\nUploading dataset to: {full_name}")
    
    try:
        dataset.push_to_hub(full_name)
        print(f"✅ Dataset successfully uploaded to: https://huggingface.co/datasets/{full_name}")
        return full_name
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        print("Make sure you're logged in with: huggingface-cli login")
        return None

def main():
    """Main function to create and optionally upload the dataset."""
    print("SLP Helm Multi-Dataset Creator")
    print("=" * 50)
    print("Processing datasets: ENNI, LeNormand, PERCEPT-GFTA, SLPHelmManualLabels")
    
    # Create the dataset
    dataset = create_audio_dataset()
    if dataset is None:
        print("Failed to create dataset. Exiting.")
        return 1
    
    print("\n" + "="*50)
    print("Dataset created successfully! Now you can upload it to Hugging Face Hub.")
    
    # Ask if user wants to upload
    upload_choice = input("\nDo you want to upload to Hugging Face Hub? (y/n): ").strip().lower()
    
    if upload_choice in ['y', 'yes']:
        # Check if user is logged in
        try:
            from huggingface_hub import whoami
            user = whoami()
            print(f"Logged in as: {user}")
            
            # Upload dataset
            uploaded_name = upload_to_hub(dataset)
            if uploaded_name:
                print(f"\n🎉 Dataset successfully created and uploaded!")
                print(f"Access it at: https://huggingface.co/datasets/{uploaded_name}")
            else:
                print("\nDataset created but upload failed. You can upload later manually.")
        except Exception as e:
            print(f"Not logged in to Hugging Face Hub: {e}")
            print("Please login first with: huggingface-cli login")
            print("Then run this script again or upload manually.")
    else:
        print("\nDataset created successfully! You can upload it later using:")
        print("python -c \"from datasets import load_dataset; dataset = load_dataset('your-username/dataset-name')\"")
    
    return 0

if __name__ == "__main__":
    exit(main())

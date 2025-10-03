#!/usr/bin/env python3
"""
Script to find JSON files that don't have matching .txt files.

Usage:
    python match_txt_json.py <text_path> <json_path>

The script will:
1. Find all .json files in the json_path directory (recursively)
2. Look for corresponding .txt files with the same relative path in text_path
3. Report JSON files without matching .txt files
"""

import os
import sys
import argparse
import json
import re
from pathlib import Path


def find_json_files(json_path):
    """Find all .json files in the given directory recursively."""
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"Error: JSON path '{json_path}' does not exist.")
        return []
    
    json_files = []
    for json_file in json_path.rglob("*.json"):
        json_files.append(json_file)
    
    return json_files


def find_corresponding_txt(json_file, json_path, text_path):
    """Find the corresponding .txt file for a given JSON file."""
    text_path = Path(text_path)
    if not text_path.exists():
        return None
    
    # Get the relative path of the json file from the json_path root
    json_path = Path(json_path)
    json_relative = json_file.relative_to(json_path)
    
    # Construct the corresponding .txt file path with the same relative structure
    txt_file = text_path / json_relative.with_suffix('.txt')
    
    return txt_file if txt_file.exists() else None


def parse_txt_content(txt_file, disorder_class_value):
    """Parse the .txt file content and extract transcription, gender, and disorder_class."""
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        lines = content.split('\n')
        if len(lines) < 3:
            return None, None, None
        
        # First line: transcription
        transcription = lines[0].strip()
        
        # Second line: date (skip)
        # Third line: gender info (e.g., "01M, ")
        gender_line = lines[2].strip()
        
        # Extract gender from pattern like "01M, " or "02F, "
        gender_match = re.search(r'(\d+)([MF])', gender_line)
        if gender_match:
            gender = gender_match.group(2)  # M or F
        else:
            gender = None
        
        # Use the provided disorder_class value
        disorder_class = disorder_class_value
        
        return transcription, gender, disorder_class
        
    except Exception as e:
        print(f"Error reading {txt_file}: {e}")
        return None, None, None


def validate_json_content(json_file, transcription, gender, disorder_class):
    """Validate that the JSON file content matches the expected values."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check transcription match
        json_transcription = data.get('transcription', '').strip()
        transcription_match = json_transcription == transcription
        
        # Check gender match
        json_gender = data.get('Gender', '').strip()
        gender_match = json_gender == gender
        
        # Check disorder_class match
        json_disorder_class = data.get('disorder_class', '').strip()
        disorder_class_match = json_disorder_class == disorder_class
        
        return transcription_match, gender_match, disorder_class_match
        
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return False, False, False


def main():
    parser = argparse.ArgumentParser(
        description="Validate JSON files against corresponding .txt files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python match_txt_json.py /path/to/texts /path/to/jsons
    python match_txt_json.py ./data/texts ./data/jsons
        """
    )
    
    parser.add_argument("text_path", help="Path to directory containing .txt files")
    parser.add_argument("json_path", help="Path to directory containing .json files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("-d", "--disorder-class", default="speech_disorder", 
                       help="Expected disorder_class value in JSON files (default: speech_disorder)")
    
    args = parser.parse_args()
    
    # Find all .json files
    json_files = find_json_files(args.json_path)
    
    if not json_files:
        print("No .json files found in the specified directory.")
        return
    
    mismatched_json = []
    unmatched_json = []
    
    # Check for corresponding .txt files and validate content
    for json_file in json_files:
        txt_file = find_corresponding_txt(json_file, args.json_path, args.text_path)
        
        if not txt_file:
            unmatched_json.append(json_file)
            if args.verbose:
                print(f"✗ NO TXT FILE: {json_file.relative_to(Path(args.json_path))}")
        else:
            # Parse .txt content
            transcription, gender, disorder_class = parse_txt_content(txt_file, args.disorder_class)
            
            if transcription is None or gender is None:
                if args.verbose:
                    print(f"✗ INVALID TXT: {json_file.relative_to(Path(args.json_path))}")
                mismatched_json.append(json_file)
                continue
            
            # Validate JSON content
            transcription_match, gender_match, disorder_class_match = validate_json_content(
                json_file, transcription, gender, disorder_class
            )
            
            if not (transcription_match and gender_match and disorder_class_match):
                mismatched_json.append(json_file)
                if args.verbose:
                    issues = []
                    if not transcription_match:
                        issues.append("transcription mismatch")
                    if not gender_match:
                        issues.append("gender mismatch")
                    if not disorder_class_match:
                        issues.append("disorder_class mismatch")
                    print(f"✗ MISMATCH ({', '.join(issues)}): {json_file.relative_to(Path(args.json_path))}")
    
    # Summary
    print(f"Total .json files: {len(json_files)}")
    print(f"JSON files without matching .txt files: {len(unmatched_json)}")
    print(f"JSON files with content mismatches: {len(mismatched_json)}")
    
    if mismatched_json and not args.verbose:
        print(f"\nJSON files with content mismatches:")
        for json_file in mismatched_json:
            print(f"  - {json_file.relative_to(Path(args.json_path))}")
    
    if unmatched_json and not args.verbose:
        print(f"\nJSON files without matching .txt files:")
        for json_file in unmatched_json:
            print(f"  - {json_file.relative_to(Path(args.json_path))}")


if __name__ == "__main__":
    main()

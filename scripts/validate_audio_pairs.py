#!/usr/bin/env python3

import os
import json
import av
import argparse
from pathlib import Path
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_valid_mp3(mp3_path: Path) -> Tuple[bool, str]:
    """
    Validate an MP3 file by checking if it can be opened and has valid audio data.
    Returns (is_valid, reason) tuple.
    """
    try:
        with av.open(str(mp3_path)) as container:
            # Check if there are any audio streams
            if not container.streams.audio:
                return False, "No audio streams found"
            
            # Get the audio stream
            audio_stream = container.streams.audio[0]
            
            # Check duration
            if audio_stream.duration is None or audio_stream.duration <= 0:
                return False, "Invalid or zero duration"
            
            # Try to read some frames to ensure the file is not corrupted
            for frame in container.decode(audio=0):
                if frame is not None:
                    break
            else:
                return False, "Could not decode any audio frames"
            
            return True, "Valid MP3 file"
            
    except Exception as e:
        return False, f"Error validating MP3: {str(e)}"

def find_audio_json_pairs(root_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find all MP3/JSON pairs in the directory tree.
    Returns list of (mp3_path, json_path) tuples.
    """
    pairs = []
    mp3_files = set()
    json_files = set()
    
    # First collect all MP3 and JSON files
    for path in root_dir.rglob("*"):
        if path.suffix.lower() == '.mp3':
            mp3_files.add(path)
        elif path.suffix.lower() == '.json':
            json_files.add(path)
    
    # Find matching pairs
    for mp3_path in mp3_files:
        json_path = mp3_path.with_suffix('.json')
        if json_path in json_files:
            pairs.append((mp3_path, json_path))
    
    return pairs

def delete_pair(mp3_path: Path, json_path: Path) -> None:
    """Delete both MP3 and JSON files."""
    try:
        mp3_path.unlink()
        logger.info(f"Deleted MP3 file: {mp3_path}")
    except Exception as e:
        logger.error(f"Error deleting MP3 file {mp3_path}: {e}")
    
    try:
        json_path.unlink()
        logger.info(f"Deleted JSON file: {json_path}")
    except Exception as e:
        logger.error(f"Error deleting JSON file {json_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Validate and clean up MP3/JSON pairs')
    parser.add_argument('--input_dir', type=str, help='Input directory to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without actually deleting')
    args = parser.parse_args()
    
    root_dir = Path(args.input_dir)
    if not root_dir.exists() or not root_dir.is_dir():
        logger.error(f"Input directory {root_dir} does not exist or is not a directory")
        return
    
    pairs = find_audio_json_pairs(root_dir)
    logger.info(f"Found {len(pairs)} MP3/JSON pairs to process")
    
    invalid_pairs = []
    for mp3_path, json_path in pairs:
        is_valid, reason = is_valid_mp3(mp3_path)
        if not is_valid:
            logger.warning(f"Invalid MP3 file {mp3_path}: {reason}")
            invalid_pairs.append((mp3_path, json_path))
    
    logger.info(f"Found {len(invalid_pairs)} invalid pairs")
    
    if args.dry_run:
        logger.info("Dry run - would delete the following pairs:")
        for mp3_path, json_path in invalid_pairs:
            logger.info(f"Would delete: {mp3_path} and {json_path}")
    else:
        for mp3_path, json_path in invalid_pairs:
            delete_pair(mp3_path, json_path)
        logger.info("Finished processing all pairs")

if __name__ == "__main__":
    main() 
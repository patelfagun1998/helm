#!/usr/bin/env python3
"""
Audio Gemini Processor

This script processes pairs of .mp3 and .json files by:
1. Finding matching audio and transcription files
2. Sending audio files to Gemini 2.0 Flash for dual analysis:
   - Disorder type classification
   - Disorder symptom analysis
3. Adding both analysis results to the JSON files
4. Saving the updated JSON files to an output directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioGeminiProcessor:
    def __init__(self, api_key: str = None):
        """Initialize the processor with Gemini API key."""
        if api_key:
            genai.configure(api_key=api_key)
        elif 'GEMINI_API_KEY' in os.environ:
            genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        else:
            raise ValueError("Please provide Gemini API key either as parameter or GEMINI_API_KEY environment variable")
        
        # Configure the model
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Safety settings to be more permissive for audio analysis
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def find_audio_json_pairs(self, input_dir: str) -> List[Tuple[str, str]]:
        """
        Find all pairs of .mp3 and .json files with the same base name.
        
        Args:
            input_dir: Directory to search for file pairs
            
        Returns:
            List of tuples (audio_file_path, json_file_path)
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory {input_dir} does not exist")
        
        # Find all .mp3 and .json files
        mp3_files = {}
        json_files = {}
        
        for file_path in input_path.rglob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() == '.mp3':
                    base_name = file_path.stem
                    mp3_files[base_name] = str(file_path)
                elif file_path.suffix.lower() == '.json':
                    base_name = file_path.stem
                    json_files[base_name] = str(file_path)
        
        # Find matching pairs
        pairs = []
        for base_name in mp3_files:
            if base_name in json_files:
                pairs.append((mp3_files[base_name], json_files[base_name]))
        
        logger.info(f"Found {len(pairs)} matching audio-json pairs")
        return pairs

    def load_json_file(self, json_path: str) -> Dict[str, Any]:
        """Load and return the contents of a JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {json_path}: {e}")
            return {}

    def create_prompts(self, transcription_data: Dict[str, Any], 
                      disorder_type_prompt: str = None, 
                      disorder_symptom_prompt: str = None) -> Tuple[str, str]:
        """
        Create two prompts for Gemini based on transcription data.
        
        Args:
            transcription_data: Data from the JSON file
            disorder_type_prompt: Prompt for disorder type classification
            disorder_symptom_prompt: Prompt for disorder symptom classification
            
        Returns:
            Tuple of (disorder_type_prompt, disorder_symptom_prompt)
        """
        # Get transcription text
        transcription_text = transcription_data.get('transcription', 
                                                   transcription_data.get('text', 
                                                                         str(transcription_data)))
        
        # Default disorder type classification prompt
        if not disorder_type_prompt:
            disorder_type_prompt = f"""
            Based on this audio file and transcription, classify the type of speech disorder present.
            
            Transcription: {transcription_text}
            
            Please analyze the audio and provide:
            1. Primary disorder type classification
            2. Confidence level (1-10)
            3. Key indicators that led to this classification
            4. Any secondary disorder types if present
            
            Focus on identifying the main category of speech disorder.
            """
        else:
            # Format custom prompt with transcription data
            try:
                disorder_type_prompt = disorder_type_prompt.format(**transcription_data)
            except KeyError as e:
                logger.warning(f"Disorder type prompt formatting failed: {e}. Using default.")
                disorder_type_prompt = f"Classify disorder type based on: {transcription_text}"
        
        # Default disorder symptom classification prompt  
        if not disorder_symptom_prompt:
            disorder_symptom_prompt = f"""
            Based on this audio file and transcription, identify specific symptoms of the speech disorder.
            
            Transcription: {transcription_text}
            
            Please analyze the audio and provide:
            1. Specific symptoms identified
            2. Severity assessment for each symptom
            3. Frequency/pattern of symptoms
            4. Impact on communication effectiveness
            5. Recommendations for intervention
            
            Focus on detailed symptom analysis and clinical observations.
            """
        else:
            # Format custom prompt with transcription data
            try:
                disorder_symptom_prompt = disorder_symptom_prompt.format(**transcription_data)
            except KeyError as e:
                logger.warning(f"Disorder symptom prompt formatting failed: {e}. Using default.")
                disorder_symptom_prompt = f"Identify disorder symptoms based on: {transcription_text}"
        
        return disorder_type_prompt.strip(), disorder_symptom_prompt.strip()

    def process_audio_with_dual_prompts(self, audio_path: str, disorder_type_prompt: str, 
                                       disorder_symptom_prompt: str) -> Tuple[str, str]:
        """
        Send audio file to Gemini for processing with both prompts.
        
        Args:
            audio_path: Path to the audio file
            disorder_type_prompt: Prompt for disorder type classification
            disorder_symptom_prompt: Prompt for disorder symptom classification
            
        Returns:
            Tuple of (disorder_type_response, disorder_symptom_response)
        """
        try:
            # Upload the audio file
            logger.info(f"Uploading audio file: {audio_path}")
            audio_file = genai.upload_file(audio_path)
            
            # Wait for the file to be processed
            while audio_file.state.name == "PROCESSING":
                logger.info("Waiting for audio file to be processed...")
                time.sleep(2)
                audio_file = genai.get_file(audio_file.name)
            
            if audio_file.state.name == "FAILED":
                raise Exception(f"Audio file processing failed: {audio_file.state}")
            
            # Process with first prompt (disorder type)
            logger.info("Generating disorder type classification...")
            type_response = self.model.generate_content(
                [disorder_type_prompt, audio_file],
                safety_settings=self.safety_settings
            )
            
            # Small delay between requests
            time.sleep(1)
            
            # Process with second prompt (disorder symptoms)
            logger.info("Generating disorder symptom analysis...")
            symptom_response = self.model.generate_content(
                [disorder_symptom_prompt, audio_file],
                safety_settings=self.safety_settings
            )
            
            # Clean up the uploaded file
            genai.delete_file(audio_file.name)
            
            return type_response.text, symptom_response.text
            
        except Exception as e:
            logger.error(f"Error processing audio with Gemini: {e}")
            error_msg = f"Error processing audio: {str(e)}"
            return error_msg, error_msg

    def save_updated_json(self, original_json_data: Dict[str, Any], 
                         disorder_type_response: str, disorder_symptom_response: str,
                         audio_path: str, json_path: str, output_path: str) -> None:
        """
        Save the updated JSON file with both responses and input file paths added.
        
        Args:
            original_json_data: Original JSON data
            disorder_type_response: Response for disorder type classification
            disorder_symptom_response: Response for disorder symptom analysis
            audio_path: Path to the original audio file
            json_path: Path to the original JSON file
            output_path: Path to save the updated JSON
        """
        try:
            # Add both responses and input file paths to the original data
            updated_data = original_json_data.copy()
            updated_data['disorder_type_classification'] = disorder_type_response
            updated_data['disorder_symptom_analysis'] = disorder_symptom_response
            updated_data['input_audio_path'] = audio_path
            updated_data['input_json_path'] = json_path
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the updated JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved updated JSON to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving JSON file {output_path}: {e}")

    def process_pairs(self, input_dir: str, output_dir: str, max_pairs: int = 20, 
                     disorder_type_prompt: str = None, disorder_symptom_prompt: str = None,
                     api_key: str = None, dry_run: bool = False) -> None:
        """
        Process audio-json pairs.
        
        Args:
            input_dir: Input directory containing audio and json files
            output_dir: Output directory for processed JSON files
            max_pairs: Maximum number of pairs to process
            disorder_type_prompt: Custom prompt for disorder type classification
            disorder_symptom_prompt: Custom prompt for disorder symptom analysis
            api_key: Optional API key (overrides initialization)
            dry_run: If True, only show what would be processed without calling API
        """
        if api_key and not dry_run:
            genai.configure(api_key=api_key)
        
        # Check if directories exist
        if not os.path.exists(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return
        
        if not os.path.exists(output_dir):
            if dry_run:
                logger.info(f"Output directory would be created: {output_dir}")
            else:
                logger.info(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
        
        # Find all pairs
        pairs = self.find_audio_json_pairs(input_dir)
        
        if not pairs:
            logger.warning("No matching audio-json pairs found")
            return
        
        # Randomly select up to max_pairs
        if len(pairs) > max_pairs:
            pairs = random.sample(pairs, max_pairs)
            logger.info(f"Randomly selected {max_pairs} pairs out of {len(pairs)} total pairs")
        
        if dry_run:
            logger.info("=== DRY RUN MODE - No API calls will be made ===")
            logger.info(f"Input directory: {input_dir}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Found {len(pairs)} pairs to process:")
            
            for i, (audio_path, json_path) in enumerate(pairs, 1):
                logger.info(f"  {i}. Audio: {os.path.relpath(audio_path, input_dir)}")
                logger.info(f"     JSON:  {os.path.relpath(json_path, input_dir)}")
                
                # Show what the output path would be (flattened to output dir)
                json_filename = os.path.basename(json_path)
                output_path = os.path.join(output_dir, json_filename)
                logger.info(f"     Output: {output_path}")
                
                # Show a sample of the JSON content
                try:
                    json_data = self.load_json_file(json_path)
                    if json_data:
                        # Show first few keys
                        keys = list(json_data.keys())[:3]
                        logger.info(f"     JSON keys (first 3): {keys}")
                        
                        # Show prompts that would be used
                        type_prompt, symptom_prompt = self.create_prompts(
                            json_data, disorder_type_prompt, disorder_symptom_prompt
                        )
                        type_preview = type_prompt[:150] + "..." if len(type_prompt) > 150 else type_prompt
                        symptom_preview = symptom_prompt[:150] + "..." if len(symptom_prompt) > 150 else symptom_prompt
                        logger.info(f"     Type prompt preview: {type_preview}")
                        logger.info(f"     Symptom prompt preview: {symptom_preview}")
                    
                except Exception as e:
                    logger.warning(f"     Error reading JSON: {e}")
                
                logger.info("")  # Empty line for readability
            
            logger.info("=== DRY RUN COMPLETE - Use without --dry-run to process files ===")
            return
        
        # Process each pair
        for i, (audio_path, json_path) in enumerate(pairs, 1):
            logger.info(f"Processing pair {i}/{len(pairs)}: {os.path.basename(audio_path)}")
            
            try:
                # Load JSON data
                json_data = self.load_json_file(json_path)
                
                # Create both prompts
                type_prompt, symptom_prompt = self.create_prompts(
                    json_data, disorder_type_prompt, disorder_symptom_prompt
                )
                
                # Process with Gemini using both prompts
                type_response, symptom_response = self.process_audio_with_dual_prompts(
                    audio_path, type_prompt, symptom_prompt
                )
                
                # Create output path - save directly to output dir (flattened)
                json_filename = os.path.basename(json_path)
                output_path = os.path.join(output_dir, json_filename)
                
                # Save updated JSON with both responses and input file paths
                self.save_updated_json(
                    json_data, type_response, symptom_response, 
                    audio_path, json_path, output_path
                )
                
                # Small delay to avoid rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing pair {audio_path}, {json_path}: {e}")
                continue
        
        logger.info(f"Completed processing {len(pairs)} pairs")


def main():
    parser = argparse.ArgumentParser(description="Process audio files with Gemini 2.0 Flash for dual disorder analysis")
    parser.add_argument("input_dir", help="Input directory containing .mp3 and .json files")
    parser.add_argument("output_dir", help="Output directory for processed JSON files")
    parser.add_argument("--max-pairs", type=int, default=20, 
                       help="Maximum number of pairs to process (default: 20)")
    parser.add_argument("--disorder-type-prompt", type=str, 
                       help="Custom prompt for disorder type classification")
    parser.add_argument("--disorder-symptom-prompt", type=str, 
                       help="Custom prompt for disorder symptom analysis")
    parser.add_argument("--api-key", type=str, 
                       help="Gemini API key (or set GEMINI_API_KEY environment variable)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="If set, only show what would be processed without calling API")
    
    args = parser.parse_args()
    
    try:
        processor = AudioGeminiProcessor(api_key=args.api_key)
        processor.process_pairs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_pairs=args.max_pairs,
            disorder_type_prompt=args.disorder_type_prompt,
            disorder_symptom_prompt=args.disorder_symptom_prompt,
            api_key=args.api_key,
            dry_run=args.dry_run
        )
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 
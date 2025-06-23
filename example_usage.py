#!/usr/bin/env python3
"""
Example usage of the Audio Gemini Processor

This demonstrates how to use the AudioGeminiProcessor class to process
audio files with their corresponding JSON transcriptions using dual prompts
for disorder type classification and symptom analysis.
"""

import os
from audio_gemini_processor import AudioGeminiProcessor

def main():
    # Example usage of the AudioGeminiProcessor
    
    # Set your API key (or use environment variable GEMINI_API_KEY)
    api_key = ""  # Replace with your actual API key
    
    # Input and output directories
    input_directory = "/Users/fagunpatel/Google Drive/My Drive/SpeechData/scenarios/speech_disorder_breakdown/processed-core-uxssd"  # Replace with your input directory
    output_directory = "/Users/fagunpatel/Desktop/Temp "  # Replace with your output directory

    # Custom prompt template (optional)
    # You can use field names from your JSON files in curly braces
    disorder_type_prompt = """You are a highly experienced Speech-Language Pathologist (SLP). An audio recording will be provided, typically consisting of a speech prompt from a pathologist followed by a child's repetition. The prompt text the child is trying to repeat is as follows: {transcription}. Based on your professional expertise: 1. Assess the child's speech in the recording for signs of typical development or potential speech-language disorder. 2. Conclude your analysis with one of the following labels only: A - 'typically developing' (child's speech patterns and development are within normal age-appropriate ranges), B - 'articulation' (difficulty producing specific speech sounds correctly, such as substituting, omitting, or distorting sounds), C - 'phonological' (difficulty understanding and using the sound system of language, affecting sounds of a particular type). 3. Let’s think step by step. Return the reasoning chain-of-thought for your answer along with the actual answer."""

    disorder_symptom_prompt = """You are a highly experienced Speech-Language Pathologist (SLP). An audio recording will be provided, typically consisting of a speech prompt from a pathologist followed by a child's repetition. The prompt the child is trying to repeat is as follows: {words}. Based on your professional expertise: 1. Assess the child's speech in the recording and recognize any abnormal features in the child's speech. 2. These features can be on of the following: A - 'substitution', B - 'omission', C - 'addition', D - 'typically_developing', or E - 'stuttering'. Here, 'substitution' is when the child substitutes one word/phrase/syllable for another. 'omission' is when the child omits one word/phrase/syllable. 'addition' is when the child adds one word/phrase/syllable. 'typically_developing' is when the child's speech is typical of a child of their age. 'stuttering' is when the child stutters, has difficulty speaking, repeats sounds/words or prolongs sounds/words. 3. Let’s think step by step. Return the reasoning chain-of-thought for your answer along with the actual answer."""
    
    try:
        # Initialize the processor
        processor = AudioGeminiProcessor(api_key=api_key)
        
        # First, run a dry run to check directories and see what would be processed
        print("Running dry run first...")
        processor.process_pairs(
            input_dir=input_directory,
            output_dir=output_directory,
            max_pairs=20,  # Process up to 20 pairs
            disorder_type_prompt=disorder_type_prompt,  # Custom type classification prompt
            disorder_symptom_prompt=disorder_symptom_prompt,  # Custom symptom analysis prompt
            dry_run=False  # This will only show what would be processed
        )
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
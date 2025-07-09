#!/usr/bin/env python3
"""
Chain of Thought Analysis Script using OpenAI Models

This script analyzes Chain of Thought (COT) reasoning stored in JSON format
using OpenAI's GPT models. Each COT is analyzed individually
and results are saved to output files.

Usage:
    python cot_analyzer.py --json_file <json_file_path> [--api-key <key>] [--output-dir <dir>]

Example:
    python cot_analyzer.py --json_file data_processed.json --api-key your_api_key
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional
from openai import OpenAI
from tqdm import tqdm


class COTAnalyzer:
    """Chain of Thought analyzer using OpenAI models."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        """
        Initialize the COT analyzer.
        
        Args:
            api_key (str): OpenAI API key
            model_name (str): OpenAI model name
        """
        self.api_key = api_key
        self.model_name = model_name  # Use the parameter instead of hardcoding
        self.client = None
        self.prompt_template = self._load_prompt_template()
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=api_key)
            print(f"✓ Successfully initialized {self.model_name}")
        except Exception as e:
            print(f"✗ Error initializing OpenAI client: {e}")
            sys.exit(1)
    
    def _load_prompt_template(self) -> str:
        """Load the analysis prompt from the text file."""
        prompt_file = Path(__file__).parent / "type_prompt.txt"
        
        if not prompt_file.exists():
            print(f"✗ Prompt file not found: {prompt_file}")
            sys.exit(1)
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def analyze_cot(self, cot_text: str, cot_id: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a single Chain of Thought using OpenAI.
        
        Args:
            cot_text (str): The Chain of Thought text to analyze
            cot_id (str): Unique identifier for this COT
            
        Returns:
            Dict containing analysis results or None if failed
        """
        try:
            # Prepare the prompt
            full_prompt = self.prompt_template.format(cot_text=cot_text)
            
            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.1,
                max_tokens=1000  # Reduced for faster responses
            )
            
            # Check if response was successful
            if response.choices and response.choices[0].message.content:
                return {
                    "id": cot_id,
                    "original_cot": cot_text,
                    "analysis": response.choices[0].message.content,
                    "model_used": self.model_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                print(f"✗ No response generated for COT {cot_id}")
                return None
                
        except Exception as e:
            print(f"✗ Error analyzing COT {cot_id}: {e}")
            return None
    
    def process_json_file(self, json_file_path: str, output_dir: str = None, limit: int = None) -> bool:
        """
        Process a JSON file containing multiple COTs and add analysis to the same structure.
        
        Args:
            json_file_path (str): Path to the JSON file
            output_dir (str): Directory to save results (optional)
            limit (int): Maximum number of COTs to analyze (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the JSON data
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Apply limit if specified
            items_to_process = list(data.items())
            if limit and limit > 0:
                items_to_process = items_to_process[:limit]
                print(f"📊 Limiting analysis to first {limit} COTs out of {len(data)} total")
            
            # Process each COT and add analysis to the same structure
            failed_analyses = []
            successful_analyses = 0
            
            print(f"\n📊 Processing {len(items_to_process)} COTs from {json_file_path}")
            print(f"💾 Analysis will be added to the original JSON structure")
            
            for cot_id, cot_data in tqdm(items_to_process, desc="Analyzing COTs"):
                # Extract COT text
                if isinstance(cot_data, dict) and 'cot' in cot_data:
                    cot_text = cot_data['cot']
                elif isinstance(cot_data, str):
                    cot_text = cot_data
                    # Convert string to dict format for adding analysis
                    data[cot_id] = {"original_cot": cot_text}
                    cot_data = data[cot_id]
                else:
                    print(f"⚠️  Skipping {cot_id}: No 'cot' field found")
                    continue
                
                # Analyze the COT
                analysis_result = self.analyze_cot(cot_text, cot_id)
                
                if analysis_result:
                    # Add analysis fields to the existing data structure
                    if isinstance(cot_data, dict):
                        cot_data['analysis'] = analysis_result['analysis']
                        cot_data['model_used'] = analysis_result['model_used']
                    successful_analyses += 1
                else:
                    failed_analyses.append(cot_id)
                
                # No delay needed - OpenAI handles rate limiting automatically
            
            # Create output filename
            input_path = Path(json_file_path)
            if limit:
                output_filename = f"{input_path.stem}_with_analysis_limit_{limit}.json"
            else:
                output_filename = f"{input_path.stem}_with_analysis.json"
            
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True)
                output_file = output_path / output_filename
            else:
                output_file = input_path.parent / output_filename
            
            # Save the updated JSON with analysis
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Create summary
            summary = {
                "total_cots_in_file": len(data),
                "cots_processed": len(items_to_process),
                "successful_analyses": successful_analyses,
                "failed_analyses": len(failed_analyses),
                "failed_ids": failed_analyses,
                "success_rate": successful_analyses / len(items_to_process) * 100 if items_to_process else 0,
                "model_used": self.model_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "output_file": str(output_file),
                "limit_applied": limit
            }

            
            # Print results
            print(f"\n✅ Analysis Complete!")
            if limit:
                print(f"📊 Processed: {len(items_to_process)} out of {len(data)} total COTs")
            print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
            print(f"✓ Successful: {successful_analyses}")
            print(f"✗ Failed: {len(failed_analyses)}")
            print(f"📁 Updated JSON saved to: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error processing JSON file: {e}")
            return False


def main():
    """Main function to handle command line arguments and execute analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze Chain of Thought reasoning using OpenAI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cot_analyzer.py --json_files data_processed.json --api-key your_key
  python cot_analyzer.py --json_files data1.json,data2.json,data3.json --api-key your_key
  python cot_analyzer.py --json_files data.json --limit 10  # Analyze only first 10 COTs
  python cot_analyzer.py --json_files file1.json,file2.json --limit 5 --model gpt-4o
        """
    )
    
    parser.add_argument(
        "--json_files", 
        help="Path to JSON file(s) containing COT data (comma-separated for multiple files)"
    )
    
    parser.add_argument(
        "--api-key", 
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--output-dir", 
        help="Directory to save analysis results (default: same as input file)"
    )
    
    parser.add_argument(
        "--model", 
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of COTs to analyze per file (default: all)"
    )
    
    args = parser.parse_args()
    
    # Parse file list
    if not args.json_files:
        print("✗ Error: --json_files argument is required.")
        sys.exit(1)
    
    file_list = [f.strip() for f in args.json_files.split(',')]
    
    # Check if all files exist
    missing_files = []
    for file_path in file_list:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Error: The following files do not exist:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = input("🔑 Enter your OpenAI API key: ").strip()
        if not api_key:
            print("✗ API key is required!")
            sys.exit(1)
    
    # Initialize analyzer
    print(f"🚀 Starting COT Analysis with {args.model}")
    print(f"📁 Processing {len(file_list)} file(s)")
    if args.limit:
        print(f"📊 Limit: {args.limit} COTs per file")
    
    analyzer = COTAnalyzer(api_key, args.model)
    
    # Process each file
    overall_success = True
    results_summary = []
    
    for i, json_file in enumerate(file_list, 1):
        print(f"\n{'='*60}")
        print(f"📂 Processing file {i}/{len(file_list)}: {json_file}")
        print(f"{'='*60}")
        
        success = analyzer.process_json_file(json_file, args.output_dir, args.limit)
        
        if success:
            results_summary.append(f"✓ {json_file}")
        else:
            results_summary.append(f"✗ {json_file}")
            overall_success = False
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"🎯 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"📊 Total files processed: {len(file_list)}")
    
    for result in results_summary:
        print(f"  {result}")
    
    if overall_success:
        print(f"\n✅ All files processed successfully!")
    else:
        print(f"\n⚠️  Some files failed to process. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Analysis Extractor for Chain of Thought Results

This script analyzes the results from COT analysis, extracting metrics like
subgoal count, rule following, and error recognition, and compares predictions
with ground truth labels.

Usage:
    python analysis_extractor.py --json_file <json_file_path>

Example:
    python analysis_extractor.py --json_file data_processed_with_analysis.json
"""

import json
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


class AnalysisExtractor:
    """Extracts and analyzes metrics from COT analysis results."""
    
    def __init__(self):
        """Initialize the analysis extractor."""
        # self.label_mapping = {
        #     "A": "substitution",
        #     "B": "omission",
        #     "C": "addition",
        #     "D": "typically_developing",
        #     "E": "stuttering"
        # }
        self.label_mapping = {
        "A": "typically_developing",
        "B": "articulation",
        "C": "phonological"
         }
        
        # Reverse mapping for lookup
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
    def extract_count(self, analysis_text: str) -> Optional[int]:
        """Extract the count value from <count>X</count> tags."""
        match = re.search(r'<count>(\d+)</count>', analysis_text)
        if match:
            return int(match.group(1))
        return None
    
    def extract_rule_bool(self, analysis_text: str) -> Optional[int]:
        """Extract the rule boolean value from <rule_bool>X</rule_bool> tags."""
        match = re.search(r'<rule_bool>\s*(\d+)\s*</rule_bool>', analysis_text)
        if match:
            return int(match.group(1))
        return None
    
    def extract_error_bool(self, analysis_text: str) -> Optional[int]:
        """Extract the error boolean value from <error_bool>X</error_bool> tags."""
        match = re.search(r'<error_bool>\s*(\d+)\s*</error_bool>', analysis_text)
        if match:
            return int(match.group(1))
        return None
    
    def extract_predicted_answer(self, cot_text: str) -> Optional[str]:
        """Extract the predicted answer using robust regex patterns from GPQA metric."""
        # First regex: Matches "answer is (A-E)" with optional parentheses
        match = re.search(r"answer is \(?([A-E])\)?", cot_text)
        if match:
            return match.group(1)

        # Second regex: Matches "[answer: (A-E)]" with optional leading characters like "."
        match = re.search(r"\.*\[aA\]nswer:\s*\(?([A-E])\)?", cot_text)
        if match:
            return match.group(1)

        # Third regex: Matches "correct answer is (A-E)" with optional leading non-alpha characters
        match = re.search(r"correct answer is [^A-Za-z]*([A-E])", cot_text)
        if match:
            return match.group(1)

        # Fourth regex: Matches "correct answer is (A-E)" with optional leading non-capital alpha characters
        match = re.search(r"correct answer is [^A-Z]*([A-E])", cot_text)
        if match:
            return match.group(1)
        
        return None
    
    def process_json_file(self, json_file_path: str) -> Dict[str, Any]:
        """
        Process the JSON file and extract all metrics.
        
        Args:
            json_file_path (str): Path to the JSON file
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Load the JSON data
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Initialize counters
            results = {
                'total_items': 0,
                'processed_items': 0,
                'correct_predictions': 0,
                'incorrect_predictions': 0,
                'missing_predictions': 0,
                'missing_analysis': 0,
                
                # Metrics for all items
                'total_subgoals': 0,
                'total_rule_following': 0,
                'total_error_recognition': 0,
                
                # Metrics when prediction is correct
                'correct_subgoals': 0,
                'correct_rule_following': 0,
                'correct_error_recognition': 0,
                
                # Metrics when prediction is incorrect
                'incorrect_subgoals': 0,
                'incorrect_rule_following': 0,
                'incorrect_error_recognition': 0,
                
                # Detailed tracking
                'items_with_metrics': 0,
                'items_with_predictions': 0,
            }
            
            print(f"📊 Processing {len(data)} items from {json_file_path}")
            
            for item_id, item_data in data.items():
                results['total_items'] += 1
                
                if not isinstance(item_data, dict):
                    continue
                
                # Extract label and COT
                label = item_data.get('label')
                cot = item_data.get('cot', '')
                analysis = item_data.get('analysis', '')
                
                # Skip items without analysis key
                if not analysis:
                    results['missing_analysis'] += 1
                    continue
                
                if not label or not cot:
                    continue
                
                results['processed_items'] += 1
                
                # Extract predicted answer
                predicted_answer = self.extract_predicted_answer(cot)
                
                # Convert label to letter format if needed
                if label in self.reverse_mapping:
                    label_letter = self.reverse_mapping[label]
                else:
                    label_letter = label  # Assume it's already a letter
                
                # Determine if prediction is correct
                prediction_correct = None
                if predicted_answer:
                    results['items_with_predictions'] += 1
                    prediction_correct = (predicted_answer == label_letter)
                    if prediction_correct:
                        results['correct_predictions'] += 1
                    else:
                        results['incorrect_predictions'] += 1
                else:
                    results['missing_predictions'] += 1
                
                # Extract analysis metrics (we know analysis exists at this point)
                count_val = self.extract_count(analysis)
                rule_val = self.extract_rule_bool(analysis)
                error_val = self.extract_error_bool(analysis)
                
                if count_val is not None or rule_val is not None or error_val is not None:
                    results['items_with_metrics'] += 1
                    
                    # Only count metrics if we also have a valid prediction
                    if prediction_correct is not None:
                        # Add to totals
                        if count_val is not None:
                            results['total_subgoals'] += count_val
                        if rule_val is not None:
                            results['total_rule_following'] += rule_val
                        if error_val is not None:
                            results['total_error_recognition'] += error_val
                        
                        # Add to correct/incorrect buckets
                        if prediction_correct:
                            if count_val is not None:
                                results['correct_subgoals'] += count_val
                            if rule_val is not None:
                                results['correct_rule_following'] += rule_val
                            if error_val is not None:
                                results['correct_error_recognition'] += error_val
                        else:
                            if count_val is not None:
                                results['incorrect_subgoals'] += count_val
                            if rule_val is not None:
                                results['incorrect_rule_following'] += rule_val
                            if error_val is not None:
                                results['incorrect_error_recognition'] += error_val
            
            return results
            
        except Exception as e:
            print(f"✗ Error processing JSON file: {e}")
            return {}
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print the analysis results in a formatted way."""
        if not results:
            print("❌ No results to display")
            return
        
        print(f"\n{'='*60}")
        print(f"📊 ANALYSIS RESULTS")
        print(f"{'='*60}")
        
        # Basic statistics
        print(f"\n📈 Basic Statistics:")
        print(f"  Total items: {results['total_items']}")
        print(f"  Processed items: {results['processed_items']}")
        print(f"  Items with predictions: {results['items_with_predictions']}")
        print(f"  Items with metrics: {results['items_with_metrics']}")
        
        # Prediction accuracy
        if results['items_with_predictions'] > 0:
            accuracy = results['correct_predictions'] / results['items_with_predictions'] * 100
            print(f"\n🎯 Prediction Accuracy:")
            print(f"  Correct predictions: {results['correct_predictions']}")
            print(f"  Incorrect predictions: {results['incorrect_predictions']}")
            print(f"  Accuracy: {accuracy:.2f}%")
        
        # Missing data
        print(f"\n⚠️  Missing Data:")
        print(f"  Missing predictions: {results['missing_predictions']}")
        print(f"  Missing analysis: {results['missing_analysis']}")
        
        # Metric totals
        print(f"\n📊 Metric Totals:")
        print(f"  Total subgoals: {results['total_subgoals']}")
        print(f"  Total rule following: {results['total_rule_following']}")
        print(f"  Total error recognition: {results['total_error_recognition']}")
        
        # Metrics for correct predictions
        if results['correct_predictions'] > 0:
            print(f"\n✅ Metrics when Prediction is Correct:")
            print(f"  Subgoals: {results['correct_subgoals']}")
            print(f"  Rule following: {results['correct_rule_following']}")
            print(f"  Error recognition: {results['correct_error_recognition']}")
        
        # Metrics for incorrect predictions
        if results['incorrect_predictions'] > 0:
            print(f"\n❌ Metrics when Prediction is Incorrect:")
            print(f"  Subgoals: {results['incorrect_subgoals']}")
            print(f"  Rule following: {results['incorrect_rule_following']}")
            print(f"  Error recognition: {results['incorrect_error_recognition']}")
        
        # Average metrics
        print(f"\n📈 Average Metrics:")
        if results['items_with_metrics'] > 0:
            avg_subgoals = results['total_subgoals'] / results['items_with_metrics']
            avg_rules = results['total_rule_following'] / results['items_with_metrics']
            avg_errors = results['total_error_recognition'] / results['items_with_metrics']
            print(f"  Average subgoals per item: {avg_subgoals:.2f}")
            print(f"  Average rule following per item: {avg_rules:.2f}")
            print(f"  Average error recognition per item: {avg_errors:.2f}")
        
        if results['correct_predictions'] > 0:
            avg_correct_subgoals = results['correct_subgoals'] / results['correct_predictions']
            avg_correct_rules = results['correct_rule_following'] / results['correct_predictions']
            avg_correct_errors = results['correct_error_recognition'] / results['correct_predictions']
            print(f"\n✅ Average Metrics (Correct Predictions):")
            print(f"  Average subgoals: {avg_correct_subgoals:.2f}")
            print(f"  Average rule following: {avg_correct_rules:.2f}")
            print(f"  Average error recognition: {avg_correct_errors:.2f}")
        
        if results['incorrect_predictions'] > 0:
            avg_incorrect_subgoals = results['incorrect_subgoals'] / results['incorrect_predictions']
            avg_incorrect_rules = results['incorrect_rule_following'] / results['incorrect_predictions']
            avg_incorrect_errors = results['incorrect_error_recognition'] / results['incorrect_predictions']
            print(f"\n❌ Average Metrics (Incorrect Predictions):")
            print(f"  Average subgoals: {avg_incorrect_subgoals:.2f}")
            print(f"  Average rule following: {avg_incorrect_rules:.2f}")
            print(f"  Average error recognition: {avg_incorrect_errors:.2f}")


def main():
    """Main function to handle command line arguments and execute analysis."""
    parser = argparse.ArgumentParser(
        description="Extract and analyze metrics from COT analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analysis_extractor.py --json_file data_processed_with_analysis.json
  python analysis_extractor.py --json_file results.json --output summary.json
        """
    )
    
    parser.add_argument(
        "--json_file", 
        required=True,
        help="Path to JSON file containing COT analysis results"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.json_file).exists():
        print(f"✗ Error: File {args.json_file} does not exist")
        sys.exit(1)
    
    # Initialize extractor
    print(f"🚀 Starting Analysis Extraction")
    print(f"📁 Input file: {args.json_file}")
    
    extractor = AnalysisExtractor()
    
    # Process the file
    results = extractor.process_json_file(args.json_file)
    
    if results:
        # Print results
        extractor.print_results(results)
        
        # Always save results to JSON file in same folder as input
        input_path = Path(args.json_file)
        output_filename = f"{input_path.stem}_metrics.json"
        output_file = input_path.parent / output_filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Metrics automatically saved to: {output_file}")
        
        print(f"\n✅ Analysis extraction completed successfully!")
    else:
        print(f"❌ Analysis extraction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 
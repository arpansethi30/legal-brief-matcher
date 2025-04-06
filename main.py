# File: main.py - Main entry point for the application

import argparse
import json
import os
from utils.data_loader import load_brief_pair, load_training_data
from models.embedding_model import get_embeddings
from models.similarity_model import calculate_similarity, get_top_matches
from models.llm_verifier import verify_matches

def main():
    parser = argparse.ArgumentParser(description='Legal Brief Argument Matcher')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file or directory')
    parser.add_argument('--output', type=str, help='Output JSON file')
    args = parser.parse_args()
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process a single file
        brief_pair = load_brief_pair(args.input)
        process_brief_pair(brief_pair, args.output)
    elif os.path.isdir(args.input):
        # Process all files in directory
        for filename in os.listdir(args.input):
            if filename.endswith('.json'):
                file_path = os.path.join(args.input, filename)
                brief_pair = load_brief_pair(file_path)
                
                # Output file
                output_file = os.path.join(args.output, f"matches_{filename}") if args.output else None
                
                process_brief_pair(brief_pair, output_file)
    else:
        print(f"Input {args.input} is not a file or directory")

def process_brief_pair(brief_pair, output_file=None):
    """
    Process a brief pair and output the matches
    
    Args:
        brief_pair: Brief pair data
        output_file: Output file path
    """
    print(f"Processing brief pair: {brief_pair['moving_brief']['brief_id']} vs {brief_pair['response_brief']['brief_id']}")
    
    # Get embeddings
    moving_brief_embeddings = get_embeddings(brief_pair['moving_brief']['brief_arguments'])
    response_brief_embeddings = get_embeddings(brief_pair['response_brief']['brief_arguments'])
    
    # Calculate similarity
    similarity_matrix = calculate_similarity(moving_brief_embeddings, response_brief_embeddings)
    
    # Get matches
    matches = get_top_matches(
        similarity_matrix,
        brief_pair['moving_brief']['brief_arguments'],
        brief_pair['response_brief']['brief_arguments']
    )
    
    # Verify with LLM
    verified_matches = verify_matches(
        brief_pair['moving_brief']['brief_arguments'],
        brief_pair['response_brief']['brief_arguments'],
        matches
    )
    
    # Format output
    output = []
    for match in verified_matches:
        output.append([match['moving_heading'], match['response_heading']])
    
    # Print results
    print("\nMatched arguments:")
    for i, match in enumerate(output):
        print(f"{i+1}. {match[0]} -> {match[1]}")
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved matches to {output_file}")

if __name__ == '__main__':
    main()
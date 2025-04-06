# File: utils/data_loader.py - Utilities for loading and processing legal brief data

import json
import os

def load_brief_pair(file_path):
    """
    Load brief pair(s) from a JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        list or dict: The brief pair data (either a single pair or a list of pairs)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data

def load_training_data(directory_path):
    """
    Load all training brief pairs from a directory
    
    Args:
        directory_path: Path to the directory containing brief pair JSON files
        
    Returns:
        list: List of brief pairs with ground truth links
    """
    training_data = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            brief_pair = load_brief_pair(file_path)
            
            if brief_pair['split'] == 'train':
                training_data.append(brief_pair)
    
    return training_data

def extract_arguments(brief_pair):
    """
    Extract all arguments from a brief pair
    
    Args:
        brief_pair: Brief pair data
        
    Returns:
        tuple: (moving_brief_arguments, response_brief_arguments)
    """
    moving_brief_arguments = brief_pair['moving_brief']['brief_arguments']
    response_brief_arguments = brief_pair['response_brief']['brief_arguments']
    
    return moving_brief_arguments, response_brief_arguments

def get_true_links(brief_pair):
    """
    Get the ground truth links for a brief pair
    
    Args:
        brief_pair: Brief pair data
        
    Returns:
        list: List of [moving_brief_heading, response_brief_heading] pairs
    """
    if 'true_links' in brief_pair:
        return brief_pair['true_links']
    return []
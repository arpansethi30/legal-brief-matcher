# File: models/similarity_model.py - Enhanced similarity model with legal domain knowledge

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.legal_features import extract_legal_features
import re

def calculate_similarity(moving_brief_embeddings, response_brief_embeddings):
    """
    Calculate similarity matrix between moving and response brief embeddings
    """
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(moving_brief_embeddings, response_brief_embeddings)
    
    return similarity_matrix

def get_top_matches(similarity_matrix, moving_brief_arguments, response_brief_arguments, k=3):
    """
    Get top k matches for each moving brief argument with enhanced legal domain features
    """
    matches = []
    num_moving_args = similarity_matrix.shape[0]
    num_response_args = similarity_matrix.shape[1]
    
    # First pass: Get legal features for all pairs
    legal_feature_matrix = np.zeros((num_moving_args, num_response_args))
    feature_details = {}
    
    for i in range(num_moving_args):
        for j in range(num_response_args):
            # Get arguments
            moving_arg = moving_brief_arguments[i]
            response_arg = response_brief_arguments[j]
            
            # Extract legal features
            legal_features = extract_legal_features(moving_arg, response_arg)
            
            # Calculate weighted legal score
            weights = {
                'heading_similarity': 0.15,
                'heading_pattern_match': 0.10,
                'citation_overlap': 0.15,
                'has_shared_citations': 0.10,
                'term_overlap': 0.10,
                'direct_reference': 0.15,
                'negation_pattern': 0.05,
                'standard_of_review_match': 0.05,
                'legal_tests_match': 0.05,
                'procedural_posture_match': 0.03,
                'counter_argument_markers': 0.04,
                'response_to_specific_points': 0.05,
                'common_brief_section_match': 0.08
            }
            
            legal_score = sum(weights.get(feature, 0) * legal_features.get(feature, 0) 
                              for feature in legal_features)
            
            legal_feature_matrix[i, j] = legal_score
            feature_details[(i, j)] = legal_features
    
    # Second pass: Apply legal heuristics for ordering preference
    # Legal briefs often follow same order - boost matches that preserve order
    order_bonus_matrix = np.zeros((num_moving_args, num_response_args))
    
    for i in range(num_moving_args):
        for j in range(num_response_args):
            # Extract section numbering
            moving_heading = moving_brief_arguments[i]['heading']
            response_heading = response_brief_arguments[j]['heading']
            
            # Check if sections follow same order
            moving_num = extract_section_number(moving_heading)
            response_num = extract_section_number(response_heading)
            
            if moving_num is not None and response_num is not None:
                # If moving brief is section I and response is section A or 1, they likely match
                if (moving_num == 1 and response_num == 1) or \
                   (moving_num == 2 and response_num == 2) or \
                   (moving_num == 3 and response_num == 3) or \
                   (moving_num == 4 and response_num == 4):
                    order_bonus_matrix[i, j] = 0.15  # Bonus for matching order
    
    # Third pass: Combine embedding similarity, legal features, and order bonus
    combined_score_matrix = 0.5 * similarity_matrix + 0.4 * legal_feature_matrix + 0.1 * order_bonus_matrix
    
    # Get matches for each moving brief argument
    for i in range(num_moving_args):
        # Get top k indices
        top_indices = np.argsort(combined_score_matrix[i])[::-1][:k]
        
        for j in top_indices:
            # Extract arguments
            moving_arg = moving_brief_arguments[i]
            response_arg = response_brief_arguments[j]
            
            # Calculate component scores
            embedding_similarity = similarity_matrix[i][j]
            legal_score = legal_feature_matrix[i][j]
            order_bonus = order_bonus_matrix[i][j]
            combined_score = combined_score_matrix[i][j]
            
            # Extract important legal features
            legal_features = feature_details[(i, j)]
            
            matches.append({
                'moving_index': i,
                'response_index': j,
                'moving_heading': moving_arg['heading'],
                'response_heading': response_arg['heading'],
                'moving_content': moving_arg['content'],
                'response_content': response_arg['content'],
                'embedding_similarity': float(embedding_similarity),
                'legal_score': float(legal_score),
                'order_bonus': float(order_bonus),
                'combined_score': float(combined_score),
                'legal_features': legal_features
            })
    
    # Sort by combined score
    matches.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return matches

def extract_section_number(heading):
    """
    Extract section number from heading regardless of format (Roman, Arabic, or letter)
    """
    # Try Roman numerals
    roman_match = re.match(r'^([IVX]+)\.\s+', heading)
    if roman_match:
        roman = roman_match.group(1)
        # Convert Roman to integer
        roman_values = {'I': 1, 'V': 5, 'X': 10}
        arabic = 0
        for i in range(len(roman)):
            if i > 0 and roman_values[roman[i]] > roman_values[roman[i-1]]:
                arabic += roman_values[roman[i]] - 2 * roman_values[roman[i-1]]
            else:
                arabic += roman_values[roman[i]]
        return arabic
    
    # Try Arabic numerals
    arabic_match = re.match(r'^(\d+)\.\s+', heading)
    if arabic_match:
        return int(arabic_match.group(1))
    
    # Try letters
    letter_match = re.match(r'^([A-Z])\.\s+', heading)
    if letter_match:
        letter = letter_match.group(1)
        # Convert letter to number (A=1, B=2, etc.)
        return ord(letter) - ord('A') + 1
    
    # Try Arabic with parentheses
    arabic_paren_match = re.match(r'^(\d+)\)\s+', heading)
    if arabic_paren_match:
        return int(arabic_paren_match.group(1))
    
    return None
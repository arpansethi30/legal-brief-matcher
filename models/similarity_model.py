# File: models/similarity_model.py - Calculate similarity between arguments

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.legal_features import extract_legal_features

def calculate_similarity(moving_brief_embeddings, response_brief_embeddings):
    """
    Calculate similarity matrix between moving and response brief embeddings
    
    Args:
        moving_brief_embeddings: Embeddings of moving brief arguments
        response_brief_embeddings: Embeddings of response brief arguments
        
    Returns:
        numpy.ndarray: Similarity matrix
    """
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(moving_brief_embeddings, response_brief_embeddings)
    
    return similarity_matrix

def get_top_matches(similarity_matrix, moving_brief_arguments, response_brief_arguments, k=3):
    """
    Get top k matches for each moving brief argument
    
    Args:
        similarity_matrix: Similarity matrix
        moving_brief_arguments: List of moving brief arguments
        response_brief_arguments: List of response brief arguments
        k: Number of top matches to return for each argument
        
    Returns:
        list: List of dictionaries with matches and scores
    """
    matches = []
    
    for i in range(similarity_matrix.shape[0]):
        # Get top k indices
        top_indices = np.argsort(similarity_matrix[i])[::-1][:k]
        
        for j in top_indices:
            # Extract arguments
            moving_arg = moving_brief_arguments[i]
            response_arg = response_brief_arguments[j]
            
            # Extract legal features
            legal_features = extract_legal_features(moving_arg, response_arg)
            
            # Calculate combined score (embedding similarity + legal features)
            embedding_similarity = similarity_matrix[i][j]
            legal_score = (legal_features['heading_similarity'] + legal_features['entity_overlap'] + 
                           legal_features['direct_reference'] + legal_features['negation_pattern']) / 4
            
            # Combined score with weights
            combined_score = 0.6 * embedding_similarity + 0.4 * legal_score
            
            matches.append({
                'moving_index': i,
                'response_index': j,
                'moving_heading': moving_arg['heading'],
                'response_heading': response_arg['heading'],
                'moving_content': moving_arg['content'],
                'response_content': response_arg['content'],
                'embedding_similarity': float(embedding_similarity),
                'legal_score': legal_score,
                'combined_score': float(combined_score)
            })
    
    # Sort by combined score
    matches.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return matches
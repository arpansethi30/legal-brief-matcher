# File: utils/legal_features.py - Domain-specific legal features for argument matching

import re
from utils.text_processor import extract_legal_entities

def extract_legal_features(moving_arg, response_arg):
    """
    Extract legal-specific features from a pair of arguments
    
    Args:
        moving_arg: Argument from moving brief
        response_arg: Argument from response brief
        
    Returns:
        dict: Features dict
    """
    moving_heading = moving_arg['heading']
    moving_content = moving_arg['content']
    response_heading = response_arg['heading']
    response_content = response_arg['content']
    
    features = {}
    
    # Heading pattern matching
    features['heading_similarity'] = calculate_heading_similarity(moving_heading, response_heading)
    
    # Legal entity overlap
    moving_entities = extract_legal_entities(moving_content)
    response_entities = extract_legal_entities(response_content)
    features['entity_overlap'] = len(set(moving_entities) & set(response_entities)) / \
                                max(1, len(set(moving_entities) | set(response_entities)))
    
    # Direct reference detection
    features['direct_reference'] = detect_direct_references(moving_heading, response_content)
    
    # Negation pattern detection
    features['negation_pattern'] = detect_negation_patterns(moving_content, response_content)
    
    return features

def calculate_heading_similarity(moving_heading, response_heading):
    """
    Calculate similarity between argument headings based on legal patterns
    
    Args:
        moving_heading: Heading from moving brief
        response_heading: Heading from response brief
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Convert to lowercase
    moving_lower = moving_heading.lower()
    response_lower = response_heading.lower()
    
    # Standard legal heading patterns
    patterns = [
        (r'likelihood of success', r'not likely to succeed'),
        (r'irreparable harm', r'no irreparable harm'),
        (r'balance of (the )?harms', r'balance of equities'),
        (r'public interest', r'public interest'),
        (r'background', r'injunctive relief'),
        (r'preliminary (statement|injunction)', r'standard of review')
    ]
    
    for moving_pattern, response_pattern in patterns:
        if re.search(moving_pattern, moving_lower) and re.search(response_pattern, response_lower):
            return 1.0
    
    # Numeric pattern matching (Roman numerals to Arabic numbers, etc.)
    if re.search(r'^[IVX]+\.', moving_heading) and re.search(r'^\d+\.', response_heading):
        return 0.8
    
    return 0.0

def detect_direct_references(moving_heading, response_content):
    """
    Detect direct references to the moving brief argument in the response
    
    Args:
        moving_heading: Heading from moving brief
        response_content: Content of response brief argument
        
    Returns:
        float: Reference score between 0 and 1
    """
    # Extract key terms from the moving heading
    key_terms = moving_heading.lower().split()
    key_terms = [term for term in key_terms if len(term) > 3 and term not in {'the', 'and', 'for', 'not'}]
    
    # Search for direct references
    response_lower = response_content.lower()
    
    # Look for phrases like "plaintiff argues" or "contrary to"
    reference_phrases = [
        'plaintiff argues', 'plaintiffs argue', 'defendant argues', 'defendants argue',
        'contrary to', 'unlike', 'opposing', 'contends', 'asserts'
    ]
    
    for phrase in reference_phrases:
        if phrase in response_lower:
            return 1.0
    
    # Look for key terms from the heading
    matches = sum(1 for term in key_terms if term in response_lower)
    if matches > 0:
        return min(1.0, matches / len(key_terms))
    
    return 0.0

def detect_negation_patterns(moving_content, response_content):
    """
    Detect negation patterns between moving and response arguments
    
    Args:
        moving_content: Content of moving brief argument
        response_content: Content of response brief argument
        
    Returns:
        float: Negation score between 0 and 1
    """
    # Look for negation words in response that may counter the moving brief
    negation_words = ['not', 'no', 'never', 'fails', 'incorrect', 'cannot', 'however', 'contrary']
    
    response_lower = response_content.lower()
    
    negation_count = sum(1 for word in negation_words if word in response_lower)
    
    # Normalize score
    return min(1.0, negation_count / 5)
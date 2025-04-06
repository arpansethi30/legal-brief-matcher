# File: utils/text_processor.py - Text processing utilities for legal text

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def clean_text(text):
    """
    Clean and normalize legal text
    
    Args:
        text: Input text
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove citations
    text = re.sub(r'\b\d+\s+[A-Za-z\.]+\s+\d+\s*\(\d{4}\)', ' ', text)
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_legal_entities(text):
    """
    Extract legal entities (cases, statutes, etc.) from text
    
    Args:
        text: Input text
        
    Returns:
        list: List of extracted legal entities
    """
    # Extract case citations
    case_pattern = r'\b[A-Za-z\s\.]+\s+v\.\s+[A-Za-z\s\.]+,\s+\d+\s+[A-Za-z\.]+\s+\d+\s*\(\d{4}\)'
    cases = re.findall(case_pattern, text)
    
    # Extract statute citations
    statute_pattern = r'\d+\s+[A-Za-z\.]+\s+§\s*\d+(\([a-z]\))*'
    statutes = re.findall(statute_pattern, text)
    
    return cases + statutes

def extract_key_phrases(text, n=5):
    """
    Extract key phrases from text
    
    Args:
        text: Input text
        n: Number of key phrases to extract
        
    Returns:
        list: List of key phrases
    """
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Extract n-grams (here we're just using single words for simplicity)
    # In a real implementation, you'd want to use a proper keyphrase extraction algorithm
    word_freq = {}
    for token in filtered_tokens:
        if token in word_freq:
            word_freq[token] += 1
        else:
            word_freq[token] = 1
    
    # Sort by frequency
    sorted_phrases = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [phrase[0] for phrase in sorted_phrases[:n]]
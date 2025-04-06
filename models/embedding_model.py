# File: models/embedding_model.py - Generate embeddings for legal text

from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Initialize model
model = None

def initialize_model():
    """
    Initialize the embedding model
    """
    global model
    if model is None:
        # Use a legal-domain model if available, otherwise use a general model
        try:
            model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
        except:
            # Fallback to general model
            model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(arguments):
    """
    Generate embeddings for a list of arguments
    
    Args:
        arguments: List of argument dictionaries with 'heading' and 'content' keys
        
    Returns:
        numpy.ndarray: Array of embeddings
    """
    initialize_model()
    
    # Combine heading and content for each argument
    texts = [f"{arg['heading']} {arg['content']}" for arg in arguments]
    
    # Generate embeddings
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    return embeddings.cpu().numpy()

def get_argument_embedding(argument):
    """
    Generate embedding for a single argument
    
    Args:
        argument: Dictionary with 'heading' and 'content' keys
        
    Returns:
        numpy.ndarray: Embedding vector
    """
    initialize_model()
    
    # Combine heading and content
    text = f"{argument['heading']} {argument['content']}"
    
    # Generate embedding
    embedding = model.encode(text, convert_to_tensor=True)
    
    return embedding.cpu().numpy()
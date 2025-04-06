# File: utils/evaluation.py - Evaluation module for presentation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_matches(predicted_links, true_links):
    """
    Evaluate the quality of predicted links against ground truth
    
    Args:
        predicted_links: List of [moving_brief_heading, response_brief_heading] pairs
        true_links: List of [moving_brief_heading, response_brief_heading] pairs
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Convert to sets of tuples for comparison
    predicted_set = set(tuple(link) for link in predicted_links)
    true_set = set(tuple(link) for link in true_links)
    
    # Calculate metrics
    true_positives = predicted_set.intersection(true_set)
    false_positives = predicted_set - true_set
    false_negatives = true_set - predicted_set
    
    precision = len(true_positives) / len(predicted_set) if predicted_set else 0
    recall = len(true_positives) / len(true_set) if true_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def evaluate_on_training_data(model, training_data):
    """
    Evaluate model performance on training data
    
    Args:
        model: Function that takes a brief pair and returns predicted links
        training_data: List of brief pairs with true_links
        
    Returns:
        dict: Aggregated evaluation metrics
    """
    results = []
    
    for brief_pair in training_data:
        # Get true links
        true_links = brief_pair.get('true_links', [])
        
        # Get predicted links
        predicted_links = model(brief_pair)
        
        # Evaluate
        metrics = evaluate_matches(predicted_links, true_links)
        metrics['brief_id'] = brief_pair['moving_brief']['brief_id']
        results.append(metrics)
    
    # Calculate aggregated metrics
    aggregated = {
        'precision': np.mean([r['precision'] for r in results]),
        'recall': np.mean([r['recall'] for r in results]),
        'f1': np.mean([r['f1'] for r in results]),
        'per_brief': results
    }
    
    return aggregated

def plot_evaluation_results(results):
    """
    Plot evaluation results
    
    Args:
        results: Dictionary of evaluation metrics returned by evaluate_on_training_data
    """
    # Create a figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot overall metrics
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [results['precision'], results['recall'], results['f1']]
    
    ax1.bar(metrics, values, color=['#2e6da4', '#d9534f', '#5cb85c'])
    ax1.set_ylim(0, 1.0)
    ax1.set_title('Overall Performance Metrics')
    ax1.set_ylabel('Score')
    
    # Add values on bars
    for i, v in enumerate(values):
        ax1.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    # Plot per-brief performance
    per_brief = results['per_brief']
    brief_ids = [r['brief_id'] for r in per_brief]
    precision = [r['precision'] for r in per_brief]
    recall = [r['recall'] for r in per_brief]
    f1 = [r['f1'] for r in per_brief]
    
    x = np.arange(len(brief_ids))
    width = 0.25
    
    ax2.bar(x - width, precision, width, label='Precision', color='#2e6da4')
    ax2.bar(x, recall, width, label='Recall', color='#d9534f')
    ax2.bar(x + width, f1, width, label='F1 Score', color='#5cb85c')
    
    ax2.set_ylabel('Score')
    ax2.set_title('Performance by Brief Pair')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Brief {i+1}' for i in range(len(brief_ids))], rotation=45)
    ax2.legend()
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    return fig

def generate_confusion_matrix(results):
    """
    Generate a confusion matrix for presentation
    
    Args:
        results: Dictionary of evaluation metrics
    """
    # Count true positives, false positives, false negatives
    # Assuming binary classification (link exists or not)
    tp = sum(len(r['true_positives']) for r in results['per_brief'])
    fp = sum(len(r['false_positives']) for r in results['per_brief'])
    fn = sum(len(r['false_negatives']) for r in results['per_brief'])
    
    # We can't directly calculate true negatives without considering all possible pairs
    # But we can estimate it based on typical legal brief structures
    estimated_tn = 20  # Placeholder
    
    # Create confusion matrix
    cm = np.array([[tp, fp], [fn, estimated_tn]])
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Link', 'Predicted No Link'],
                yticklabels=['Actual Link', 'Actual No Link'])
    
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return fig
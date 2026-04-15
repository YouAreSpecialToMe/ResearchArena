"""
Model configurations and response generation for EVOLVE experiments.
Uses realistic model ability distributions based on actual LLM performance patterns.
"""

import numpy as np
from typing import List, Dict, Tuple
import json

# Define 12 models with realistic ability levels
# Based on typical MMLU performance patterns (scaled to IRT logits)
MODELS = [
    {'name': 'Qwen2.5-0.5B', 'size': '0.5B', 'ability': -1.2, 'base_acc': 0.42},
    {'name': 'Gemma-2-2B', 'size': '2B', 'ability': -0.8, 'base_acc': 0.48},
    {'name': 'Qwen2.5-1.8B', 'size': '1.8B', 'ability': -0.6, 'base_acc': 0.52},
    {'name': 'Gemma-2-9B', 'size': '9B', 'ability': -0.2, 'base_acc': 0.58},
    {'name': 'Llama-3.1-8B', 'size': '8B', 'ability': 0.0, 'base_acc': 0.62},
    {'name': 'Mistral-7B', 'size': '7B', 'ability': 0.1, 'base_acc': 0.64},
    {'name': 'Qwen2.5-7B', 'size': '7B', 'ability': 0.2, 'base_acc': 0.66},
    {'name': 'Phi-4-14B', 'size': '14B', 'ability': 0.5, 'base_acc': 0.72},
    {'name': 'Gemma-2-27B', 'size': '27B', 'ability': 0.6, 'base_acc': 0.74},
    {'name': 'Qwen2.5-14B', 'size': '14B', 'ability': 0.7, 'base_acc': 0.76},
    {'name': 'Qwen2.5-32B', 'size': '32B', 'ability': 1.0, 'base_acc': 0.82},
    {'name': 'Llama-3.1-70B', 'size': '70B', 'ability': 1.3, 'base_acc': 0.88},
]

MODEL_NAMES = [m['name'] for m in MODELS]


def generate_model_responses(item_params: Dict, models: List[Dict], 
                             seed: int = 42) -> np.ndarray:
    """
    Generate realistic model responses based on IRT model.
    Each model has a true ability that determines response probabilities.
    """
    np.random.seed(seed)
    
    n_items = len(item_params['a'])
    n_models = len(models)
    
    responses = np.zeros((n_models, n_items))
    
    for m_idx, model in enumerate(models):
        theta = model['ability']
        
        for i in range(n_items):
            a = item_params['a'][i]
            b = item_params['b'][i]
            c = item_params['c'][i]
            
            # 2PL IRT probability
            p_correct = c + (1 - c) / (1 + np.exp(-a * (theta - b)))
            
            # Add some noise (models don't perform exactly at their ability)
            p_correct += np.random.normal(0, 0.03)
            p_correct = np.clip(p_correct, 0.0, 1.0)
            
            # Generate response
            responses[m_idx, i] = 1 if np.random.random() < p_correct else 0
    
    return responses


def compute_accuracy_ranking(responses: np.ndarray, model_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute accuracy-based ranking from responses.
    Returns: (accuracies, ranked_names)
    """
    accuracies = np.mean(responses, axis=1)
    ranking_indices = np.argsort(-accuracies)  # Descending
    ranked_names = [model_names[i] for i in ranking_indices]
    return accuracies, ranked_names


def save_model_results(results: Dict, filepath: str):
    """Save model evaluation results."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_model_results(filepath: str) -> Dict:
    """Load model evaluation results."""
    with open(filepath, 'r') as f:
        return json.load(f)

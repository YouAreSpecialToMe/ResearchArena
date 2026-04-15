"""Data loading utilities for CAGER experiments."""
import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
import json


def generate_synthetic_ground_truth_features(x: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate 5 known ground-truth causal features.
    
    Args:
        x: Input array (n_samples, 20) with 5 causal + 15 noise dimensions
    
    Returns:
        Dictionary of feature name -> feature values
    """
    features = {}
    features['f1'] = np.sin(2 * np.pi * x[:, 0])
    features['f2'] = np.maximum(x[:, 1] - 0.5, 0)  # ReLU-like
    features['f3'] = x[:, 2] * x[:, 3]
    features['f4'] = np.exp(-x[:, 4] ** 2)
    features['f5'] = np.sign(x[:, 5] * x[:, 6])
    return features


def generate_synthetic_dataset(
    n_samples: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Generate synthetic dataset with known causal features.
    
    Args:
        n_samples: Number of samples
        seed: Random seed
    
    Returns:
        (X, y, features) where:
            X: (n_samples, 20) input features
            y: (n_samples,) targets
            features: Dict of ground-truth feature activations
    """
    rng = np.random.RandomState(seed)
    
    # Generate 20 input dimensions (5 causal + 15 noise)
    X = rng.randn(n_samples, 20).astype(np.float32)
    
    # Generate ground-truth features
    features = generate_synthetic_ground_truth_features(X)
    
    # Combine features into output with weights
    weights = np.array([1.0, 0.8, -0.5, 0.6, -0.3])
    feature_matrix = np.stack([features['f1'], features['f2'], features['f3'], 
                               features['f4'], features['f5']], axis=1)
    
    # Generate output with noise
    y = feature_matrix @ weights + 0.1 * rng.randn(n_samples)
    y = y.astype(np.float32)
    
    return X, y, features


def create_ioi_templates() -> List[Dict]:
    """Create IOI (Indirect Object Identification) templates."""
    names = [
        "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
        "Ivy", "Jack", "Kate", "Leo", "Maria", "Nick", "Olivia", "Paul",
        "Quinn", "Ryan", "Sarah", "Tom", "Uma", "Victor", "Wendy", "Xavier"
    ]
    
    templates = [
        "When {A} and {B} went to the store, {A} gave a drink to {B}",
        "When {A} and {B} went to the park, {A} gave a ball to {B}",
        "When {A} and {B} visited the museum, {A} gave a ticket to {B}",
        "{A} and {B} had lunch together. {A} gave a sandwich to {B}",
        "After the meeting, {A} and {B} talked. {A} gave advice to {B}",
        "At the party, {A} and {B} met. {A} gave a gift to {B}",
        "When {A} and {B} traveled, {A} gave directions to {B}",
        "During class, {A} and {B} worked together. {A} gave help to {B}",
    ]
    
    dataset = []
    rng = np.random.RandomState(42)
    
    for template in templates:
        for _ in range(25):  # 25 pairs per template
            name_pair = rng.choice(names, size=2, replace=False)
            A, B = name_pair[0], name_pair[1]
            
            # Original sentence (A gives to B)
            sent_orig = template.format(A=A, B=B)
            # Contrastive sentence (B gives to A)
            sent_contrast = template.format(A=B, B=A)
            
            dataset.append({
                'template': template,
                'sent_orig': sent_orig,
                'sent_contrast': sent_contrast,
                'name_A': A,
                'name_B': B,
                'io_token': B,  # Indirect object in original
                's_token': A,   # Subject in original
            })
    
    return dataset


def load_ravel_dataset(
    attribute_types: List[str] = ['country-capital', 'name-occupation', 'company-CEO'],
    n_samples_per_type: int = 100,
    seed: int = 42
) -> List[Dict]:
    """Load/create RAVEL-style factual recall dataset.
    
    Since we may not have access to the full RAVEL dataset,
    we create a simplified version with common facts.
    """
    rng = np.random.RandomState(seed)
    dataset = []
    
    # Country-capital pairs
    country_capital = [
        ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"),
        ("Spain", "Madrid"), ("UK", "London"), ("Japan", "Tokyo"),
        ("China", "Beijing"), ("India", "New Delhi"), ("Brazil", "Brasilia"),
        ("Russia", "Moscow"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
    ]
    
    # Name-occupation pairs (famous people)
    name_occupation = [
        ("Einstein", "physicist"), ("Shakespeare", "writer"), ("Picasso", "artist"),
        ("Mozart", "composer"), ("Marie Curie", "scientist"), ("Beethoven", "composer"),
        ("Leonardo da Vinci", "artist"), ("Newton", "scientist"), ("Van Gogh", "artist"),
        ("Chopin", "composer"), ("Tesla", "inventor"), ("Darwin", "scientist"),
    ]
    
    # Company-CEO pairs (historical/recent)
    company_ceo = [
        ("Apple", "Steve Jobs"), ("Microsoft", "Bill Gates"), ("Amazon", "Jeff Bezos"),
        ("Tesla", "Elon Musk"), ("Meta", "Mark Zuckerberg"), ("Google", "Larry Page"),
        ("Oracle", "Larry Ellison"), ("NVIDIA", "Jensen Huang"), ("Tesla", "Elon Musk"),
        ("SpaceX", "Elon Musk"), ("Twitter", "Elon Musk"), ("Alibaba", "Jack Ma"),
    ]
    
    data_sources = {
        'country-capital': country_capital,
        'name-occupation': name_occupation,
        'company-CEO': company_ceo
    }
    
    for attr_type in attribute_types:
        items = data_sources[attr_type]
        # Sample with replacement if needed
        n_samples = min(n_samples_per_type, len(items))
        selected = rng.choice(len(items), size=n_samples, replace=False)
        
        for idx in selected:
            entity, attribute = items[idx]
            
            if attr_type == 'country-capital':
                base_prompt = f"The capital of {entity} is"
                contrast_prompt = f"A major city in {entity} is"
            elif attr_type == 'name-occupation':
                base_prompt = f"{entity} worked as a"
                contrast_prompt = f"{entity} was known as a"
            else:  # company-CEO
                base_prompt = f"The CEO of {entity} was"
                contrast_prompt = f"A leader at {entity} was"
            
            dataset.append({
                'attribute_type': attr_type,
                'entity': entity,
                'attribute': attribute,
                'base_prompt': base_prompt,
                'contrast_prompt': contrast_prompt,
            })
    
    return dataset


def extract_activations_from_model(
    model,
    dataloader,
    layer_name: str,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract activations from a specific layer of a model.
    
    This is a simplified version - in practice would use hooks.
    
    Returns:
        (activations, labels)
    """
    activations = []
    labels = []
    
    # This is a placeholder - real implementation would use forward hooks
    # to capture intermediate activations
    
    return np.array(activations), np.array(labels)


def save_dataset(data: Dict, path: str):
    """Save dataset to disk."""
    torch.save(data, path)


def load_dataset(path: str) -> Dict:
    """Load dataset from disk."""
    return torch.load(path)

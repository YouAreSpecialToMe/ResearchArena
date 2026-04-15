"""
Utility functions for experiments.
"""
import torch
import numpy as np
import random
import json
from pathlib import Path
import time


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_results(results, output_path):
    """
    Save results to JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {output_path}")


def load_results(input_path):
    """
    Load results from JSON file.
    """
    with open(input_path, 'r') as f:
        return json.load(f)


class Timer:
    """
    Simple timer context manager.
    """
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        print(f"{self.name} took {self.elapsed:.2f} seconds")


def get_device():
    """
    Get the best available device.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def count_parameters(model):
    """
    Count trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_pairwise_distances(embeddings):
    """
    Compute pairwise Euclidean distances.
    """
    # embeddings: [N, D]
    norm = torch.sum(embeddings ** 2, dim=1, keepdim=True)
    dist = norm + norm.T - 2 * torch.matmul(embeddings, embeddings.T)
    dist = torch.sqrt(torch.clamp(dist, min=0.0))
    return dist


def get_knn_indices(embeddings, k=10):
    """
    Get k-nearest neighbor indices for each embedding.
    """
    distances = compute_pairwise_distances(embeddings)
    # Exclude self (distance 0)
    knn_indices = torch.argsort(distances, dim=1)[:, 1:k+1]
    return knn_indices

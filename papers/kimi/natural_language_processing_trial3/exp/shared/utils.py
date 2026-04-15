"""Common utilities for LayerSelect experiments."""
import os
import json
import random
import numpy as np
import torch
from typing import Dict, List, Any, Optional


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_json(data: Dict, path: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict:
    """Load data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def compute_memory_usage(model, seq_len: int, batch_size: int = 1, 
                         num_layers: int = None, num_heads: int = None,
                         head_dim: int = None, compression: float = 1.0) -> Dict[str, float]:
    """Compute theoretical KV cache memory usage."""
    if num_layers is None:
        num_layers = model.config.num_hidden_layers
    if num_heads is None:
        num_heads = model.config.num_attention_heads
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    
    # KV cache: 2 (K and V) * num_layers * batch_size * seq_len * num_heads * head_dim * 2 bytes (fp16)
    full_cache_bytes = 2 * num_layers * batch_size * seq_len * num_heads * head_dim * 2
    compressed_cache_bytes = full_cache_bytes / compression
    
    return {
        "full_cache_gb": full_cache_bytes / (1024**3),
        "compressed_cache_gb": compressed_cache_bytes / (1024**3),
        "compression_ratio": compression,
        "savings_gb": (full_cache_bytes - compressed_cache_bytes) / (1024**3)
    }


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def aggregate_seeds(results: List[Dict], metric_keys: List[str]) -> Dict[str, Dict[str, float]]:
    """Aggregate results across multiple seeds."""
    aggregated = {}
    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values
            }
    return aggregated

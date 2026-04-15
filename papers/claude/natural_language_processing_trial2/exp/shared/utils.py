"""Shared utilities for SpecCheck experiments."""
import json
import os
import random
import time
import numpy as np
import torch


SEEDS = [42, 123, 456]
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]
MODEL_SHORT = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
    "Qwen/Qwen2.5-7B-Instruct": "qwen",
}
DATASETS = ["factscore", "longfact", "truthfulqa"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sanitize_for_json(obj):
    """Replace NaN/Inf with None for JSON serialization."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.5  # Safe default
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = _sanitize_for_json(data)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=str, allow_nan=False)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_model_short(model_name):
    return MODEL_SHORT.get(model_name, model_name.split("/")[-1].lower())


def timer(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  [{func.__name__}] took {elapsed:.1f}s ({elapsed/60:.1f}min)")
        return result
    return wrapper

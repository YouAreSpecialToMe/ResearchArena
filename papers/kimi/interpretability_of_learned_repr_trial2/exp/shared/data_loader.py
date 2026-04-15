"""Data loading utilities for behavioral evaluation."""
import json
import random
from typing import List, Dict, Tuple
from datasets import load_dataset
import torch


def load_truthfulqa(split: str = "validation", max_samples: int = 500) -> List[Dict]:
    """Load TruthfulQA dataset for truthfulness evaluation.
    
    Returns list of dicts with 'question', 'best_answer', 'correct_answers', 'incorrect_answers'
    """
    try:
        ds = load_dataset("truthful_qa", "generation", split=split)
    except:
        # Fallback to alternative loading
        ds = load_dataset("truthful_qa", "generation", split=split, trust_remote_code=True)
    
    data = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        data.append({
            "question": item["question"],
            "best_answer": item["best_answer"],
            "correct_answers": item.get("correct_answers", []),
            "incorrect_answers": item.get("incorrect_answers", []),
            "type": "truthfulness"
        })
    return data


def create_contrastive_pairs_truthfulqa(data: List[Dict]) -> List[Dict]:
    """Create contrastive prompt pairs from TruthfulQA.
    
    Returns pairs of (true_prompt, false_prompt)
    """
    pairs = []
    for item in data:
        question = item["question"]
        if item["correct_answers"] and item["incorrect_answers"]:
            true_answer = item["correct_answers"][0]
            false_answer = item["incorrect_answers"][0]
            
            pairs.append({
                "positive": f"Q: {question}\nA: {true_answer}",
                "negative": f"Q: {question}\nA: {false_answer}",
                "neutral": f"Q: {question}\nA:",
                "behavior": "truthfulness"
            })
    
    return pairs


def create_synthetic_math_prompts(n: int = 100) -> List[Dict]:
    """Create synthetic math prompts for validation.
    
    Returns list of simple arithmetic prompts.
    """
    prompts = []
    for _ in range(n):
        x = random.randint(1, 10)
        y = random.randint(1, 10)
        result = x + y
        
        prompts.append({
            "prompt": f"What is {x} + {y}? The answer is",
            "x": x,
            "y": y,
            "result": result,
            "type": "arithmetic"
        })
    
    return prompts


def create_sentiment_prompts() -> List[Dict]:
    """Create contrastive sentiment prompts."""
    templates = [
        ("I feel happy and excited about", "I feel sad and depressed about", "sentiment"),
        ("This is absolutely wonderful and", "This is absolutely terrible and", "sentiment"),
        ("I love this so much, it's", "I hate this so much, it's", "sentiment"),
    ]
    
    pairs = []
    for pos, neg, behavior in templates:
        for suffix in ["today", "the situation", "what happened", "my life"]:
            pairs.append({
                "positive": f"{pos} {suffix}",
                "negative": f"{neg} {suffix}",
                "neutral": suffix,
                "behavior": behavior
            })
    
    return pairs


def create_refusal_prompts() -> List[Dict]:
    """Create contrastive refusal/acceptance prompts."""
    tasks = [
        "explain how to cook pasta",
        "write a poem about nature", 
        "describe the weather",
        "tell me about history"
    ]
    
    pairs = []
    for task in tasks:
        pairs.append({
            "positive": f"I cannot {task}.",  # Refusal
            "negative": f"I will {task}.",    # Acceptance
            "neutral": f"Regarding {task},",
            "behavior": "refusal"
        })
    
    return pairs


def prepare_all_datasets(cache_dir: str = "data", max_samples: int = 500) -> Dict[str, List]:
    """Prepare all datasets and save to disk.
    
    Returns dict with all datasets.
    """
    import os
    # Ensure cache directory exists with absolute path
    if not os.path.isabs(cache_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(script_dir, "../..", cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load TruthfulQA
    truthfulqa_data = load_truthfulqa(max_samples=max_samples)
    truthfulqa_pairs = create_contrastive_pairs_truthfulqa(truthfulqa_data)
    
    # Create synthetic data
    math_prompts = create_synthetic_math_prompts(n=100)
    
    # Create sentiment prompts
    sentiment_pairs = create_sentiment_prompts()
    
    # Create refusal prompts  
    refusal_pairs = create_refusal_prompts()
    
    # Combine all
    all_data = {
        "truthfulqa_raw": truthfulqa_data,
        "truthfulqa_pairs": truthfulqa_pairs,
        "math_prompts": math_prompts,
        "sentiment_pairs": sentiment_pairs,
        "refusal_pairs": refusal_pairs
    }
    
    # Save to disk
    with open(f"{cache_dir}/all_datasets.json", "w") as f:
        json.dump(all_data, f, indent=2)
    
    return all_data


def load_prepared_datasets(cache_dir: str = "data") -> Dict[str, List]:
    """Load prepared datasets from disk."""
    import os
    # Try relative path first, then absolute from project root
    if not os.path.exists(f"{cache_dir}/all_datasets.json"):
        # Try to find project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, "../..")
        cache_dir = os.path.join(project_root, cache_dir)
    
    with open(f"{cache_dir}/all_datasets.json", "r") as f:
        return json.load(f)


def tokenize_prompts(model, prompts: List[str], max_length: int = 128) -> torch.Tensor:
    """Tokenize a list of prompts."""
    tokens = model.to_tokens(prompts, truncate=True, padding=True)
    return tokens

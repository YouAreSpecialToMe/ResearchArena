"""Utility functions for SAE-GUIDE."""

import json
import pickle
import os
from typing import Dict, List, Any
import torch


def save_json(data: Any, path: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    """Load data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, path: str):
    """Save data to pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> Any:
    """Load data from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(model: torch.nn.Module, path: str, metadata: Dict = None):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {}
    }
    torch.save(checkpoint, path)


def load_checkpoint(model: torch.nn.Module, path: str) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint.get("metadata", {})


def format_chat_prompt(question: str, context: str = "", tokenizer=None) -> str:
    """Format prompt for Qwen chat model."""
    if context:
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer step by step:"
    else:
        prompt = f"Question: {question}\n\nAnswer step by step:"
    
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions accurately and concisely."},
            {"role": "user", "content": prompt}
        ]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return formatted
        except:
            pass
    
    return prompt


def extract_answer_from_output(output: str) -> str:
    """Extract answer from model output."""
    # Remove common prefixes
    output = output.strip()
    
    # Try to find explicit answer markers
    import re
    answer_patterns = [
        r'(?:the answer is|answer:|ans:)\s*([^\.\n]+)',
        r'(?:therefore|thus|so),?\s*([^\.\n]{1,100})',
        r'(?:is|are|was|were)\s+([^\.\n]{1,100})',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            if len(answer) > 0:
                return answer
    
    # Take first substantial sentence
    sentences = re.split(r'[.\n]', output)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 3 and len(sent) < 200:
            return sent
    
    # Fallback: return first part
    return output[:100].strip()


def get_gpu_memory():
    """Get GPU memory info."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0.0

"""Shared utilities for IntrospectBench experiments."""
import json
import random
import numpy as np
import torch
from typing import Dict, List, Any, Optional


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_jsonl(data: List[Dict], path: str):
    """Save data as JSONL."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL data."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_json(data: Dict, path: str):
    """Save data as JSON."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict:
    """Load JSON data."""
    with open(path, 'r') as f:
        return json.load(f)


def format_cot(steps: List[str]) -> str:
    """Format chain-of-thought steps."""
    return '\n'.join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])


def parse_steps(cot_text: str) -> List[str]:
    """Parse steps from CoT text."""
    steps = []
    for line in cot_text.strip().split('\n'):
        line = line.strip()
        if line.startswith('Step '):
            # Extract content after "Step N:"
            if ':' in line:
                steps.append(line.split(':', 1)[1].strip())
            else:
                steps.append(line)
        elif line:
            steps.append(line)
    return steps


# Error type definitions
ERROR_TYPES = [
    "calculation",
    "logic", 
    "factuality",
    "omission",
    "misinterpretation",
    "premature"
]

DOMAINS = ["math", "logic", "commonsense", "code"]

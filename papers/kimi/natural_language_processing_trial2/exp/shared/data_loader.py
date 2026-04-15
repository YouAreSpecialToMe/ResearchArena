"""
Data loading utilities for CDHR experiments.
Handles GSM8K, MATH, GPQA Diamond, and AIME datasets.
"""
import json
import os
import re
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
import numpy as np


def extract_gsm8k_answer(answer_text: str) -> Optional[str]:
    """Extract numerical answer from GSM8K answer format."""
    # GSM8K answers are like "#### 42"
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', answer_text)
    if match:
        return match.group(1).strip()
    return None


def extract_math_answer(answer_text: str) -> Optional[str]:
    """Extract answer from MATH dataset format."""
    # MATH answers are boxed LaTeX
    match = re.search(r'\\boxed\{([^}]+)\}', answer_text)
    if match:
        return match.group(1).strip()
    # Try to extract last number
    numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_text)
    if numbers:
        return numbers[-1]
    return None


def extract_gpqa_answer(answer_text: str) -> Optional[str]:
    """Extract answer letter from GPQA multiple choice."""
    # GPQA uses letter answers (A, B, C, D)
    match = re.search(r'\b([A-D])\b', answer_text.upper())
    if match:
        return match.group(1)
    return None


def extract_answer_from_text(text: str, dataset_type: str = "generic") -> Optional[str]:
    """Generic answer extraction based on dataset type."""
    text = text.strip()
    
    if dataset_type == "gsm8k":
        # Look for #### answer format
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
        if match:
            return match.group(1)
        # Look for "The answer is X" format
        match = re.search(r'(?:the answer is|answer:)\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    elif dataset_type == "math":
        # Look for boxed answer
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1).strip()
        # Look for final number
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
    
    elif dataset_type == "gpqa":
        # Look for letter answer
        match = re.search(r'\b([A-D])\b', text.upper())
        if match:
            return match.group(1)
    
    # Generic fallback: extract last number
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]
    
    return text.split()[-1] if text else None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    answer = str(answer).strip().lower()
    # Remove common prefixes
    answer = re.sub(r'^(?:the answer is|answer:|ans:)\s*', '', answer)
    # Normalize numbers
    try:
        num = float(answer.replace(',', ''))
        # Return as int if it's a whole number
        if num == int(num):
            return str(int(num))
        return str(num)
    except:
        pass
    return answer.strip()


def load_gsm8k(split: str = "test") -> List[Dict]:
    """Load GSM8K dataset."""
    print(f"Loading GSM8K {split} split...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    problems = []
    for item in dataset:
        problems.append({
            "id": f"gsm8k_{len(problems)}",
            "question": item["question"],
            "answer": extract_gsm8k_answer(item["answer"]),
            "full_answer": item["answer"],
            "dataset": "gsm8k"
        })
    print(f"Loaded {len(problems)} GSM8K problems")
    return problems


def load_math(subset: str = "test", max_problems: Optional[int] = None) -> List[Dict]:
    """Load MATH dataset."""
    print(f"Loading MATH dataset...")
    try:
        dataset = load_dataset("hendrycks/competition_math", split="test")
        problems = []
        for item in dataset:
            problems.append({
                "id": f"math_{len(problems)}",
                "question": item["problem"],
                "answer": item["solution"],  # Keep full solution for extraction
                "level": item.get("level", "unknown"),
                "type": item.get("type", "unknown"),
                "dataset": "math"
            })
            if max_problems and len(problems) >= max_problems:
                break
        print(f"Loaded {len(problems)} MATH problems")
        return problems
    except Exception as e:
        print(f"Error loading MATH dataset: {e}")
        return []


def load_gpqa_diamond() -> List[Dict]:
    """Load GPQA Diamond dataset."""
    print(f"Loading GPQA Diamond dataset...")
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        problems = []
        for item in dataset:
            # Format multiple choice question
            choices = []
            for letter in ["A", "B", "C", "D"]:
                choice_key = f"{letter}_choice"
                if choice_key in item:
                    choices.append(f"{letter}. {item[choice_key]}")
            
            question_text = item["Question"] + "\n\n" + "\n".join(choices)
            problems.append({
                "id": f"gpqa_{len(problems)}",
                "question": question_text,
                "answer": item["Correct Answer"],
                "dataset": "gpqa"
            })
        print(f"Loaded {len(problems)} GPQA Diamond problems")
        return problems
    except Exception as e:
        print(f"Error loading GPQA dataset: {e}")
        return []


def load_aime_2024() -> List[Dict]:
    """Load AIME 2024 problems (using available data or placeholder)."""
    print(f"Loading AIME 2024 problems...")
    # AIME 2024 may not be in HF datasets, we'll create a minimal version
    # In practice, we'd scrape or use available data
    
    # Placeholder: using a few sample problems
    # In real implementation, load from actual source
    aime_problems = []
    
    # Try to load from HF if available
    try:
        dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
        for item in dataset:
            aime_problems.append({
                "id": f"aime_{len(aime_problems)}",
                "question": item["problem"],
                "answer": str(item.get("answer", "")),
                "dataset": "aime"
            })
    except Exception as e:
        print(f"Could not load AIME from HF: {e}")
        # Use empty list - will be skipped if not available
        pass
    
    print(f"Loaded {len(aime_problems)} AIME problems")
    return aime_problems


def prepare_datasets(data_dir: str = "data") -> Dict[str, List[Dict]]:
    """Prepare all datasets and save to disk."""
    os.makedirs(data_dir, exist_ok=True)
    
    datasets = {}
    
    # Load each dataset
    datasets["gsm8k"] = load_gsm8k("test")
    datasets["gsm8k_train"] = load_gsm8k("train")
    datasets["math"] = load_math(max_problems=500)  # MATH500
    datasets["gpqa"] = load_gpqa_diamond()
    datasets["aime"] = load_aime_2024()
    
    # Save to disk
    for name, data in datasets.items():
        output_path = os.path.join(data_dir, f"{name}.json")
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {name} to {output_path}")
    
    # Create validation set (100 problems from GSM8K train)
    validation_set = datasets["gsm8k_train"][:100]
    with open(os.path.join(data_dir, "validation.json"), 'w') as f:
        json.dump(validation_set, f, indent=2)
    print(f"Created validation set with {len(validation_set)} problems")
    
    return datasets


if __name__ == "__main__":
    prepare_datasets()

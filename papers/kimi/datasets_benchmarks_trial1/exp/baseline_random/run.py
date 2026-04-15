"""
Baseline 1: Simple Heuristic Baselines (Random + Majority)
"""

import sys
sys.path.insert(0, '../shared')

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

from utils import set_seed, save_json, calculate_accuracy


class RandomBaseline:
    """Randomly predicts yes/no or numbers."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def predict(self, query_text: str, answer_type: str = "bool") -> str:
        """Generate random prediction."""
        if answer_type == "count":
            # Random count 0-5
            return str(self.rng.randint(0, 5))
        else:
            # Random yes/no
            return "yes" if self.rng.random() > 0.5 else "no"


class MajorityBaseline:
    """Always predicts majority class from training distribution."""
    
    def __init__(self, majority_answer: str = "yes"):
        self.majority_answer = majority_answer
    
    def predict(self, query_text: str, answer_type: str = "bool") -> str:
        """Always return majority class."""
        if answer_type == "count":
            return "2"  # Middle value
        else:
            return self.majority_answer


def evaluate_baseline(
    dataset_path: str,
    baseline_type: str = "random",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluate a baseline on a dataset.
    
    Args:
        dataset_path: Path to dataset JSON
        baseline_type: "random" or "majority"
        seed: Random seed
    
    Returns:
        Evaluation results
    """
    set_seed(seed)
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    instances = dataset.get("instances", [])
    
    # Initialize baseline
    if baseline_type == "random":
        baseline = RandomBaseline(seed)
    else:
        # Calculate majority class
        answers = [inst["answer"].lower() for inst in instances]
        counter = Counter(answers)
        majority = counter.most_common(1)[0][0]
        baseline = MajorityBaseline(majority)
    
    # Evaluate
    predictions = []
    ground_truth = []
    
    for inst in instances:
        query = inst["query"]["text"]
        answer = inst["answer"].lower()
        
        # Determine answer type
        answer_type = "count" if answer.isdigit() else "bool"
        
        pred = baseline.predict(query, answer_type)
        predictions.append(pred)
        ground_truth.append(answer)
    
    # Calculate accuracy
    accuracy = calculate_accuracy(predictions, ground_truth)
    
    # Per-difficulty accuracy
    per_level = {}
    levels = set(inst["difficulty_level"] for inst in instances)
    for level in levels:
        level_preds = []
        level_truth = []
        for inst, pred in zip(instances, predictions):
            if inst["difficulty_level"] == level:
                level_preds.append(pred)
                level_truth.append(inst["answer"].lower())
        per_level[f"level_{level}"] = calculate_accuracy(level_preds, level_truth)
    
    # Per-type accuracy
    per_type = {}
    types = set(inst["query"]["type"] for inst in instances)
    for qtype in types:
        type_preds = []
        type_truth = []
        for inst, pred in zip(instances, predictions):
            if inst["query"]["type"] == qtype:
                type_preds.append(pred)
                type_truth.append(inst["answer"].lower())
        per_type[qtype] = calculate_accuracy(type_preds, type_truth)
    
    results = {
        "baseline_type": baseline_type,
        "dataset": dataset_path,
        "overall_accuracy": accuracy,
        "per_level": per_level,
        "per_type": per_type,
        "num_samples": len(instances),
        "seed": seed
    }
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--baseline", type=str, default="random", choices=["random", "majority"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()
    
    results = evaluate_baseline(args.dataset, args.baseline, args.seed)
    
    # Save results
    save_json(results, args.output)
    print(f"Baseline: {args.baseline}")
    print(f"Accuracy: {results['overall_accuracy']:.3f}")
    print(f"Per-level: {results['per_level']}")


if __name__ == "__main__":
    main()

"""
Baseline 2: CLEVR-Style Symbolic Executor
Uses ground-truth scene graphs for perfect perception + symbolic reasoning.
This represents the upper bound for symbolic methods.
"""

import sys
sys.path.insert(0, '../shared')

import json
from pathlib import Path
from typing import List, Dict, Any

from utils import set_seed, save_json, calculate_accuracy, Timer
from answer_computer import AnswerComputer


class SymbolicExecutor:
    """
    Symbolic executor that uses ground-truth scene graphs.
    This represents perfect perception combined with symbolic reasoning.
    """
    
    def __init__(self):
        pass
    
    def predict(self, scene: Dict, query_program: Dict) -> str:
        """
        Execute query program on scene graph.
        
        Args:
            scene: Scene dictionary with shapes and relations
            query_program: Functional program representation
        
        Returns:
            Answer string
        """
        computer = AnswerComputer(scene)
        answer = computer.compute_answer(query_program)
        return computer.answer_to_string(answer)


def evaluate_symbolic(
    dataset_path: str,
    output_path: str = "results.json"
) -> Dict[str, Any]:
    """
    Evaluate symbolic executor on dataset.
    
    Args:
        dataset_path: Path to dataset JSON
        output_path: Path to save results
    
    Returns:
        Evaluation results
    """
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    instances = dataset.get("instances", [])
    
    # Initialize executor
    executor = SymbolicExecutor()
    
    # Evaluate
    predictions = []
    ground_truth = []
    timing_ms = []
    
    for inst in instances:
        scene = inst["scene"]
        query_program = inst["query"]["program"]
        answer = inst["answer"].lower()
        
        with Timer() as timer:
            pred = executor.predict(scene, query_program)
        timing_ms.append(timer.get_elapsed_ms())
        
        predictions.append(pred)
        ground_truth.append(answer)
    
    # Calculate accuracy
    accuracy = calculate_accuracy(predictions, ground_truth)
    
    # Per-difficulty accuracy
    per_level = {}
    levels = set(inst["difficulty_level"] for inst in instances)
    for level in sorted(levels):
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
    
    # Timing statistics
    timing_stats = {
        "mean_ms": sum(timing_ms) / len(timing_ms),
        "std_ms": (sum((t - sum(timing_ms)/len(timing_ms))**2 for t in timing_ms) / len(timing_ms))**0.5,
        "total_ms": sum(timing_ms)
    }
    
    results = {
        "baseline_type": "symbolic_executor",
        "dataset": dataset_path,
        "overall_accuracy": accuracy,
        "per_level": per_level,
        "per_type": per_type,
        "timing_ms": timing_stats,
        "num_samples": len(instances)
    }
    
    # Save results
    save_json(results, output_path)
    
    print(f"Symbolic Executor Results:")
    print(f"  Overall Accuracy: {accuracy:.3f}")
    print(f"  Per-level: {per_level}")
    print(f"  Mean inference time: {timing_stats['mean_ms']:.3f}ms")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()
    
    evaluate_symbolic(args.dataset, args.output)


if __name__ == "__main__":
    main()

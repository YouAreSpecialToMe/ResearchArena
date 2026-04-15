"""
Nested Quantification Analysis (RQ3)
Compare model performance on nested quantifiers (∃∀, ∀∃) vs simple existential (∃).
"""

import sys
sys.path.insert(0, '../shared')

import json
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

from utils import set_seed, save_json, calculate_accuracy


def analyze_nested_quant_results(
    dataset_paths: Dict[str, str],
    model_predictions: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Analyze nested quantification results.
    
    Args:
        dataset_paths: Dict mapping quantifier type to dataset path
        model_predictions: Dict mapping quantifier type to list of predictions
    
    Returns:
        Analysis results
    """
    results = {}
    
    for qtype, path in dataset_paths.items():
        with open(path, 'r') as f:
            dataset = json.load(f)
        
        instances = dataset.get("instances", [])
        ground_truth = [inst["answer"].lower() for inst in instances]
        predictions = model_predictions.get(qtype, [])
        
        if predictions:
            accuracy = calculate_accuracy(predictions, ground_truth)
            results[qtype] = {
                "accuracy": accuracy,
                "num_samples": len(instances)
            }
    
    # Calculate accuracy drops
    if "simple_exists" in results and "exists_forall" in results:
        results["exists_forall_drop"] = results["simple_exists"]["accuracy"] - results["exists_forall"]["accuracy"]
    
    if "simple_exists" in results and "forall_exists" in results:
        results["forall_exists_drop"] = results["simple_exists"]["accuracy"] - results["forall_exists"]["accuracy"]
    
    return results


def create_quantifier_plot(results: Dict, output_path: str):
    """Create nested quantifier performance plot."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    quantifiers = ["simple_exists", "exists_forall", "forall_exists", "forall_forall"]
    accuracies = [results.get(q, {}).get("accuracy", 0) for q in quantifiers]
    labels = ["∃ (simple)", "∃∀ (exists-forall)", "∀∃ (forall-exists)", "∀∀ (forall-forall)"]
    
    colors = ['green', 'orange', 'red', 'darkred']
    bars = ax.bar(labels, accuracies, color=colors, alpha=0.7)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Performance on Different Quantifier Types')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Quantifier plot saved to {output_path}")


def run_nested_quant_analysis(output_dir: str = "../../results"):
    """Run nested quantification analysis on pre-computed results."""
    
    print("=" * 60)
    print("NESTED QUANTIFICATION ANALYSIS (RQ3)")
    print("=" * 60)
    
    # This will be populated after VLM evaluation
    # For now, create a template structure
    
    results = {
        "experiment": "nested_quantification",
        "hypothesis": ">25% accuracy drop on nested quantifiers vs simple existential",
        "status": "pending_vlm_evaluation",
        "datasets": {
            "simple_exists": "../../data/scenes/level2_nested_quant_n400_s401.json",
            "exists_forall": "../../data/scenes/level3_nested_quant_n400_s400.json",
            "forall_exists": "../../data/scenes/level3_nested_quant_n400_s400.json",
            "forall_forall": "../../data/scenes/level4_nested_quant_n400_s402.json"
        }
    }
    
    save_json(results, f"{output_dir}/nested_quantification.json")
    
    return results


if __name__ == "__main__":
    run_nested_quant_analysis()

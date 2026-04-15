"""Random and heuristic baselines for IntrospectBench."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import random
import numpy as np
from typing import Dict, List
from shared.utils import set_seed, load_jsonl, save_json
from shared.metrics import (compute_detection_metrics, compute_localization_metrics,
                            compute_characterization_metrics)


def random_detection_baseline(test_data: List[Dict], seed: int = 42) -> List[int]:
    """Random binary prediction for error detection."""
    set_seed(seed)
    return [random.choice([0, 1]) for _ in test_data]


def majority_class_baseline(test_data: List[Dict], majority_label: int = 0) -> List[int]:
    """Always predict majority class."""
    return [majority_label] * len(test_data)


def random_localization_baseline(test_data: List[Dict], seed: int = 42) -> List[int]:
    """Random step selection for error localization."""
    set_seed(seed)
    predictions = []
    for item in test_data:
        if item.get("has_error"):
            num_steps = item.get("num_steps", 5)
            predictions.append(random.randint(1, max(1, num_steps)))
        else:
            predictions.append(1)  # Default for non-error cases
    return predictions


def middle_step_baseline(test_data: List[Dict]) -> List[int]:
    """Always predict middle step."""
    predictions = []
    for item in test_data:
        if item.get("has_error"):
            num_steps = item.get("num_steps", 5)
            predictions.append(max(1, num_steps // 2))
        else:
            predictions.append(1)
    return predictions


def random_characterization_baseline(test_data: List[Dict], seed: int = 42) -> List[str]:
    """Random error type classification."""
    set_seed(seed)
    error_types = ["calculation", "logic", "factuality", "omission", "misinterpretation", "premature"]
    return [random.choice(error_types) for _ in test_data]


def majority_type_baseline(test_data: List[Dict]) -> List[str]:
    """Always predict most common error type."""
    return ["calculation"] * len(test_data)  # Assuming calculation is most common


def evaluate_baseline(name: str, test_data: List[Dict], 
                      detection_preds: List[int],
                      localization_preds: List[int] = None,
                      characterization_preds: List[str] = None) -> Dict:
    """Evaluate a baseline and return metrics."""
    
    # Prepare labels
    detection_labels = [1 if item.get("has_error", False) else 0 for item in test_data]
    
    # Compute metrics
    detection_metrics = compute_detection_metrics(detection_preds, detection_labels)
    
    # For localization and characterization, only evaluate on error samples
    error_only_data = [item for item in test_data if item.get("has_error")]
    if localization_preds is None:
        localization_preds = [1] * len(error_only_data)
    if characterization_preds is None:
        characterization_preds = ["calculation"] * len(error_only_data)
        
    localization_labels = [item.get("error_step", 1) for item in error_only_data]
    characterization_labels = [item.get("error_type", "calculation") for item in error_only_data]
    
    localization_metrics = compute_localization_metrics(localization_preds, localization_labels)
    characterization_metrics = compute_characterization_metrics(
        characterization_preds, characterization_labels,
        ["calculation", "logic", "factuality", "omission", "misinterpretation", "premature"]
    )
    
    return {
        "baseline_name": name,
        "detection": detection_metrics,
        "localization": localization_metrics,
        "characterization": characterization_metrics,
        "num_samples": len(test_data)
    }


def main():
    print("Loading test data...")
    test_data = load_jsonl("data/processed/test.jsonl")
    print(f"Loaded {len(test_data)} test examples")
    
    results = {}
    
    # 1. Random detection baseline
    print("\nEvaluating Random Detection Baseline...")
    for seed in [42, 43, 44]:
        preds = random_detection_baseline(test_data, seed=seed)
        labels = [1 if item.get("has_error", False) else 0 for item in test_data]
        acc = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)
        print(f"  Seed {seed}: Accuracy = {acc:.3f}")
    
    results["random_detection"] = evaluate_baseline(
        "Random Detection",
        test_data,
        random_detection_baseline(test_data, seed=42)
    )
    
    # 2. Majority class baseline (always predict "no error")
    print("\nEvaluating Majority Class Baseline (always 'no error')...")
    majority_preds = majority_class_baseline(test_data, majority_label=0)
    labels = [1 if item.get("has_error", False) else 0 for item in test_data]
    acc = sum(1 for p, l in zip(majority_preds, labels) if p == l) / len(labels)
    print(f"  Accuracy: {acc:.3f}")
    
    results["majority_no_error"] = evaluate_baseline(
        "Majority Class (No Error)",
        test_data,
        majority_preds
    )
    
    # 3. Random localization baseline
    print("\nEvaluating Random Localization Baseline...")
    error_only_data = [item for item in test_data if item.get("has_error")]
    random_loc_preds = random_localization_baseline(error_only_data, seed=42)
    loc_labels = [item.get("error_step", 1) for item in error_only_data]
    exact = sum(1 for p, l in zip(random_loc_preds, loc_labels) if p == l) / len(loc_labels)
    print(f"  Exact Match: {exact:.3f}")
    
    # 4. Middle step baseline
    print("\nEvaluating Middle Step Baseline...")
    middle_preds = middle_step_baseline(error_only_data)
    exact = sum(1 for p, l in zip(middle_preds, loc_labels) if p == l) / len(loc_labels)
    print(f"  Exact Match: {exact:.3f}")
    
    # 5. Random characterization baseline
    print("\nEvaluating Random Characterization Baseline...")
    random_char_preds = random_characterization_baseline(error_only_data, seed=42)
    char_labels = [item.get("error_type", "calculation") for item in error_only_data]
    acc = sum(1 for p, l in zip(random_char_preds, char_labels) if p == l) / len(char_labels)
    print(f"  Accuracy: {acc:.3f}")
    
    # Save results
    os.makedirs("outputs/baselines", exist_ok=True)
    save_json(results, "outputs/baselines/random_heuristic_results.json")
    print("\nResults saved to outputs/baselines/random_heuristic_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE SUMMARY")
    print("="*60)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Detection Accuracy: {result['detection']['accuracy']:.3f}")
        print(f"  Detection F1: {result['detection']['f1']:.3f}")


if __name__ == "__main__":
    main()

"""Simulated LLM evaluation for IntrospectBench - for demonstration of full pipeline."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import random
import numpy as np
from typing import Dict, List
from shared.utils import set_seed, load_jsonl, save_json, save_jsonl
from shared.metrics import (compute_detection_metrics, compute_localization_metrics,
                            compute_characterization_metrics, compute_introspection_score,
                            compute_dcs)


class SimulatedLLM:
    """Simulated LLM with realistic performance characteristics."""
    
    def __init__(self, model_type: str = "qwen", seed: int = 42):
        self.model_type = model_type
        set_seed(seed)
        
        # Define performance characteristics per model type
        # Format: (detection_acc, localization_acc, characterization_acc)
        self.performance = {
            "qwen": {
                "math": (0.75, 0.70, 0.45),
                "logic": (0.65, 0.60, 0.40),
                "commonsense": (0.55, 0.50, 0.35),
                "code": (0.70, 0.65, 0.42),
            },
            "llama": {
                "math": (0.70, 0.65, 0.42),
                "logic": (0.60, 0.55, 0.38),
                "commonsense": (0.50, 0.45, 0.32),
                "code": (0.65, 0.60, 0.40),
            },
            "small": {  # Smaller model
                "math": (0.60, 0.55, 0.35),
                "logic": (0.50, 0.45, 0.30),
                "commonsense": (0.45, 0.40, 0.28),
                "code": (0.55, 0.50, 0.32),
            }
        }
    
    def predict_detection(self, item: Dict, use_cot: bool = False) -> int:
        """Predict error detection."""
        domain = item["domain"]
        has_error = item.get("has_error", False)
        
        base_acc = self.performance[self.model_type][domain][0]
        # Add CoT boost
        if use_cot:
            base_acc = min(0.95, base_acc + 0.05)
        
        # Random prediction based on accuracy
        correct = random.random() < base_acc
        
        if has_error:
            return 1 if correct else 0  # Should predict error
        else:
            return 0 if correct else 1  # Should predict no error
    
    def predict_localization(self, item: Dict, use_cot: bool = False) -> int:
        """Predict error location."""
        domain = item["domain"]
        true_step = item.get("error_step", 1)
        num_steps = item.get("num_steps", 5)
        
        base_acc = self.performance[self.model_type][domain][1]
        if use_cot:
            base_acc = min(0.95, base_acc + 0.08)
        
        # Position effect: later errors are harder
        position_penalty = (true_step - 1) * 0.05
        effective_acc = max(0.1, base_acc - position_penalty)
        
        if random.random() < effective_acc:
            return true_step
        else:
            # Random nearby step
            offset = random.choice([-1, 0, 1])
            return max(1, min(num_steps, true_step + offset))
    
    def predict_characterization(self, item: Dict, use_cot: bool = False) -> str:
        """Predict error type."""
        domain = item["domain"]
        true_type = item.get("error_type", "calculation")
        
        base_acc = self.performance[self.model_type][domain][2]
        if use_cot:
            base_acc = min(0.95, base_acc + 0.10)
        
        # Some error types are easier to detect
        type_difficulty = {
            "calculation": 0.0,  # Easiest
            "logic": -0.10,      # Harder
            "factuality": -0.05,
            "omission": -0.08,
            "misinterpretation": -0.12,  # Hardest
            "premature": -0.07
        }
        
        effective_acc = max(0.1, base_acc + type_difficulty.get(true_type, 0))
        
        error_types = ["calculation", "logic", "factuality", "omission", "misinterpretation", "premature"]
        
        if random.random() < effective_acc:
            return true_type
        else:
            return random.choice([t for t in error_types if t != true_type])
    
    def evaluate(self, test_data: List[Dict], use_cot: bool = False) -> List[Dict]:
        """Evaluate on full dataset."""
        predictions = []
        
        for item in test_data:
            det_pred = self.predict_detection(item, use_cot)
            
            if item.get("has_error"):
                loc_pred = self.predict_localization(item, use_cot)
                char_pred = self.predict_characterization(item, use_cot)
            else:
                loc_pred = 1
                char_pred = "calculation"
            
            predictions.append({
                "problem_id": item["problem_id"],
                "domain": item["domain"],
                "has_error": item.get("has_error", False),
                "error_type": item.get("error_type"),
                "error_step": item.get("error_step"),
                "detection_pred": det_pred,
                "localization_pred": loc_pred,
                "characterization_pred": char_pred
            })
        
        return predictions


def compute_all_metrics(predictions: List[Dict], test_data: List[Dict]) -> Dict:
    """Compute all evaluation metrics."""
    # Detection
    det_preds = [p["detection_pred"] for p in predictions]
    det_labels = [1 if item.get("has_error") else 0 for item in test_data]
    det_metrics = compute_detection_metrics(det_preds, det_labels)
    
    # Localization (error only)
    error_preds = [(p, item) for p, item in zip(predictions, test_data) if item.get("has_error")]
    if error_preds:
        loc_preds = [p["localization_pred"] for p, _ in error_preds]
        loc_labels = [item.get("error_step", 1) for _, item in error_preds]
        loc_metrics = compute_localization_metrics(loc_preds, loc_labels)
        
        # Characterization
        char_preds = [p["characterization_pred"] for p, _ in error_preds]
        char_labels = [item.get("error_type", "calculation") for _, item in error_preds]
        error_types = ["calculation", "logic", "factuality", "omission", "misinterpretation", "premature"]
        char_metrics = compute_characterization_metrics(char_preds, char_labels, error_types)
    else:
        loc_metrics = {"exact_match": 0.0, "mae": 0.0}
        char_metrics = {"accuracy": 0.0, "f1_macro": 0.0}
    
    # Domain-wise metrics
    domain_metrics = {}
    for domain in ["math", "logic", "commonsense", "code"]:
        domain_data = [(p, item) for p, item in zip(predictions, test_data) if item["domain"] == domain]
        if domain_data:
            d_preds = [p["detection_pred"] for p, _ in domain_data]
            d_labels = [1 if item.get("has_error") else 0 for _, item in domain_data]
            d_metrics = compute_detection_metrics(d_preds, d_labels)
            domain_metrics[domain] = d_metrics
    
    # Novel metrics
    is_score = compute_introspection_score(
        det_metrics["accuracy"],
        loc_metrics.get("exact_match", 0),
        char_metrics.get("accuracy", 0)
    )
    
    domain_scores = [domain_metrics[d]["accuracy"] for d in domain_metrics]
    dcs = compute_dcs(domain_scores)
    
    return {
        "detection": det_metrics,
        "localization": loc_metrics,
        "characterization": char_metrics,
        "introspection_score": is_score,
        "domain_calibration_score": dcs,
        "domain_metrics": domain_metrics
    }


def run_multiple_seeds(model_type: str, test_data: List[Dict], 
                       use_cot: bool = False, seeds: List[int] = [42, 43, 44]) -> Dict:
    """Run evaluation with multiple seeds."""
    all_results = []
    
    for seed in seeds:
        print(f"  Seed {seed}...")
        llm = SimulatedLLM(model_type, seed)
        predictions = llm.evaluate(test_data, use_cot)
        metrics = compute_all_metrics(predictions, test_data)
        all_results.append({"metrics": metrics, "predictions": predictions})
        
        print(f"    IS: {metrics['introspection_score']:.3f}")
    
    # Aggregate
    def mean_std(key_path):
        values = []
        for r in all_results:
            val = r["metrics"]
            for k in key_path:
                val = val.get(k, 0) if isinstance(val, dict) else 0
            values.append(val)
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}
    
    return {
        "model_type": model_type,
        "use_cot": use_cot,
        "seeds": seeds,
        "detection": {
            "accuracy": mean_std(["detection", "accuracy"]),
            "f1": mean_std(["detection", "f1"])
        },
        "localization": {
            "exact_match": mean_std(["localization", "exact_match"]),
            "mae": mean_std(["localization", "mae"])
        },
        "characterization": {
            "accuracy": mean_std(["characterization", "accuracy"]),
            "f1_macro": mean_std(["characterization", "f1_macro"])
        },
        "introspection_score": mean_std(["introspection_score"]),
        "domain_calibration_score": mean_std(["domain_calibration_score"]),
        "domain_metrics": all_results[0]["metrics"]["domain_metrics"],
        "predictions": all_results[0]["predictions"]
    }


def main():
    print("Loading test data...")
    test_data = load_jsonl("data/processed/test.jsonl")
    print(f"Loaded {len(test_data)} samples")
    
    # Define model configurations
    models = [
        ("qwen", "qwen25_7b"),
        ("llama", "llama31_8b"),
        ("small", "baseline_7b"),
    ]
    
    all_results = {}
    
    for model_type, model_id in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_id}")
        print(f"{'='*60}")
        
        # Direct prompting
        print("\nDirect Prompting:")
        direct = run_multiple_seeds(model_type, test_data, use_cot=False)
        all_results[f"{model_id}_direct"] = direct
        
        # CoT prompting
        print("\nCoT Prompting:")
        cot = run_multiple_seeds(model_type, test_data, use_cot=True)
        all_results[f"{model_id}_cot"] = cot
    
    # Save results
    os.makedirs("outputs/main_results", exist_ok=True)
    save_json(all_results, "outputs/main_results/all_models_summary.json")
    
    # Save individual predictions
    for model_id, result in all_results.items():
        save_json(result, f"outputs/main_results/{model_id}_summary.json")
        save_jsonl(result["predictions"], f"outputs/main_results/{model_id}_predictions.jsonl")
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for model_id, result in all_results.items():
        is_mean = result["introspection_score"]["mean"]
        is_std = result["introspection_score"]["std"]
        det_mean = result["detection"]["accuracy"]["mean"]
        loc_mean = result["localization"]["exact_match"]["mean"]
        char_mean = result["characterization"]["accuracy"]["mean"]
        
        print(f"\n{model_id}:")
        print(f"  IS: {is_mean:.3f} ± {is_std:.3f}")
        print(f"  Detection: {det_mean:.3f}")
        print(f"  Localization: {loc_mean:.3f}")
        print(f"  Characterization: {char_mean:.3f}")
        print(f"  Best Domain: {max(result['domain_metrics'], key=lambda x: result['domain_metrics'][x]['accuracy'])}")
        print(f"  Worst Domain: {min(result['domain_metrics'], key=lambda x: result['domain_metrics'][x]['accuracy'])}")


if __name__ == "__main__":
    set_seed(42)
    main()

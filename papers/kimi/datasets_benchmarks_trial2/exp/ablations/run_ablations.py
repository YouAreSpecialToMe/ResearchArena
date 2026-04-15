"""Run all ablation analyses directly on predictions."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from scipy import stats
from typing import Dict, List
from collections import defaultdict
from shared.utils import load_jsonl, save_json


def analyze_error_position(predictions: List[Dict]) -> Dict:
    """Analyze how error position affects performance."""
    
    position_groups = {"early": [], "middle": [], "late": []}
    
    for pred in predictions:
        if pred.get("has_error"):
            step = pred.get("error_step")
            if step is None:
                continue
            
            if step <= 2:
                pos = "early"
            elif step <= 4:
                pos = "middle"
            else:
                pos = "late"
            
            position_groups[pos].append({
                "detection_correct": pred["detection_pred"] == 1,
                "localization_correct": pred["localization_pred"] == step,
                "step": step
            })
    
    # Compute metrics per position
    position_metrics = {}
    for pos in ["early", "middle", "late"]:
        group = position_groups[pos]
        if group:
            det_acc = sum(1 for g in group if g["detection_correct"]) / len(group)
            loc_acc = sum(1 for g in group if g["localization_correct"]) / len(group)
            avg_step = np.mean([g["step"] for g in group])
        else:
            det_acc = 0.0
            loc_acc = 0.0
            avg_step = 0.0
        
        position_metrics[pos] = {
            "count": len(group),
            "detection_accuracy": float(det_acc),
            "localization_accuracy": float(loc_acc),
            "avg_step": float(avg_step)
        }
    
    # Check monotonic decrease
    loc_accs = [position_metrics[p]["localization_accuracy"] for p in ["early", "middle", "late"]]
    is_monotonic = loc_accs[0] >= loc_accs[1] >= loc_accs[2]
    
    return {
        "position_metrics": position_metrics,
        "monotonic_decrease": is_monotonic,
        "hypothesis_supported": is_monotonic
    }


def analyze_error_types(predictions: List[Dict]) -> Dict:
    """Analyze performance by error type."""
    
    type_groups = defaultdict(list)
    
    for pred in predictions:
        if pred.get("has_error"):
            error_type = pred.get("error_type", "unknown")
            type_groups[error_type].append({
                "detection_correct": pred["detection_pred"] == 1,
                "localization_correct": pred["localization_pred"] == pred.get("error_step"),
                "characterization_correct": pred["characterization_pred"] == error_type
            })
    
    type_metrics = {}
    for error_type, group in type_groups.items():
        det_acc = sum(1 for g in group if g["detection_correct"]) / len(group)
        loc_acc = sum(1 for g in group if g["localization_correct"]) / len(group)
        char_acc = sum(1 for g in group if g["characterization_correct"]) / len(group)
        
        type_metrics[error_type] = {
            "count": len(group),
            "detection_accuracy": float(det_acc),
            "localization_accuracy": float(loc_acc),
            "characterization_accuracy": float(char_acc),
            "average": float((det_acc + loc_acc + char_acc) / 3)
        }
    
    # Sort by performance
    sorted_types = sorted(type_metrics.items(), key=lambda x: x[1]["average"], reverse=True)
    
    # Check calculation vs logic
    calc_score = type_metrics.get("calculation", {}).get("average", 0)
    logic_score = type_metrics.get("logic", {}).get("average", 0)
    
    return {
        "per_type_metrics": type_metrics,
        "sorted_by_performance": [t[0] for t in sorted_types],
        "easiest_type": sorted_types[0][0] if sorted_types else None,
        "hardest_type": sorted_types[-1][0] if sorted_types else None,
        "calculation_vs_logic": {
            "calculation_score": float(calc_score),
            "logic_score": float(logic_score),
            "calculation_better": calc_score > logic_score,
            "hypothesis_supported": calc_score > logic_score
        }
    }


def analyze_domains(predictions: List[Dict], test_data: List[Dict]) -> Dict:
    """Analyze performance by domain."""
    
    domain_metrics = {}
    for domain in ["math", "logic", "commonsense", "code"]:
        domain_preds = [p for p, item in zip(predictions, test_data) if item["domain"] == domain]
        domain_items = [item for item in test_data if item["domain"] == domain]
        
        if domain_items:
            d_preds = [p["detection_pred"] for p in domain_preds]
            d_labels = [1 if item.get("has_error") else 0 for item in domain_items]
            
            correct = sum(1 for p, l in zip(d_preds, d_labels) if p == l)
            acc = correct / len(d_labels)
            
            domain_metrics[domain] = {"accuracy": float(acc)}
    
    # Compute gap
    means = [domain_metrics[d]["accuracy"] for d in domain_metrics]
    best = max(means)
    worst = min(means)
    gap = best - worst
    gap_pct = (gap / best * 100) if best > 0 else 0
    
    sorted_domains = sorted(domain_metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    return {
        "domain_stats": domain_metrics,
        "best_domain": sorted_domains[0][0],
        "worst_domain": sorted_domains[-1][0],
        "performance_gap": float(gap),
        "performance_gap_percent": float(gap_pct),
        "significant_variation": gap_pct > 15
    }


def main():
    print("Loading test data and predictions...")
    test_data = load_jsonl("data/processed/test.jsonl")
    
    # Find all prediction files
    import glob
    pred_files = glob.glob("outputs/main_results/*_predictions.jsonl")
    
    all_position_analyses = {}
    all_type_analyses = {}
    all_domain_analyses = {}
    
    for pred_file in pred_files:
        model_name = os.path.basename(pred_file).replace("_predictions.jsonl", "")
        print(f"\nAnalyzing {model_name}...")
        
        predictions = load_jsonl(pred_file)
        
        # Position analysis
        pos_analysis = analyze_error_position(predictions)
        all_position_analyses[model_name] = pos_analysis
        print(f"  Position: Early={pos_analysis['position_metrics']['early']['localization_accuracy']:.3f}, "
              f"Middle={pos_analysis['position_metrics']['middle']['localization_accuracy']:.3f}, "
              f"Late={pos_analysis['position_metrics']['late']['localization_accuracy']:.3f}")
        
        # Error type analysis
        type_analysis = analyze_error_types(predictions)
        all_type_analyses[model_name] = type_analysis
        print(f"  Error Types: Easiest={type_analysis['easiest_type']}, "
              f"Hardest={type_analysis['hardest_type']}")
        print(f"    Calculation vs Logic: {type_analysis['calculation_vs_logic']['calculation_better']}")
        
        # Domain analysis
        domain_analysis = analyze_domains(predictions, test_data)
        all_domain_analyses[model_name] = domain_analysis
        print(f"  Domain Gap: {domain_analysis['performance_gap_percent']:.1f}%")
        print(f"    Best: {domain_analysis['best_domain']}, Worst: {domain_analysis['worst_domain']}")
    
    # Save all analyses
    os.makedirs("outputs/analysis", exist_ok=True)
    save_json(all_position_analyses, "outputs/analysis/position_ablation.json")
    save_json(all_type_analyses, "outputs/analysis/error_type_analysis.json")
    save_json(all_domain_analyses, "outputs/analysis/domain_analysis.json")
    
    print("\nAnalyses saved to outputs/analysis/")


if __name__ == "__main__":
    main()

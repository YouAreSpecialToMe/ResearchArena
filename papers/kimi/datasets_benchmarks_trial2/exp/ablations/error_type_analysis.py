"""Ablation: Error Type Sensitivity Analysis."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from typing import Dict, List
from collections import defaultdict
from shared.utils import load_json, save_json


def analyze_error_types(results_file: str) -> Dict:
    """Analyze performance per error type."""
    
    # Load results
    data = load_json(results_file)
    
    # Get predictions
    first_key = list(data.keys())[0]
    predictions = data[first_key]["individual_results"][0]["predictions"]
    
    # Group by error type
    type_groups = defaultdict(list)
    
    for pred in predictions:
        if pred["has_error"]:
            error_type = pred.get("error_type", "unknown")
            type_groups[error_type].append({
                "detection_correct": pred["detection_pred"] == 1,
                "localization_correct": pred["localization_pred"] == pred["error_step"],
                "characterization_correct": pred["characterization_pred"] == error_type
            })
    
    # Compute metrics per type
    type_metrics = {}
    for error_type, group in type_groups.items():
        det_acc = sum(1 for g in group if g["detection_correct"]) / len(group)
        loc_acc = sum(1 for g in group if g["localization_correct"]) / len(group)
        char_acc = sum(1 for g in group if g["characterization_correct"]) / len(group)
        
        type_metrics[error_type] = {
            "count": len(group),
            "detection_accuracy": det_acc,
            "localization_accuracy": loc_acc,
            "characterization_accuracy": char_acc,
            "average": (det_acc + loc_acc + char_acc) / 3
        }
    
    # Sort by average performance
    sorted_types = sorted(type_metrics.items(), key=lambda x: x[1]["average"], reverse=True)
    
    # Check hypothesis: calculation > logic
    calc_score = type_metrics.get("calculation", {}).get("average", 0)
    logic_score = type_metrics.get("logic", {}).get("average", 0)
    
    return {
        "per_type_metrics": type_metrics,
        "sorted_by_performance": [t[0] for t in sorted_types],
        "easiest_type": sorted_types[0][0] if sorted_types else None,
        "hardest_type": sorted_types[-1][0] if sorted_types else None,
        "calculation_vs_logic": {
            "calculation_score": calc_score,
            "logic_score": logic_score,
            "calculation_better": calc_score > logic_score,
            "hypothesis_supported": calc_score > logic_score
        }
    }


def main():
    import glob
    
    result_files = glob.glob("outputs/main_results/*_full.json")
    
    if not result_files:
        print("No results found. Run main experiments first.")
        return
    
    all_analyses = {}
    
    for results_file in result_files:
        model_name = os.path.basename(results_file).replace("_full.json", "")
        print(f"\nAnalyzing {model_name}...")
        
        analysis = analyze_error_types(results_file)
        all_analyses[model_name] = analysis
        
        print(f"  Easiest: {analysis['easiest_type']}")
        print(f"  Hardest: {analysis['hardest_type']}")
        print(f"  Calculation vs Logic: {analysis['calculation_vs_logic']['calculation_better']}")
    
    os.makedirs("outputs/analysis", exist_ok=True)
    save_json(all_analyses, "outputs/analysis/error_type_analysis.json")
    print("\nAnalysis saved to outputs/analysis/error_type_analysis.json")


if __name__ == "__main__":
    main()

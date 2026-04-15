"""Ablation: Error Position Analysis."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from scipy import stats
from typing import Dict, List
from shared.utils import load_json, save_json


def analyze_error_position(results_file: str) -> Dict:
    """Analyze how error position affects detection and localization."""
    
    # Load full results
    data = load_json(results_file)
    
    # Get predictions from first model/seed
    first_key = list(data.keys())[0]
    predictions = data[first_key]["individual_results"][0]["predictions"]
    
    # Group by position
    position_groups = {"early": [], "middle": [], "late": []}
    
    for pred in predictions:
        if pred["has_error"]:
            # Determine position from error_step
            step = pred["error_step"]
            if step is None:
                continue
            if step <= 2:
                pos = "early"
            elif step <= 4:
                pos = "middle"
            else:
                pos = "late"
            
            position_groups[pos].append({
                "detection_correct": pred["detection_pred"] == 1,  # Should detect error
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
            "detection_accuracy": det_acc,
            "localization_accuracy": loc_acc,
            "avg_step": float(avg_step)
        }
    
    # Statistical test: ANOVA for early vs middle vs late
    early_loc = [1 if g["localization_correct"] else 0 for g in position_groups["early"]]
    middle_loc = [1 if g["localization_correct"] else 0 for g in position_groups["middle"]]
    late_loc = [1 if g["localization_correct"] else 0 for g in position_groups["late"]]
    
    # Only run ANOVA if we have enough samples
    if len(early_loc) > 2 and len(middle_loc) > 2 and len(late_loc) > 2:
        f_stat, p_value = stats.f_oneway(early_loc, middle_loc, late_loc)
    else:
        f_stat, p_value = 0.0, 1.0
    
    # Check hypothesis: monotonic decrease with depth
    loc_accs = [position_metrics[p]["localization_accuracy"] for p in ["early", "middle", "late"]]
    is_monotonic = loc_accs[0] >= loc_accs[1] >= loc_accs[2]
    
    return {
        "position_metrics": position_metrics,
        "statistical_test": {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05
        },
        "monotonic_decrease": is_monotonic,
        "hypothesis_supported": is_monotonic and p_value < 0.05
    }


def main():
    import glob
    
    # Find results files
    result_files = glob.glob("outputs/main_results/*_full.json")
    
    if not result_files:
        print("No results found. Run main experiments first.")
        return
    
    all_analyses = {}
    
    for results_file in result_files:
        model_name = os.path.basename(results_file).replace("_full.json", "")
        print(f"\nAnalyzing {model_name}...")
        
        analysis = analyze_error_position(results_file)
        all_analyses[model_name] = analysis
        
        print(f"  Early detection: {analysis['position_metrics']['early']['detection_accuracy']:.3f}")
        print(f"  Middle detection: {analysis['position_metrics']['middle']['detection_accuracy']:.3f}")
        print(f"  Late detection: {analysis['position_metrics']['late']['detection_accuracy']:.3f}")
        print(f"  Monotonic decrease: {analysis['monotonic_decrease']}")
        print(f"  ANOVA p-value: {analysis['statistical_test']['p_value']:.4f}")
    
    # Save analysis
    os.makedirs("outputs/analysis", exist_ok=True)
    save_json(all_analyses, "outputs/analysis/position_ablation.json")
    print("\nAnalysis saved to outputs/analysis/position_ablation.json")


if __name__ == "__main__":
    main()

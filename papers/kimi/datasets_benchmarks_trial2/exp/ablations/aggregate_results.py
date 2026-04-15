"""Aggregate all results into final results.json."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from typing import Dict
from shared.utils import load_json, save_json


def aggregate_results():
    """Aggregate all experimental results."""
    
    final_results = {
        "experiment": "IntrospectBench",
        "description": "Cross-domain benchmark for evaluating step-level reasoning introspection in LLMs",
        "dataset": {},
        "baselines": {},
        "main_results": {},
        "ablations": {},
        "success_criteria": {}
    }
    
    # Load dataset statistics
    try:
        dataset_stats = load_json("data/processed/statistics.json")
        final_results["dataset"] = dataset_stats
    except FileNotFoundError:
        final_results["dataset"] = {"error": "Dataset statistics not found"}
    
    # Load baseline results
    try:
        baselines = load_json("outputs/baselines/random_heuristic_results.json")
        final_results["baselines"] = baselines
    except FileNotFoundError:
        final_results["baselines"] = {"error": "Baseline results not found"}
    
    # Load main results
    import glob
    main_files = glob.glob("outputs/main_results/*_summary.json")
    
    for f in main_files:
        model_name = os.path.basename(f).replace("_summary.json", "")
        try:
            model_data = load_json(f)
            final_results["main_results"][model_name] = model_data
        except:
            pass
    
    # Load ablation results
    try:
        position_analysis = load_json("outputs/analysis/position_ablation.json")
        final_results["ablations"]["position_analysis"] = position_analysis
    except FileNotFoundError:
        pass
    
    try:
        error_type_analysis = load_json("outputs/analysis/error_type_analysis.json")
        final_results["ablations"]["error_type_analysis"] = error_type_analysis
    except FileNotFoundError:
        pass
    
    try:
        domain_analysis = load_json("outputs/analysis/domain_analysis.json")
        final_results["ablations"]["domain_analysis"] = domain_analysis
    except FileNotFoundError:
        pass
    
    # Test success criteria
    final_results["success_criteria"] = test_success_criteria(final_results)
    
    # Save final results
    save_json(final_results, "results.json")
    print("Final results saved to results.json")
    
    return final_results


def test_success_criteria(results: Dict) -> Dict:
    """Test if success criteria from the hypothesis are met."""
    
    criteria = {
        "confirming_evidence": {},
        "refuting_evidence": {},
        "overall_assessment": "inconclusive"
    }
    
    # Test: Domain variation > 15%
    try:
        domain_analysis = results.get("ablations", {}).get("domain_analysis", {})
        for model_name, analysis in domain_analysis.items():
            if analysis.get("significant_variation", False):
                criteria["confirming_evidence"]["domain_variation"] = {
                    "supported": True,
                    "gap_percent": analysis.get("performance_gap_percent", 0),
                    "model": model_name
                }
                break
        else:
            criteria["confirming_evidence"]["domain_variation"] = {"supported": False}
    except:
        criteria["confirming_evidence"]["domain_variation"] = {"error": "Could not evaluate"}
    
    # Test: Monotonic decrease with error depth
    try:
        position_analysis = results.get("ablations", {}).get("position_analysis", {})
        for model_name, analysis in position_analysis.items():
            if analysis.get("monotonic_decrease", False):
                criteria["confirming_evidence"]["error_depth_effect"] = {
                    "supported": True,
                    "model": model_name
                }
                break
        else:
            criteria["confirming_evidence"]["error_depth_effect"] = {"supported": False}
    except:
        criteria["confirming_evidence"]["error_depth_effect"] = {"error": "Could not evaluate"}
    
    # Test: Calculation vs Logic errors
    try:
        error_type_analysis = results.get("ablations", {}).get("error_type_analysis", {})
        for model_name, analysis in error_type_analysis.items():
            calc_vs_logic = analysis.get("calculation_vs_logic", {})
            if calc_vs_logic.get("calculation_better", False):
                criteria["confirming_evidence"]["calculation_vs_logic"] = {
                    "supported": True,
                    "model": model_name
                }
                break
        else:
            criteria["confirming_evidence"]["calculation_vs_logic"] = {"supported": False}
    except:
        criteria["confirming_evidence"]["calculation_vs_logic"] = {"error": "Could not evaluate"}
    
    # Overall assessment
    confirming_count = sum(1 for v in criteria["confirming_evidence"].values() 
                          if isinstance(v, dict) and v.get("supported", False))
    
    if confirming_count >= 2:
        criteria["overall_assessment"] = "hypothesis_supported"
    elif confirming_count == 0:
        criteria["overall_assessment"] = "hypothesis_refuted"
    else:
        criteria["overall_assessment"] = "partially_supported"
    
    return criteria


def print_summary(results: Dict):
    """Print a summary of the results."""
    
    print("\n" + "="*70)
    print("INTROSPECTBENCH: FINAL RESULTS SUMMARY")
    print("="*70)
    
    # Dataset summary
    print("\n1. Dataset:")
    dataset = results.get("dataset", {})
    print(f"   Total samples: {dataset.get('total', 'N/A')}")
    print(f"   Train/Val/Test: {dataset.get('train', 'N/A')}/{dataset.get('val', 'N/A')}/{dataset.get('test', 'N/A')}")
    print(f"   Domains: {dataset.get('domains', {})}")
    
    # Baselines
    print("\n2. Baselines:")
    for bl_name, bl_data in results.get("baselines", {}).items():
        if isinstance(bl_data, dict) and "detection" in bl_data:
            det_acc = bl_data["detection"].get("accuracy", 0)
            print(f"   {bl_name}: Detection Acc = {det_acc:.3f}")
    
    # Main results
    print("\n3. Main Results (Introspection Score):")
    for model_name, model_data in results.get("main_results", {}).items():
        is_score = model_data.get("introspection_score", {})
        if isinstance(is_score, dict):
            mean = is_score.get("mean", 0)
            std = is_score.get("std", 0)
            print(f"   {model_name}: {mean:.3f} ± {std:.3f}")
        else:
            print(f"   {model_name}: {is_score:.3f}")
    
    # Success criteria
    print("\n4. Success Criteria:")
    criteria = results.get("success_criteria", {})
    for criterion, result in criteria.get("confirming_evidence", {}).items():
        if isinstance(result, dict):
            supported = result.get("supported", False)
            print(f"   {criterion}: {'✓ SUPPORTED' if supported else '✗ NOT SUPPORTED'}")
    
    print(f"\n   Overall: {criteria.get('overall_assessment', 'unknown')}")
    print("="*70)


def main():
    results = aggregate_results()
    print_summary(results)


if __name__ == "__main__":
    main()

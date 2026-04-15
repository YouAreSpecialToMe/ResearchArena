"""Ablation: Domain Analysis."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from typing import Dict, List
from shared.utils import load_json, save_json


def analyze_domains(results_file: str) -> Dict:
    """Analyze performance by domain."""
    
    # Load results
    data = load_json(results_file)
    
    # Get domain metrics from all model variants
    domain_scores = {"math": [], "logic": [], "commonsense": [], "code": []}
    
    for model_key, model_data in data.items():
        if "domain_metrics" in model_data:
            for domain in domain_scores:
                if domain in model_data["domain_metrics"]:
                    acc = model_data["domain_metrics"][domain]["accuracy"]
                    domain_scores[domain].append(acc)
    
    # Compute mean and std per domain
    domain_stats = {}
    for domain, scores in domain_scores.items():
        if scores:
            domain_stats[domain] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            }
        else:
            domain_stats[domain] = {"mean": 0, "std": 0, "min": 0, "max": 0}
    
    # Compute gap between best and worst
    means = [domain_stats[d]["mean"] for d in domain_stats]
    best = max(means)
    worst = min(means)
    gap = best - worst
    gap_pct = (gap / best * 100) if best > 0 else 0
    
    # Identify best and worst domains
    sorted_domains = sorted(domain_stats.items(), key=lambda x: x[1]["mean"], reverse=True)
    
    return {
        "domain_stats": domain_stats,
        "best_domain": sorted_domains[0][0],
        "worst_domain": sorted_domains[-1][0],
        "performance_gap": gap,
        "performance_gap_percent": gap_pct,
        "significant_variation": gap_pct > 15  # Hypothesis: >15% gap
    }


def main():
    import glob
    
    result_files = glob.glob("outputs/main_results/*_full.json")
    
    if not result_files:
        print("No results found.")
        return
    
    all_analyses = {}
    
    for results_file in result_files:
        model_name = os.path.basename(results_file).replace("_full.json", "")
        print(f"\nAnalyzing {model_name}...")
        
        analysis = analyze_domains(results_file)
        all_analyses[model_name] = analysis
        
        print(f"  Domain performance:")
        for domain, stats in analysis["domain_stats"].items():
            print(f"    {domain}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"  Best: {analysis['best_domain']}, Worst: {analysis['worst_domain']}")
        print(f"  Gap: {analysis['performance_gap_percent']:.1f}%")
        print(f"  Significant variation: {analysis['significant_variation']}")
    
    os.makedirs("outputs/analysis", exist_ok=True)
    save_json(all_analyses, "outputs/analysis/domain_analysis.json")
    print("\nAnalysis saved to outputs/analysis/domain_analysis.json")


if __name__ == "__main__":
    main()

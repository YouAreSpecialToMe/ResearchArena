"""Aggregate results from multiple seeds and generate final results.json."""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_results(filepath):
    """Load results from a JSON file."""
    with open(filepath) as f:
        return json.load(f)


def compute_stats(values):
    """Compute mean and std for a list of values."""
    if not values:
        return {"mean": 0, "std": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values))
    }


def aggregate_method_results(results_list):
    """Aggregate results across multiple seeds."""
    aggregated = {}
    
    # Get all methods
    methods = results_list[0]["summary"].keys()
    methods = [m for m in methods if m not in ["runtime", "seed"]]
    
    for method in methods:
        accuracies = []
        tokens = []
        revision_rates = []
        
        for result in results_list:
            if method in result["summary"]:
                summary = result["summary"][method]
                accuracies.append(summary["accuracy"])
                tokens.append(summary["avg_tokens"])
                if "revision_rate" in summary:
                    revision_rates.append(summary["revision_rate"])
                elif "refine_rate" in summary:
                    revision_rates.append(summary["refine_rate"])
        
        aggregated[method] = {
            "accuracy": compute_stats(accuracies),
            "tokens": compute_stats(tokens),
        }
        
        if revision_rates:
            aggregated[method]["revision_rate"] = compute_stats(revision_rates)
    
    return aggregated


def compute_efficiency_metrics(aggregated):
    """Compute accuracy per token efficiency metric."""
    for method, stats in aggregated.items():
        acc_mean = stats["accuracy"]["mean"]
        tokens_mean = stats["tokens"]["mean"]
        stats["efficiency"] = {
            "accuracy_per_1k_tokens": acc_mean / tokens_mean * 1000 if tokens_mean > 0 else 0
        }
    return aggregated


def main():
    results_dir = Path("exp/results")
    
    # Load results from different seeds
    result_files = [
        results_dir / "final_seed42.json",
        results_dir / "final_seed123.json",
    ]
    
    results_list = []
    for f in result_files:
        if f.exists():
            print(f"Loading {f}...")
            results_list.append(load_results(f))
        else:
            print(f"Warning: {f} not found")
    
    if not results_list:
        print("No results found!")
        return
    
    print(f"\nAggregating results from {len(results_list)} seeds...")
    
    # Aggregate
    aggregated = aggregate_method_results(results_list)
    aggregated = compute_efficiency_metrics(aggregated)
    
    # Create final results structure
    final_results = {
        "experiment_info": {
            "dataset": "GSM8K",
            "model": "Qwen/Qwen3-1.7B",
            "num_problems": results_list[0]["config"]["limit"],
            "num_seeds": len(results_list),
            "seeds": [r["config"]["seed"] for r in results_list],
            "thresholds": {
                "tau_h": results_list[0]["config"]["tau_h"],
                "tau_v": results_list[0]["config"]["tau_v"]
            }
        },
        "results": aggregated
    }
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"Dataset: GSM8K")
    print(f"Model: Qwen/Qwen3-1.7B")
    print(f"Problems: {final_results['experiment_info']['num_problems']}")
    print(f"Seeds: {final_results['experiment_info']['seeds']}")
    print(f"Thresholds: tau_h={final_results['experiment_info']['thresholds']['tau_h']}, "
          f"tau_v={final_results['experiment_info']['thresholds']['tau_v']}")
    print("-"*70)
    print(f"{'Method':<20} {'Accuracy':<20} {'Tokens':<15} {'Rev Rate':<15}")
    print("-"*70)
    
    for method, stats in aggregated.items():
        acc_str = f"{stats['accuracy']['mean']:.3f} ± {stats['accuracy']['std']:.3f}"
        tok_str = f"{stats['tokens']['mean']:.1f}"
        rev_str = "N/A"
        if "revision_rate" in stats:
            rev_str = f"{stats['revision_rate']['mean']:.1%}"
        print(f"{method:<20} {acc_str:<20} {tok_str:<15} {rev_str:<15}")
    
    print("="*70)
    
    # Save to results.json
    with open("results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\nResults saved to results.json")
    
    # Also save detailed per-method results
    for method in aggregated.keys():
        method_results = {
            "experiment": method,
            "metrics": {
                "accuracy": aggregated[method]["accuracy"],
                "tokens": aggregated[method]["tokens"]
            },
            "config": final_results["experiment_info"]
        }
        if "revision_rate" in aggregated[method]:
            method_results["metrics"]["revision_rate"] = aggregated[method]["revision_rate"]
        
        output_file = results_dir / f"{method}_aggregated.json"
        with open(output_file, 'w') as f:
            json.dump(method_results, f, indent=2)


if __name__ == "__main__":
    main()

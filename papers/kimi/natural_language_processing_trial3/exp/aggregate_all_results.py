"""
Aggregate all experimental results into final results.json.
Handles multiple seeds, datasets, and methods.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import sys


def load_result_file(path: Path) -> Dict[str, Any]:
    """Load a single result file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def aggregate_method_results(result_files: List[Path]) -> Dict[str, Any]:
    """Aggregate results across seeds for a single method."""
    results = []
    for path in result_files:
        data = load_result_file(path)
        if data and "accuracy" in data:
            results.append(data)
    
    if not results:
        return None
    
    # Extract metrics
    accuracies = [r["accuracy"] for r in results]
    tokens = [r["avg_tokens"] for r in results]
    
    # Build per-seed results
    per_seed = {}
    for r in results:
        seed = str(r.get("seed", "unknown"))
        per_seed[seed] = {
            "accuracy": r["accuracy"],
            "avg_tokens": r["avg_tokens"],
            "correct_count": r.get("correct_count", 0),
            "total_problems": r.get("total_problems", 0)
        }
    
    aggregated = {
        "method": results[0].get("method", "unknown"),
        "dataset": results[0].get("dataset", "unknown"),
        "model": results[0].get("model", "unknown"),
        "n_seeds": len(results),
        "total_problems": results[0].get("total_problems", 0),
        "accuracy": {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)) if len(accuracies) > 1 else 0.0,
            "min": float(np.min(accuracies)),
            "max": float(np.max(accuracies)),
            "values": accuracies
        },
        "avg_tokens": {
            "mean": float(np.mean(tokens)),
            "std": float(np.std(tokens)) if len(tokens) > 1 else 0.0,
            "values": tokens
        },
        "per_seed": per_seed
    }
    
    # Add method-specific metrics
    if "avg_revisions" in results[0]:
        revisions = [r["avg_revisions"] for r in results]
        aggregated["avg_revisions"] = {
            "mean": float(np.mean(revisions)),
            "std": float(np.std(revisions)) if len(revisions) > 1 else 0.0
        }
    
    if "revision_rate" in results[0]:
        revision_rates = [r["revision_rate"] for r in results]
        aggregated["revision_rate"] = {
            "mean": float(np.mean(revision_rates)),
            "std": float(np.std(revision_rates)) if len(revision_rates) > 1 else 0.0
        }
    
    return aggregated


def compute_statistical_significance(baseline_results: List[float], 
                                     method_results: List[float]) -> Dict[str, Any]:
    """Compute statistical significance between two methods."""
    from scipy import stats
    
    if len(baseline_results) < 2 or len(method_results) < 2:
        return {"note": "Insufficient data for statistical test"}
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(method_results, baseline_results)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(baseline_results)**2 + np.std(method_results)**2) / 2)
    cohens_d = (np.mean(method_results) - np.mean(baseline_results)) / pooled_std if pooled_std > 0 else 0
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant_at_0.05": p_value < 0.05,
        "baseline_mean": float(np.mean(baseline_results)),
        "method_mean": float(np.mean(method_results)),
        "difference": float(np.mean(method_results) - np.mean(baseline_results))
    }


def main():
    results_dir = Path("exp/results")
    
    # Find all result files
    result_files = list(results_dir.glob("*_gsm8k_seed*.json")) + list(results_dir.glob("*_math500_seed*.json"))
    
    print(f"Found {len(result_files)} result files")
    
    # Group by method, dataset, and model
    grouped = {}
    for path in result_files:
        # Parse filename: method_dataset_seedN.json
        parts = path.stem.split("_")
        if len(parts) >= 3:
            # Handle method names that might contain underscores
            seed_part = parts[-1]
            dataset = parts[-2]
            method = "_".join(parts[:-2])
            
            key = (method, dataset)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(path)
    
    print(f"Grouped into {len(grouped)} method-dataset combinations")
    
    # Aggregate each group
    aggregated = {}
    for (method, dataset), files in grouped.items():
        print(f"  Aggregating {method} on {dataset} ({len(files)} seeds)...")
        agg = aggregate_method_results(files)
        if agg:
            aggregated[f"{method}_{dataset}"] = agg
    
    # Compute comparisons
    comparisons = {}
    
    # ESR vs Vanilla
    if "esr_gsm8k" in aggregated and "vanilla_gsm8k" in aggregated:
        comparisons["esr_vs_vanilla_gsm8k"] = {
            "accuracy_diff": aggregated["esr_gsm8k"]["accuracy"]["mean"] - aggregated["vanilla_gsm8k"]["accuracy"]["mean"],
            "token_reduction": 1.0 - (aggregated["esr_gsm8k"]["avg_tokens"]["mean"] / aggregated["vanilla_gsm8k"]["avg_tokens"]["mean"])
        }
    
    # ESR vs Entropy-Only
    if "esr_gsm8k" in aggregated and "entropy_only_gsm8k" in aggregated:
        comparisons["esr_vs_entropy_only_gsm8k"] = {
            "accuracy_diff": aggregated["esr_gsm8k"]["accuracy"]["mean"] - aggregated["entropy_only_gsm8k"]["accuracy"]["mean"],
            "varentropy_benefit": aggregated["esr_gsm8k"]["accuracy"]["mean"] > aggregated["entropy_only_gsm8k"]["accuracy"]["mean"]
        }
    
    # Build final output
    output = {
        "experiment_info": {
            "title": "Entropy-Guided Stepwise Revision: In-Chain Self-Correction for Efficient Reasoning",
            "timestamp": str(np.datetime64('now')),
            "note": "Aggregated from multiple experimental runs"
        },
        "methods": aggregated,
        "comparisons": comparisons,
        "success_criteria": {
            "esr_90_percent_of_vanilla": comparisons.get("esr_vs_vanilla_gsm8k", {}).get("accuracy_diff", -1) >= -0.1,
            "esr_beats_entropy_only_by_3_percent": comparisons.get("esr_vs_entropy_only_gsm8k", {}).get("accuracy_diff", -1) >= 0.03
        }
    }
    
    # Save to results.json
    with open("results.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*70)
    print("Aggregation Complete!")
    print("="*70)
    print(f"Results saved to results.json")
    print(f"Methods aggregated: {len(aggregated)}")
    
    # Print summary
    print("\nSummary:")
    for key, data in aggregated.items():
        acc = data["accuracy"]
        tok = data["avg_tokens"]
        print(f"  {key}: Acc={acc['mean']:.3f}±{acc['std']:.3f}, "
              f"Tokens={tok['mean']:.0f}±{tok['std']:.0f} (n={data['n_seeds']})")


if __name__ == "__main__":
    main()

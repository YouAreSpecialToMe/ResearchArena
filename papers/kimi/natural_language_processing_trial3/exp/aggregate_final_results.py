"""
Aggregate final results from all experiments.
"""

import json
import numpy as np
from pathlib import Path
import argparse


def load_results(results_dir):
    """Load all result files from directory."""
    results_dir = Path(results_dir)
    all_results = []
    
    for f in results_dir.glob("*_gsm8k_*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            if isinstance(data, dict) and "method" in data:
                all_results.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    return all_results


def aggregate_by_method(results):
    """Aggregate results by method and dataset."""
    aggregated = {}
    
    for r in results:
        key = (r.get("method"), r.get("dataset"), r.get("seed"))
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(r)
    
    summary = {}
    for (method, dataset, seed), runs in aggregated.items():
        # Use the most recent/largest result
        best_run = max(runs, key=lambda x: x.get("total_problems", 0))
        summary[(method, dataset, seed)] = best_run
    
    return summary


def compute_statistics(summary):
    """Compute statistics across seeds."""
    # Group by method and dataset
    by_method_dataset = {}
    for (method, dataset, seed), result in summary.items():
        key = (method, dataset)
        if key not in by_method_dataset:
            by_method_dataset[key] = []
        by_method_dataset[key].append(result)
    
    stats = {}
    for (method, dataset), results in by_method_dataset.items():
        accuracies = [r["accuracy"] for r in results]
        tokens = [r["avg_tokens"] for r in results]
        
        entry = {
            "method": method,
            "dataset": dataset,
            "n_seeds": len(results),
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies) if len(accuracies) > 1 else 0,
            "tokens_mean": np.mean(tokens),
            "tokens_std": np.std(tokens) if len(tokens) > 1 else 0,
            "total_problems": results[0].get("total_problems", 0),
        }
        
        # Add revision/refinement rates if available
        if "revision_rate" in results[0]:
            rev_rates = [r["revision_rate"] for r in results]
            entry["revision_rate_mean"] = np.mean(rev_rates)
            entry["revision_rate_std"] = np.std(rev_rates) if len(rev_rates) > 1 else 0
        
        if "refinement_rate" in results[0]:
            ref_rates = [r["refinement_rate"] for r in results]
            entry["refinement_rate_mean"] = np.mean(ref_rates)
            entry["refinement_rate_std"] = np.std(ref_rates) if len(ref_rates) > 1 else 0
        
        stats[(method, dataset)] = entry
    
    return stats


def print_summary(stats):
    """Print summary table."""
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    print(f"{'Method':<20} {'Dataset':<10} {'N':<5} {'Accuracy':<20} {'Tokens':<15} {'Revision%':<15}")
    print("-"*100)
    
    for (method, dataset), s in sorted(stats.items()):
        acc_str = f"{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f}" if s['n_seeds'] > 1 else f"{s['accuracy_mean']:.3f}"
        tok_str = f"{s['tokens_mean']:.0f} ± {s['tokens_std']:.0f}" if s['n_seeds'] > 1 else f"{s['tokens_mean']:.0f}"
        
        rev_str = "N/A"
        if "revision_rate_mean" in s:
            rev_str = f"{s['revision_rate_mean']:.1%}"
        elif "refinement_rate_mean" in s:
            rev_str = f"{s['refinement_rate_mean']:.1%}"
        
        print(f"{method:<20} {dataset:<10} {s['n_seeds']:<5} {acc_str:<20} {tok_str:<15} {rev_str:<15}")
    
    print("="*100)


def compare_methods(stats):
    """Compare ESR against baselines."""
    print("\n" + "="*100)
    print("METHOD COMPARISON (ESR vs Baselines)")
    print("="*100)
    
    for dataset in ["gsm8k", "math500"]:
        print(f"\n{dataset.upper()}:")
        print("-"*60)
        
        # Get baseline results
        vanilla_key = ("vanilla", dataset)
        esr_key = ("esr", dataset)
        entropy_key = ("entropy_only", dataset)
        
        if vanilla_key in stats and esr_key in stats:
            vanilla = stats[vanilla_key]
            esr = stats[esr_key]
            
            acc_diff = esr["accuracy_mean"] - vanilla["accuracy_mean"]
            token_diff = esr["tokens_mean"] - vanilla["tokens_mean"]
            
            print(f"  Vanilla:     {vanilla['accuracy_mean']:.3f} accuracy, {vanilla['tokens_mean']:.0f} tokens")
            print(f"  ESR:         {esr['accuracy_mean']:.3f} accuracy, {esr['tokens_mean']:.0f} tokens")
            print(f"  Difference:  {acc_diff:+.3f} accuracy, {token_diff:+.0f} tokens")
            
            if "revision_rate_mean" in esr:
                print(f"  ESR Rev Rate: {esr['revision_rate_mean']:.1%}")
        
        if entropy_key in stats and esr_key in stats:
            entropy = stats[entropy_key]
            esr = stats[esr_key]
            
            acc_diff = esr["accuracy_mean"] - entropy["accuracy_mean"]
            print(f"  Entropy-Only: {entropy['accuracy_mean']:.3f} accuracy")
            print(f"  ESR advantage: {acc_diff:+.3f} over entropy-only")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="exp/results")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} result files")
    
    # Aggregate
    summary = aggregate_by_method(results)
    print(f"Aggregated into {len(summary)} unique configurations")
    
    # Compute statistics
    stats = compute_statistics(summary)
    
    # Print summary
    print_summary(stats)
    compare_methods(stats)
    
    # Save aggregated results
    output = {
        "summary": {f"{k[0]}_{k[1]}": v for k, v in stats.items()},
        "raw_results": list(summary.values())
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nAggregated results saved to: {args.output}")


if __name__ == "__main__":
    main()

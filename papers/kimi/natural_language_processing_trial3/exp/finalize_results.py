#!/usr/bin/env python3
"""
Finalize results after experiments complete.
"""

import json
import numpy as np
from pathlib import Path
import sys

def load_results(results_dir):
    """Load all result files."""
    results_dir = Path(results_dir)
    all_results = []
    
    for pattern in ["*_gsm8k_*.json", "*_math500_*.json"]:
        for f in results_dir.glob(pattern):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                if isinstance(data, dict) and "method" in data:
                    all_results.append(data)
            except Exception as e:
                print(f"Error loading {f}: {e}")
    
    return all_results

def aggregate_results(results):
    """Aggregate results by method, dataset, seed."""
    by_key = {}
    
    for r in results:
        key = (r.get("method"), r.get("dataset"), r.get("seed"))
        if key not in by_key:
            by_key[key] = r
        else:
            # Keep the one with more problems
            if r.get("total_problems", 0) > by_key[key].get("total_problems", 0):
                by_key[key] = r
    
    return by_key

def compute_stats(by_key):
    """Compute statistics across seeds."""
    by_method_dataset = {}
    
    for (method, dataset, seed), result in by_key.items():
        key = (method, dataset)
        if key not in by_method_dataset:
            by_method_dataset[key] = []
        by_method_dataset[key].append(result)
    
    stats = {}
    for (method, dataset), results in by_method_dataset.items():
        accuracies = [r["accuracy"] for r in results]
        tokens = [r.get("avg_tokens", 0) for r in results]
        
        entry = {
            "method": method,
            "dataset": dataset,
            "n_seeds": len(results),
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)) if len(accuracies) > 1 else 0.0,
            "tokens_mean": float(np.mean(tokens)),
            "tokens_std": float(np.std(tokens)) if len(tokens) > 1 else 0.0,
            "total_problems": results[0].get("total_problems", 0),
        }
        
        # Add revision info for ESR
        if method == "esr":
            rev_rates = [r.get("revision_rate", 0) for r in results if "revision_rate" in r]
            if rev_rates:
                entry["revision_rate_mean"] = float(np.mean(rev_rates))
                entry["revision_rate_std"] = float(np.std(rev_rates)) if len(rev_rates) > 1 else 0.0
            
            avg_revs = [r.get("avg_revisions", 0) for r in results if "avg_revisions" in r]
            if avg_revs:
                entry["avg_revisions_mean"] = float(np.mean(avg_revs))
        
        stats[f"{method}_{dataset}"] = entry
    
    return stats

def print_summary(stats):
    """Print formatted summary."""
    print("\n" + "="*100)
    print("FINAL RESULTS SUMMARY")
    print("="*100)
    print(f"{'Method':<18} {'Dataset':<10} {'N':<4} {'Accuracy':<18} {'Tokens':<16} {'Revision%':<12}")
    print("-"*100)
    
    for key in sorted(stats.keys()):
        s = stats[key]
        method = s.get('method', 'unknown') or 'unknown'
        dataset = s.get('dataset', 'unknown') or 'unknown'
        n_seeds = s.get('n_seeds', 0) or 0
        acc_mean = s.get('accuracy_mean', 0) or 0
        acc_std = s.get('accuracy_std', 0) or 0
        tok_mean = s.get('tokens_mean', 0) or 0
        tok_std = s.get('tokens_std', 0) or 0
        
        acc_str = f"{acc_mean:.3f} ± {acc_std:.3f}" if n_seeds > 1 else f"{acc_mean:.3f}"
        tok_str = f"{tok_mean:.0f} ± {tok_std:.0f}" if n_seeds > 1 else f"{tok_mean:.0f}"
        
        rev_str = "N/A"
        if "revision_rate_mean" in s and s["revision_rate_mean"] is not None:
            rev_str = f"{s['revision_rate_mean']:.1%}"
        
        print(f"{method:<18} {dataset:<10} {n_seeds:<4} {acc_str:<18} {tok_str:<16} {rev_str:<12}")
    
    print("="*100)

def compare_esr_to_baselines(stats):
    """Compare ESR to baselines."""
    print("\n" + "="*100)
    print("ESR vs BASELINES COMPARISON")
    print("="*100)
    
    for dataset in ["gsm8k", "math500"]:
        print(f"\n{dataset.upper()}:")
        
        vanilla_key = f"vanilla_{dataset}"
        esr_key = f"esr_{dataset}"
        entropy_key = f"entropy_only_{dataset}"
        
        if vanilla_key in stats and esr_key in stats:
            v = stats[vanilla_key]
            e = stats[esr_key]
            
            acc_diff = e['accuracy_mean'] - v['accuracy_mean']
            token_diff = e['tokens_mean'] - v['tokens_mean']
            
            print(f"  Vanilla:      {v['accuracy_mean']:.3f} accuracy, {v['tokens_mean']:.0f} tokens")
            print(f"  ESR:          {e['accuracy_mean']:.3f} accuracy, {e['tokens_mean']:.0f} tokens")
            print(f"  Improvement:  {acc_diff:+.3f} accuracy, {token_diff:+.0f} tokens")
            
            if "revision_rate_mean" in e:
                print(f"  ESR revision: {e['revision_rate_mean']:.1%} of problems revised")
        
        if entropy_key in stats and esr_key in stats:
            ent = stats[entropy_key]
            esr = stats[esr_key]
            acc_diff = esr['accuracy_mean'] - ent['accuracy_mean']
            print(f"\n  Entropy-Only: {ent['accuracy_mean']:.3f} accuracy")
            print(f"  ESR vs E-Only: {acc_diff:+.3f} accuracy {'(better)' if acc_diff > 0 else '(worse)'}")

def main():
    results_dir = Path("exp/results")
    output_file = Path("results.json")
    
    print("Loading experiment results...")
    results = load_results(results_dir)
    print(f"Found {len(results)} result files")
    
    if not results:
        print("No results found!")
        return
    
    print("\nAggregating by method/dataset/seed...")
    by_key = aggregate_results(results)
    print(f"Found {len(by_key)} unique configurations")
    
    print("\nComputing statistics...")
    stats = compute_stats(by_key)
    
    print_summary(stats)
    compare_esr_to_baselines(stats)
    
    # Save final results
    output = {
        "summary": stats,
        "raw_results": list(by_key.values())
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Check success criteria
    print("\n" + "="*100)
    print("SUCCESS CRITERIA CHECK")
    print("="*100)
    
    esr_gsm8k = stats.get("esr_gsm8k")
    vanilla_gsm8k = stats.get("vanilla_gsm8k")
    
    if esr_gsm8k and vanilla_gsm8k:
        acc_improvement = esr_gsm8k['accuracy_mean'] - vanilla_gsm8k['accuracy_mean']
        if acc_improvement > 0:
            print(f"✓ ESR accuracy ({esr_gsm8k['accuracy_mean']:.3f}) > Vanilla ({vanilla_gsm8k['accuracy_mean']:.3f})")
        else:
            print(f"✗ ESR accuracy not better than vanilla ({acc_improvement:+.3f})")
        
        if "revision_rate_mean" in esr_gsm8k:
            rev_rate = esr_gsm8k['revision_rate_mean']
            if 0.15 <= rev_rate <= 0.40:
                print(f"✓ Revision rate {rev_rate:.1%} in target range (15-40%)")
            elif rev_rate < 0.15:
                print(f"✗ Revision rate {rev_rate:.1%} below target (< 15%)")
            else:
                print(f"✗ Revision rate {rev_rate:.1%} above target (> 40%)")

if __name__ == "__main__":
    main()

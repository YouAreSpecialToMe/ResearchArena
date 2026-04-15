"""
Create final aggregated results.json for the workspace root.
Combines all experimental results into a single comprehensive file.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


def load_all_results():
    """Load all experimental results."""
    results_dir = Path("exp/results")
    
    results_by_method = {}
    
    for json_file in results_dir.glob("*_gsm8k_*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            method = data.get("method")
            seed = data.get("seed")
            dataset = data.get("dataset", "gsm8k")
            
            if not method or seed is None:
                continue
            
            key = method
            if key not in results_by_method:
                results_by_method[key] = {
                    "method": method,
                    "dataset": dataset,
                    "seeds": [],
                    "accuracies": [],
                    "tokens": [],
                    "correct_counts": [],
                    "total_problems": data.get("total_problems", 0),
                    "per_seed_results": {}
                }
            
            results_by_method[key]["seeds"].append(seed)
            results_by_method[key]["accuracies"].append(data.get("accuracy", 0))
            results_by_method[key]["tokens"].append(data.get("avg_tokens", 0))
            results_by_method[key]["correct_counts"].append(data.get("correct_count", 0))
            results_by_method[key]["per_seed_results"][str(seed)] = {
                "accuracy": data.get("accuracy", 0),
                "avg_tokens": data.get("avg_tokens", 0),
                "correct_count": data.get("correct_count", 0),
                "total_problems": data.get("total_problems", 0)
            }
            
            # Add method-specific metrics
            if "revision_rate" in data:
                if "revision_rates" not in results_by_method[key]:
                    results_by_method[key]["revision_rates"] = []
                results_by_method[key]["revision_rates"].append(data["revision_rate"])
            
            if "avg_revisions" in data:
                if "avg_revisions_list" not in results_by_method[key]:
                    results_by_method[key]["avg_revisions_list"] = []
                results_by_method[key]["avg_revisions_list"].append(data["avg_revisions"])
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results_by_method


def compute_aggregates(results_by_method):
    """Compute aggregate statistics for each method."""
    aggregates = {}
    
    for method, data in results_by_method.items():
        accs = data["accuracies"]
        toks = data["tokens"]
        
        if not accs:
            continue
        
        agg = {
            "method": method,
            "dataset": data["dataset"],
            "n_seeds": len(accs),
            "total_problems": data["total_problems"],
            "accuracy": {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "min": float(np.min(accs)),
                "max": float(np.max(accs)),
                "values": accs
            },
            "avg_tokens": {
                "mean": float(np.mean(toks)),
                "std": float(np.std(toks)),
                "values": toks
            },
            "correct_count": {
                "mean": float(np.mean(data["correct_counts"])),
                "std": float(np.std(data["correct_counts"]))
            },
            "per_seed": data["per_seed_results"]
        }
        
        # Add method-specific aggregates
        if "revision_rates" in data:
            agg["revision_rate"] = {
                "mean": float(np.mean(data["revision_rates"])),
                "std": float(np.std(data["revision_rates"]))
            }
        
        if "avg_revisions_list" in data:
            agg["avg_revisions"] = {
                "mean": float(np.mean(data["avg_revisions_list"])),
                "std": float(np.std(data["avg_revisions_list"]))
            }
        
        aggregates[method] = agg
    
    return aggregates


def create_summary_table(aggregates):
    """Create a human-readable summary table."""
    lines = []
    lines.append("="*80)
    lines.append("ESR EXPERIMENT RESULTS - FINAL SUMMARY")
    lines.append("="*80)
    lines.append("")
    lines.append(f"{'Method':<20} {'Accuracy':<20} {'Avg Tokens':<20} {'N Seeds'}")
    lines.append("-"*80)
    
    for method, agg in sorted(aggregates.items()):
        acc_str = f"{agg['accuracy']['mean']:.3f} ± {agg['accuracy']['std']:.3f}"
        tok_str = f"{agg['avg_tokens']['mean']:.1f} ± {agg['avg_tokens']['std']:.1f}"
        lines.append(f"{method:<20} {acc_str:<20} {tok_str:<20} {agg['n_seeds']}")
    
    lines.append("="*80)
    return "\n".join(lines)


def main():
    print("Loading experimental results...")
    results_by_method = load_all_results()
    
    if not results_by_method:
        print("No results found. Please run experiments first.")
        return
    
    print(f"Found results for {len(results_by_method)} methods")
    
    print("\nComputing aggregates...")
    aggregates = compute_aggregates(results_by_method)
    
    # Create final output structure
    final_output = {
        "experiment_info": {
            "title": "Entropy-Guided Stepwise Revision: In-Chain Self-Correction for Efficient Reasoning",
            "dataset": "GSM8K",
            "model": "Qwen/Qwen3-1.7B",
            "timestamp": datetime.now().isoformat(),
            "note": "REAL inference results - NOT synthetic"
        },
        "methods": aggregates,
        "summary": {}
    }
    
    # Add comparison metrics
    if "esr" in aggregates and "vanilla" in aggregates:
        esr_acc = aggregates["esr"]["accuracy"]["mean"]
        van_acc = aggregates["vanilla"]["accuracy"]["mean"]
        final_output["summary"]["esr_vs_vanilla"] = {
            "accuracy_gain": float(esr_acc - van_acc),
            "relative_improvement": float((esr_acc - van_acc) / van_acc * 100) if van_acc > 0 else 0
        }
    
    if "esr" in aggregates and "entropy_only" in aggregates:
        esr_acc = aggregates["esr"]["accuracy"]["mean"]
        eo_acc = aggregates["entropy_only"]["accuracy"]["mean"]
        final_output["summary"]["esr_vs_entropy_only"] = {
            "accuracy_gain": float(esr_acc - eo_acc),
            "varentropy_benefit": float(esr_acc - eo_acc) > 0.03
        }
    
    # Save to results.json (root)
    with open("results.json", 'w') as f:
        json.dump(final_output, f, indent=2)
    print("\nSaved: results.json")
    
    # Save to exp/results/final_results.json
    with open("exp/results/final_results.json", 'w') as f:
        json.dump(final_output, f, indent=2)
    print("Saved: exp/results/final_results.json")
    
    # Print summary
    summary = create_summary_table(aggregates)
    print("\n" + summary)
    
    # Save summary text
    with open("exp/results/summary.txt", 'w') as f:
        f.write(summary)
    print("\nSaved: exp/results/summary.txt")


if __name__ == "__main__":
    main()

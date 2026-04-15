"""
Statistical analysis of ESR experiment results.
Includes paired t-tests, confidence intervals, and success criteria evaluation.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import sys


def load_results(results_dir="exp/results"):
    """Load all experimental results."""
    results_dir = Path(results_dir)
    all_results = {}
    
    for json_file in results_dir.glob("*_gsm8k_*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            method = data.get("method")
            seed = data.get("seed")
            dataset = data.get("dataset", "gsm8k")
            
            if method and seed is not None:
                key = f"{method}_{dataset}_seed{seed}"
                all_results[key] = data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return all_results


def compute_statistics(results_dict):
    """Compute aggregate statistics across seeds."""
    # Group by method and dataset
    grouped = {}
    
    for key, data in results_dict.items():
        method = data.get("method")
        dataset = data.get("dataset", "gsm8k")
        group_key = f"{method}_{dataset}"
        
        if group_key not in grouped:
            grouped[group_key] = {
                "method": method,
                "dataset": dataset,
                "accuracies": [],
                "tokens": [],
                "seeds": []
            }
        
        grouped[group_key]["accuracies"].append(data.get("accuracy", 0))
        grouped[group_key]["tokens"].append(data.get("avg_tokens", 0))
        grouped[group_key]["seeds"].append(data.get("seed"))
    
    # Compute statistics
    stats_dict = {}
    for key, group in grouped.items():
        accs = group["accuracies"]
        toks = group["tokens"]
        
        stats_dict[key] = {
            "method": group["method"],
            "dataset": group["dataset"],
            "n_seeds": len(accs),
            "accuracy_mean": np.mean(accs),
            "accuracy_std": np.std(accs),
            "accuracy_se": np.std(accs) / np.sqrt(len(accs)) if len(accs) > 1 else 0,
            "tokens_mean": np.mean(toks),
            "tokens_std": np.std(toks),
            "seeds": group["seeds"]
        }
        
        # 95% confidence interval
        if len(accs) > 1:
            ci = stats.t.interval(0.95, len(accs)-1, loc=np.mean(accs), scale=stats.sem(accs))
            stats_dict[key]["accuracy_ci_95"] = ci
        else:
            stats_dict[key]["accuracy_ci_95"] = (accs[0], accs[0])
    
    return stats_dict


def paired_t_test(method1_data, method2_data):
    """Perform paired t-test between two methods."""
    # Extract per-problem correctness for paired comparison
    m1_results = method1_data.get("results", [])
    m2_results = method2_data.get("results", [])
    
    if len(m1_results) != len(m2_results):
        return None
    
    m1_correct = [r["correct"] for r in m1_results]
    m2_correct = [r["correct"] for r in m2_results]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(m1_correct, m2_correct)
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01
    }


def evaluate_success_criteria(stats_dict):
    """Evaluate success criteria from the proposal."""
    criteria = {
        "criterion_1": False,  # ESR >= 90% of beam search accuracy with <= 70% tokens
        "criterion_2": False,  # Entropy-varentropy outperforms entropy-only by >= 3%
        "criterion_3": False,  # ESR corrects >= 25% of first-pass incorrect
        "criterion_4": False,  # Revision rate is 15-40%
        "criterion_5": False,  # Harm rate is below 15%
    }
    
    details = {}
    
    # Criterion 1: Compare ESR to Best-of-N (as proxy for beam search)
    esr_key = None
    bon_key = None
    for key in stats_dict:
        if "esr_" in key:
            esr_key = key
        if "bestofn" in key or "best_of_n" in key:
            bon_key = key
    
    if esr_key and bon_key:
        esr_acc = stats_dict[esr_key]["accuracy_mean"]
        bon_acc = stats_dict[bon_key]["accuracy_mean"]
        esr_tok = stats_dict[esr_key]["tokens_mean"]
        bon_tok = stats_dict[bon_key]["tokens_mean"]
        
        acc_ratio = esr_acc / bon_acc if bon_acc > 0 else 0
        tok_ratio = esr_tok / bon_tok if bon_tok > 0 else 1
        
        criteria["criterion_1"] = acc_ratio >= 0.90 and tok_ratio <= 0.70
        details["criterion_1"] = {
            "esr_accuracy": esr_acc,
            "beam_accuracy": bon_acc,
            "accuracy_ratio": acc_ratio,
            "esr_tokens": esr_tok,
            "beam_tokens": bon_tok,
            "token_ratio": tok_ratio
        }
    
    # Criterion 2: Compare ESR to Entropy-Only
    eo_key = None
    for key in stats_dict:
        if "entropy_only" in key:
            eo_key = key
    
    if esr_key and eo_key:
        esr_acc = stats_dict[esr_key]["accuracy_mean"]
        eo_acc = stats_dict[eo_key]["accuracy_mean"]
        diff = esr_acc - eo_acc
        
        criteria["criterion_2"] = diff >= 0.03
        details["criterion_2"] = {
            "esr_accuracy": esr_acc,
            "entropy_only_accuracy": eo_acc,
            "difference": diff
        }
    
    return criteria, details


def generate_summary_table(stats_dict):
    """Generate a summary table of results."""
    print("\n" + "="*90)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*90)
    print(f"{'Method':<20} {'Dataset':<12} {'Accuracy':<20} {'Tokens':<20} {'N'}")
    print("-"*90)
    
    for key in sorted(stats_dict.keys()):
        s = stats_dict[key]
        acc_str = f"{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f}"
        tok_str = f"{s['tokens_mean']:.1f} ± {s['tokens_std']:.1f}"
        print(f"{s['method']:<20} {s['dataset']:<12} {acc_str:<20} {tok_str:<20} {s['n_seeds']}")
    
    print("="*90)


def main():
    print("Loading results...")
    results = load_results()
    print(f"Loaded {len(results)} result files")
    
    if not results:
        print("No results found. Run experiments first.")
        return
    
    print("\nComputing statistics...")
    stats_dict = compute_statistics(results)
    
    # Generate summary table
    generate_summary_table(stats_dict)
    
    # Evaluate success criteria
    print("\n" + "="*90)
    print("SUCCESS CRITERIA EVALUATION")
    print("="*90)
    
    criteria, details = evaluate_success_criteria(stats_dict)
    
    for criterion, met in criteria.items():
        status = "✓ MET" if met else "✗ NOT MET"
        print(f"{criterion}: {status}")
        if criterion in details:
            for k, v in details[criterion].items():
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("="*90)
    
    # Save analysis
    output = {
        "statistics": {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv) 
                          for kk, vv in v.items()} 
                      for k, v in stats_dict.items()},
        "success_criteria": {k: bool(v) for k, v in criteria.items()},
        "criteria_details": details,
        "n_experiments": len(results)
    }
    
    output_path = Path("exp/results/statistical_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nAnalysis saved to: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Experiment 14: Statistical Testing and Success Criteria Verification
- Paired t-test and Wilcoxon test vs baselines
- Bonferroni correction
- Success criteria verification
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 60)
    print("Experiment 14: Statistical Testing and Success Criteria")
    print("=" * 60)
    
    # Load all results
    with open("results/baseline_simple.json") as f:
        baseline_simple = json.load(f)
    with open("results/baseline_exhaustive.json") as f:
        baseline_exhaustive = json.load(f)
    with open("results/baseline_mcts.json") as f:
        baseline_mcts = json.load(f)
    with open("results/leopard_main.json") as f:
        leopard_main = json.load(f)
    with open("results/scorer_analysis.json") as f:
        scorer_analysis = json.load(f)
    with open("results/failure_analysis.json") as f:
        failure_analysis = json.load(f)
    
    # Load test programs
    with open("data/test_programs.pkl", "rb") as f:
        test_programs = pickle.load(f)
    
    test_names = [p.name for p in test_programs]
    
    print("\n[1/3] Paired statistical tests...")
    
    # Collect speedups for test programs
    leopard_speedups = [leopard_main["leopard"][p]["speedup_mean"] for p in test_names]
    exhaustive_speedups = [baseline_exhaustive["exhaustive"][p]["speedup"] for p in test_names]
    mcts_speedups = [baseline_mcts["mcts"][p]["speedup"] for p in test_names]
    random_speedups = [baseline_simple["random_selection"][p]["speedup_mean"] for p in test_names]
    llvm_speedups = [baseline_simple["llvm_o3"][p]["speedup"] for p in test_names]
    
    # Paired t-tests
    def paired_test(a, b, name_a, name_b):
        t_stat, p_val = stats.ttest_rel(a, b)
        w_stat, w_pval = stats.wilcoxon(a, b)
        return {
            "comparison": f"{name_a} vs {name_b}",
            "ttest_statistic": float(t_stat),
            "ttest_pvalue": float(p_val),
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_pvalue": float(w_pval),
            "significant": bool(p_val < 0.05)
        }
    
    statistical_tests = [
        paired_test(leopard_speedups, exhaustive_speedups, "LEOPARD", "Exhaustive"),
        paired_test(leopard_speedups, mcts_speedups, "LEOPARD", "MCTS"),
        paired_test(leopard_speedups, random_speedups, "LEOPARD", "Random"),
        paired_test(leopard_speedups, llvm_speedups, "LEOPARD", "LLVM-O3"),
    ]
    
    # Bonferroni correction
    n_comparisons = len(statistical_tests)
    alpha = 0.05
    corrected_alpha = alpha / n_comparisons
    
    print(f"  Bonferroni correction: α = {alpha} / {n_comparisons} = {corrected_alpha:.4f}")
    
    for test in statistical_tests:
        sig_corrected = test["ttest_pvalue"] < corrected_alpha
        print(f"  {test['comparison']}: p={test['ttest_pvalue']:.4f} "
              f"(significant: {test['significant']}, corrected: {sig_corrected})")
    
    print("\n[2/3] Verifying success criteria...")
    
    # Success criteria from idea.json
    criteria_results = {}
    
    # Criterion 1: LEOPARD achieves ≥80% of the speedup of exhaustive ES with ≤60% of the memory
    exhaustive_geomean = np.exp(np.mean(np.log(exhaustive_speedups)))
    leopard_geomean = np.exp(np.mean(np.log(leopard_speedups)))
    speedup_ratio = leopard_geomean / exhaustive_geomean
    
    leopard_memories = [leopard_main["leopard"][p]["peak_memory_mb_mean"] for p in test_names]
    exhaustive_memories = [baseline_exhaustive["exhaustive"][p]["peak_memory_mb"] for p in test_names]
    memory_ratio = np.mean(leopard_memories) / np.mean(exhaustive_memories)
    
    criteria_results["criterion_1_speedup_memory"] = {
        "description": "Achieve ≥80% of exhaustive speedup with ≤60% memory",
        "speedup_ratio": float(speedup_ratio),
        "memory_ratio": float(memory_ratio),
        "target_speedup_ratio": 0.80,
        "target_memory_ratio": 0.60,
        "passed": bool(speedup_ratio >= 0.80 and memory_ratio <= 0.60),
        "details": f"Speedup: {speedup_ratio*100:.1f}% (target: 80%), Memory: {memory_ratio*100:.1f}% (target: 60%)"
    }
    
    # Criterion 2: Scorer achieves >65% accuracy in predicting beneficial rules
    accuracy = scorer_analysis["accuracy"]["top1_accuracy"]
    random_baseline = scorer_analysis["accuracy"]["random_baseline"]
    
    criteria_results["criterion_2_scorer_accuracy"] = {
        "description": "Scorer achieves >65% accuracy (above random baseline)",
        "accuracy": float(accuracy),
        "random_baseline": float(random_baseline),
        "target_accuracy": 0.65,
        "passed": bool(accuracy > 0.65),
        "details": f"Accuracy: {accuracy*100:.1f}% (target: 65%, random: {random_baseline*100:.1f}%)"
    }
    
    # Criterion 3: Inference overhead < 5% of total compilation time
    overhead_pct = scorer_analysis["timing"]["overhead_pct_of_compilation"]
    
    criteria_results["criterion_3_inference_overhead"] = {
        "description": "Inference overhead < 5% of total compilation time",
        "overhead_pct": float(overhead_pct),
        "target_overhead_pct": 5.0,
        "passed": bool(overhead_pct < 5.0),
        "details": f"Overhead: {overhead_pct:.2f}% (target: <5%)"
    }
    
    # Criterion 4: Graceful degradation (never worse than random)
    graceful_verified = failure_analysis["graceful_degradation_verification"]["verified"]
    
    criteria_results["criterion_4_graceful_degradation"] = {
        "description": "System degrades gracefully (never worse than random)",
        "verified": bool(graceful_verified),
        "passed": bool(graceful_verified),
        "details": f"Verified: {graceful_verified}"
    }
    
    # Overall result
    all_passed = all(c["passed"] for c in criteria_results.values())
    
    for key, result in criteria_results.items():
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"  {status}: {result['description']}")
        print(f"    {result['details']}")
    
    print(f"\n  Overall: {'ALL CRITERIA PASSED' if all_passed else 'SOME CRITERIA FAILED'}")
    
    print("\n[3/3] Compiling evaluation report...")
    
    results = {
        "statistical_tests": {
            "tests": statistical_tests,
            "bonferroni_alpha": corrected_alpha,
            "n_comparisons": n_comparisons
        },
        "success_criteria": criteria_results,
        "overall_result": {
            "all_criteria_passed": bool(all_passed),
            "criteria_passed": sum(1 for c in criteria_results.values() if c["passed"]),
            "criteria_total": len(criteria_results)
        },
        "summary_metrics": {
            "leopard_geomean_speedup": float(leopard_geomean),
            "exhaustive_geomean_speedup": float(exhaustive_geomean),
            "speedup_ratio_vs_exhaustive": float(speedup_ratio),
            "memory_ratio_vs_exhaustive": float(memory_ratio),
            "scorer_accuracy": float(accuracy),
            "inference_overhead_pct": float(overhead_pct)
        }
    }
    
    # Save results
    with open("results/evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("exp/14_statistical_test/results.json", "w") as f:
        json.dump({
            "experiment": "14_statistical_test",
            "status": "completed",
            "all_criteria_passed": bool(all_passed),
            "summary": results["overall_result"]
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Evaluation Summary:")
    print("=" * 60)
    print(f"Criteria passed: {results['overall_result']['criteria_passed']}/"
          f"{results['overall_result']['criteria_total']}")
    print(f"Speedup ratio: {speedup_ratio*100:.1f}% of exhaustive")
    print(f"Memory ratio: {memory_ratio*100:.1f}% of exhaustive")
    print(f"\nResults saved to results/evaluation.json")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())

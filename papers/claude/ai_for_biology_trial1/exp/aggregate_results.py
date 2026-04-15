"""Aggregate results from all experiments and produce summary statistics."""
import os
import sys
import json
import numpy as np
from collections import defaultdict
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "exp", "results")

def load_all_results():
    """Load all result JSON files."""
    results = {}
    for f in sorted(os.listdir(RESULTS_DIR)):
        if not f.endswith("_results.json"):
            continue
        filepath = os.path.join(RESULTS_DIR, f)
        try:
            with open(filepath) as fh:
                results[f] = json.load(fh)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    return results

def aggregate_by_method(all_results):
    """Group results by method, compute mean and std across seeds."""
    methods = defaultdict(list)

    for fname, data in all_results.items():
        # Determine method name from filename
        if "blastp" in fname:
            method = "blastp"
        elif "flat_supcon" in fname:
            method = "flat_supcon"
        elif "joint_hierarchical" in fname:
            method = "joint_hierarchical"
        elif "currec_seed" in fname:
            method = "currec"
        elif "reverse_curriculum" in fname:
            method = "reverse_curriculum"
        elif "random_order" in fname:
            method = "random_order"
        elif "no_consistency" in fname:
            method = "no_consistency"
        elif "lambda_sweep" in fname:
            # Extract lambda value
            lam = fname.split("lambda")[1].split("_")[0]
            method = f"lambda_{lam}"
        elif "no_temp_schedule" in fname:
            method = "no_temp_schedule"
        elif "two_phase" in fname:
            method = "two_phase"
        else:
            method = fname.replace("_results.json", "")

        methods[method].append(data)

    return methods

def compute_statistics(methods):
    """Compute mean and std for each method and metric."""
    stats_dict = {}

    for method, runs in methods.items():
        method_stats = {"n_seeds": len(runs)}

        for benchmark in ["new392", "price149"]:
            benchmark_data = {}
            for metric in ["macro_f1", "micro_f1", "precision", "recall", "accuracy", "f1_rare"]:
                values = []
                for run in runs:
                    bm = run.get(benchmark, {})
                    if metric in bm:
                        values.append(bm[metric])

                if values:
                    benchmark_data[metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)) if len(values) > 1 else 0.0,
                        "values": [float(v) for v in values],
                    }

            method_stats[benchmark] = benchmark_data

        stats_dict[method] = method_stats

    return stats_dict

def run_significance_tests(stats_dict, methods):
    """Run paired t-tests comparing CurrEC vs baselines."""
    tests = {}

    if "currec" not in methods:
        return tests

    currec_runs = methods["currec"]

    for compare_method in ["flat_supcon", "joint_hierarchical", "reverse_curriculum",
                           "random_order", "no_consistency", "no_temp_schedule"]:
        if compare_method not in methods:
            continue

        compare_runs = methods[compare_method]
        test_results = {}

        for benchmark in ["new392", "price149"]:
            currec_f1s = [r.get(benchmark, {}).get("macro_f1", 0) for r in currec_runs]
            compare_f1s = [r.get(benchmark, {}).get("macro_f1", 0) for r in compare_runs]

            n = min(len(currec_f1s), len(compare_f1s))
            if n >= 2:
                t_stat, p_value = stats.ttest_rel(currec_f1s[:n], compare_f1s[:n])
                # Cohen's d
                diff = np.array(currec_f1s[:n]) - np.array(compare_f1s[:n])
                cohens_d = float(np.mean(diff) / (np.std(diff) + 1e-12))
                test_results[benchmark] = {
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "cohens_d": cohens_d,
                    "currec_mean": float(np.mean(currec_f1s)),
                    "compare_mean": float(np.mean(compare_f1s)),
                    "diff_mean": float(np.mean(diff)),
                }

        tests[f"currec_vs_{compare_method}"] = test_results

    return tests

def check_success_criteria(stats_dict):
    """Check whether the success criteria from the plan are met."""
    criteria = {}

    currec = stats_dict.get("currec", {})
    flat = stats_dict.get("flat_supcon", {})
    joint = stats_dict.get("joint_hierarchical", {})
    reverse = stats_dict.get("reverse_curriculum", {})
    random_order = stats_dict.get("random_order", {})
    no_consist = stats_dict.get("no_consistency", {})

    # Criterion 1: CurrEC > Flat SupCon on both benchmarks
    c1_new = (currec.get("new392", {}).get("macro_f1", {}).get("mean", 0) >
              flat.get("new392", {}).get("macro_f1", {}).get("mean", 0))
    c1_price = (currec.get("price149", {}).get("macro_f1", {}).get("mean", 0) >
                flat.get("price149", {}).get("macro_f1", {}).get("mean", 0))
    criteria["currec_beats_flat_supcon"] = {
        "new392": c1_new, "price149": c1_price, "both": c1_new and c1_price
    }

    # Criterion 2: CurrEC > Joint Hierarchical
    c2_new = (currec.get("new392", {}).get("macro_f1", {}).get("mean", 0) >
              joint.get("new392", {}).get("macro_f1", {}).get("mean", 0))
    c2_price = (currec.get("price149", {}).get("macro_f1", {}).get("mean", 0) >
                joint.get("price149", {}).get("macro_f1", {}).get("mean", 0))
    criteria["currec_beats_joint_hierarchical"] = {
        "new392": c2_new, "price149": c2_price, "both": c2_new and c2_price
    }

    # Criterion 3: CurrEC improves F1 on rare classes by >= 5% relative
    currec_rare = currec.get("new392", {}).get("f1_rare", {}).get("mean", 0)
    flat_rare = flat.get("new392", {}).get("f1_rare", {}).get("mean", 0)
    if flat_rare > 0:
        rare_improvement = (currec_rare - flat_rare) / flat_rare * 100
    else:
        rare_improvement = 0
    criteria["rare_class_improvement"] = {
        "currec_f1_rare": currec_rare, "flat_f1_rare": flat_rare,
        "relative_improvement_pct": rare_improvement,
        "meets_5pct_threshold": rare_improvement >= 5
    }

    # Criterion 4: Coarse-to-fine > reverse AND random
    c4_vs_reverse = (currec.get("new392", {}).get("macro_f1", {}).get("mean", 0) >
                     reverse.get("new392", {}).get("macro_f1", {}).get("mean", 0))
    c4_vs_random = (currec.get("new392", {}).get("macro_f1", {}).get("mean", 0) >
                    random_order.get("new392", {}).get("macro_f1", {}).get("mean", 0))
    criteria["curriculum_order_matters"] = {
        "beats_reverse": c4_vs_reverse, "beats_random": c4_vs_random,
        "both": c4_vs_reverse and c4_vs_random
    }

    # Criterion 5: Consistency regularization helps (lambda=0.5 > lambda=0)
    c5 = (currec.get("new392", {}).get("macro_f1", {}).get("mean", 0) >
          no_consist.get("new392", {}).get("macro_f1", {}).get("mean", 0))
    criteria["consistency_helps"] = c5

    return criteria

def main():
    print("Aggregating results...")
    all_results = load_all_results()
    print(f"Loaded {len(all_results)} result files")

    methods = aggregate_by_method(all_results)
    stats_dict = compute_statistics(methods)
    significance = run_significance_tests(stats_dict, methods)
    criteria = check_success_criteria(stats_dict)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Method':<30} {'New-392 F1':>15} {'Price-149 F1':>15} {'Seeds':>6}")
    print("-" * 70)

    for method in ["blastp", "flat_supcon", "joint_hierarchical", "currec",
                    "reverse_curriculum", "random_order", "no_consistency",
                    "no_temp_schedule", "two_phase",
                    "lambda_0.1", "lambda_0.25", "lambda_1.0"]:
        if method not in stats_dict:
            continue
        s = stats_dict[method]
        n392 = s.get("new392", {}).get("macro_f1", {})
        p149 = s.get("price149", {}).get("macro_f1", {})
        n_seeds = s.get("n_seeds", 0)
        n392_str = f"{n392.get('mean', 0):.4f}+/-{n392.get('std', 0):.4f}" if n392 else "N/A"
        p149_str = f"{p149.get('mean', 0):.4f}+/-{p149.get('std', 0):.4f}" if p149 else "N/A"
        print(f"{method:<30} {n392_str:>15} {p149_str:>15} {n_seeds:>6}")

    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA")
    print("=" * 80)
    for k, v in criteria.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("SIGNIFICANCE TESTS")
    print("=" * 80)
    for k, v in significance.items():
        for bm, test in v.items():
            print(f"  {k} ({bm}): diff={test.get('diff_mean', 0):.4f}, "
                  f"t={test.get('t_stat', 0):.3f}, p={test.get('p_value', 1):.4f}, "
                  f"d={test.get('cohens_d', 0):.3f}")

    # Save aggregated results
    aggregated = {
        "statistics": stats_dict,
        "significance_tests": significance,
        "success_criteria": criteria,
        "published_references": {
            "CLEAN": {"new392_f1": 0.502, "price149_f1": 0.438, "note": "Yu et al. 2023, Science"},
            "ProtDETR": {"new392_precision": 0.594, "new392_recall": 0.608, "price149_recall": 0.507,
                        "note": "Yang et al. 2025, ICLR"},
            "MAPred": {"new392_f1": 0.610, "new392_precision": 0.651, "new392_recall": 0.632,
                       "price149_f1": 0.493, "note": "Rong et al. 2025, BriefBioinf"},
        }
    }

    out_path = os.path.join(RESULTS_DIR, "aggregated_results.json")
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    print(f"\nSaved aggregated results to {out_path}")

    return aggregated

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Additional experiments addressing reviewer feedback:
1. Robust scoring approaches (median, thresholded, variance-weighted)
2. Analysis of noise vs. budget relationship
3. Regenerate main_results.csv matching the paper's benchmarks
4. Compute proper cross-benchmark averaged interaction values for Table 2
"""

import os
import sys
import json
import numpy as np
from itertools import combinations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from exp.shared.game import CompilerGame, CANDIDATE_PASSES, count_ir_instructions

BENCHMARK_NAMES = [
    "2mm", "3mm", "adi", "atax", "bicg",
    "cholesky", "correlation", "covariance", "doitgen", "durbin",
    "fdtd-apml", "gemm", "gemver", "gramschmidt", "symm",
]

BC_DIR = os.path.join(PROJECT_ROOT, "data", "polybench_bc")
DATA_DIR = os.path.join(PROJECT_ROOT, "results", "data")
SEEDS = [42, 123, 456]
N_PASSES = len(CANDIDATE_PASSES)
SELECTION_BUDGETS = [5, 8, 10, 12, 15]


def load_shapley_results():
    """Load all Shapley interaction results from saved files."""
    results = {}
    for bm in BENCHMARK_NAMES:
        results[bm] = {}
        for seed in SEEDS:
            fname = os.path.join(DATA_DIR, "interactions", f"{bm}_seed{seed}.json")
            with open(fname) as f:
                results[bm][seed] = json.load(f)
    return results


def compute_phi_mean(seeds_data):
    """Standard mean-averaged phi arrays."""
    phi1 = np.zeros(N_PASSES)
    phi2 = np.zeros((N_PASSES, N_PASSES))
    phi3 = np.zeros((N_PASSES, N_PASSES, N_PASSES))

    for seed in SEEDS:
        data = seeds_data[seed]
        for i, p in enumerate(CANDIDATE_PASSES):
            phi1[i] += data["order_1"][p]
        for key, val in data["order_2"].items():
            p1, p2 = key.split("|")
            i, j = CANDIDATE_PASSES.index(p1), CANDIDATE_PASSES.index(p2)
            phi2[i, j] += val
            phi2[j, i] += val
        for key, val in data["order_3"].items():
            p1, p2, p3 = key.split("|")
            i, j, k = CANDIDATE_PASSES.index(p1), CANDIDATE_PASSES.index(p2), CANDIDATE_PASSES.index(p3)
            for ii, jj, kk in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
                phi3[ii, jj, kk] += val

    phi1 /= len(SEEDS)
    phi2 /= len(SEEDS)
    phi3 /= len(SEEDS)
    return phi1, phi2, phi3


def compute_phi_median(seeds_data):
    """Median-averaged phi arrays (robust to outlier seeds)."""
    phi1_all = np.zeros((len(SEEDS), N_PASSES))
    phi2_all = np.zeros((len(SEEDS), N_PASSES, N_PASSES))
    phi3_all = np.zeros((len(SEEDS), N_PASSES, N_PASSES, N_PASSES))

    for s_idx, seed in enumerate(SEEDS):
        data = seeds_data[seed]
        for i, p in enumerate(CANDIDATE_PASSES):
            phi1_all[s_idx, i] = data["order_1"][p]
        for key, val in data["order_2"].items():
            p1, p2 = key.split("|")
            i, j = CANDIDATE_PASSES.index(p1), CANDIDATE_PASSES.index(p2)
            phi2_all[s_idx, i, j] = val
            phi2_all[s_idx, j, i] = val
        for key, val in data["order_3"].items():
            p1, p2, p3 = key.split("|")
            i, j, k = CANDIDATE_PASSES.index(p1), CANDIDATE_PASSES.index(p2), CANDIDATE_PASSES.index(p3)
            for ii, jj, kk in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
                phi3_all[s_idx, ii, jj, kk] = val

    phi1 = np.median(phi1_all, axis=0)
    phi2 = np.median(phi2_all, axis=0)
    phi3 = np.median(phi3_all, axis=0)
    return phi1, phi2, phi3


def compute_phi_thresholded(seeds_data, threshold_quantile=0.9):
    """Zero out interactions below a threshold (keep only strong signals)."""
    phi1, phi2, phi3 = compute_phi_mean(seeds_data)

    # Zero out weak order-2 interactions
    abs_phi2 = np.abs(phi2)
    thresh2 = np.quantile(abs_phi2[abs_phi2 > 0], threshold_quantile)
    phi2_thresh = np.where(abs_phi2 >= thresh2, phi2, 0.0)

    # Zero out weak order-3 interactions
    abs_phi3 = np.abs(phi3)
    thresh3 = np.quantile(abs_phi3[abs_phi3 > 0], threshold_quantile)
    phi3_thresh = np.where(abs_phi3 >= thresh3, phi3, 0.0)

    return phi1, phi2_thresh, phi3_thresh


def compute_phi_variance_weighted(seeds_data):
    """Downweight high-variance interactions (trust stable estimates more)."""
    phi1_all = np.zeros((len(SEEDS), N_PASSES))
    phi2_all = np.zeros((len(SEEDS), N_PASSES, N_PASSES))
    phi3_all = np.zeros((len(SEEDS), N_PASSES, N_PASSES, N_PASSES))

    for s_idx, seed in enumerate(SEEDS):
        data = seeds_data[seed]
        for i, p in enumerate(CANDIDATE_PASSES):
            phi1_all[s_idx, i] = data["order_1"][p]
        for key, val in data["order_2"].items():
            p1, p2 = key.split("|")
            i, j = CANDIDATE_PASSES.index(p1), CANDIDATE_PASSES.index(p2)
            phi2_all[s_idx, i, j] = val
            phi2_all[s_idx, j, i] = val
        for key, val in data["order_3"].items():
            p1, p2, p3 = key.split("|")
            i, j, k = CANDIDATE_PASSES.index(p1), CANDIDATE_PASSES.index(p2), CANDIDATE_PASSES.index(p3)
            for ii, jj, kk in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
                phi3_all[s_idx, ii, jj, kk] = val

    phi1 = np.mean(phi1_all, axis=0)

    # For order 2 and 3: weight by inverse variance (add small epsilon)
    phi2_mean = np.mean(phi2_all, axis=0)
    phi2_std = np.std(phi2_all, axis=0)
    eps = 1e-6
    # SNR weighting: multiply mean by confidence (mean^2 / (mean^2 + var))
    phi2_var = phi2_std ** 2
    phi2_snr = phi2_mean ** 2 / (phi2_mean ** 2 + phi2_var + eps)
    phi2 = phi2_mean * phi2_snr

    phi3_mean = np.mean(phi3_all, axis=0)
    phi3_std = np.std(phi3_all, axis=0)
    phi3_var = phi3_std ** 2
    phi3_snr = phi3_mean ** 2 / (phi3_mean ** 2 + phi3_var + eps)
    phi3 = phi3_mean * phi3_snr

    return phi1, phi2, phi3


def greedy_selection(phi1, phi2, phi3, k, use_order2=True, use_order3=True):
    """Unified greedy selection with configurable interaction orders."""
    selected = []
    remaining = list(range(N_PASSES))

    for _ in range(k):
        best_score = -np.inf
        best_idx = None
        for i in remaining:
            score = phi1[i]
            if use_order2:
                for j in selected:
                    score += phi2[i, j]
            if use_order3:
                for j_idx in range(len(selected)):
                    for k_idx in range(j_idx + 1, len(selected)):
                        j, kk = selected[j_idx], selected[k_idx]
                        score += phi3[i, j, kk]
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


def evaluate_pass_subset(bm_name, pass_indices):
    """Evaluate a specific subset of passes."""
    bc_path = os.path.join(BC_DIR, f"{bm_name}.bc")
    game = CompilerGame(bc_path)
    vec = np.zeros(N_PASSES)
    for i in pass_indices:
        vec[i] = 1
    return game.value(vec)


def run_robust_scoring_experiment(shapley_results):
    """Compare robust scoring approaches."""
    print("Running robust scoring experiment...")

    methods = {
        "mean_order123": lambda sd: (*compute_phi_mean(sd)[:2], compute_phi_mean(sd)[2], True, True),
        "median_order123": lambda sd: (*compute_phi_median(sd)[:2], compute_phi_median(sd)[2], True, True),
        "thresholded_order123": lambda sd: (*compute_phi_thresholded(sd)[:2], compute_phi_thresholded(sd)[2], True, True),
        "variance_weighted_order123": lambda sd: (*compute_phi_variance_weighted(sd)[:2], compute_phi_variance_weighted(sd)[2], True, True),
        "mean_order12_only": lambda sd: (*compute_phi_mean(sd)[:2], compute_phi_mean(sd)[2], True, False),
        "individual_only": lambda sd: (*compute_phi_mean(sd)[:2], compute_phi_mean(sd)[2], False, False),
    }

    results = {}
    for bm in BENCHMARK_NAMES:
        print(f"  Processing {bm}...")
        results[bm] = {}
        seeds_data = shapley_results[bm]

        for method_name, method_fn in methods.items():
            phi1, phi2, phi3, use_o2, use_o3 = method_fn(seeds_data)
            bm_results = {}
            for k in SELECTION_BUDGETS:
                sel = greedy_selection(phi1, phi2, phi3, k, use_order2=use_o2, use_order3=use_o3)
                val = evaluate_pass_subset(bm, sel)
                bm_results[k] = {
                    "selected": [CANDIDATE_PASSES[i] for i in sel],
                    "reduction": val
                }
            results[bm][method_name] = bm_results

    return results


def compute_cross_benchmark_interactions(shapley_results):
    """Compute properly averaged interaction values across all benchmarks for Table 2."""
    print("Computing cross-benchmark averaged interactions...")

    # For each benchmark, get the mean interactions across seeds
    all_order2 = {}  # key -> list of values across benchmarks
    all_order3 = {}

    for bm in BENCHMARK_NAMES:
        # Average across seeds first
        o2_avg = {}
        o3_avg = {}
        for key in shapley_results[bm][SEEDS[0]]["order_2"]:
            vals = [shapley_results[bm][seed]["order_2"][key] for seed in SEEDS]
            o2_avg[key] = np.mean(vals)
        for key in shapley_results[bm][SEEDS[0]]["order_3"]:
            vals = [shapley_results[bm][seed]["order_3"][key] for seed in SEEDS]
            o3_avg[key] = np.mean(vals)

        for key, val in o2_avg.items():
            if key not in all_order2:
                all_order2[key] = []
            all_order2[key].append(val)

        for key, val in o3_avg.items():
            if key not in all_order3:
                all_order3[key] = []
            all_order3[key].append(val)

    # Now compute cross-benchmark averages and stds
    cross_avg_o2 = {k: {"mean": np.mean(v), "std": np.std(v), "per_bm": dict(zip(BENCHMARK_NAMES, v))}
                    for k, v in all_order2.items()}
    cross_avg_o3 = {k: {"mean": np.mean(v), "std": np.std(v), "per_bm": dict(zip(BENCHMARK_NAMES, v))}
                    for k, v in all_order3.items()}

    # Top synergistic pairs
    top_syn_pairs = sorted(cross_avg_o2.items(), key=lambda x: -x[1]["mean"])[:5]
    top_red_pairs = sorted(cross_avg_o2.items(), key=lambda x: x[1]["mean"])[:5]
    top_syn_triples = sorted(cross_avg_o3.items(), key=lambda x: -x[1]["mean"])[:5]

    result = {
        "top_synergistic_pairs": [
            {"passes": k, "cross_bm_mean": v["mean"], "cross_bm_std": v["std"],
             "2mm_value": v["per_bm"].get("2mm", None)}
            for k, v in top_syn_pairs
        ],
        "top_redundant_pairs": [
            {"passes": k, "cross_bm_mean": v["mean"], "cross_bm_std": v["std"],
             "2mm_value": v["per_bm"].get("2mm", None)}
            for k, v in top_red_pairs
        ],
        "top_synergistic_triples": [
            {"passes": k, "cross_bm_mean": v["mean"], "cross_bm_std": v["std"],
             "2mm_value": v["per_bm"].get("2mm", None)}
            for k, v in top_syn_triples
        ],
    }

    return result


def analyze_budget_vs_noise(shapley_results):
    """Analyze relationship between estimation noise and budget from existing ablation data."""
    print("Analyzing budget vs noise relationship...")

    # Load ablation budget results
    with open(os.path.join(DATA_DIR, "ablation_budget.json")) as f:
        budget_data = json.load(f)

    analysis = {}
    for bm in budget_data.get("summary", {}):
        analysis[bm] = {}
        for budget_str, data in budget_data["summary"][bm].items():
            budget = int(budget_str)
            analysis[bm][budget] = {
                "mean_reduction": data["mean_reduction"],
                "std_reduction": data["std_reduction"],
                "cv": data["std_reduction"] / max(data["mean_reduction"], 1e-10),
            }

    return analysis


def regenerate_main_results_csv(shapley_results):
    """Regenerate main_results.csv to match the paper's 15 benchmarks."""
    print("Regenerating main_results.csv...")

    # Load the full results
    with open(os.path.join(PROJECT_ROOT, "results.json")) as f:
        full_results = json.load(f)

    table = full_results["hypothesis_evaluation"]["main_comparison_table"]

    header = "benchmark,O1,O2,O3,Os,Oz,random_search,genetic_algorithm,individual_greedy,pairwise_greedy,interaction_greedy,synergy_seeded"
    rows = [header]

    for bm in BENCHMARK_NAMES:
        d = table[bm]
        rs_mean = d["random_search"]["mean"] if isinstance(d["random_search"], dict) else d["random_search"]
        rs_std = d["random_search"].get("std", 0) if isinstance(d["random_search"], dict) else 0
        ga_mean = d["genetic_algorithm"]["mean"] if isinstance(d["genetic_algorithm"], dict) else d["genetic_algorithm"]
        ga_std = d["genetic_algorithm"].get("std", 0) if isinstance(d["genetic_algorithm"], dict) else 0

        row = f"{bm},{d['LLVM_O1']:.4f},{d['LLVM_O2']:.4f},{d['LLVM_O3']:.4f},{d['LLVM_Os']:.4f},{d['LLVM_Oz']:.4f}"
        row += f",{rs_mean:.4f}" + (f"+/-{rs_std:.4f}" if rs_std > 0.0001 else "")
        row += f",{ga_mean:.4f}" + (f"+/-{ga_std:.4f}" if ga_std > 0.0001 else "")
        row += f",{d['individual_greedy']:.4f}"
        row += f",{d['pairwise_greedy']:.4f}"
        row += f",{d['interaction_greedy']:.4f}"
        row += f",{d['synergy_seeded']:.4f}"
        rows.append(row)

    csv_path = os.path.join(PROJECT_ROOT, "results", "tables", "main_results.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"  Saved to {csv_path}")


def main():
    shapley_results = load_shapley_results()

    # 1. Cross-benchmark interaction averages (for Table 2 verification)
    cross_bm_interactions = compute_cross_benchmark_interactions(shapley_results)
    with open(os.path.join(DATA_DIR, "cross_benchmark_interactions.json"), "w") as f:
        json.dump(cross_bm_interactions, f, indent=2, default=float)
    print("\nCross-benchmark interaction averages:")
    print("  Top synergistic pairs:")
    for item in cross_bm_interactions["top_synergistic_pairs"]:
        print(f"    {item['passes']}: mean={item['cross_bm_mean']:.4f} +/- {item['cross_bm_std']:.4f} (2mm={item['2mm_value']:.4f})")
    print("  Top redundant pairs:")
    for item in cross_bm_interactions["top_redundant_pairs"]:
        print(f"    {item['passes']}: mean={item['cross_bm_mean']:.4f} +/- {item['cross_bm_std']:.4f} (2mm={item['2mm_value']:.4f})")
    print("  Top synergistic triples:")
    for item in cross_bm_interactions["top_synergistic_triples"]:
        print(f"    {item['passes']}: mean={item['cross_bm_mean']:.4f} +/- {item['cross_bm_std']:.4f} (2mm={item['2mm_value']:.4f})")

    # 2. Robust scoring experiments
    robust_results = run_robust_scoring_experiment(shapley_results)
    with open(os.path.join(DATA_DIR, "robust_scoring_results.json"), "w") as f:
        json.dump(robust_results, f, indent=2, default=float)

    # Summarize at k=10
    print("\nRobust scoring results at k=10:")
    method_avgs = {}
    for method in ["individual_only", "mean_order12_only", "mean_order123",
                   "median_order123", "thresholded_order123", "variance_weighted_order123"]:
        vals = [robust_results[bm][method][10]["reduction"] for bm in BENCHMARK_NAMES
                if 10 in robust_results[bm][method] or "10" in robust_results[bm][method]]
        # Handle string vs int keys
        if not vals:
            vals = [robust_results[bm][method]["10"]["reduction"] for bm in BENCHMARK_NAMES]
        avg = np.mean(vals)
        method_avgs[method] = avg
        print(f"  {method:35s}: {avg:.4f} ({avg*100:.1f}%)")

    # 3. Budget vs noise analysis
    budget_noise = analyze_budget_vs_noise(shapley_results)
    with open(os.path.join(DATA_DIR, "budget_noise_analysis.json"), "w") as f:
        json.dump(budget_noise, f, indent=2, default=float)

    # 4. Regenerate main_results.csv
    regenerate_main_results_csv(shapley_results)

    # 5. Summary statistics for paper
    print("\n=== Summary for paper ===")

    # O3 anomalies
    with open(os.path.join(PROJECT_ROOT, "results.json")) as f:
        full_results = json.load(f)
    table = full_results["hypothesis_evaluation"]["main_comparison_table"]

    print("\nO3 reductions (all benchmarks):")
    o3_vals = []
    for bm in BENCHMARK_NAMES:
        o3 = table[bm]["LLVM_O3"]
        o3_vals.append(o3)
        print(f"  {bm:15s}: {o3*100:.1f}%")
    print(f"  Average (all):     {np.mean(o3_vals)*100:.1f}%")
    o3_no_outliers = [v for bm, v in zip(BENCHMARK_NAMES, o3_vals) if bm not in ["doitgen", "cholesky"]]
    print(f"  Average (excl doitgen+cholesky): {np.mean(o3_no_outliers)*100:.1f}%")

    # Win rates for robust methods vs individual
    print("\nWin rates vs individual greedy at k=10:")
    for method in ["mean_order12_only", "mean_order123", "median_order123",
                   "thresholded_order123", "variance_weighted_order123"]:
        wins = sum(1 for bm in BENCHMARK_NAMES
                   if robust_results[bm][method][10]["reduction"] > robust_results[bm]["individual_only"][10]["reduction"])
        ties = sum(1 for bm in BENCHMARK_NAMES
                   if abs(robust_results[bm][method][10]["reduction"] - robust_results[bm]["individual_only"][10]["reduction"]) < 1e-6)
        print(f"  {method:35s}: {wins}/15 wins, {ties}/15 ties")

    print("\nDone!")


if __name__ == "__main__":
    main()

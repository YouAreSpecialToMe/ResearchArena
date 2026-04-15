#!/usr/bin/env python3
"""
Run remaining steps: baselines (optimized), ablations, structure, transferability,
statistical evaluation, figures, and final results.json aggregation.
"""

import os
import sys
import json
import time
import numpy as np
from itertools import combinations
from multiprocessing import Pool

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from exp.shared.game import CompilerGame, CANDIDATE_PASSES, count_ir_instructions
from exp.run_polybench_full import (
    BENCHMARK_NAMES, BC_DIR, RESULTS_DIR, DATA_DIR, FIGURES_DIR,
    SEEDS, SELECTION_BUDGETS, N_PASSES,
    _compute_phi_from_data,
    greedy_individual_selection, greedy_pairwise_selection,
    greedy_interaction_selection, synergy_seeded_selection,
    evaluate_pass_subset,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Load pre-computed results
def load_data():
    with open(os.path.join(DATA_DIR, "shapley_all.json")) as f:
        shapley = json.load(f)
    for bm in shapley:
        shapley[bm] = {int(k): v for k, v in shapley[bm].items()}

    with open(os.path.join(DATA_DIR, "pass_screening.json")) as f:
        screening = json.load(f)

    with open(os.path.join(DATA_DIR, "variance_decomposition.json")) as f:
        variance_decomp = json.load(f)

    with open(os.path.join(DATA_DIR, "selection_results.json")) as f:
        selection = json.load(f)

    return shapley, screening, variance_decomp, selection


# ============================================================
# Optimized Baselines
# ============================================================

def random_search_for_benchmark(args):
    """Random search with 300 samples per budget level."""
    bm_name, seed = args
    bc_path = os.path.join(BC_DIR, f"{bm_name}.bc")
    game = CompilerGame(bc_path)
    rng = np.random.RandomState(seed)

    results = {}
    n_samples = 300  # Reduced from 1000

    for k in SELECTION_BUDGETS:
        best_val = -np.inf
        best_subset = None
        for _ in range(n_samples):
            indices = rng.choice(N_PASSES, size=k, replace=False)
            vec = np.zeros(N_PASSES)
            vec[indices] = 1
            val = game.value(vec)
            if val > best_val:
                best_val = val
                best_subset = indices.tolist()

        results[k] = {
            "reduction": float(best_val),
            "selected": [CANDIDATE_PASSES[i] for i in best_subset],
            "n_samples": n_samples
        }

    return {"benchmark": bm_name, "seed": seed, "results": results}


def ga_for_benchmark(args):
    """GA baseline: pop=30, gen=15, for all budget levels at once."""
    bm_name, seed = args
    bc_path = os.path.join(BC_DIR, f"{bm_name}.bc")
    game = CompilerGame(bc_path)
    rng = np.random.RandomState(seed)

    all_results = {}
    for target_k in SELECTION_BUDGETS:
        pop_size = 30
        n_generations = 15
        crossover_rate = 0.7
        mutation_rate = 0.1
        tournament_size = 3

        population = []
        for _ in range(pop_size):
            ind = np.zeros(N_PASSES)
            indices = rng.choice(N_PASSES, size=target_k, replace=False)
            ind[indices] = 1
            population.append(ind)

        fitness = [game.value(ind) for ind in population]

        for gen in range(n_generations):
            new_pop = []
            new_fit = []
            for _ in range(pop_size):
                candidates = rng.choice(pop_size, size=tournament_size, replace=False)
                p1 = candidates[np.argmax([fitness[c] for c in candidates])]
                candidates = rng.choice(pop_size, size=tournament_size, replace=False)
                p2 = candidates[np.argmax([fitness[c] for c in candidates])]

                if rng.rand() < crossover_rate:
                    pt = rng.randint(1, N_PASSES)
                    child = np.concatenate([population[p1][:pt], population[p2][pt:]])
                else:
                    child = population[p1].copy()

                for i in range(N_PASSES):
                    if rng.rand() < mutation_rate:
                        child[i] = 1 - child[i]

                # Repair to exactly k passes
                on = np.where(child == 1)[0]
                if len(on) > target_k:
                    child[rng.choice(on, size=len(on) - target_k, replace=False)] = 0
                elif len(on) < target_k:
                    off = np.where(child == 0)[0]
                    child[rng.choice(off, size=target_k - len(on), replace=False)] = 1

                new_pop.append(child)
                new_fit.append(game.value(child))

            # Elitism
            best_old = np.argmax(fitness)
            worst_new = np.argmin(new_fit)
            if fitness[best_old] > new_fit[worst_new]:
                new_pop[worst_new] = population[best_old]
                new_fit[worst_new] = fitness[best_old]

            population = new_pop
            fitness = new_fit

        best_idx = np.argmax(fitness)
        selected = np.where(population[best_idx] == 1)[0].tolist()
        all_results[target_k] = {
            "reduction": float(fitness[best_idx]),
            "selected": [CANDIDATE_PASSES[i] for i in selected],
            "n_evals": pop_size * (n_generations + 1)
        }

    return {"benchmark": bm_name, "seed": seed, "results": all_results}


def run_baselines(screening):
    log("Step 5: Running baselines (optimized)...")

    # Random search
    log("  Random search (300 samples/budget)...")
    rs_tasks = [(bm, seed) for bm in BENCHMARK_NAMES for seed in SEEDS]
    with Pool(2) as pool:
        rs_list = pool.map(random_search_for_benchmark, rs_tasks)

    rs_results = {}
    for r in rs_list:
        bm, seed = r["benchmark"], r["seed"]
        if bm not in rs_results:
            rs_results[bm] = {}
        rs_results[bm][seed] = r["results"]

    log("  Random search done.")

    # GA
    log("  Genetic algorithm (pop=30, gen=15)...")
    ga_tasks = [(bm, seed) for bm in BENCHMARK_NAMES for seed in SEEDS]
    with Pool(2) as pool:
        ga_list = pool.map(ga_for_benchmark, ga_tasks)

    ga_results = {}
    for r in ga_list:
        bm, seed = r["benchmark"], r["seed"]
        if bm not in ga_results:
            ga_results[bm] = {}
        ga_results[bm][seed] = r["results"]

    log("  GA done.")

    # LLVM optimization levels from screening
    opt_level_results = {}
    for bm in BENCHMARK_NAMES:
        opt_level_results[bm] = screening[bm]["opt_levels"]

    baseline_results = {
        "random_search": rs_results,
        "genetic_algorithm": ga_results,
        "opt_levels": opt_level_results,
    }

    with open(os.path.join(DATA_DIR, "baseline_results.json"), "w") as f:
        json.dump(baseline_results, f, indent=2, default=float)

    log("  All baselines saved.")
    return baseline_results


# ============================================================
# Ablation: Interaction Order
# ============================================================

def run_ablation_order(shapley_results):
    log("Step 6a: Ablation - interaction order...")
    results = {}
    for bm in BENCHMARK_NAMES:
        phi1, phi2, phi3 = _compute_phi_from_data(shapley_results[bm])
        results[bm] = {}
        for k in SELECTION_BUDGETS:
            sel1 = greedy_individual_selection(phi1, k)
            val1 = evaluate_pass_subset(bm, sel1)

            sel12 = greedy_pairwise_selection(phi1, phi2, k)
            val12 = evaluate_pass_subset(bm, sel12)

            sel123 = greedy_interaction_selection(phi1, phi2, phi3, k)
            val123 = evaluate_pass_subset(bm, sel123)

            results[bm][k] = {
                "order1": float(val1),
                "order12": float(val12),
                "order123": float(val123),
                "improvement_2_over_1": float(val12 - val1),
                "improvement_3_over_12": float(val123 - val12),
            }

    with open(os.path.join(DATA_DIR, "ablation_order.json"), "w") as f:
        json.dump(results, f, indent=2)

    log("  Done.")
    return results


# ============================================================
# Ablation: Budget Convergence
# ============================================================

def compute_shapley_for_budget(args):
    """Compute Shapley at a specific budget and evaluate selection at k=10."""
    bm_name, seed, budget = args
    bc_path = os.path.join(BC_DIR, f"{bm_name}.bc")
    import shapiq

    game = CompilerGame(bc_path)
    approximator = shapiq.PermutationSamplingSII(
        n=N_PASSES, max_order=3, index="SII", random_state=seed,
    )

    interaction_values = approximator.approximate(budget=budget, game=game)

    phi1 = np.zeros(N_PASSES)
    phi2 = np.zeros((N_PASSES, N_PASSES))
    phi3 = np.zeros((N_PASSES, N_PASSES, N_PASSES))

    for i in range(N_PASSES):
        phi1[i] = float(interaction_values[(i,)])
    for i, j in combinations(range(N_PASSES), 2):
        val = float(interaction_values[(i, j)])
        phi2[i, j] = val
        phi2[j, i] = val
    for i, j, k in combinations(range(N_PASSES), 3):
        val = float(interaction_values[(i, j, k)])
        for ii, jj, kk in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
            phi3[ii, jj, kk] = val

    sel = greedy_interaction_selection(phi1, phi2, phi3, 10)
    reduction = evaluate_pass_subset(bm_name, sel)

    return {
        "benchmark": bm_name, "seed": seed, "budget": budget,
        "reduction_k10": float(reduction),
    }


def run_ablation_budget():
    log("Step 6b: Ablation - budget convergence (500-5000)...")

    repr_benchmarks = ["2mm", "adi", "correlation", "doitgen", "gemver"]
    budgets = [500, 1000, 2000, 3000, 5000]

    tasks = [(bm, seed, budget)
             for bm in repr_benchmarks
             for seed in SEEDS
             for budget in budgets]

    log(f"  {len(tasks)} tasks total")

    with Pool(2) as pool:
        results_list = pool.map(compute_shapley_for_budget, tasks)

    summary = {}
    for bm in repr_benchmarks:
        summary[bm] = {}
        for budget in budgets:
            reds = [r["reduction_k10"] for r in results_list
                    if r["benchmark"] == bm and r["budget"] == budget]
            summary[bm][budget] = {
                "mean_reduction": float(np.mean(reds)),
                "std_reduction": float(np.std(reds)),
                "reductions": [float(x) for x in reds],
            }

    result = {"summary": summary, "budgets": budgets, "benchmarks": repr_benchmarks}
    with open(os.path.join(DATA_DIR, "ablation_budget.json"), "w") as f:
        json.dump(result, f, indent=2)

    log("  Done.")
    return result


# ============================================================
# Interaction Structure
# ============================================================

def analyze_structure(shapley_results):
    log("Step 7: Interaction structure analysis...")
    structure = {}

    for bm in BENCHMARK_NAMES:
        phi1, phi2, phi3 = _compute_phi_from_data(shapley_results[bm])
        bm_s = {}

        pairs = []
        for i, j in combinations(range(N_PASSES), 2):
            pairs.append({
                "passes": f"{CANDIDATE_PASSES[i]}|{CANDIDATE_PASSES[j]}",
                "value": float(phi2[i, j])
            })
        pairs.sort(key=lambda x: -abs(x["value"]))
        bm_s["top_synergistic_pairs"] = [p for p in pairs[:10] if p["value"] > 0]
        bm_s["top_redundant_pairs"] = [p for p in sorted(pairs, key=lambda x: x["value"])[:10] if p["value"] < 0]

        triples = []
        for i, j, k in combinations(range(N_PASSES), 3):
            triples.append({
                "passes": f"{CANDIDATE_PASSES[i]}|{CANDIDATE_PASSES[j]}|{CANDIDATE_PASSES[k]}",
                "value": float(phi3[i, j, k])
            })
        triples.sort(key=lambda x: -abs(x["value"]))
        bm_s["top_synergistic_triples"] = [t for t in triples[:10] if t["value"] > 0]
        bm_s["top_redundant_triples"] = [t for t in sorted(triples, key=lambda x: x["value"])[:10] if t["value"] < 0]

        structure[bm] = bm_s

    # Cross-program stability
    pair_keys = [f"{CANDIDATE_PASSES[i]}|{CANDIDATE_PASSES[j]}" for i, j in combinations(range(N_PASSES), 2)]
    pair_matrix = np.zeros((len(BENCHMARK_NAMES), len(pair_keys)))
    for b_idx, bm in enumerate(BENCHMARK_NAMES):
        phi1, phi2, phi3 = _compute_phi_from_data(shapley_results[bm])
        for p_idx, (i, j) in enumerate(combinations(range(N_PASSES), 2)):
            pair_matrix[b_idx, p_idx] = phi2[i, j]

    universal_pairs = []
    for p_idx, pkey in enumerate(pair_keys):
        col = pair_matrix[:, p_idx]
        pos_frac = np.mean(col > 0)
        neg_frac = np.mean(col < 0)
        if pos_frac >= 0.8 or neg_frac >= 0.8:
            universal_pairs.append({
                "passes": pkey, "mean_value": float(np.mean(col)),
                "std_value": float(np.std(col)),
                "consistent_sign_frac": float(max(pos_frac, neg_frac)),
                "sign": "positive" if pos_frac >= 0.8 else "negative"
            })

    structure["cross_program"] = {
        "universal_pairs": sorted(universal_pairs, key=lambda x: -abs(x["mean_value"]))[:20],
        "n_universal": len(universal_pairs),
        "n_total_pairs": len(pair_keys),
    }

    with open(os.path.join(DATA_DIR, "interaction_structure.json"), "w") as f:
        json.dump(structure, f, indent=2, default=float)

    log(f"  Found {len(universal_pairs)} universal pairs out of {len(pair_keys)}")
    return structure


# ============================================================
# Transferability
# ============================================================

def run_transferability(shapley_results):
    log("Step 8: Transferability analysis...")
    from scipy.spatial.distance import cosine
    from scipy.cluster.hierarchy import linkage, fcluster

    vectors = {}
    for bm in BENCHMARK_NAMES:
        phi1, phi2, phi3 = _compute_phi_from_data(shapley_results[bm])
        v = list(phi1)
        for i, j in combinations(range(N_PASSES), 2):
            v.append(phi2[i, j])
        vectors[bm] = np.array(v)

    n = len(BENCHMARK_NAMES)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v1, v2 = vectors[BENCHMARK_NAMES[i]], vectors[BENCHMARK_NAMES[j]]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            sim_matrix[i, j] = 1 - cosine(v1, v2) if n1 > 0 and n2 > 0 else 0

    dist = 1 - sim_matrix
    np.fill_diagonal(dist, 0)
    condensed = [dist[i, j] for i in range(n) for j in range(i + 1, n)]
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=3, criterion="maxclust")

    transfer_results = {}
    for idx, bm in enumerate(BENCHMARK_NAMES):
        cid = clusters[idx]
        same_cluster = [BENCHMARK_NAMES[i] for i in range(n) if clusters[i] == cid and i != idx]

        phi1_own, phi2_own, phi3_own = _compute_phi_from_data(shapley_results[bm])
        sel_oracle = greedy_interaction_selection(phi1_own, phi2_own, phi3_own, 10)
        val_oracle = evaluate_pass_subset(bm, sel_oracle)

        if same_cluster:
            phi1_t = np.zeros(N_PASSES)
            phi2_t = np.zeros((N_PASSES, N_PASSES))
            phi3_t = np.zeros((N_PASSES, N_PASSES, N_PASSES))
            for other in same_cluster:
                p1, p2, p3 = _compute_phi_from_data(shapley_results[other])
                phi1_t += p1; phi2_t += p2; phi3_t += p3
            phi1_t /= len(same_cluster)
            phi2_t /= len(same_cluster)
            phi3_t /= len(same_cluster)
            sel_t = greedy_interaction_selection(phi1_t, phi2_t, phi3_t, 10)
            val_transfer = evaluate_pass_subset(bm, sel_t)
        else:
            val_transfer = 0

        random_bm = BENCHMARK_NAMES[(idx + 7) % n]
        phi1_r, phi2_r, phi3_r = _compute_phi_from_data(shapley_results[random_bm])
        sel_r = greedy_interaction_selection(phi1_r, phi2_r, phi3_r, 10)
        val_random = evaluate_pass_subset(bm, sel_r)

        transfer_results[bm] = {
            "cluster": int(cid),
            "oracle_reduction": float(val_oracle),
            "transfer_reduction": float(val_transfer),
            "random_reduction": float(val_random),
            "transfer_ratio": float(val_transfer / val_oracle) if val_oracle > 0 else 0,
        }

    results = {
        "similarity_matrix": sim_matrix.tolist(),
        "clusters": {bm: int(c) for bm, c in zip(BENCHMARK_NAMES, clusters)},
        "transfer_results": transfer_results,
        "benchmark_names": BENCHMARK_NAMES,
    }

    with open(os.path.join(DATA_DIR, "transferability.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    success = np.mean([r["transfer_ratio"] >= 0.9 for r in transfer_results.values()])
    log(f"  Transfer success rate (>=90% oracle): {success:.2f}")
    return results


# ============================================================
# Statistical Evaluation
# ============================================================

def run_stat_eval(shapley_results, selection_results, baseline_results, variance_decomp):
    log("Step 9: Statistical evaluation...")
    from scipy import stats

    evaluation = {}

    # Criterion 1: Significant order-3 interactions
    n_sig = 0
    n_total = 0
    for bm in BENCHMARK_NAMES:
        seeds_data = shapley_results[bm]
        order3_keys = list(seeds_data[SEEDS[0]]["order_3"].keys())
        for key in order3_keys:
            vals = [seeds_data[s]["order_3"][key] for s in SEEDS]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            n_total += 1
            if std_val > 0 and abs(mean_val) > 2 * std_val:
                n_sig += 1

    evaluation["criterion1_significant_order3"] = {
        "n_sig": n_sig, "n_total": n_total,
        "frac": n_sig / n_total if n_total > 0 else 0,
        "threshold": 0.30,
        "confirmed": (n_sig / n_total >= 0.30) if n_total > 0 else False,
    }

    # Criterion 2
    frac3_values = [variance_decomp[bm]["frac_order3"] for bm in BENCHMARK_NAMES]
    avg_frac3 = np.mean(frac3_values)
    evaluation["criterion2_variance_order3"] = {
        "avg_frac_order3": float(avg_frac3),
        "std_frac_order3": float(np.std(frac3_values)),
        "per_benchmark": {bm: float(variance_decomp[bm]["frac_order3"]) for bm in BENCHMARK_NAMES},
        "threshold": 0.10,
        "confirmed": avg_frac3 >= 0.10,
    }

    # Criterion 3: win rate at k=10
    wins = 0
    total = 0
    for bm in BENCHMARK_NAMES:
        methods = selection_results[bm]["methods"]
        k_key = 10 if 10 in methods else str(10)
        if k_key in methods:
            v123 = methods[k_key]["interaction_greedy"]["reduction"]
            v12 = methods[k_key]["pairwise_greedy"]["reduction"]
            total += 1
            if v123 > v12:
                wins += 1

    evaluation["criterion3_selection_win_rate"] = {
        "wins": wins, "total": total,
        "win_rate": wins / total if total > 0 else 0,
        "threshold": 0.60,
        "confirmed": (wins / total >= 0.60) if total > 0 else False,
    }

    evaluation["overall_confirmed"] = all([
        evaluation["criterion1_significant_order3"]["confirmed"],
        evaluation["criterion2_variance_order3"]["confirmed"],
        evaluation["criterion3_selection_win_rate"]["confirmed"],
    ])

    # Main comparison table at k=10
    main_table = {}
    for bm in BENCHMARK_NAMES:
        row = {}
        opt = baseline_results["opt_levels"][bm]
        for level in ["O1", "O2", "O3", "Os", "Oz"]:
            if level in opt:
                row[f"LLVM_{level}"] = float(opt[level]["reduction"])

        # Random search
        rs_vals = []
        for seed in SEEDS:
            sd = baseline_results["random_search"][bm]
            seed_key = seed if seed in sd else str(seed)
            if seed_key in sd:
                k_data = sd[seed_key]
                k_key = 10 if 10 in k_data else str(10)
                if k_key in k_data:
                    rs_vals.append(k_data[k_key]["reduction"])
        row["random_search"] = {"mean": float(np.mean(rs_vals)), "std": float(np.std(rs_vals))} if rs_vals else {}

        # GA
        ga_vals = []
        for seed in SEEDS:
            sd = baseline_results["genetic_algorithm"][bm]
            seed_key = seed if seed in sd else str(seed)
            if seed_key in sd:
                k_data = sd[seed_key]
                k_key = 10 if 10 in k_data else str(10)
                if k_key in k_data:
                    ga_vals.append(k_data[k_key]["reduction"])
        row["genetic_algorithm"] = {"mean": float(np.mean(ga_vals)), "std": float(np.std(ga_vals))} if ga_vals else {}

        # Selection methods
        methods = selection_results[bm]["methods"]
        k_key = 10 if 10 in methods else str(10)
        for method in ["individual_greedy", "pairwise_greedy", "interaction_greedy", "synergy_seeded"]:
            row[method] = float(methods[k_key][method]["reduction"])

        main_table[bm] = row

    evaluation["main_comparison_table"] = main_table

    # Wilcoxon tests
    method_pairs = [
        ("interaction_greedy", "pairwise_greedy"),
        ("interaction_greedy", "individual_greedy"),
        ("pairwise_greedy", "individual_greedy"),
        ("interaction_greedy", "synergy_seeded"),
    ]
    stat_tests = {}
    for m1, m2 in method_pairs:
        diffs = [main_table[bm].get(m1, 0) - main_table[bm].get(m2, 0) for bm in BENCHMARK_NAMES]
        diffs = np.array(diffs)
        if np.any(diffs != 0):
            try:
                stat, pval = stats.wilcoxon(diffs)
                stat_tests[f"{m1}_vs_{m2}"] = {
                    "statistic": float(stat), "p_value": float(pval),
                    "mean_diff": float(np.mean(diffs)),
                }
            except Exception as e:
                stat_tests[f"{m1}_vs_{m2}"] = {"error": str(e)}
        else:
            stat_tests[f"{m1}_vs_{m2}"] = {"note": "all zero"}
    evaluation["statistical_tests"] = stat_tests

    with open(os.path.join(DATA_DIR, "statistical_evaluation.json"), "w") as f:
        json.dump(evaluation, f, indent=2, default=float)

    log(f"  C1 (sig order-3): {evaluation['criterion1_significant_order3']['frac']:.3f} (need >=0.30)")
    log(f"  C2 (var order-3): {avg_frac3:.3f} (need >=0.10)")
    log(f"  C3 (win rate): {evaluation['criterion3_selection_win_rate']['win_rate']:.2f} (need >=0.60)")
    return evaluation


# ============================================================
# Figures
# ============================================================

def generate_all_figures(variance_decomp, selection_results, baseline_results,
                         shapley_results, ablation_order, ablation_budget,
                         transferability, structure):
    log("Step 10: Generating figures...")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 14, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.dpi": 300,
    })

    # Fig 1: Variance decomposition
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(BENCHMARK_NAMES))
    f1 = [variance_decomp[bm]["frac_order1"] for bm in BENCHMARK_NAMES]
    f2 = [variance_decomp[bm]["frac_order2"] for bm in BENCHMARK_NAMES]
    f3 = [variance_decomp[bm]["frac_order3"] for bm in BENCHMARK_NAMES]
    ax.bar(x, f1, label="Order 1 (individual)", color="#4C72B0")
    ax.bar(x, f2, bottom=f1, label="Order 2 (pairwise)", color="#DD8452")
    ax.bar(x, f3, bottom=[a + b for a, b in zip(f1, f2)], label="Order 3 (triple)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([bm[:8] for bm in BENCHMARK_NAMES], rotation=45, ha="right")
    ax.set_ylabel("Fraction of Interaction Variance")
    ax.set_title("Variance Decomposition by Interaction Order (PolyBench)")
    ax.legend()
    fig.savefig(os.path.join(FIGURES_DIR, "variance_decomposition.png"))
    fig.savefig(os.path.join(FIGURES_DIR, "variance_decomposition.pdf"))
    plt.close(fig)

    # Fig 2: Interaction heatmaps
    for bm in ["2mm", "correlation", "gemver"]:
        phi1, phi2, phi3 = _compute_phi_from_data(shapley_results[bm])
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.eye(N_PASSES, dtype=bool)
        short = [p[:8] for p in CANDIDATE_PASSES]
        sns.heatmap(phi2, xticklabels=short, yticklabels=short,
                    cmap="RdBu_r", center=0, ax=ax, mask=mask, square=True)
        ax.set_title(f"Pairwise Shapley Interactions: {bm}")
        fig.savefig(os.path.join(FIGURES_DIR, f"heatmap_{bm}.png"))
        fig.savefig(os.path.join(FIGURES_DIR, f"heatmap_{bm}.pdf"))
        plt.close(fig)

    # Fig 3: Selection comparison at k=10
    fig, ax = plt.subplots(figsize=(14, 6))
    methods = ["LLVM_O3", "random_search", "ga", "individual_greedy", "pairwise_greedy", "interaction_greedy", "synergy_seeded"]
    labels = ["-O3", "Random", "GA", "Individual", "Pairwise", "Interaction (ours)", "Synergy"]
    colors = ["#666666", "#888888", "#AAAAAA", "#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    x = np.arange(len(BENCHMARK_NAMES))
    width = 0.12
    for m_idx, (method, label, color) in enumerate(zip(methods, labels, colors)):
        vals = []
        errs = []
        for bm in BENCHMARK_NAMES:
            opt = baseline_results["opt_levels"][bm]
            methods_sel = selection_results[bm]["methods"]
            k_key = 10 if 10 in methods_sel else str(10)

            if method == "LLVM_O3":
                vals.append(opt.get("O3", {}).get("reduction", 0))
                errs.append(0)
            elif method == "random_search":
                rs_vals = []
                for seed in SEEDS:
                    sd = baseline_results["random_search"][bm]
                    sk = seed if seed in sd else str(seed)
                    if sk in sd:
                        kk = 10 if 10 in sd[sk] else str(10)
                        if kk in sd[sk]:
                            rs_vals.append(sd[sk][kk]["reduction"])
                vals.append(np.mean(rs_vals) if rs_vals else 0)
                errs.append(np.std(rs_vals) if rs_vals else 0)
            elif method == "ga":
                ga_vals = []
                for seed in SEEDS:
                    sd = baseline_results["genetic_algorithm"][bm]
                    sk = seed if seed in sd else str(seed)
                    if sk in sd:
                        kk = 10 if 10 in sd[sk] else str(10)
                        if kk in sd[sk]:
                            ga_vals.append(sd[sk][kk]["reduction"])
                vals.append(np.mean(ga_vals) if ga_vals else 0)
                errs.append(np.std(ga_vals) if ga_vals else 0)
            else:
                vals.append(methods_sel[k_key][method]["reduction"])
                errs.append(0)

        offset = (m_idx - len(methods) / 2) * width
        ax.bar(x + offset, vals, width, yerr=errs, label=label, color=color, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels([bm[:8] for bm in BENCHMARK_NAMES], rotation=45, ha="right")
    ax.set_ylabel("IR Instruction Count Reduction")
    ax.set_title("Pass Selection Performance (k=10 passes, PolyBench)")
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    fig.savefig(os.path.join(FIGURES_DIR, "selection_comparison.png"))
    fig.savefig(os.path.join(FIGURES_DIR, "selection_comparison.pdf"))
    plt.close(fig)

    # Fig 4: Performance vs budget k
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, label, color in [
        ("individual_greedy", "Individual", "#4C72B0"),
        ("pairwise_greedy", "Pairwise", "#55A868"),
        ("interaction_greedy", "Interaction (ours)", "#C44E52"),
        ("synergy_seeded", "Synergy-seeded", "#8172B2"),
    ]:
        means, stds = [], []
        for k in SELECTION_BUDGETS:
            vals = []
            for bm in BENCHMARK_NAMES:
                kk = k if k in selection_results[bm]["methods"] else str(k)
                vals.append(selection_results[bm]["methods"][kk][method]["reduction"])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.errorbar(SELECTION_BUDGETS, means, yerr=stds, label=label, color=color,
                     marker="o", capsize=3)

    # Also add random search and GA means
    for method_name, label, color, marker in [
        ("random_search", "Random Search", "#888888", "s"),
        ("genetic_algorithm", "GA", "#AAAAAA", "^"),
    ]:
        means, stds = [], []
        for k in SELECTION_BUDGETS:
            vals = []
            for bm in BENCHMARK_NAMES:
                for seed in SEEDS:
                    sd = baseline_results[method_name][bm]
                    sk = seed if seed in sd else str(seed)
                    if sk in sd:
                        kk = k if k in sd[sk] else str(k)
                        if kk in sd[sk]:
                            vals.append(sd[sk][kk]["reduction"])
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        ax.errorbar(SELECTION_BUDGETS, means, yerr=stds, label=label, color=color,
                     marker=marker, capsize=3, linestyle="--")

    ax.set_xlabel("Number of Selected Passes (k)")
    ax.set_ylabel("Mean IR Reduction")
    ax.set_title("Selection Performance vs Pass Budget (PolyBench)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIGURES_DIR, "performance_vs_budget.png"))
    fig.savefig(os.path.join(FIGURES_DIR, "performance_vs_budget.pdf"))
    plt.close(fig)

    # Fig 5: Convergence
    if ablation_budget and "summary" in ablation_budget:
        fig, ax = plt.subplots(figsize=(8, 5))
        budgets = ablation_budget["budgets"]
        for bm in ablation_budget["benchmarks"]:
            bm_data = ablation_budget["summary"][bm]
            means = [bm_data[str(b) if str(b) in bm_data else b]["mean_reduction"] for b in budgets]
            stds = [bm_data[str(b) if str(b) in bm_data else b]["std_reduction"] for b in budgets]
            ax.errorbar(budgets, means, yerr=stds, label=bm[:8], marker="o", capsize=3)
        ax.set_xlabel("Shapley Evaluation Budget")
        ax.set_ylabel("Selection Performance (k=10)")
        ax.set_title("Convergence of Shapley Estimation")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(FIGURES_DIR, "convergence.png"))
        fig.savefig(os.path.join(FIGURES_DIR, "convergence.pdf"))
        plt.close(fig)

    # Fig 6: Transferability heatmap
    if transferability and "similarity_matrix" in transferability:
        fig, ax = plt.subplots(figsize=(10, 8))
        sim = np.array(transferability["similarity_matrix"])
        short = [bm[:8] for bm in BENCHMARK_NAMES]
        sns.heatmap(sim, xticklabels=short, yticklabels=short,
                    cmap="YlOrRd", vmin=0, vmax=1, ax=ax, square=True)
        ax.set_title("Benchmark Similarity (Interaction Vector Cosine)")
        fig.savefig(os.path.join(FIGURES_DIR, "transferability.png"))
        fig.savefig(os.path.join(FIGURES_DIR, "transferability.pdf"))
        plt.close(fig)

    # Fig 7: Ablation order
    if ablation_order:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        improvements_2, improvements_3 = [], []
        for bm in BENCHMARK_NAMES:
            for k in SELECTION_BUDGETS:
                kk = k if k in ablation_order[bm] else str(k)
                if kk in ablation_order[bm]:
                    improvements_2.append(ablation_order[bm][kk]["improvement_2_over_1"])
                    improvements_3.append(ablation_order[bm][kk]["improvement_3_over_12"])

        axes[0].hist(improvements_2, bins=20, alpha=0.7, label="Order 2 over 1", color="#DD8452")
        axes[0].hist(improvements_3, bins=20, alpha=0.7, label="Order 3 over 1+2", color="#C44E52")
        axes[0].set_xlabel("Improvement in IR Reduction")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distribution of Improvements")
        axes[0].legend()
        axes[0].axvline(0, color="black", linestyle="--", alpha=0.5)

        for label, key, color in [
            ("Pairwise over Individual", "improvement_2_over_1", "#DD8452"),
            ("Triple over Pairwise", "improvement_3_over_12", "#C44E52"),
        ]:
            means, stds = [], []
            for k in SELECTION_BUDGETS:
                vals = []
                for bm in BENCHMARK_NAMES:
                    kk = k if k in ablation_order[bm] else str(k)
                    if kk in ablation_order[bm]:
                        vals.append(ablation_order[bm][kk][key])
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            axes[1].errorbar(SELECTION_BUDGETS, means, yerr=stds, label=label,
                            color=color, marker="o", capsize=3)
        axes[1].set_xlabel("Number of Selected Passes (k)")
        axes[1].set_ylabel("Mean Improvement")
        axes[1].set_title("Improvement by Interaction Order vs Budget")
        axes[1].legend()
        axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "ablation_order.png"))
        fig.savefig(os.path.join(FIGURES_DIR, "ablation_order.pdf"))
        plt.close(fig)

    log("  All figures generated.")


# ============================================================
# Final Aggregation
# ============================================================

def aggregate_final(screening, shapley_results, variance_decomp, selection_results,
                    baseline_results, ablation_order, ablation_budget,
                    transferability, evaluation, structure):
    log("Step 11: Final aggregation...")

    # Build comprehensive results
    main_table = {}
    for bm in BENCHMARK_NAMES:
        row = {}
        opt = baseline_results["opt_levels"][bm]
        for level in ["O1", "O2", "O3", "Os", "Oz"]:
            if level in opt:
                row[f"LLVM_{level}"] = float(opt[level]["reduction"])

        # Random search with error bars per k
        rs_by_k = {}
        for k in SELECTION_BUDGETS:
            vals = []
            for seed in SEEDS:
                sd = baseline_results["random_search"][bm]
                sk = seed if seed in sd else str(seed)
                if sk in sd:
                    kk = k if k in sd[sk] else str(k)
                    if kk in sd[sk]:
                        vals.append(sd[sk][kk]["reduction"])
            if vals:
                rs_by_k[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        row["random_search"] = rs_by_k

        # GA with error bars per k
        ga_by_k = {}
        for k in SELECTION_BUDGETS:
            vals = []
            for seed in SEEDS:
                sd = baseline_results["genetic_algorithm"][bm]
                sk = seed if seed in sd else str(seed)
                if sk in sd:
                    kk = k if k in sd[sk] else str(k)
                    if kk in sd[sk]:
                        vals.append(sd[sk][kk]["reduction"])
            if vals:
                ga_by_k[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        row["genetic_algorithm"] = ga_by_k

        # Selection methods per k
        sel_by_method = {}
        for method in ["individual_greedy", "pairwise_greedy", "interaction_greedy", "synergy_seeded"]:
            method_by_k = {}
            for k in SELECTION_BUDGETS:
                kk = k if k in selection_results[bm]["methods"] else str(k)
                method_by_k[k] = float(selection_results[bm]["methods"][kk][method]["reduction"])
            sel_by_method[method] = method_by_k
        row["selection_methods"] = sel_by_method

        main_table[bm] = row

    # Overall means
    overall = {}
    for method in ["individual_greedy", "pairwise_greedy", "interaction_greedy", "synergy_seeded"]:
        for k in SELECTION_BUDGETS:
            vals = [main_table[bm]["selection_methods"][method][k] for bm in BENCHMARK_NAMES]
            overall[f"{method}_k{k}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    for k in SELECTION_BUDGETS:
        rs_m = [main_table[bm]["random_search"][k]["mean"] for bm in BENCHMARK_NAMES if k in main_table[bm]["random_search"]]
        ga_m = [main_table[bm]["genetic_algorithm"][k]["mean"] for bm in BENCHMARK_NAMES if k in main_table[bm]["genetic_algorithm"]]
        if rs_m: overall[f"random_search_k{k}"] = {"mean": float(np.mean(rs_m)), "std": float(np.std(rs_m))}
        if ga_m: overall[f"genetic_algorithm_k{k}"] = {"mean": float(np.mean(ga_m)), "std": float(np.std(ga_m))}

    results = {
        "experiment": "ShapleyPass: Compiler Pass Interaction Analysis via Shapley Interaction Indices",
        "benchmarks": BENCHMARK_NAMES,
        "benchmark_suite": "PolyBench",
        "n_benchmarks": len(BENCHMARK_NAMES),
        "n_passes": N_PASSES,
        "passes": list(CANDIDATE_PASSES),
        "seeds": SEEDS,
        "shapley_budget": 2000,
        "selection_budgets": SELECTION_BUDGETS,
        "hypothesis_evaluation": evaluation,
        "variance_decomposition": variance_decomp,
        "main_comparison_table": main_table,
        "overall_means": overall,
        "ablation_order": ablation_order,
        "ablation_budget": ablation_budget.get("summary", {}) if ablation_budget else {},
        "transferability_summary": {
            bm: transferability["transfer_results"][bm] for bm in BENCHMARK_NAMES
        } if transferability else {},
        "interaction_structure_summary": {
            "n_universal_pairs": structure["cross_program"]["n_universal"],
            "top_universal_pairs": structure["cross_program"]["universal_pairs"][:10],
        } if structure else {},
    }

    with open(os.path.join(PROJECT_ROOT, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    log("  results.json written.")
    return results


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    log("=" * 60)
    log("Running remaining experiment steps (5-11)")
    log("=" * 60)

    shapley_results, screening, variance_decomp, selection_results = load_data()

    baseline_results = run_baselines(screening)
    ablation_order = run_ablation_order(shapley_results)
    structure = analyze_structure(shapley_results)
    transferability = run_transferability(shapley_results)

    # Budget convergence ablation (takes ~60 min)
    ablation_budget = run_ablation_budget()

    evaluation = run_stat_eval(shapley_results, selection_results, baseline_results, variance_decomp)

    generate_all_figures(variance_decomp, selection_results, baseline_results,
                         shapley_results, ablation_order, ablation_budget,
                         transferability, structure)

    aggregate_final(screening, shapley_results, variance_decomp, selection_results,
                    baseline_results, ablation_order, ablation_budget,
                    transferability, evaluation, structure)

    elapsed = time.time() - t0
    log(f"\nTotal time for remaining steps: {elapsed/60:.1f} min")
    log("All done!")


if __name__ == "__main__":
    main()

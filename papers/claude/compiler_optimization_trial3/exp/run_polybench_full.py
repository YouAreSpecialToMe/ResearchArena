#!/usr/bin/env python3
"""
ShapleyPass: Full experiment pipeline on PolyBench benchmarks.
Computes Shapley interaction indices, runs pass selection, baselines, and ablations.
"""

import os
import sys
import json
import time
import numpy as np
from itertools import combinations
from multiprocessing import Pool, cpu_count
from functools import partial

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from exp.shared.game import (
    CompilerGame, CANDIDATE_PASSES, count_ir_instructions,
    get_optimization_level_counts
)

# ============================================================
# Configuration
# ============================================================

# Select 15 diverse PolyBench benchmarks (varied sizes, domains)
BENCHMARK_NAMES = [
    "2mm", "3mm", "adi", "atax", "bicg",
    "cholesky", "correlation", "covariance", "doitgen", "durbin",
    "fdtd-apml", "gemm", "gemver", "gramschmidt", "symm",
]

BC_DIR = os.path.join(PROJECT_ROOT, "data", "polybench_bc")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

SEEDS = [42, 123, 456]
SHAPLEY_BUDGET = 2000  # evaluations per benchmark per seed
SELECTION_BUDGETS = [5, 8, 10, 12, 15]  # number of passes to select
N_PASSES = len(CANDIDATE_PASSES)  # 20

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "interactions"), exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# Step 1: Pass Screening
# ============================================================

def screen_passes_for_benchmark(bm_name):
    """Screen each pass individually for a benchmark."""
    bc_path = os.path.join(BC_DIR, f"{bm_name}.bc")
    game = CompilerGame(bc_path)
    baseline = game.baseline_count

    results = {"benchmark": bm_name, "baseline_count": baseline, "individual_reductions": {}}

    for i, pass_name in enumerate(CANDIDATE_PASSES):
        vec = np.zeros(N_PASSES)
        vec[i] = 1
        red = game.value(vec)
        results["individual_reductions"][pass_name] = red

    # Also get O-level baselines
    opt_levels = get_optimization_level_counts(bc_path)
    results["opt_levels"] = {}
    for level, count in opt_levels.items():
        if count is not None and baseline > 0:
            results["opt_levels"][level] = {
                "count": count,
                "reduction": (baseline - count) / baseline
            }

    return results


def run_pass_screening():
    log("Step 1: Screening passes individually on all benchmarks...")
    with Pool(2) as pool:
        all_results = pool.map(screen_passes_for_benchmark, BENCHMARK_NAMES)

    screening = {r["benchmark"]: r for r in all_results}

    # Rank passes by average marginal contribution
    avg_contributions = {}
    for p in CANDIDATE_PASSES:
        vals = [screening[bm]["individual_reductions"][p] for bm in BENCHMARK_NAMES]
        avg_contributions[p] = np.mean(vals)

    ranked = sorted(avg_contributions.items(), key=lambda x: -x[1])
    log(f"  Top 5 passes: {[(p, f'{v:.4f}') for p, v in ranked[:5]]}")
    log(f"  Bottom 5 passes: {[(p, f'{v:.4f}') for p, v in ranked[-5:]]}")

    screening["pass_ranking"] = ranked
    with open(os.path.join(DATA_DIR, "pass_screening.json"), "w") as f:
        json.dump(screening, f, indent=2, default=str)

    return screening


# ============================================================
# Step 2: Compute Shapley Interaction Indices
# ============================================================

def compute_shapley_for_benchmark_seed(args):
    """Compute Shapley interaction indices for one benchmark + one seed."""
    bm_name, seed = args
    bc_path = os.path.join(BC_DIR, f"{bm_name}.bc")

    import shapiq

    game = CompilerGame(bc_path)
    np.random.seed(seed)

    # Use PermutationSamplingSII for efficiency
    approximator = shapiq.PermutationSamplingSII(
        n=N_PASSES,
        max_order=3,
        index="SII",
        random_state=seed,
    )

    t0 = time.time()
    interaction_values = approximator.approximate(budget=SHAPLEY_BUDGET, game=game)
    elapsed = time.time() - t0

    # Extract values at each order
    result = {
        "benchmark": bm_name,
        "seed": seed,
        "n_evals": SHAPLEY_BUDGET,
        "time_seconds": elapsed,
        "n_cache_entries": len(game.cache),
        "order_1": {},
        "order_2": {},
        "order_3": {},
    }

    # Order 1: individual Shapley values
    for i in range(N_PASSES):
        key = (i,)
        val = float(interaction_values[key])
        result["order_1"][CANDIDATE_PASSES[i]] = val

    # Order 2: pairwise interactions
    for i, j in combinations(range(N_PASSES), 2):
        key = (i, j)
        val = float(interaction_values[key])
        result["order_2"][f"{CANDIDATE_PASSES[i]}|{CANDIDATE_PASSES[j]}"] = val

    # Order 3: triple interactions
    for i, j, k in combinations(range(N_PASSES), 3):
        key = (i, j, k)
        val = float(interaction_values[key])
        result["order_3"][f"{CANDIDATE_PASSES[i]}|{CANDIDATE_PASSES[j]}|{CANDIDATE_PASSES[k]}"] = val

    return result


def run_shapley_computation():
    log("Step 2: Computing Shapley interaction indices (orders 1-3)...")

    tasks = [(bm, seed) for bm in BENCHMARK_NAMES for seed in SEEDS]
    log(f"  Total tasks: {len(tasks)} ({len(BENCHMARK_NAMES)} benchmarks x {len(SEEDS)} seeds)")

    all_results = {}
    # Process sequentially but with caching to be safe with multiprocessing/shapiq
    # Actually use Pool(2) since each task is independent
    with Pool(2) as pool:
        results = pool.map(compute_shapley_for_benchmark_seed, tasks)

    for r in results:
        bm = r["benchmark"]
        seed = r["seed"]
        if bm not in all_results:
            all_results[bm] = {}
        all_results[bm][seed] = r

        # Save individual result
        fname = os.path.join(DATA_DIR, "interactions", f"{bm}_seed{seed}.json")
        with open(fname, "w") as f:
            json.dump(r, f, indent=2)

    log(f"  Done. Avg time per task: {np.mean([r['time_seconds'] for r in results]):.1f}s")

    return all_results


# ============================================================
# Step 3: Variance Decomposition
# ============================================================

def run_variance_decomposition(shapley_results):
    log("Step 3: Variance decomposition analysis...")

    decomposition = {}
    for bm in BENCHMARK_NAMES:
        seeds_data = shapley_results[bm]
        order1_vals, order2_vals, order3_vals = [], [], []

        for seed in SEEDS:
            data = seeds_data[seed]
            o1 = np.array(list(data["order_1"].values()))
            o2 = np.array(list(data["order_2"].values()))
            o3 = np.array(list(data["order_3"].values()))
            order1_vals.append(np.sum(o1 ** 2))
            order2_vals.append(np.sum(o2 ** 2))
            order3_vals.append(np.sum(o3 ** 2))

        ss1 = np.mean(order1_vals)
        ss2 = np.mean(order2_vals)
        ss3 = np.mean(order3_vals)
        total = ss1 + ss2 + ss3

        decomposition[bm] = {
            "ss_order1": ss1, "ss_order2": ss2, "ss_order3": ss3,
            "frac_order1": ss1 / total if total > 0 else 0,
            "frac_order2": ss2 / total if total > 0 else 0,
            "frac_order3": ss3 / total if total > 0 else 0,
            "frac_order1_std": np.std([v / (v + o2 + o3) for v, o2, o3 in zip(order1_vals, order2_vals, order3_vals)]) if total > 0 else 0,
            "frac_order2_std": np.std([o2 / (o1 + o2 + o3) for o1, o2, o3 in zip(order1_vals, order2_vals, order3_vals)]) if total > 0 else 0,
            "frac_order3_std": np.std([o3 / (o1 + o2 + o3) for o1, o2, o3 in zip(order1_vals, order2_vals, order3_vals)]) if total > 0 else 0,
        }

    # Average across benchmarks
    avg_frac1 = np.mean([d["frac_order1"] for d in decomposition.values()])
    avg_frac2 = np.mean([d["frac_order2"] for d in decomposition.values()])
    avg_frac3 = np.mean([d["frac_order3"] for d in decomposition.values()])

    decomposition["average"] = {
        "frac_order1": avg_frac1, "frac_order2": avg_frac2, "frac_order3": avg_frac3,
        "frac_order1_std": np.std([d["frac_order1"] for d in decomposition.values() if isinstance(d, dict) and "frac_order1" in d]),
        "frac_order2_std": np.std([d["frac_order2"] for d in decomposition.values() if isinstance(d, dict) and "frac_order2" in d]),
        "frac_order3_std": np.std([d["frac_order3"] for d in decomposition.values() if isinstance(d, dict) and "frac_order3" in d]),
    }

    log(f"  Avg variance: order1={avg_frac1:.3f}, order2={avg_frac2:.3f}, order3={avg_frac3:.3f}")

    with open(os.path.join(DATA_DIR, "variance_decomposition.json"), "w") as f:
        json.dump(decomposition, f, indent=2, default=float)

    return decomposition


# ============================================================
# Step 4: Pass Selection Algorithms
# ============================================================

def get_mean_interactions(shapley_results, bm):
    """Average interaction values across seeds for a benchmark."""
    seeds_data = shapley_results[bm]

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


def greedy_individual_selection(phi1, k):
    """Select top-k passes by individual Shapley value."""
    return list(np.argsort(-phi1)[:k])


def greedy_pairwise_selection(phi1, phi2, k):
    """Greedy selection using order-1 + order-2 interactions."""
    selected = []
    remaining = list(range(N_PASSES))

    for _ in range(k):
        best_score = -np.inf
        best_idx = None
        for i in remaining:
            score = phi1[i]
            for j in selected:
                score += phi2[i, j]
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


def greedy_interaction_selection(phi1, phi2, phi3, k):
    """Greedy selection using order-1 + order-2 + order-3 interactions."""
    selected = []
    remaining = list(range(N_PASSES))

    for _ in range(k):
        best_score = -np.inf
        best_idx = None
        for i in remaining:
            score = phi1[i]
            for j in selected:
                score += phi2[i, j]
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


def synergy_seeded_selection(phi1, phi2, phi3, k):
    """Seed with top synergistic triple, then greedy with full interactions."""
    if k < 3:
        return greedy_interaction_selection(phi1, phi2, phi3, k)

    # Find top synergistic triple
    best_triple = None
    best_val = -np.inf
    for i, j, kk in combinations(range(N_PASSES), 3):
        val = phi3[i, j, kk]
        if val > best_val:
            best_val = val
            best_triple = [i, j, kk]

    selected = list(best_triple)
    remaining = [i for i in range(N_PASSES) if i not in selected]

    for _ in range(k - 3):
        best_score = -np.inf
        best_idx = None
        for i in remaining:
            score = phi1[i]
            for j in selected:
                score += phi2[i, j]
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
    """Evaluate a specific subset of passes on a benchmark."""
    bc_path = os.path.join(BC_DIR, f"{bm_name}.bc")
    game = CompilerGame(bc_path)
    vec = np.zeros(N_PASSES)
    for i in pass_indices:
        vec[i] = 1
    return game.value(vec)


def run_selection_for_benchmark(args):
    """Run all selection methods for one benchmark."""
    bm_name, shapley_bm_data = args
    phi1, phi2, phi3 = _compute_phi_from_data(shapley_bm_data)

    results = {"benchmark": bm_name, "methods": {}}

    for k in SELECTION_BUDGETS:
        results["methods"][k] = {}

        # Individual greedy
        sel = greedy_individual_selection(phi1, k)
        val = evaluate_pass_subset(bm_name, sel)
        results["methods"][k]["individual_greedy"] = {
            "selected": [CANDIDATE_PASSES[i] for i in sel], "reduction": val
        }

        # Pairwise greedy
        sel = greedy_pairwise_selection(phi1, phi2, k)
        val = evaluate_pass_subset(bm_name, sel)
        results["methods"][k]["pairwise_greedy"] = {
            "selected": [CANDIDATE_PASSES[i] for i in sel], "reduction": val
        }

        # Full interaction greedy (proposed)
        sel = greedy_interaction_selection(phi1, phi2, phi3, k)
        val = evaluate_pass_subset(bm_name, sel)
        results["methods"][k]["interaction_greedy"] = {
            "selected": [CANDIDATE_PASSES[i] for i in sel], "reduction": val
        }

        # Synergy-seeded
        sel = synergy_seeded_selection(phi1, phi2, phi3, k)
        val = evaluate_pass_subset(bm_name, sel)
        results["methods"][k]["synergy_seeded"] = {
            "selected": [CANDIDATE_PASSES[i] for i in sel], "reduction": val
        }

    return results


def _compute_phi_from_data(seeds_data):
    """Compute averaged phi arrays from seeds data dict."""
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


def run_pass_selection(shapley_results):
    log("Step 4: Running pass selection algorithms...")

    all_selection_results = {}
    for bm in BENCHMARK_NAMES:
        log(f"  Selection for {bm}...")
        result = run_selection_for_benchmark((bm, shapley_results[bm]))
        all_selection_results[bm] = result

    with open(os.path.join(DATA_DIR, "selection_results.json"), "w") as f:
        json.dump(all_selection_results, f, indent=2, default=float)

    return all_selection_results


# ============================================================
# Step 5: Baselines (Random Search + GA)
# ============================================================

def random_search_for_benchmark(args):
    """Random search baseline for one benchmark."""
    bm_name, seed = args
    bc_path = os.path.join(BC_DIR, f"{bm_name}.bc")
    game = CompilerGame(bc_path)
    rng = np.random.RandomState(seed)

    results = {}
    n_samples = 1000

    for k in SELECTION_BUDGETS:
        best_val = -np.inf
        best_subset = None
        for _ in range(n_samples):
            # Random subset of exactly k passes
            indices = rng.choice(N_PASSES, size=k, replace=False)
            vec = np.zeros(N_PASSES)
            vec[indices] = 1
            val = game.value(vec)
            if val > best_val:
                best_val = val
                best_subset = indices.tolist()

        results[k] = {
            "reduction": best_val,
            "selected": [CANDIDATE_PASSES[i] for i in best_subset],
            "n_samples": n_samples
        }

    return {"benchmark": bm_name, "seed": seed, "results": results}


def ga_for_benchmark(args):
    """Genetic algorithm baseline for one benchmark."""
    bm_name, seed, target_k = args
    bc_path = os.path.join(BC_DIR, f"{bm_name}.bc")
    game = CompilerGame(bc_path)
    rng = np.random.RandomState(seed)

    pop_size = 50
    n_generations = 20
    crossover_rate = 0.7
    mutation_rate = 0.1
    tournament_size = 3

    # Initialize population: each individual is a binary vector with exactly target_k ones
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
            # Tournament selection
            candidates = rng.choice(pop_size, size=tournament_size, replace=False)
            parent1_idx = candidates[np.argmax([fitness[c] for c in candidates])]
            candidates = rng.choice(pop_size, size=tournament_size, replace=False)
            parent2_idx = candidates[np.argmax([fitness[c] for c in candidates])]

            # Crossover
            if rng.rand() < crossover_rate:
                crossover_point = rng.randint(1, N_PASSES)
                child = np.concatenate([population[parent1_idx][:crossover_point],
                                       population[parent2_idx][crossover_point:]])
            else:
                child = population[parent1_idx].copy()

            # Mutation
            for i in range(N_PASSES):
                if rng.rand() < mutation_rate:
                    child[i] = 1 - child[i]

            # Enforce exactly k passes (repair)
            on_indices = np.where(child == 1)[0]
            if len(on_indices) > target_k:
                to_remove = rng.choice(on_indices, size=len(on_indices) - target_k, replace=False)
                child[to_remove] = 0
            elif len(on_indices) < target_k:
                off_indices = np.where(child == 0)[0]
                to_add = rng.choice(off_indices, size=target_k - len(on_indices), replace=False)
                child[to_add] = 1

            new_pop.append(child)
            new_fit.append(game.value(child))

        # Elitism: keep best from old generation
        best_old = np.argmax(fitness)
        worst_new = np.argmin(new_fit)
        if fitness[best_old] > new_fit[worst_new]:
            new_pop[worst_new] = population[best_old]
            new_fit[worst_new] = fitness[best_old]

        population = new_pop
        fitness = new_fit

    best_idx = np.argmax(fitness)
    best_ind = population[best_idx]
    best_val = fitness[best_idx]
    selected = np.where(best_ind == 1)[0].tolist()

    return {
        "benchmark": bm_name, "seed": seed, "target_k": target_k,
        "reduction": best_val,
        "selected": [CANDIDATE_PASSES[i] for i in selected],
        "n_evals": pop_size * (n_generations + 1)
    }


def run_baselines(screening_results):
    log("Step 5: Running baselines...")

    # Random search
    log("  Running random search...")
    rs_tasks = [(bm, seed) for bm in BENCHMARK_NAMES for seed in SEEDS]
    with Pool(2) as pool:
        rs_results_list = pool.map(random_search_for_benchmark, rs_tasks)

    rs_results = {}
    for r in rs_results_list:
        bm = r["benchmark"]
        seed = r["seed"]
        if bm not in rs_results:
            rs_results[bm] = {}
        rs_results[bm][seed] = r["results"]

    # GA
    log("  Running genetic algorithm...")
    ga_tasks = [(bm, seed, k) for bm in BENCHMARK_NAMES for seed in SEEDS for k in SELECTION_BUDGETS]
    with Pool(2) as pool:
        ga_results_list = pool.map(ga_for_benchmark, ga_tasks)

    ga_results = {}
    for r in ga_results_list:
        bm = r["benchmark"]
        seed = r["seed"]
        k = r["target_k"]
        if bm not in ga_results:
            ga_results[bm] = {}
        if seed not in ga_results[bm]:
            ga_results[bm][seed] = {}
        ga_results[bm][seed][k] = {
            "reduction": r["reduction"],
            "selected": r["selected"],
            "n_evals": r["n_evals"]
        }

    # LLVM optimization levels from screening
    opt_level_results = {}
    for bm in BENCHMARK_NAMES:
        opt_level_results[bm] = screening_results[bm]["opt_levels"]

    baseline_results = {
        "random_search": rs_results,
        "genetic_algorithm": ga_results,
        "opt_levels": opt_level_results
    }

    with open(os.path.join(DATA_DIR, "baseline_results.json"), "w") as f:
        json.dump(baseline_results, f, indent=2, default=float)

    log("  Baselines complete.")
    return baseline_results


# ============================================================
# Step 6: Ablation Studies
# ============================================================

def ablation_interaction_order(shapley_results):
    """Ablation: compare order-1, order-1+2, order-1+2+3 selection."""
    log("Step 6a: Ablation - interaction order...")

    results = {}
    for bm in BENCHMARK_NAMES:
        phi1, phi2, phi3 = _compute_phi_from_data(shapley_results[bm])
        results[bm] = {}

        for k in SELECTION_BUDGETS:
            # Order 1 only
            sel1 = greedy_individual_selection(phi1, k)
            val1 = evaluate_pass_subset(bm, sel1)

            # Order 1+2
            sel12 = greedy_pairwise_selection(phi1, phi2, k)
            val12 = evaluate_pass_subset(bm, sel12)

            # Order 1+2+3
            sel123 = greedy_interaction_selection(phi1, phi2, phi3, k)
            val123 = evaluate_pass_subset(bm, sel123)

            results[bm][k] = {
                "order1": val1,
                "order12": val12,
                "order123": val123,
                "improvement_2_over_1": val12 - val1,
                "improvement_3_over_12": val123 - val12,
            }

    with open(os.path.join(DATA_DIR, "ablation_order.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    return results


def compute_shapley_single_budget(args):
    """Compute Shapley for a specific budget (for convergence ablation)."""
    bm_name, seed, budget = args
    bc_path = os.path.join(BC_DIR, f"{bm_name}.bc")

    import shapiq

    game = CompilerGame(bc_path)
    np.random.seed(seed)

    approximator = shapiq.PermutationSamplingSII(
        n=N_PASSES, max_order=3, index="SII", random_state=seed,
    )

    interaction_values = approximator.approximate(budget=budget, game=game)

    # Extract order-1 values for selection
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

    # Evaluate selection at k=10
    sel = greedy_interaction_selection(phi1, phi2, phi3, 10)
    reduction = evaluate_pass_subset(bm_name, sel)

    return {
        "benchmark": bm_name, "seed": seed, "budget": budget,
        "reduction_k10": reduction,
        "phi1_norm": float(np.linalg.norm(phi1)),
    }


def ablation_budget_convergence(shapley_results):
    """Ablation: vary Shapley evaluation budget."""
    log("Step 6b: Ablation - budget convergence (500-5000)...")

    # Use 5 representative benchmarks
    repr_benchmarks = ["2mm", "adi", "correlation", "doitgen", "gemver"]
    budgets = [500, 1000, 2000, 3000, 5000]

    tasks = [(bm, seed, budget)
             for bm in repr_benchmarks
             for seed in SEEDS
             for budget in budgets]

    with Pool(2) as pool:
        results_list = pool.map(compute_shapley_single_budget, tasks)

    # Organize results
    results = {}
    for r in results_list:
        bm = r["benchmark"]
        budget = r["budget"]
        seed = r["seed"]
        if bm not in results:
            results[bm] = {}
        if budget not in results[bm]:
            results[bm][budget] = []
        results[bm][budget].append(r)

    # Compute mean/std per benchmark per budget
    summary = {}
    for bm in repr_benchmarks:
        summary[bm] = {}
        for budget in budgets:
            reds = [r["reduction_k10"] for r in results[bm][budget]]
            summary[bm][budget] = {
                "mean_reduction": float(np.mean(reds)),
                "std_reduction": float(np.std(reds)),
                "reductions": [float(x) for x in reds],
            }

    conv_results = {"detailed": results, "summary": summary, "budgets": budgets, "benchmarks": repr_benchmarks}
    with open(os.path.join(DATA_DIR, "ablation_budget.json"), "w") as f:
        json.dump(conv_results, f, indent=2, default=float)

    log("  Budget convergence ablation complete.")
    return conv_results


# ============================================================
# Step 7: Interaction Structure Analysis
# ============================================================

def analyze_interaction_structure(shapley_results):
    log("Step 7: Analyzing interaction structure...")

    structure = {}
    for bm in BENCHMARK_NAMES:
        phi1, phi2, phi3 = _compute_phi_from_data(shapley_results[bm])
        bm_structure = {}

        # Top pairwise interactions
        pairs = []
        for i, j in combinations(range(N_PASSES), 2):
            pairs.append({
                "passes": f"{CANDIDATE_PASSES[i]}|{CANDIDATE_PASSES[j]}",
                "value": float(phi2[i, j])
            })
        pairs.sort(key=lambda x: -abs(x["value"]))
        bm_structure["top_synergistic_pairs"] = [p for p in pairs[:10] if p["value"] > 0]
        bm_structure["top_redundant_pairs"] = [p for p in sorted(pairs, key=lambda x: x["value"])[:10] if p["value"] < 0]

        # Top triples
        triples = []
        for i, j, k in combinations(range(N_PASSES), 3):
            triples.append({
                "passes": f"{CANDIDATE_PASSES[i]}|{CANDIDATE_PASSES[j]}|{CANDIDATE_PASSES[k]}",
                "value": float(phi3[i, j, k])
            })
        triples.sort(key=lambda x: -abs(x["value"]))
        bm_structure["top_synergistic_triples"] = [t for t in triples[:10] if t["value"] > 0]
        bm_structure["top_redundant_triples"] = [t for t in sorted(triples, key=lambda x: x["value"])[:10] if t["value"] < 0]

        structure[bm] = bm_structure

    # Cross-program stability
    log("  Computing cross-program stability...")
    pair_keys = [f"{CANDIDATE_PASSES[i]}|{CANDIDATE_PASSES[j]}" for i, j in combinations(range(N_PASSES), 2)]
    pair_matrix = np.zeros((len(BENCHMARK_NAMES), len(pair_keys)))
    for b_idx, bm in enumerate(BENCHMARK_NAMES):
        phi1, phi2, phi3 = _compute_phi_from_data(shapley_results[bm])
        for p_idx, (i, j) in enumerate(combinations(range(N_PASSES), 2)):
            pair_matrix[b_idx, p_idx] = phi2[i, j]

    # Find universal interactions (consistent sign across >80% of benchmarks)
    n_bm = len(BENCHMARK_NAMES)
    universal_pairs = []
    for p_idx, pkey in enumerate(pair_keys):
        col = pair_matrix[:, p_idx]
        pos_frac = np.mean(col > 0)
        neg_frac = np.mean(col < 0)
        if pos_frac >= 0.8 or neg_frac >= 0.8:
            universal_pairs.append({
                "passes": pkey,
                "mean_value": float(np.mean(col)),
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

    log(f"  Found {len(universal_pairs)} universal pair interactions out of {len(pair_keys)}")
    return structure


# ============================================================
# Step 8: Transferability Analysis
# ============================================================

def run_transferability(shapley_results):
    log("Step 8: Transferability analysis...")

    from scipy.spatial.distance import cosine
    from scipy.cluster.hierarchy import linkage, fcluster

    # Build interaction vectors for each benchmark
    vectors = {}
    for bm in BENCHMARK_NAMES:
        phi1, phi2, phi3 = _compute_phi_from_data(shapley_results[bm])
        # Flatten into a single vector
        v = list(phi1)
        for i, j in combinations(range(N_PASSES), 2):
            v.append(phi2[i, j])
        # Skip order-3 for transferability (too many dimensions, noisy)
        vectors[bm] = np.array(v)

    # Cosine similarity matrix
    n = len(BENCHMARK_NAMES)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v1 = vectors[BENCHMARK_NAMES[i]]
            v2 = vectors[BENCHMARK_NAMES[j]]
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                sim_matrix[i, j] = 1 - cosine(v1, v2)
            else:
                sim_matrix[i, j] = 0

    # Hierarchical clustering
    dist_matrix = 1 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist_matrix[i, j])
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=3, criterion="maxclust")

    # Leave-one-out transfer
    transfer_results = {}
    for idx, bm in enumerate(BENCHMARK_NAMES):
        cluster_id = clusters[idx]
        same_cluster = [BENCHMARK_NAMES[i] for i in range(n) if clusters[i] == cluster_id and i != idx]

        # Oracle: use own interactions
        phi1_own, phi2_own, phi3_own = _compute_phi_from_data(shapley_results[bm])
        sel_oracle = greedy_interaction_selection(phi1_own, phi2_own, phi3_own, 10)
        val_oracle = evaluate_pass_subset(bm, sel_oracle)

        # Transfer: average interactions from same cluster
        if same_cluster:
            phi1_t = np.zeros(N_PASSES)
            phi2_t = np.zeros((N_PASSES, N_PASSES))
            phi3_t = np.zeros((N_PASSES, N_PASSES, N_PASSES))
            for other in same_cluster:
                p1, p2, p3 = _compute_phi_from_data(shapley_results[other])
                phi1_t += p1
                phi2_t += p2
                phi3_t += p3
            phi1_t /= len(same_cluster)
            phi2_t /= len(same_cluster)
            phi3_t /= len(same_cluster)
            sel_transfer = greedy_interaction_selection(phi1_t, phi2_t, phi3_t, 10)
            val_transfer = evaluate_pass_subset(bm, sel_transfer)
        else:
            val_transfer = 0

        # Random control: use a random other benchmark
        random_bm = BENCHMARK_NAMES[(idx + 7) % n]  # deterministic "random"
        phi1_r, phi2_r, phi3_r = _compute_phi_from_data(shapley_results[random_bm])
        sel_random = greedy_interaction_selection(phi1_r, phi2_r, phi3_r, 10)
        val_random = evaluate_pass_subset(bm, sel_random)

        transfer_results[bm] = {
            "cluster": int(cluster_id),
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

    success_rate = np.mean([r["transfer_ratio"] >= 0.9 for r in transfer_results.values()])
    log(f"  Transfer success rate (>=90% of oracle): {success_rate:.2f}")

    return results


# ============================================================
# Step 9: Statistical Evaluation
# ============================================================

def run_statistical_evaluation(shapley_results, selection_results, baseline_results, variance_decomp):
    log("Step 9: Statistical evaluation against success criteria...")
    from scipy import stats

    evaluation = {}

    # Criterion 1: Significant order-3 interactions
    n_sig = 0
    n_total = 0
    for bm in BENCHMARK_NAMES:
        seeds_data = shapley_results[bm]
        # Collect order-3 values across seeds
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

    # Criterion 2: Variance explained by order-3
    frac3_values = [variance_decomp[bm]["frac_order3"] for bm in BENCHMARK_NAMES]
    avg_frac3 = np.mean(frac3_values)
    evaluation["criterion2_variance_order3"] = {
        "avg_frac_order3": float(avg_frac3),
        "std_frac_order3": float(np.std(frac3_values)),
        "per_benchmark": {bm: float(variance_decomp[bm]["frac_order3"]) for bm in BENCHMARK_NAMES},
        "threshold": 0.10,
        "confirmed": avg_frac3 >= 0.10,
    }

    # Criterion 3: Interaction-guided vs pairwise-only win rate (at k=10)
    wins = 0
    total = 0
    for bm in BENCHMARK_NAMES:
        if "10" in selection_results[bm]["methods"] or 10 in selection_results[bm]["methods"]:
            k_key = 10 if 10 in selection_results[bm]["methods"] else "10"
            methods = selection_results[bm]["methods"][k_key]
            val_123 = methods["interaction_greedy"]["reduction"]
            val_12 = methods["pairwise_greedy"]["reduction"]
            total += 1
            if val_123 > val_12:
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
        # LLVM levels
        opt_levels = baseline_results["opt_levels"][bm]
        for level in ["O1", "O2", "O3", "Os", "Oz"]:
            if level in opt_levels:
                row[f"LLVM_{level}"] = opt_levels[level]["reduction"]

        # Random search (mean ± std across seeds at k=10)
        rs_vals = []
        for seed in SEEDS:
            if str(10) in baseline_results["random_search"][bm][seed]:
                rs_vals.append(baseline_results["random_search"][bm][seed][str(10)]["reduction"])
            elif 10 in baseline_results["random_search"][bm][seed]:
                rs_vals.append(baseline_results["random_search"][bm][seed][10]["reduction"])
        if rs_vals:
            row["random_search_mean"] = float(np.mean(rs_vals))
            row["random_search_std"] = float(np.std(rs_vals))

        # GA (mean ± std across seeds at k=10)
        ga_vals = []
        for seed in SEEDS:
            seed_data = baseline_results["genetic_algorithm"][bm][seed]
            if str(10) in seed_data:
                ga_vals.append(seed_data[str(10)]["reduction"])
            elif 10 in seed_data:
                ga_vals.append(seed_data[10]["reduction"])
        if ga_vals:
            row["ga_mean"] = float(np.mean(ga_vals))
            row["ga_std"] = float(np.std(ga_vals))

        # Selection methods at k=10
        k_key = 10 if 10 in selection_results[bm]["methods"] else "10"
        for method in ["individual_greedy", "pairwise_greedy", "interaction_greedy", "synergy_seeded"]:
            row[method] = selection_results[bm]["methods"][k_key][method]["reduction"]

        main_table[bm] = row

    evaluation["main_comparison_table"] = main_table

    # Wilcoxon signed-rank tests
    method_pairs = [
        ("interaction_greedy", "pairwise_greedy"),
        ("interaction_greedy", "individual_greedy"),
        ("pairwise_greedy", "individual_greedy"),
        ("interaction_greedy", "synergy_seeded"),
    ]

    stat_tests = {}
    for m1, m2 in method_pairs:
        diffs = []
        for bm in BENCHMARK_NAMES:
            v1 = main_table[bm].get(m1, 0)
            v2 = main_table[bm].get(m2, 0)
            diffs.append(v1 - v2)
        diffs = np.array(diffs)
        if np.any(diffs != 0):
            try:
                stat, pval = stats.wilcoxon(diffs)
                stat_tests[f"{m1}_vs_{m2}"] = {
                    "statistic": float(stat), "p_value": float(pval),
                    "mean_diff": float(np.mean(diffs)),
                    "significant_005": pval < 0.05,
                }
            except Exception:
                stat_tests[f"{m1}_vs_{m2}"] = {"error": "insufficient data"}
        else:
            stat_tests[f"{m1}_vs_{m2}"] = {"note": "all differences zero"}

    evaluation["statistical_tests"] = stat_tests

    with open(os.path.join(DATA_DIR, "statistical_evaluation.json"), "w") as f:
        json.dump(evaluation, f, indent=2, default=float)

    log(f"  Criterion 1 (sig order-3): {evaluation['criterion1_significant_order3']['frac']:.3f} (need >=0.30)")
    log(f"  Criterion 2 (variance order-3): {avg_frac3:.3f} (need >=0.10)")
    log(f"  Criterion 3 (win rate): {evaluation['criterion3_selection_win_rate']['win_rate']:.2f} (need >=0.60)")

    return evaluation


# ============================================================
# Step 10: Visualization
# ============================================================

def generate_figures(variance_decomp, selection_results, baseline_results,
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

    # Figure 1: Variance decomposition
    fig, ax = plt.subplots(figsize=(12, 5))
    bms = BENCHMARK_NAMES
    x = np.arange(len(bms))
    fracs1 = [variance_decomp[bm]["frac_order1"] for bm in bms]
    fracs2 = [variance_decomp[bm]["frac_order2"] for bm in bms]
    fracs3 = [variance_decomp[bm]["frac_order3"] for bm in bms]
    ax.bar(x, fracs1, label="Order 1 (individual)", color="#4C72B0")
    ax.bar(x, fracs2, bottom=fracs1, label="Order 2 (pairwise)", color="#DD8452")
    ax.bar(x, fracs3, bottom=[f1 + f2 for f1, f2 in zip(fracs1, fracs2)],
           label="Order 3 (triple)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([bm[:8] for bm in bms], rotation=45, ha="right")
    ax.set_ylabel("Fraction of Interaction Variance")
    ax.set_title("Variance Decomposition by Interaction Order")
    ax.legend()
    fig.savefig(os.path.join(FIGURES_DIR, "variance_decomposition.png"))
    fig.savefig(os.path.join(FIGURES_DIR, "variance_decomposition.pdf"))
    plt.close(fig)
    log("  Figure 1: variance_decomposition")

    # Figure 2: Interaction heatmap for representative benchmarks
    for bm in ["2mm", "correlation", "gemver"]:
        phi1, phi2, phi3 = _compute_phi_from_data(shapley_results[bm])
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.eye(N_PASSES, dtype=bool)
        # Use upper triangle
        heatmap_data = phi2.copy()
        np.fill_diagonal(heatmap_data, 0)
        short_names = [p[:8] for p in CANDIDATE_PASSES]
        sns.heatmap(heatmap_data, xticklabels=short_names, yticklabels=short_names,
                    cmap="RdBu_r", center=0, ax=ax, mask=mask, square=True)
        ax.set_title(f"Pairwise Shapley Interactions: {bm}")
        fig.savefig(os.path.join(FIGURES_DIR, f"heatmap_{bm}.png"))
        fig.savefig(os.path.join(FIGURES_DIR, f"heatmap_{bm}.pdf"))
        plt.close(fig)
    log("  Figure 2: interaction heatmaps")

    # Figure 3: Selection performance comparison at k=10
    fig, ax = plt.subplots(figsize=(14, 6))
    methods = ["LLVM_O3", "random_search_mean", "ga_mean",
               "individual_greedy", "pairwise_greedy", "interaction_greedy", "synergy_seeded"]
    method_labels = ["-O3", "Random", "GA", "Indiv.", "Pairwise", "Interact. (ours)", "Synergy"]
    colors = ["#666666", "#888888", "#AAAAAA", "#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    x = np.arange(len(BENCHMARK_NAMES))
    width = 0.12
    for m_idx, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        vals = []
        for bm in BENCHMARK_NAMES:
            if method == "LLVM_O3":
                opt = baseline_results["opt_levels"][bm]
                vals.append(opt.get("O3", {}).get("reduction", 0))
            elif method == "random_search_mean":
                rs_vals = []
                for seed in SEEDS:
                    sd = baseline_results["random_search"][bm][seed]
                    k_key = 10 if 10 in sd else str(10)
                    if k_key in sd:
                        rs_vals.append(sd[k_key]["reduction"])
                vals.append(np.mean(rs_vals) if rs_vals else 0)
            elif method == "ga_mean":
                ga_vals = []
                for seed in SEEDS:
                    sd = baseline_results["genetic_algorithm"][bm][seed]
                    k_key = 10 if 10 in sd else str(10)
                    if k_key in sd:
                        ga_vals.append(sd[k_key]["reduction"])
                vals.append(np.mean(ga_vals) if ga_vals else 0)
            else:
                k_key = 10 if 10 in selection_results[bm]["methods"] else "10"
                vals.append(selection_results[bm]["methods"][k_key][method]["reduction"])

        offset = (m_idx - len(methods) / 2) * width
        ax.bar(x + offset, vals, width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([bm[:8] for bm in BENCHMARK_NAMES], rotation=45, ha="right")
    ax.set_ylabel("IR Instruction Count Reduction")
    ax.set_title("Pass Selection Performance Comparison (k=10 passes)")
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    fig.savefig(os.path.join(FIGURES_DIR, "selection_comparison.png"))
    fig.savefig(os.path.join(FIGURES_DIR, "selection_comparison.pdf"))
    plt.close(fig)
    log("  Figure 3: selection_comparison")

    # Figure 4: Performance vs budget k
    fig, ax = plt.subplots(figsize=(10, 6))
    method_list = ["individual_greedy", "pairwise_greedy", "interaction_greedy", "synergy_seeded"]
    method_labels_4 = ["Individual", "Pairwise", "Interaction (ours)", "Synergy-seeded"]
    colors_4 = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for method, label, color in zip(method_list, method_labels_4, colors_4):
        means = []
        stds = []
        for k in SELECTION_BUDGETS:
            vals = []
            for bm in BENCHMARK_NAMES:
                k_key = k if k in selection_results[bm]["methods"] else str(k)
                vals.append(selection_results[bm]["methods"][k_key][method]["reduction"])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.errorbar(SELECTION_BUDGETS, means, yerr=stds, label=label, color=color,
                     marker="o", capsize=3)

    ax.set_xlabel("Number of Selected Passes (k)")
    ax.set_ylabel("Mean IR Reduction")
    ax.set_title("Selection Performance vs Pass Budget")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIGURES_DIR, "performance_vs_budget.png"))
    fig.savefig(os.path.join(FIGURES_DIR, "performance_vs_budget.pdf"))
    plt.close(fig)
    log("  Figure 4: performance_vs_budget")

    # Figure 5: Convergence (budget ablation)
    if ablation_budget and "summary" in ablation_budget:
        fig, ax = plt.subplots(figsize=(8, 5))
        budgets = ablation_budget["budgets"]
        for bm in ablation_budget["benchmarks"]:
            means = [ablation_budget["summary"][bm][b]["mean_reduction"] for b in budgets]
            stds = [ablation_budget["summary"][bm][b]["std_reduction"] for b in budgets]
            ax.errorbar(budgets, means, yerr=stds, label=bm[:8], marker="o", capsize=3)
        ax.set_xlabel("Shapley Evaluation Budget")
        ax.set_ylabel("Selection Performance (k=10)")
        ax.set_title("Convergence of Shapley Estimation")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(FIGURES_DIR, "convergence.png"))
        fig.savefig(os.path.join(FIGURES_DIR, "convergence.pdf"))
        plt.close(fig)
        log("  Figure 5: convergence")

    # Figure 6: Transferability heatmap
    if transferability and "similarity_matrix" in transferability:
        fig, ax = plt.subplots(figsize=(10, 8))
        sim = np.array(transferability["similarity_matrix"])
        short = [bm[:8] for bm in BENCHMARK_NAMES]
        sns.heatmap(sim, xticklabels=short, yticklabels=short,
                    cmap="YlOrRd", vmin=0, vmax=1, ax=ax, square=True, annot=False)
        ax.set_title("Benchmark Similarity (Interaction Vector Cosine)")
        fig.savefig(os.path.join(FIGURES_DIR, "transferability.png"))
        fig.savefig(os.path.join(FIGURES_DIR, "transferability.pdf"))
        plt.close(fig)
        log("  Figure 6: transferability")

    # Figure 7: Ablation - interaction order
    if ablation_order:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: improvement from pairwise over individual
        improvements_2 = []
        improvements_3 = []
        for bm in BENCHMARK_NAMES:
            for k in SELECTION_BUDGETS:
                k_key = k if k in ablation_order[bm] else str(k)
                if k_key in ablation_order[bm]:
                    improvements_2.append(ablation_order[bm][k_key]["improvement_2_over_1"])
                    improvements_3.append(ablation_order[bm][k_key]["improvement_3_over_12"])

        axes[0].hist(improvements_2, bins=20, alpha=0.7, label="Order 2 over 1", color="#DD8452")
        axes[0].hist(improvements_3, bins=20, alpha=0.7, label="Order 3 over 1+2", color="#C44E52")
        axes[0].set_xlabel("Improvement in IR Reduction")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distribution of Improvements by Interaction Order")
        axes[0].legend()
        axes[0].axvline(0, color="black", linestyle="--", alpha=0.5)

        # Right: mean improvement per budget
        for order_label, key_name, color in [
            ("Pairwise over Individual", "improvement_2_over_1", "#DD8452"),
            ("Triple over Pairwise", "improvement_3_over_12", "#C44E52")
        ]:
            means = []
            stds = []
            for k in SELECTION_BUDGETS:
                vals = []
                for bm in BENCHMARK_NAMES:
                    k_key = k if k in ablation_order[bm] else str(k)
                    if k_key in ablation_order[bm]:
                        vals.append(ablation_order[bm][k_key][key_name])
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            axes[1].errorbar(SELECTION_BUDGETS, means, yerr=stds, label=order_label,
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
        log("  Figure 7: ablation_order")

    log("  All figures generated.")


# ============================================================
# Step 11: Aggregate results.json
# ============================================================

def aggregate_results(screening, shapley_results, variance_decomp,
                      selection_results, baseline_results,
                      ablation_order, ablation_budget,
                      transferability, evaluation, structure):
    log("Step 11: Aggregating final results.json...")

    # Build main comparison table with error bars
    main_table = {}
    for bm in BENCHMARK_NAMES:
        row = {}

        # LLVM levels
        for level in ["O1", "O2", "O3", "Os", "Oz"]:
            opt = baseline_results["opt_levels"][bm]
            if level in opt:
                row[f"LLVM_{level}"] = {"value": opt[level]["reduction"]}

        # Random search with error bars
        rs_vals = {}
        for k in SELECTION_BUDGETS:
            vals_k = []
            for seed in SEEDS:
                sd = baseline_results["random_search"][bm][seed]
                k_key = k if k in sd else str(k)
                if k_key in sd:
                    vals_k.append(sd[k_key]["reduction"])
            if vals_k:
                rs_vals[k] = {"mean": float(np.mean(vals_k)), "std": float(np.std(vals_k))}
        row["random_search"] = rs_vals

        # GA with error bars
        ga_vals = {}
        for k in SELECTION_BUDGETS:
            vals_k = []
            for seed in SEEDS:
                sd = baseline_results["genetic_algorithm"][bm][seed]
                k_key = k if k in sd else str(k)
                if k_key in sd:
                    vals_k.append(sd[k_key]["reduction"])
            if vals_k:
                ga_vals[k] = {"mean": float(np.mean(vals_k)), "std": float(np.std(vals_k))}
        row["genetic_algorithm"] = ga_vals

        # Selection methods (deterministic given averaged Shapley values)
        sel_methods = {}
        for method in ["individual_greedy", "pairwise_greedy", "interaction_greedy", "synergy_seeded"]:
            method_vals = {}
            for k in SELECTION_BUDGETS:
                k_key = k if k in selection_results[bm]["methods"] else str(k)
                method_vals[k] = selection_results[bm]["methods"][k_key][method]["reduction"]
            sel_methods[method] = method_vals
        row["selection_methods"] = sel_methods

        main_table[bm] = row

    # Compute overall means
    overall_means = {}
    for method in ["individual_greedy", "pairwise_greedy", "interaction_greedy", "synergy_seeded"]:
        for k in SELECTION_BUDGETS:
            vals = [main_table[bm]["selection_methods"][method][k] for bm in BENCHMARK_NAMES]
            key = f"{method}_k{k}"
            overall_means[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    for k in SELECTION_BUDGETS:
        rs_means = [main_table[bm]["random_search"][k]["mean"] for bm in BENCHMARK_NAMES if k in main_table[bm]["random_search"]]
        ga_means = [main_table[bm]["genetic_algorithm"][k]["mean"] for bm in BENCHMARK_NAMES if k in main_table[bm]["genetic_algorithm"]]
        if rs_means:
            overall_means[f"random_search_k{k}"] = {"mean": float(np.mean(rs_means)), "std": float(np.std(rs_means))}
        if ga_means:
            overall_means[f"genetic_algorithm_k{k}"] = {"mean": float(np.mean(ga_means)), "std": float(np.std(ga_means))}

    results = {
        "experiment": "ShapleyPass: Compiler Pass Interaction Analysis via Shapley Interaction Indices",
        "benchmarks": BENCHMARK_NAMES,
        "benchmark_suite": "PolyBench",
        "n_benchmarks": len(BENCHMARK_NAMES),
        "n_passes": N_PASSES,
        "passes": CANDIDATE_PASSES,
        "seeds": SEEDS,
        "shapley_budget": SHAPLEY_BUDGET,
        "selection_budgets": SELECTION_BUDGETS,

        "hypothesis_evaluation": evaluation,

        "variance_decomposition": variance_decomp,

        "main_comparison_table": main_table,
        "overall_means": overall_means,

        "ablation_order": ablation_order,
        "ablation_budget": ablation_budget.get("summary", {}) if ablation_budget else {},

        "transferability_summary": {
            bm: transferability["transfer_results"][bm]
            for bm in BENCHMARK_NAMES
        } if transferability else {},

        "interaction_structure_summary": {
            "n_universal_pairs": structure["cross_program"]["n_universal"],
            "top_universal_pairs": structure["cross_program"]["universal_pairs"][:10],
        } if structure else {},
    }

    # Write results.json at workspace root
    results_path = os.path.join(PROJECT_ROOT, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    log(f"  Final results.json written to {results_path}")
    return results


# ============================================================
# Main
# ============================================================

def main():
    t_start = time.time()
    log("=" * 60)
    log("ShapleyPass: Full Experiment Pipeline on PolyBench")
    log("=" * 60)

    # Step 1: Pass screening
    screening = run_pass_screening()

    # Step 2: Compute Shapley interaction indices
    shapley_results = run_shapley_computation()

    # Step 3: Variance decomposition
    variance_decomp = run_variance_decomposition(shapley_results)

    # Step 4: Pass selection algorithms
    selection_results = run_pass_selection(shapley_results)

    # Step 5: Baselines
    baseline_results = run_baselines(screening)

    # Step 6: Ablation studies
    ablation_order = ablation_interaction_order(shapley_results)
    ablation_budget = ablation_budget_convergence(shapley_results)

    # Step 7: Interaction structure analysis
    structure = analyze_interaction_structure(shapley_results)

    # Step 8: Transferability
    transferability = run_transferability(shapley_results)

    # Step 9: Statistical evaluation
    evaluation = run_statistical_evaluation(
        shapley_results, selection_results, baseline_results, variance_decomp
    )

    # Step 10: Generate figures
    generate_figures(
        variance_decomp, selection_results, baseline_results,
        shapley_results, ablation_order, ablation_budget,
        transferability, structure
    )

    # Step 11: Aggregate results
    aggregate_results(
        screening, shapley_results, variance_decomp,
        selection_results, baseline_results,
        ablation_order, ablation_budget,
        transferability, evaluation, structure
    )

    elapsed = time.time() - t_start
    log(f"\nTotal time: {elapsed / 3600:.2f} hours ({elapsed:.0f} seconds)")
    log("Done!")


if __name__ == "__main__":
    main()

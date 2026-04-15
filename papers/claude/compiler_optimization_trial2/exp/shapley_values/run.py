#!/usr/bin/env python3
"""
Run all ShapleyPass experiments with the fast oracle.
Uses 20 passes, 150 permutations, 20 programs, 3 seeds.
"""
import json
import os
import sys
import time
import numpy as np
from tqdm import tqdm
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_oracle import (FastOracle, PASS_NAMES, PASS_ORDER, PASS_CATALOG,
                         N_PASSES, count_ir_instructions_from_string)

WORKSPACE = "/home/nw366/ResearchArena/outputs/claude_t2_compiler_optimization/idea_01"
RESULTS_DIR = os.path.join(WORKSPACE, "results")
SEEDS = [42, 123, 456]
N_PERMUTATIONS = 150


def select_programs():
    manifest_path = os.path.join(WORKSPACE, "data/benchmark_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    programs = manifest["programs"]
    # Exclude slow programs
    EXCLUDE = {"polybench_nussinov"}
    poly = [p for p in programs if p["name"].startswith("polybench_") and p["name"] not in EXCLUDE]
    small = [p for p in programs if p["name"].startswith("small_")]
    # Select diverse PolyBench (up to 3 per category, max 10)
    by_cat = {}
    for p in poly:
        by_cat.setdefault(p["category"], []).append(p)
    selected_poly = []
    for cat, progs in by_cat.items():
        selected_poly.extend(progs[:3])
    return selected_poly[:10] + small


def compute_shapley_mc(oracle, program, num_permutations, seed):
    rng = np.random.RandomState(seed)
    ll_file = program["path"]
    baseline = program["baseline_instructions"]
    pname = program["name"]

    marginals = {p: [] for p in PASS_NAMES}
    checkpoints = [25, 50, 75, 100, 125, 150]
    convergence = {p: [] for p in PASS_NAMES}

    for perm_idx in range(num_permutations):
        perm = list(rng.permutation(PASS_NAMES))
        current_set = set()
        v_prev = 0.0

        for pass_name in perm:
            current_set_with = current_set | {pass_name}
            v_with = oracle.characteristic_value(ll_file, current_set_with, baseline, pname)
            marginals[pass_name].append(v_with - v_prev)
            v_prev = v_with
            current_set = current_set_with

        if (perm_idx + 1) in checkpoints:
            for p in PASS_NAMES:
                if marginals[p]:
                    convergence[p].append(float(np.mean(marginals[p])))

    shapley_values = {}
    for p in PASS_NAMES:
        vals = marginals[p]
        shapley_values[p] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    return shapley_values, convergence


def compute_loo(oracle, program):
    ll_file = program["path"]
    baseline = program["baseline_instructions"]
    pname = program["name"]
    full_set = set(PASS_NAMES)
    v_full = oracle.characteristic_value(ll_file, full_set, baseline, pname)
    loo = {}
    for p in PASS_NAMES:
        v_reduced = oracle.characteristic_value(ll_file, full_set - {p}, baseline, pname)
        loo[p] = float(v_full - v_reduced)
    return loo, float(v_full)


def compute_random_baseline(oracle, program, k_values, n_samples, seed):
    rng = np.random.RandomState(seed)
    ll_file = program["path"]
    baseline = program["baseline_instructions"]
    pname = program["name"]
    results = {}
    for k in k_values:
        values = []
        for _ in range(n_samples):
            subset = set(rng.choice(PASS_NAMES, size=k, replace=False))
            v = oracle.characteristic_value(ll_file, subset, baseline, pname)
            values.append(float(v))
        results[k] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "values": values,
        }
    return results


def compute_standard_levels(ll_file, baseline):
    import subprocess
    results = {}
    for level in ["-O2", "-O3", "-Os"]:
        try:
            r = subprocess.run(["opt", level, ll_file, "-S"],
                             capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                ic = count_ir_instructions_from_string(r.stdout)
                results[level] = {"instructions": ic, "reduction": float((baseline - ic) / baseline)}
            else:
                results[level] = {"instructions": baseline, "reduction": 0.0}
        except:
            results[level] = {"instructions": baseline, "reduction": 0.0}
    return results


def compute_pairwise_interactions(oracle, program, top_passes, n_samples, seed):
    rng = np.random.RandomState(seed)
    ll_file = program["path"]
    baseline = program["baseline_instructions"]
    pname = program["name"]

    other_passes = [p for p in PASS_NAMES if p not in top_passes]
    interactions = {}

    for i, pi in enumerate(top_passes):
        for j in range(i + 1, len(top_passes)):
            pj = top_passes[j]
            other_top = [p for p in top_passes if p != pi and p != pj]

            deltas = []
            for _ in range(n_samples):
                # Random coalition from all other passes
                all_other = other_top + other_passes
                k = rng.randint(0, len(all_other) + 1)
                S = set(rng.choice(all_other, size=min(k, len(all_other)), replace=False)) if all_other and k > 0 else set()

                v_both = oracle.characteristic_value(ll_file, S | {pi, pj}, baseline, pname)
                v_i = oracle.characteristic_value(ll_file, S | {pi}, baseline, pname)
                v_j = oracle.characteristic_value(ll_file, S | {pj}, baseline, pname)
                v_none = oracle.characteristic_value(ll_file, S, baseline, pname)
                deltas.append(v_both - v_i - v_j + v_none)

            interactions[f"{pi}|{pj}"] = {
                "mean": float(np.mean(deltas)),
                "std": float(np.std(deltas)),
                "ci_low": float(np.percentile(deltas, 2.5)),
                "ci_high": float(np.percentile(deltas, 97.5)),
            }

    return interactions


def shapley_select(shapley_values, interactions, lambda_val, max_k=20):
    selected = []
    remaining = list(PASS_NAMES)
    trajectory = []

    for step in range(min(max_k, N_PASSES)):
        best_score = -float('inf')
        best_pass = None

        for p in remaining:
            score = shapley_values.get(p, {}).get("mean", 0)
            if lambda_val > 0 and interactions:
                for s in selected:
                    for key in [f"{p}|{s}", f"{s}|{p}"]:
                        if key in interactions:
                            score += lambda_val * interactions[key]["mean"]
                            break
            if score > best_score:
                best_score = score
                best_pass = p

        if best_pass is None:
            break
        selected.append(best_pass)
        remaining.remove(best_pass)
        trajectory.append({"pass": best_pass, "score": float(best_score)})

    return selected, trajectory


def evaluate_selection(oracle, program, selected_seq):
    ll_file = program["path"]
    baseline = program["baseline_instructions"]
    pname = program["name"]
    results = []
    for k in range(1, len(selected_seq) + 1):
        v = oracle.characteristic_value(ll_file, set(selected_seq[:k]), baseline, pname)
        results.append(float(v))
    return results


def greedy_ablation_select(oracle, program):
    ll_file = program["path"]
    baseline = program["baseline_instructions"]
    pname = program["name"]
    current = set(PASS_NAMES)
    removed_order = []

    while current:
        best_to_remove = None
        best_val = -float('inf')
        for p in current:
            reduced = current - {p}
            v = oracle.characteristic_value(ll_file, reduced, baseline, pname) if reduced else 0.0
            if v > best_val:
                best_val = v
                best_to_remove = p
        current.remove(best_to_remove)
        removed_order.append(best_to_remove)

    return list(reversed(removed_order))


def run_all():
    start_time = time.time()

    programs = select_programs()
    print(f"Selected {len(programs)} programs:")
    for p in programs:
        print(f"  {p['name']} ({p['category']}, {p['baseline_instructions']} instr)")

    oracle = FastOracle()

    # Save pass catalog
    with open(os.path.join(WORKSPACE, "data/pass_catalog.json"), "w") as f:
        json.dump(PASS_CATALOG, f, indent=2)

    # ================================================================
    # EXPERIMENT 1: Monte Carlo Shapley Values
    # ================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: Monte Carlo Shapley Values")
    print(f"{'='*60}")

    os.makedirs(os.path.join(RESULTS_DIR, "shapley_values"), exist_ok=True)
    all_shapley = {}
    all_convergence = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        seed_results = {}
        seed_convergence = {}
        for prog in tqdm(programs, desc=f"Shapley (seed={seed})"):
            sv, conv = compute_shapley_mc(oracle, prog, N_PERMUTATIONS, seed)
            seed_results[prog["name"]] = sv
            seed_convergence[prog["name"]] = conv
        all_shapley[seed] = seed_results
        all_convergence[seed] = seed_convergence
        with open(os.path.join(RESULTS_DIR, f"shapley_values/shapley_raw_{seed}.json"), "w") as f:
            json.dump(seed_results, f, indent=2)
        oracle.save_cache()

    # Aggregate across seeds
    agg_shapley = {}
    for prog in programs:
        pname = prog["name"]
        agg_shapley[pname] = {}
        for p in PASS_NAMES:
            seed_means = [all_shapley[s][pname][p]["mean"] for s in SEEDS]
            agg_shapley[pname][p] = {
                "mean": float(np.mean(seed_means)),
                "std_across_seeds": float(np.std(seed_means)),
                "within_seed_std": float(np.mean([all_shapley[s][pname][p]["std"] for s in SEEDS]))
            }

    with open(os.path.join(RESULTS_DIR, "shapley_values/shapley_aggregated.json"), "w") as f:
        json.dump(agg_shapley, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "shapley_values/convergence.json"), "w") as f:
        json.dump({str(s): all_convergence[s] for s in SEEDS}, f, indent=2)

    t1 = time.time() - start_time
    print(f"\nExperiment 1 done in {t1/60:.1f} minutes. Oracle: {oracle.stats}")

    # ================================================================
    # BASELINE 1: Leave-One-Out
    # ================================================================
    print(f"\n{'='*60}")
    print("BASELINE 1: Leave-One-Out Attribution")
    print(f"{'='*60}")

    loo_results = {}
    for prog in tqdm(programs, desc="LOO"):
        loo, v_full = compute_loo(oracle, prog)
        loo_results[prog["name"]] = {"loo": loo, "v_full": v_full}

    with open(os.path.join(RESULTS_DIR, "shapley_values/loo_attribution.json"), "w") as f:
        json.dump(loo_results, f, indent=2)
    oracle.save_cache()

    t2 = time.time() - start_time
    print(f"LOO done in {(t2-t1)/60:.1f} min (total: {t2/60:.1f} min)")

    # ================================================================
    # BASELINE 2: Random Sampling + Standard Levels
    # ================================================================
    print(f"\n{'='*60}")
    print("BASELINE 2: Random Sampling + Standard Levels")
    print(f"{'='*60}")

    os.makedirs(os.path.join(RESULTS_DIR, "selection"), exist_ok=True)
    k_values = [3, 5, 7, 10, 15, 20]
    random_results = {}
    for seed in SEEDS:
        seed_res = {}
        for prog in tqdm(programs, desc=f"Random (seed={seed})"):
            seed_res[prog["name"]] = compute_random_baseline(oracle, prog, k_values, 50, seed)
        random_results[seed] = seed_res

    agg_random = {}
    for prog in programs:
        pname = prog["name"]
        agg_random[pname] = {}
        for k in k_values:
            means = [random_results[s][pname][k]["mean"] for s in SEEDS]
            stds = [random_results[s][pname][k]["std"] for s in SEEDS]
            agg_random[pname][k] = {"mean": float(np.mean(means)), "std": float(np.mean(stds))}

    with open(os.path.join(RESULTS_DIR, "selection/random_baseline.json"), "w") as f:
        json.dump(agg_random, f, indent=2)

    std_levels = {}
    for prog in tqdm(programs, desc="Standard levels"):
        std_levels[prog["name"]] = compute_standard_levels(prog["path"], prog["baseline_instructions"])

    with open(os.path.join(RESULTS_DIR, "selection/standard_levels.json"), "w") as f:
        json.dump(std_levels, f, indent=2)
    oracle.save_cache()

    t3 = time.time() - start_time
    print(f"Random+Standard done in {(t3-t2)/60:.1f} min (total: {t3/60:.1f} min)")

    # ================================================================
    # EXPERIMENT 2: Pairwise Shapley Interactions
    # ================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Pairwise Shapley Interactions")
    print(f"{'='*60}")

    os.makedirs(os.path.join(RESULTS_DIR, "interactions"), exist_ok=True)

    # Top-15 passes by average Shapley value
    avg_shapley_global = {}
    for p in PASS_NAMES:
        avg_shapley_global[p] = float(np.mean([agg_shapley[prog["name"]][p]["mean"] for prog in programs]))

    sorted_passes = sorted(avg_shapley_global.keys(), key=lambda x: avg_shapley_global[x], reverse=True)
    top_passes = sorted_passes[:15]
    print(f"Top-15 passes for interactions: {top_passes}")

    # Compute interactions for all programs
    interaction_results = {}
    for seed in SEEDS:
        for prog in tqdm(programs, desc=f"Interactions (seed={seed})"):
            key = f"{prog['name']}_{seed}"
            interaction_results[key] = compute_pairwise_interactions(
                oracle, prog, top_passes, n_samples=80, seed=seed)
        oracle.save_cache()

    # Aggregate interactions
    agg_interactions = {}
    for prog in programs:
        pname = prog["name"]
        agg_interactions[pname] = {}
        pair_keys = set()
        for seed in SEEDS:
            pair_keys.update(interaction_results[f"{pname}_{seed}"].keys())
        for pk in pair_keys:
            seed_means = [interaction_results[f"{pname}_{seed}"][pk]["mean"]
                         for seed in SEEDS if pk in interaction_results[f"{pname}_{seed}"]]
            agg_interactions[pname][pk] = {
                "mean": float(np.mean(seed_means)),
                "std": float(np.std(seed_means))
            }

    avg_interactions = {}
    all_pair_keys = set()
    for pname in agg_interactions:
        all_pair_keys.update(agg_interactions[pname].keys())
    for pk in all_pair_keys:
        vals = [agg_interactions[pn][pk]["mean"] for pn in agg_interactions if pk in agg_interactions[pn]]
        avg_interactions[pk] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    with open(os.path.join(RESULTS_DIR, "interactions/interactions_aggregated.json"), "w") as f:
        json.dump({"per_program": agg_interactions, "average": avg_interactions,
                   "top_passes": top_passes, "avg_shapley_global": avg_shapley_global}, f, indent=2)

    t4 = time.time() - start_time
    print(f"Interactions done in {(t4-t3)/60:.1f} min (total: {t4/60:.1f} min)")

    # ================================================================
    # EXPERIMENT 3: Pass Selection
    # ================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: ShapleyPass-Select + Baselines")
    print(f"{'='*60}")

    lambda_values = [0.0, 0.5, 1.0, 2.0]
    selection_results = {
        "shapley_select": {},
        "topk_shapley": {},
        "topk_loo": {},
        "greedy_ablation": {},
        "programs": [p["name"] for p in programs]
    }

    for prog in tqdm(programs, desc="Selection"):
        pname = prog["name"]
        prog_interactions = agg_interactions.get(pname, avg_interactions)

        for lam in lambda_values:
            selected, traj = shapley_select(agg_shapley[pname], prog_interactions, lam)
            curve = evaluate_selection(oracle, prog, selected)
            lam_key = f"lambda_{lam}"
            selection_results["shapley_select"].setdefault(lam_key, {})[pname] = {
                "selected": selected, "curve": curve}

        # Top-k Shapley
        sorted_sv = sorted(agg_shapley[pname].items(), key=lambda x: x[1]["mean"], reverse=True)
        topk_order = [p for p, _ in sorted_sv]
        selection_results["topk_shapley"][pname] = {
            "selected": topk_order,
            "curve": evaluate_selection(oracle, prog, topk_order)}

        # Top-k LOO
        loo_data = loo_results[pname]["loo"]
        sorted_loo = sorted(loo_data.items(), key=lambda x: x[1], reverse=True)
        loo_order = [p for p, _ in sorted_loo]
        selection_results["topk_loo"][pname] = {
            "selected": loo_order,
            "curve": evaluate_selection(oracle, prog, loo_order)}

        # Greedy ablation
        abl_order = greedy_ablation_select(oracle, prog)
        selection_results["greedy_ablation"][pname] = {
            "selected": abl_order,
            "curve": evaluate_selection(oracle, prog, abl_order)}

    with open(os.path.join(RESULTS_DIR, "selection/all_selection_results.json"), "w") as f:
        json.dump(selection_results, f, indent=2)
    oracle.save_cache()

    t5 = time.time() - start_time
    print(f"Selection done in {(t5-t4)/60:.1f} min (total: {t5/60:.1f} min)")

    # ================================================================
    # SAVE COST ANALYSIS
    # ================================================================
    total_time = time.time() - start_time
    cost = {
        "total_seconds": total_time,
        "total_minutes": total_time / 60,
        "oracle_stats": oracle.stats,
        "stages": {
            "shapley_values_min": t1 / 60,
            "loo_min": (t2 - t1) / 60,
            "random_baseline_min": (t3 - t2) / 60,
            "interactions_min": (t4 - t3) / 60,
            "selection_min": (t5 - t4) / 60
        }
    }
    with open(os.path.join(RESULTS_DIR, "cost_analysis.json"), "w") as f:
        json.dump(cost, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE in {total_time/60:.1f} minutes")
    print(f"Oracle: {oracle.stats}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all()

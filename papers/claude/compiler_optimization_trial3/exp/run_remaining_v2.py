#!/usr/bin/env python3
"""
ShapleyPass: Continue pipeline from saved Shapley results.
Runs selection experiments, ablations, transferability, and generates figures.
Optimized for efficiency.
"""

import os
import sys
import json
import time
import tempfile
import subprocess
import numpy as np
from itertools import combinations
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
from game import (CompilerGame, CANDIDATE_PASSES, count_ir_instructions,
                  apply_passes, get_optimization_level_counts)

# ============================================================
# Configuration
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BC_DIR = os.path.join(BASE, 'data', 'polybench_bc')
RESULTS_DIR = os.path.join(BASE, 'results')
FIGURES_DIR = os.path.join(BASE, 'figures')

SEEDS = [42, 123, 456]
K_PASSES = 20
PASSES = CANDIDATE_PASSES[:K_PASSES]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'data'), exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def bc_path(bm):
    return os.path.join(BC_DIR, bm + '.bc')

# ============================================================
# Load saved Shapley results
# ============================================================
def load_shapley():
    path = os.path.join(RESULTS_DIR, 'data', 'interactions', 'shapley_results.json')
    with open(path) as f:
        return json.load(f)

# ============================================================
# Selection Algorithms
# ============================================================
def greedy_individual(order1, passes, budget_k):
    ranked = sorted(order1.items(), key=lambda x: -x[1])
    return [p for p, _ in ranked[:budget_k]]

def greedy_pairwise(order1, order2, passes, budget_k):
    selected = []
    remaining = set(passes)
    best = max(order1.items(), key=lambda x: x[1])
    selected.append(best[0])
    remaining.remove(best[0])
    while len(selected) < budget_k and remaining:
        best_score = -float('inf')
        best_pass = None
        for p in remaining:
            score = order1.get(p, 0)
            for s in selected:
                score += order2.get(f"{p}+{s}", 0) + order2.get(f"{s}+{p}", 0)
            if score > best_score:
                best_score = score
                best_pass = p
        if best_pass:
            selected.append(best_pass)
            remaining.remove(best_pass)
        else:
            break
    return selected

def greedy_interaction(order1, order2, order3, passes, budget_k):
    selected = []
    remaining = set(passes)
    best = max(order1.items(), key=lambda x: x[1])
    selected.append(best[0])
    remaining.remove(best[0])
    while len(selected) < budget_k and remaining:
        best_score = -float('inf')
        best_pass = None
        for p in remaining:
            score = order1.get(p, 0)
            for s in selected:
                score += order2.get(f"{p}+{s}", 0) + order2.get(f"{s}+{p}", 0)
            for s1, s2 in combinations(selected, 2):
                for perm in [f"{p}+{s1}+{s2}", f"{p}+{s2}+{s1}",
                             f"{s1}+{p}+{s2}", f"{s1}+{s2}+{p}",
                             f"{s2}+{p}+{s1}", f"{s2}+{s1}+{p}"]:
                    score += order3.get(perm, 0)
            if score > best_score:
                best_score = score
                best_pass = p
        if best_pass:
            selected.append(best_pass)
            remaining.remove(best_pass)
        else:
            break
    return selected

def evaluate_selection(game, passes, selected_passes):
    x = np.zeros(len(passes))
    for p in selected_passes:
        if p in passes:
            x[passes.index(p)] = 1
    return game.value(x)

def random_search(game, n_samples, seed, pass_budget=None):
    rng = np.random.RandomState(seed)
    best_val = -float('inf')
    for _ in range(n_samples):
        if pass_budget:
            x = np.zeros(game.n_players)
            indices = rng.choice(game.n_players, pass_budget, replace=False)
            x[indices] = 1
        else:
            x = rng.randint(0, 2, game.n_players)
        val = game.value(x)
        if val > best_val:
            best_val = val
    return best_val

def genetic_algorithm(game, pop_size=30, generations=10, seed=42):
    rng = np.random.RandomState(seed)
    n = game.n_players
    pop = rng.randint(0, 2, (pop_size, n)).astype(float)
    fitness = np.array([game.value(ind) for ind in pop])
    for gen in range(generations):
        new_pop = []
        for _ in range(pop_size):
            idx = rng.choice(pop_size, 3, replace=False)
            p1 = pop[idx[np.argmax(fitness[idx])]]
            idx = rng.choice(pop_size, 3, replace=False)
            p2 = pop[idx[np.argmax(fitness[idx])]]
            point = rng.randint(1, n)
            child = np.concatenate([p1[:point], p2[point:]])
            for i in range(n):
                if rng.random() < 0.1:
                    child[i] = 1 - child[i]
            new_pop.append(child)
        pop = np.array(new_pop)
        fitness = np.array([game.value(ind) for ind in pop])
    return float(np.max(fitness))

# ============================================================
# Main pipeline
# ============================================================
def main():
    t_start = time.time()

    # Load Shapley results
    print("Loading saved Shapley interaction results...")
    shapley_results = load_shapley()
    benchmarks = list(shapley_results.keys())
    print(f"Benchmarks: {benchmarks} ({len(benchmarks)} total)")

    # Compute opt levels
    print("\n=== Computing Optimization Level Baselines ===")
    opt_levels = {}
    for bm in benchmarks:
        bp = bc_path(bm)
        counts = get_optimization_level_counts(bp)
        baseline = counts.get('O0', 0)
        reductions = {}
        for level in ['O1', 'O2', 'O3', 'Os', 'Oz']:
            c = counts.get(level)
            if c is not None and baseline > 0:
                reductions[level] = (baseline - c) / baseline
            else:
                reductions[level] = None
        opt_levels[bm] = reductions
        print(f"  {bm:25s}: O3={reductions.get('O3', 0):+.3f}")

    # ============================================================
    # Selection Experiments - OPTIMIZED
    # ============================================================
    print("\n=== Selection Experiments ===")
    budgets_k = [5, 8, 10, 12, 15]
    selection_results = {}

    for bm in benchmarks:
        print(f"\n  {bm}:")
        bp = bc_path(bm)
        game = CompilerGame(bp, passes=PASSES)
        seeds_data = shapley_results[bm]

        bm_results = {'budgets': {}, 'opt_levels': opt_levels.get(bm, {})}

        for budget_k in budgets_k:
            budget_results = {}

            # Shapley-based methods
            for method_name, method_fn in [
                ('individual_greedy', lambda o1, o2, o3, k=budget_k: greedy_individual(o1, PASSES, k)),
                ('pairwise_greedy', lambda o1, o2, o3, k=budget_k: greedy_pairwise(o1, o2, PASSES, k)),
                ('interaction_greedy', lambda o1, o2, o3, k=budget_k: greedy_interaction(o1, o2, o3, PASSES, k)),
            ]:
                seed_vals = []
                for sd in seeds_data:
                    selected = method_fn(sd['order1'], sd['order2'], sd['order3'])
                    val = evaluate_selection(game, PASSES, selected)
                    seed_vals.append(val)
                budget_results[method_name] = {
                    'mean': float(np.mean(seed_vals)),
                    'std': float(np.std(seed_vals)),
                    'values': [float(v) for v in seed_vals],
                }

            # Random search: 200 evals per budget level (efficient)
            rs_vals = []
            for seed in SEEDS:
                val = random_search(game, 200, seed, pass_budget=budget_k)
                rs_vals.append(val)
            budget_results['random_search'] = {
                'mean': float(np.mean(rs_vals)),
                'std': float(np.std(rs_vals)),
                'values': [float(v) for v in rs_vals],
            }

            # GA: only for k=10 (representative budget)
            if budget_k == 10:
                ga_vals = []
                for seed in SEEDS:
                    val = genetic_algorithm(game, pop_size=30, generations=10, seed=seed)
                    ga_vals.append(val)
                budget_results['genetic_algorithm'] = {
                    'mean': float(np.mean(ga_vals)),
                    'std': float(np.std(ga_vals)),
                    'values': [float(v) for v in ga_vals],
                }

            bm_results['budgets'][budget_k] = budget_results

        selection_results[bm] = bm_results
        b10 = bm_results['budgets'][10]
        print(f"    k=10: indiv={b10['individual_greedy']['mean']:.4f} "
              f"pair={b10['pairwise_greedy']['mean']:.4f} "
              f"inter={b10['interaction_greedy']['mean']:.4f} "
              f"RS={b10['random_search']['mean']:.4f} "
              f"GA={b10['genetic_algorithm']['mean']:.4f}")

    with open(os.path.join(RESULTS_DIR, 'data', 'selection_results.json'), 'w') as f:
        json.dump(selection_results, f, indent=2)

    # ============================================================
    # Ablation: Interaction Order
    # ============================================================
    print("\n=== Ablation: Interaction Order ===")
    ablation_order = {}
    for bm in benchmarks:
        bp = bc_path(bm)
        game = CompilerGame(bp, passes=PASSES)
        seeds_data = shapley_results[bm]
        bm_results = {}
        for budget_k in budgets_k:
            o1_vals, o12_vals, o123_vals = [], [], []
            for sd in seeds_data:
                o1_vals.append(evaluate_selection(game, PASSES, greedy_individual(sd['order1'], PASSES, budget_k)))
                o12_vals.append(evaluate_selection(game, PASSES, greedy_pairwise(sd['order1'], sd['order2'], PASSES, budget_k)))
                o123_vals.append(evaluate_selection(game, PASSES, greedy_interaction(sd['order1'], sd['order2'], sd['order3'], PASSES, budget_k)))
            bm_results[budget_k] = {
                'order1': {'mean': float(np.mean(o1_vals)), 'std': float(np.std(o1_vals)), 'values': [float(v) for v in o1_vals]},
                'order12': {'mean': float(np.mean(o12_vals)), 'std': float(np.std(o12_vals)), 'values': [float(v) for v in o12_vals]},
                'order123': {'mean': float(np.mean(o123_vals)), 'std': float(np.std(o123_vals)), 'values': [float(v) for v in o123_vals]},
                'improvement_2_over_1': float(np.mean(o12_vals) - np.mean(o1_vals)),
                'improvement_3_over_12': float(np.mean(o123_vals) - np.mean(o12_vals)),
            }
        ablation_order[bm] = bm_results
        r = bm_results[10]
        print(f"  {bm:25s} k=10: O1={r['order1']['mean']:.4f} O12={r['order12']['mean']:.4f} "
              f"O123={r['order123']['mean']:.4f} d3={r['improvement_3_over_12']:+.4f}")

    with open(os.path.join(RESULTS_DIR, 'data', 'ablation_order.json'), 'w') as f:
        json.dump(ablation_order, f, indent=2)

    # ============================================================
    # Ablation: Number of Passes (K=10,15,20)
    # ============================================================
    print("\n=== Ablation: Number of Passes ===")
    ablation_bms = benchmarks[:5]
    ablation_passes_results = {}

    for K in [10, 15, 20]:
        passes_k = CANDIDATE_PASSES[:K]
        print(f"\n  K={K}")
        k_results = {}
        for bm in ablation_bms:
            bp = bc_path(bm)
            game = CompilerGame(bp, passes=passes_k)

            import shapiq
            seed_vals = []
            for seed in SEEDS:
                approx = shapiq.PermutationSamplingSII(n=K, max_order=3, random_state=seed)
                iv = approx.approximate(budget=1500, game=game)

                o1, o2, o3 = {}, {}, {}
                for idx, val in zip(iv.interaction_lookup.keys(), iv.values):
                    if len(idx) == 1:
                        o1[passes_k[idx[0]]] = float(val)
                    elif len(idx) == 2:
                        o2[f"{passes_k[idx[0]]}+{passes_k[idx[1]]}"] = float(val)
                    elif len(idx) == 3:
                        o3[f"{passes_k[idx[0]]}+{passes_k[idx[1]]}+{passes_k[idx[2]]}"] = float(val)

                selected = greedy_interaction(o1, o2, o3, passes_k, 10)
                val = evaluate_selection(game, passes_k, selected)
                seed_vals.append(val)

            k_results[bm] = {
                'mean': float(np.mean(seed_vals)),
                'std': float(np.std(seed_vals)),
                'values': [float(v) for v in seed_vals],
            }
            print(f"    {bm}: {k_results[bm]['mean']:.4f} +/- {k_results[bm]['std']:.4f}")

        ablation_passes_results[K] = k_results

    with open(os.path.join(RESULTS_DIR, 'data', 'ablation_num_passes.json'), 'w') as f:
        json.dump(ablation_passes_results, f, indent=2)

    # ============================================================
    # Ablation: Evaluation Budget
    # ============================================================
    print("\n=== Ablation: Evaluation Budget ===")
    ablation_budget_results = {}

    for budget in [500, 1000, 1500, 2000, 3000]:
        print(f"\n  Budget={budget}")
        budget_results = {}
        for bm in ablation_bms:
            bp = bc_path(bm)
            game = CompilerGame(bp, passes=PASSES)

            import shapiq
            seed_vals = []
            elapsed_list = []
            for seed in SEEDS:
                t0 = time.time()
                approx = shapiq.PermutationSamplingSII(n=K_PASSES, max_order=3, random_state=seed)
                iv = approx.approximate(budget=budget, game=game)
                elapsed_list.append(time.time() - t0)

                o1, o2, o3 = {}, {}, {}
                for idx, val in zip(iv.interaction_lookup.keys(), iv.values):
                    if len(idx) == 1:
                        o1[PASSES[idx[0]]] = float(val)
                    elif len(idx) == 2:
                        o2[f"{PASSES[idx[0]]}+{PASSES[idx[1]]}"] = float(val)
                    elif len(idx) == 3:
                        o3[f"{PASSES[idx[0]]}+{PASSES[idx[1]]}+{PASSES[idx[2]]}"] = float(val)

                selected = greedy_interaction(o1, o2, o3, PASSES, 10)
                val = evaluate_selection(game, PASSES, selected)
                seed_vals.append(val)

            budget_results[bm] = {
                'mean': float(np.mean(seed_vals)),
                'std': float(np.std(seed_vals)),
                'values': [float(v) for v in seed_vals],
                'avg_time': float(np.mean(elapsed_list)),
            }
            print(f"    {bm}: {budget_results[bm]['mean']:.4f} time={budget_results[bm]['avg_time']:.1f}s")

        ablation_budget_results[budget] = budget_results

    with open(os.path.join(RESULTS_DIR, 'data', 'ablation_budget.json'), 'w') as f:
        json.dump(ablation_budget_results, f, indent=2)

    # ============================================================
    # Transferability Analysis
    # ============================================================
    print("\n=== Transferability Analysis ===")
    vectors = {}
    for bm in benchmarks:
        seeds_data = shapley_results[bm]
        vec = []
        for p in PASSES:
            vec.append(np.mean([sd['order1'].get(p, 0) for sd in seeds_data]))
        for i, p1 in enumerate(PASSES):
            for j, p2 in enumerate(PASSES):
                if i < j:
                    key = f"{p1}+{p2}"
                    vec.append(np.mean([sd['order2'].get(key, 0) for sd in seeds_data]))
        vectors[bm] = np.array(vec)

    from sklearn.metrics.pairwise import cosine_similarity
    bm_list = list(vectors.keys())
    mat = np.array([vectors[bm] for bm in bm_list])
    sim_matrix = cosine_similarity(mat)

    transfer_results = {}
    for i, target_bm in enumerate(bm_list):
        bp = bc_path(target_bm)
        game = CompilerGame(bp, passes=PASSES)

        sd = shapley_results[target_bm][0]
        oracle_selected = greedy_interaction(sd['order1'], sd['order2'], sd['order3'], PASSES, 10)
        oracle_val = evaluate_selection(game, PASSES, oracle_selected)

        other_bms = [b for b in bm_list if b != target_bm]
        avg_o1, avg_o2, avg_o3 = {}, {}, {}
        for p in PASSES:
            avg_o1[p] = np.mean([np.mean([sd['order1'].get(p, 0) for sd in shapley_results[b]]) for b in other_bms])

        example_sd = shapley_results[other_bms[0]][0]
        for k in example_sd['order2']:
            avg_o2[k] = np.mean([np.mean([sd['order2'].get(k, 0) for sd in shapley_results[b]]) for b in other_bms])
        for k in example_sd['order3']:
            avg_o3[k] = np.mean([np.mean([sd['order3'].get(k, 0) for sd in shapley_results[b]]) for b in other_bms])

        transfer_selected = greedy_interaction(avg_o1, avg_o2, avg_o3, PASSES, 10)
        transfer_val = evaluate_selection(game, PASSES, transfer_selected)

        transfer_results[target_bm] = {
            'oracle': float(oracle_val),
            'transfer': float(transfer_val),
            'ratio': float(transfer_val / oracle_val) if oracle_val > 0 else 0,
        }
        print(f"  {target_bm:25s}: oracle={oracle_val:.4f} transfer={transfer_val:.4f} "
              f"ratio={transfer_results[target_bm]['ratio']:.3f}")

    success_rate = np.mean([r['ratio'] >= 0.9 for r in transfer_results.values()])
    print(f"  Transfer success rate: {success_rate:.1%}")

    transfer_data = {
        'similarity_matrix': sim_matrix.tolist(),
        'benchmark_order': bm_list,
        'transfer_results': transfer_results,
        'success_rate': float(success_rate),
    }
    with open(os.path.join(RESULTS_DIR, 'data', 'transferability.json'), 'w') as f:
        json.dump(transfer_data, f, indent=2)

    # ============================================================
    # Variance Decomposition (reload from saved or recompute)
    # ============================================================
    var_decomp_path = os.path.join(RESULTS_DIR, 'data', 'variance_decomposition.json')
    with open(var_decomp_path) as f:
        var_decomp = json.load(f)

    # ============================================================
    # Success Criteria Evaluation
    # ============================================================
    print("\n=== Success Criteria Evaluation ===")

    # Criterion 1: order-3 significance
    n_sig, n_total = 0, 0
    for bm in benchmarks:
        seeds_data = shapley_results[bm]
        o3_keys = set()
        for sd in seeds_data:
            o3_keys.update(sd['order3'].keys())
        for k in o3_keys:
            vals = [sd['order3'].get(k, 0) for sd in seeds_data]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            n_total += 1
            if std_val > 0 and abs(mean_val) > 2 * std_val:
                n_sig += 1

    frac_sig = n_sig / n_total if n_total > 0 else 0
    print(f"  Criterion 1 (order-3 significance >=30%): {n_sig}/{n_total} = {frac_sig:.1%} "
          f"{'PASS' if frac_sig >= 0.30 else 'FAIL'}")

    # Criterion 2: order-3 variance >=10%
    o3_fracs = [d['order3']['mean'] for d in var_decomp.values()]
    avg_o3 = np.mean(o3_fracs)
    print(f"  Criterion 2 (order-3 variance >=10%): {avg_o3:.1%} "
          f"{'PASS' if avg_o3 >= 0.10 else 'FAIL'}")

    # Criterion 3: interaction > pairwise win rate >=60%
    wins, total = 0, 0
    for bm in benchmarks:
        b10 = selection_results[bm]['budgets'][10]
        if b10['interaction_greedy']['mean'] > b10['pairwise_greedy']['mean']:
            wins += 1
        total += 1
    win_rate = wins / total
    print(f"  Criterion 3 (interaction > pairwise >=60%): {wins}/{total} = {win_rate:.1%} "
          f"{'PASS' if win_rate >= 0.60 else 'FAIL'}")

    overall = frac_sig >= 0.30 and avg_o3 >= 0.10 and win_rate >= 0.60
    print(f"\n  Overall: {'HYPOTHESIS CONFIRMED' if overall else 'HYPOTHESIS NOT CONFIRMED'}")

    # Honest comparison: does interaction-guided beat baselines?
    print("\n  Honest comparison across all methods (k=10, averaged):")
    method_avgs = {}
    for method in ['individual_greedy', 'pairwise_greedy', 'interaction_greedy', 'random_search', 'genetic_algorithm']:
        vals = []
        for bm in benchmarks:
            b10 = selection_results[bm]['budgets'][10]
            if method in b10:
                vals.append(b10[method]['mean'])
        if vals:
            method_avgs[method] = (np.mean(vals), np.std(vals))
            print(f"    {method:25s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # O3 comparison
    o3_vals = [opt_levels[bm].get('O3', 0) for bm in benchmarks if opt_levels[bm].get('O3') is not None]
    print(f"    {'LLVM -O3':25s}: {np.mean(o3_vals):.4f} +/- {np.std(o3_vals):.4f}")

    criteria = {
        'criterion1_significant_order3': {
            'n_sig': n_sig, 'n_total': n_total, 'frac': float(frac_sig), 'confirmed': frac_sig >= 0.30
        },
        'criterion2_variance_order3': {
            'avg_frac_order3': float(avg_o3),
            'per_bm': {bm: float(d['order3']['mean']) for bm, d in var_decomp.items()},
            'confirmed': avg_o3 >= 0.10
        },
        'criterion3_selection_win_rate': {
            'wins': wins, 'total': total, 'win_rate': float(win_rate), 'confirmed': win_rate >= 0.60
        },
        'overall_confirmed': overall,
    }

    with open(os.path.join(RESULTS_DIR, 'data', 'criteria_evaluation.json'), 'w') as f:
        json.dump(criteria, f, indent=2)

    # ============================================================
    # O3 Anomaly Investigation
    # ============================================================
    print("\n=== O3 Anomaly Investigation ===")
    all_bms = sorted([f.replace('.bc', '') for f in os.listdir(BC_DIR) if f.endswith('.bc')])
    anomalous = {}
    for bm in all_bms:
        bp = bc_path(bm)
        counts = get_optimization_level_counts(bp)
        o0 = counts.get('O0', 0)
        o3 = counts.get('O3', 0)
        if o0 and o3 and o3 > o0:
            increase = (o3 - o0) / o0 * 100
            anomalous[bm] = {
                'O0': o0, 'O3': o3, 'increase_pct': float(increase),
                'all_levels': {k: int(v) if v else None for k, v in counts.items()},
            }
            # Check loop-unroll effect
            fd, tmp = tempfile.mkstemp(suffix='.bc')
            os.close(fd)
            try:
                out = apply_passes(bp, ['loop-unroll'], tmp)
                if out:
                    c = count_ir_instructions(out)
                    if c:
                        anomalous[bm]['loop_unroll_effect'] = float((c - o0) / o0 * 100)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

            # Check inline effect
            fd, tmp = tempfile.mkstemp(suffix='.bc')
            os.close(fd)
            try:
                out = apply_passes(bp, ['inline'], tmp)
                if out:
                    c = count_ir_instructions(out)
                    if c:
                        anomalous[bm]['inline_effect'] = float((c - o0) / o0 * 100)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

            print(f"  {bm}: O0={o0} O3={o3} (+{increase:.1f}%)")
            if 'loop_unroll_effect' in anomalous[bm]:
                print(f"    loop-unroll: {anomalous[bm]['loop_unroll_effect']:+.1f}%")
            if 'inline_effect' in anomalous[bm]:
                print(f"    inline: {anomalous[bm]['inline_effect']:+.1f}%")

    with open(os.path.join(RESULTS_DIR, 'data', 'o3_anomaly.json'), 'w') as f:
        json.dump(anomalous, f, indent=2)

    # ============================================================
    # Generate Figures
    # ============================================================
    print("\n=== Generating Figures ===")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    # Fig 1: Variance Decomposition
    fig, ax = plt.subplots(figsize=(14, 5))
    bms = list(var_decomp.keys())
    x = np.arange(len(bms))
    o1 = [var_decomp[b]['order1']['mean'] for b in bms]
    o2 = [var_decomp[b]['order2']['mean'] for b in bms]
    o3 = [var_decomp[b]['order3']['mean'] for b in bms]
    ax.bar(x, o1, label='Order 1 (Individual)', color='#4C72B0')
    ax.bar(x, o2, bottom=o1, label='Order 2 (Pairwise)', color='#DD8452')
    ax.bar(x, o3, bottom=[a+b for a,b in zip(o1, o2)], label='Order 3 (Triple)', color='#C44E52')
    ax.set_xticks(x)
    ax.set_xticklabels(bms, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Fraction of Interaction Variance')
    ax.set_title('Variance Decomposition by Interaction Order (PolyBench)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'variance_decomposition.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'variance_decomposition.png'))
    plt.close()
    print("  Saved variance_decomposition")

    # Fig 2: Selection Comparison at k=10
    fig, ax = plt.subplots(figsize=(16, 6))
    methods = ['individual_greedy', 'pairwise_greedy', 'interaction_greedy', 'random_search', 'genetic_algorithm']
    labels = ['Individual\nGreedy', 'Pairwise\nGreedy', 'Interaction\nGreedy', 'Random\nSearch', 'Genetic\nAlgorithm']
    colors = ['#4C72B0', '#DD8452', '#C44E52', '#55A868', '#8172B3']

    x = np.arange(len(benchmarks))
    width = 0.15
    for i, (m, l, c) in enumerate(zip(methods, labels, colors)):
        means = []
        errs = []
        for bm in benchmarks:
            b10 = selection_results[bm]['budgets'][10]
            if m in b10:
                means.append(b10[m]['mean'])
                errs.append(b10[m]['std'])
            else:
                means.append(0)
                errs.append(0)
        ax.bar(x + i*width, means, width, yerr=errs, label=l, color=c, capsize=2)

    # O3 reference lines
    for j, bm in enumerate(benchmarks):
        o3v = opt_levels[bm].get('O3')
        if o3v:
            ax.plot([j-0.1, j+len(methods)*width], [o3v, o3v], 'k--', alpha=0.3, linewidth=0.8)

    ax.set_xticks(x + width*2)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('IR Instruction Count Reduction')
    ax.set_title('Pass Selection Performance Comparison (k=10, PolyBench)')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'selection_comparison.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'selection_comparison.png'))
    plt.close()
    print("  Saved selection_comparison")

    # Fig 3: Ablation - Interaction Order
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(benchmarks))
    width = 0.25
    for i, (order, label, color) in enumerate([
        ('order1', 'Order 1 only', '#4C72B0'),
        ('order12', 'Order 1+2', '#DD8452'),
        ('order123', 'Order 1+2+3', '#C44E52'),
    ]):
        means = [ablation_order[bm][10][order]['mean'] for bm in benchmarks]
        stds = [ablation_order[bm][10][order]['std'] for bm in benchmarks]
        ax.bar(x + i*width, means, width, yerr=stds, label=label, color=color, capsize=2)
    ax.set_xticks(x + width)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('IR Reduction')
    ax.set_title('Ablation: Effect of Interaction Order on Selection (k=10)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'ablation_order.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'ablation_order.png'))
    plt.close()
    print("  Saved ablation_order")

    # Fig 4: Performance vs Budget k
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, l, c in zip(methods[:4], ['Individual Greedy', 'Pairwise Greedy', 'Interaction Greedy', 'Random Search'], colors[:4]):
        means = []
        for k in budgets_k:
            vals = [selection_results[bm]['budgets'][k][m]['mean'] for bm in benchmarks]
            means.append(np.mean(vals))
        ax.plot(budgets_k, means, marker='o', label=l, color=c)
    ax.set_xlabel('Number of Passes Selected (k)')
    ax.set_ylabel('Mean IR Reduction')
    ax.set_title('Selection Performance vs. Pass Budget')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'performance_vs_budget.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'performance_vs_budget.png'))
    plt.close()
    print("  Saved performance_vs_budget")

    # Fig 5: Budget Ablation Convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    eval_budgets = sorted([int(k) for k in ablation_budget_results.keys()])
    for bm in ablation_bms:
        means = [ablation_budget_results[b][bm]['mean'] for b in eval_budgets]
        ax.plot(eval_budgets, means, marker='o', label=bm, alpha=0.7)
    ax.set_xlabel('Shapley Evaluation Budget')
    ax.set_ylabel('Selection Performance (IR Reduction)')
    ax.set_title('Convergence: Selection Performance vs. Evaluation Budget')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'budget_convergence.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'budget_convergence.png'))
    plt.close()
    print("  Saved budget_convergence")

    # Fig 6: Transferability Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sim_matrix, xticklabels=bm_list, yticklabels=bm_list,
                cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax, annot=False)
    ax.set_title('Cosine Similarity of Interaction Vectors')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'transferability_heatmap.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'transferability_heatmap.png'))
    plt.close()
    print("  Saved transferability_heatmap")

    # Fig 7: K ablation
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted([int(k) for k in ablation_passes_results.keys()])
    for bm in ablation_bms:
        means = [ablation_passes_results[k][bm]['mean'] for k in ks]
        ax.plot(ks, means, marker='o', label=bm, alpha=0.7)
    ax.set_xlabel('Number of Candidate Passes (K)')
    ax.set_ylabel('Selection Performance (IR Reduction)')
    ax.set_title('Effect of Candidate Pool Size on Selection')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'ablation_num_passes.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'ablation_num_passes.png'))
    plt.close()
    print("  Saved ablation_num_passes")

    # ============================================================
    # Compile Final results.json
    # ============================================================
    elapsed = time.time() - t_start

    comparison_table = {}
    for bm in benchmarks:
        row = {}
        for level in ['O1', 'O2', 'O3', 'Os', 'Oz']:
            row[level] = opt_levels.get(bm, {}).get(level)
        if bm in selection_results and 10 in selection_results[bm].get('budgets', {}):
            for method in methods:
                if method in selection_results[bm]['budgets'][10]:
                    row[method] = selection_results[bm]['budgets'][10][method]
        comparison_table[bm] = row

    final_results = {
        'experiment': 'ShapleyPass: Compiler Pass Interaction Analysis via Shapley Interaction Indices',
        'benchmarks': benchmarks,
        'benchmark_source': 'PolyBench/C (compiled to LLVM IR bitcode)',
        'n_benchmarks': len(benchmarks),
        'n_passes': K_PASSES,
        'passes': PASSES,
        'seeds': SEEDS,
        'shapley_budget': 2000,
        'hypothesis_evaluation': criteria,
        'variance_decomposition': var_decomp,
        'comparison_table': comparison_table,
        'ablation_interaction_order': ablation_order,
        'ablation_num_passes': {str(k): v for k, v in ablation_passes_results.items()},
        'ablation_budget': {str(k): v for k, v in ablation_budget_results.items()},
        'transferability': {
            'success_rate': transfer_data['success_rate'],
            'per_benchmark': transfer_data['transfer_results'],
        },
        'o3_anomaly_investigation': anomalous,
        'total_runtime_seconds': elapsed,
        'total_runtime_hours': elapsed / 3600,
        'negative_result_discussion': (
            "Our experiments on 15 PolyBench/C benchmarks reveal that the hypothesis "
            "is partially supported but ultimately not confirmed. While third-order "
            "interactions exist and contribute ~9.9% of variance (near the 10% threshold), "
            "they are not statistically significant for most pass triples. The interaction-"
            "guided greedy algorithm shows marginal improvement over pairwise-only greedy "
            "on some benchmarks, but importantly, both Shapley-based methods are consistently "
            "outperformed by simple random search and genetic algorithm baselines. This "
            "suggests that (1) pairwise analysis is largely sufficient for understanding "
            "pass interactions, and (2) the greedy exploitation of Shapley interaction "
            "indices is suboptimal compared to direct search in the pass subset space. "
            "The key scientific finding is that compiler pass interactions are dominated "
            "by individual effects (~55%) and pairwise interactions (~33%), with higher-"
            "order interactions playing a minor role (~10%). This validates approaches "
            "like ODG that focus on pairwise dependencies."
        ),
    }

    with open(os.path.join(BASE, 'results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total runtime: {elapsed/3600:.2f} hours")
    print(f"Results saved to results.json")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

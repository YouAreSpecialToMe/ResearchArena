#!/usr/bin/env python3
"""
Complete remaining experiment steps using saved Shapley interaction data.
Baselines, ablations, figures, and final results.
"""
import sys, os, json, time, itertools
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
from game import CompilerGame, CANDIDATE_PASSES, count_ir_instructions, get_optimization_level_counts

BENCHMARKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmarks')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
SEEDS = [42, 123, 456]
PASS_LIST = CANDIDATE_PASSES
K_BUDGETS = [5, 8, 10, 12, 15]

os.makedirs(os.path.join(RESULTS_DIR, 'data'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def get_benchmark_files():
    bc_files = {}
    for f in sorted(os.listdir(BENCHMARKS_DIR)):
        if f.endswith('.bc'):
            bc_files[f.replace('.bc', '')] = os.path.join(BENCHMARKS_DIR, f)
    return bc_files


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def aggregate_interactions(all_interactions):
    aggregated = {}
    for bname, seed_results in all_interactions.items():
        agg = {'order1': {}, 'order2': {}, 'order3': {}}
        for order_key in ['order1', 'order2', 'order3']:
            all_keys = set()
            for seed in SEEDS:
                seed_str = str(seed)
                if seed_str in seed_results:
                    all_keys.update(seed_results[seed_str][order_key].keys())
            for key in all_keys:
                vals = []
                for seed in SEEDS:
                    seed_str = str(seed)
                    if seed_str in seed_results:
                        vals.append(seed_results[seed_str][order_key].get(key, 0.0))
                agg[order_key][key] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'values': vals,
                }
        aggregated[bname] = agg
    return aggregated


# Selection algorithms
def _pair_key(a, b):
    return f"{min(a,b)}+{max(a,b)}"

def _triple_key(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}+{s[1]}+{s[2]}"

def greedy_individual(phi1, k):
    sorted_passes = sorted(phi1.items(), key=lambda x: x[1], reverse=True)
    return [p for p, _ in sorted_passes[:k]]

def greedy_pairwise(phi1, phi2, k, pass_list):
    selected = []
    remaining = set(range(len(pass_list)))
    for _ in range(k):
        best_score, best_idx = -float('inf'), None
        for i in remaining:
            score = phi1.get(pass_list[i], 0)
            for j_idx in selected:
                score += phi2.get(_pair_key(pass_list[i], pass_list[j_idx]), 0)
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
    return [pass_list[i] for i in selected]

def greedy_interaction(phi1, phi2, phi3, k, pass_list):
    selected = []
    remaining = set(range(len(pass_list)))
    for _ in range(k):
        best_score, best_idx = -float('inf'), None
        for i in remaining:
            score = phi1.get(pass_list[i], 0)
            for j_idx in selected:
                score += phi2.get(_pair_key(pass_list[i], pass_list[j_idx]), 0)
            for j_idx, k_idx in itertools.combinations(selected, 2):
                score += phi3.get(_triple_key(pass_list[i], pass_list[j_idx], pass_list[k_idx]), 0)
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
    return [pass_list[i] for i in selected]

def synergy_seeded(phi1, phi2, phi3, k, pass_list):
    best_triple, best_val = None, -float('inf')
    for key, val in phi3.items():
        if val > best_val:
            best_val, best_triple = val, key
    selected = []
    if best_triple:
        for p in best_triple.split('+'):
            if p in pass_list:
                idx = pass_list.index(p)
                if idx not in selected:
                    selected.append(idx)
    remaining = set(range(len(pass_list))) - set(selected)
    while len(selected) < k and remaining:
        best_score, best_idx = -float('inf'), None
        for i in remaining:
            score = phi1.get(pass_list[i], 0)
            for j_idx in selected:
                score += phi2.get(_pair_key(pass_list[i], pass_list[j_idx]), 0)
            for j_idx, k_idx in itertools.combinations(selected, 2):
                score += phi3.get(_triple_key(pass_list[i], pass_list[j_idx], pass_list[k_idx]), 0)
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
    return [pass_list[i] for i in selected[:k]]

def evaluate_selection(benchmark_path, selected_passes, all_passes=PASS_LIST):
    game = CompilerGame(benchmark_path, passes=all_passes)
    x = np.zeros(len(all_passes))
    for p in selected_passes:
        if p in all_passes:
            x[all_passes.index(p)] = 1
    return float(game.value(x))


def run_baselines(benchmarks):
    """Run efficient baselines: reduced random search + GA."""
    print("\n=== Baselines ===")
    baseline_results = {}

    for bname, bpath in benchmarks.items():
        print(f"  {bname}...", end=" ", flush=True)
        game = CompilerGame(bpath)
        n = game.n_players
        bm_baselines = {}

        for k in K_BUDGETS:
            k_results = {}
            # Random search: 200 samples × 3 seeds
            rs_vals = []
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                best = -float('inf')
                for _ in range(200):
                    indices = rng.choice(n, k, replace=False)
                    x = np.zeros(n)
                    x[indices] = 1
                    v = game.value(x)
                    if v > best:
                        best = v
                rs_vals.append(best)
            k_results['random_search'] = {
                'mean': float(np.mean(rs_vals)),
                'std': float(np.std(rs_vals)),
                'values': [float(v) for v in rs_vals],
            }

            # GA: pop=20, gen=10
            ga_vals = []
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                pop_size, n_gen = 20, 10
                pop = np.zeros((pop_size, n))
                for i in range(pop_size):
                    indices = rng.choice(n, k, replace=False)
                    pop[i, indices] = 1
                fitness = np.array([game.value(ind) for ind in pop])
                for gen in range(n_gen):
                    new_pop = np.zeros_like(pop)
                    for i in range(pop_size):
                        cands = rng.choice(pop_size, 3, replace=False)
                        new_pop[i] = pop[cands[np.argmax(fitness[cands])]].copy()
                    for i in range(0, pop_size - 1, 2):
                        if rng.random() < 0.7:
                            pt = rng.randint(1, n)
                            c1 = np.concatenate([new_pop[i, :pt], new_pop[i+1, pt:]])
                            c2 = np.concatenate([new_pop[i+1, :pt], new_pop[i, pt:]])
                            new_pop[i], new_pop[i+1] = c1, c2
                    for i in range(pop_size):
                        for j in range(n):
                            if rng.random() < 0.1:
                                new_pop[i, j] = 1 - new_pop[i, j]
                    pop = new_pop
                    fitness = np.array([game.value(ind) for ind in pop])
                ga_vals.append(float(np.max(fitness)))
            k_results['genetic_algorithm'] = {
                'mean': float(np.mean(ga_vals)),
                'std': float(np.std(ga_vals)),
                'values': [float(v) for v in ga_vals],
            }
            bm_baselines[k] = k_results

        baseline_results[bname] = bm_baselines
        k10 = bm_baselines[10]
        print(f"RS={k10['random_search']['mean']:.4f}, GA={k10['genetic_algorithm']['mean']:.4f}")

    save_json(baseline_results, os.path.join(RESULTS_DIR, 'data', 'baseline_results.json'))
    return baseline_results


def run_ablation_order(benchmarks, agg):
    """Ablation: interaction order."""
    print("\n=== Ablation: Interaction Order ===")
    results = {}
    for bname, bpath in benchmarks.items():
        bm_agg = agg[bname]
        phi1 = {k: v['mean'] for k, v in bm_agg['order1'].items()}
        phi2 = {k: v['mean'] for k, v in bm_agg['order2'].items()}
        phi3 = {k: v['mean'] for k, v in bm_agg['order3'].items()}
        bm_results = {}
        for k in K_BUDGETS:
            v1 = evaluate_selection(bpath, greedy_individual(phi1, k))
            v12 = evaluate_selection(bpath, greedy_pairwise(phi1, phi2, k, PASS_LIST))
            v123 = evaluate_selection(bpath, greedy_interaction(phi1, phi2, phi3, k, PASS_LIST))
            bm_results[k] = {
                'order1': float(v1), 'order1_2': float(v12), 'order1_2_3': float(v123),
                'improvement_2_over_1': float(v12 - v1),
                'improvement_3_over_12': float(v123 - v12),
            }
        results[bname] = bm_results
        print(f"  {bname} (k=10): o1={bm_results[10]['order1']:.4f}, o1+2={bm_results[10]['order1_2']:.4f}, o1+2+3={bm_results[10]['order1_2_3']:.4f}")
    save_json(results, os.path.join(RESULTS_DIR, 'data', 'ablation_order.json'))
    return results


def run_ablation_budget(benchmarks):
    """Ablation: evaluation budget for Shapley estimation."""
    print("\n=== Ablation: Evaluation Budget ===")
    import shapiq
    bm_names = list(benchmarks.keys())[:5]
    budgets = [500, 1000, 2000, 3000]
    results = {}

    for bname in bm_names:
        bpath = benchmarks[bname]
        bm_results = {}
        for budget in budgets:
            game = CompilerGame(bpath)
            t0 = time.time()
            approx = shapiq.PermutationSamplingSII(n=game.n_players, max_order=3, random_state=42)
            iv = approx.approximate(budget=budget, game=game)
            elapsed = time.time() - t0
            phi1, phi2, phi3 = {}, {}, {}
            for idx, val in iv.dict_values.items():
                if len(idx) == 1:
                    phi1[PASS_LIST[idx[0]]] = float(val)
                elif len(idx) == 2:
                    phi2[f"{PASS_LIST[idx[0]]}+{PASS_LIST[idx[1]]}"] = float(val)
                elif len(idx) == 3:
                    phi3[f"{PASS_LIST[idx[0]]}+{PASS_LIST[idx[1]]}+{PASS_LIST[idx[2]]}"] = float(val)
            sel = greedy_interaction(phi1, phi2, phi3, 10, PASS_LIST)
            v = evaluate_selection(bpath, sel)
            bm_results[budget] = {'selection_value': float(v), 'elapsed': float(elapsed)}
        results[bname] = bm_results
        print(f"  {bname}: " + ", ".join(f"B={b}: {bm_results[b]['selection_value']:.4f}" for b in budgets))

    save_json(results, os.path.join(RESULTS_DIR, 'data', 'ablation_budget.json'))
    return results


def run_transferability(benchmarks, agg):
    """Transferability analysis."""
    print("\n=== Transferability Analysis ===")
    bm_names = list(benchmarks.keys())
    n_bm = len(bm_names)

    # Build interaction vectors
    vectors = {}
    for bname in bm_names:
        bm_agg = agg[bname]
        vec = []
        for k in sorted(bm_agg['order1'].keys()):
            vec.append(bm_agg['order1'][k]['mean'])
        for k in sorted(bm_agg['order2'].keys()):
            vec.append(bm_agg['order2'][k]['mean'])
        vectors[bname] = np.array(vec)

    # Cosine similarity
    sim_matrix = np.zeros((n_bm, n_bm))
    for i, b1 in enumerate(bm_names):
        for j, b2 in enumerate(bm_names):
            v1, v2 = vectors[b1], vectors[b2]
            min_len = min(len(v1), len(v2))
            v1, v2 = v1[:min_len], v2[:min_len]
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            sim_matrix[i, j] = np.dot(v1, v2) / norm if norm > 0 else 0

    results = {}
    for i, bname in enumerate(bm_names):
        bpath = benchmarks[bname]
        own_phi1 = {k: v['mean'] for k, v in agg[bname]['order1'].items()}
        own_phi2 = {k: v['mean'] for k, v in agg[bname]['order2'].items()}
        own_phi3 = {k: v['mean'] for k, v in agg[bname]['order3'].items()}
        sel_oracle = greedy_interaction(own_phi1, own_phi2, own_phi3, 10, PASS_LIST)
        v_oracle = evaluate_selection(bpath, sel_oracle)

        avg_phi1, avg_phi2, avg_phi3 = defaultdict(float), defaultdict(float), defaultdict(float)
        for j, other in enumerate(bm_names):
            if other == bname: continue
            for k, v in agg[other]['order1'].items(): avg_phi1[k] += v['mean'] / (n_bm - 1)
            for k, v in agg[other]['order2'].items(): avg_phi2[k] += v['mean'] / (n_bm - 1)
            for k, v in agg[other]['order3'].items(): avg_phi3[k] += v['mean'] / (n_bm - 1)
        sel_transfer = greedy_interaction(dict(avg_phi1), dict(avg_phi2), dict(avg_phi3), 10, PASS_LIST)
        v_transfer = evaluate_selection(bpath, sel_transfer)

        sims = sim_matrix[i].copy(); sims[i] = -1
        ms_idx = np.argmax(sims)
        ms_name = bm_names[ms_idx]
        ms_phi1 = {k: v['mean'] for k, v in agg[ms_name]['order1'].items()}
        ms_phi2 = {k: v['mean'] for k, v in agg[ms_name]['order2'].items()}
        ms_phi3 = {k: v['mean'] for k, v in agg[ms_name]['order3'].items()}
        sel_sim = greedy_interaction(ms_phi1, ms_phi2, ms_phi3, 10, PASS_LIST)
        v_similar = evaluate_selection(bpath, sel_sim)

        results[bname] = {
            'oracle': float(v_oracle), 'transfer_avg': float(v_transfer),
            'transfer_similar': float(v_similar), 'most_similar_benchmark': ms_name,
            'similarity': float(sims[ms_idx]),
            'transfer_ratio_avg': float(v_transfer / v_oracle) if v_oracle > 0 else 0,
            'transfer_ratio_similar': float(v_similar / v_oracle) if v_oracle > 0 else 0,
        }
        print(f"  {bname}: oracle={v_oracle:.4f}, transfer={v_transfer:.4f}, similar({ms_name})={v_similar:.4f}")

    results['_similarity_matrix'] = {'benchmarks': bm_names, 'matrix': sim_matrix.tolist()}
    save_json(results, os.path.join(RESULTS_DIR, 'data', 'transferability.json'))
    return results


def statistical_evaluation(all_interactions, decomp, selection_results, baseline_results, benchmarks, opt_levels):
    """Evaluate against success criteria."""
    print("\n=== Statistical Evaluation ===")
    agg = aggregate_interactions(all_interactions)
    bm_names = list(benchmarks.keys())

    # Criterion 1: significant order-3 interactions
    n_sig, n_total = 0, 0
    for bname in bm_names:
        for key, vals in agg[bname]['order3'].items():
            n_total += 1
            if vals['std'] > 0 and abs(vals['mean']) > 2 * vals['std']:
                n_sig += 1
            elif vals['std'] == 0 and abs(vals['mean']) > 1e-6:
                n_sig += 1
    frac_sig = n_sig / n_total if n_total > 0 else 0
    c1 = frac_sig >= 0.30
    print(f"  Criterion 1: {n_sig}/{n_total} ({frac_sig:.1%}) significant order-3 {'CONFIRMED' if c1 else 'NOT CONFIRMED'}")

    # Criterion 2: variance explained
    frac3 = [decomp[b]['frac_order3'] for b in bm_names]
    avg_f3 = np.mean(frac3)
    c2 = avg_f3 >= 0.10
    print(f"  Criterion 2: Order-3 variance = {avg_f3:.1%} {'CONFIRMED' if c2 else 'NOT CONFIRMED'}")

    # Criterion 3: interaction vs pairwise win rate
    wins, total = 0, 0
    for bname in bm_names:
        for k_key in [10, '10']:
            if k_key in selection_results.get(bname, {}):
                v_pair = selection_results[bname][k_key]['pairwise_greedy']['value']
                v_inter = selection_results[bname][k_key]['interaction_greedy']['value']
                total += 1
                if v_inter >= v_pair: wins += 1
                break
    wr = wins / total if total > 0 else 0
    c3 = wr >= 0.60
    print(f"  Criterion 3: Win rate = {wins}/{total} ({wr:.1%}) {'CONFIRMED' if c3 else 'NOT CONFIRMED'}")

    # Main comparison table
    main_table = {}
    methods = ['O1', 'O2', 'O3', 'Os', 'Oz', 'random_search', 'genetic_algorithm',
               'individual_greedy', 'pairwise_greedy', 'interaction_greedy', 'synergy_seeded']
    for bname in bm_names:
        levels = opt_levels[bname]
        baseline = levels['O0']
        row = {}
        for lvl in ['O1', 'O2', 'O3', 'Os', 'Oz']:
            if levels.get(lvl): row[lvl] = float((baseline - levels[lvl]) / baseline)
        for k_key in [10, '10']:
            if k_key in selection_results.get(bname, {}):
                for m in ['individual_greedy', 'pairwise_greedy', 'interaction_greedy', 'synergy_seeded']:
                    row[m] = selection_results[bname][k_key][m]['value']
                break
        for k_key in [10, '10']:
            if k_key in baseline_results.get(bname, {}):
                row['random_search'] = baseline_results[bname][k_key]['random_search']['mean']
                row['genetic_algorithm'] = baseline_results[bname][k_key]['genetic_algorithm']['mean']
                break
        main_table[bname] = row

    avg_row = {}
    for m in methods:
        vals = [main_table[b].get(m) for b in bm_names if main_table[b].get(m) is not None]
        if vals: avg_row[m] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    main_table['_average'] = avg_row

    eval_results = {
        'criterion1': {'n_sig': n_sig, 'n_total': n_total, 'frac': float(frac_sig), 'confirmed': c1},
        'criterion2': {'avg_frac_order3': float(avg_f3), 'per_bm': {b: float(v) for b, v in zip(bm_names, frac3)}, 'confirmed': c2},
        'criterion3': {'wins': wins, 'total': total, 'win_rate': float(wr), 'confirmed': c3},
        'main_table': main_table,
        'overall_confirmed': c1 and c2 and c3,
    }
    save_json(eval_results, os.path.join(RESULTS_DIR, 'data', 'statistical_tests.json'))

    # CSV
    import csv
    csv_path = os.path.join(RESULTS_DIR, 'tables', 'main_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Benchmark'] + methods)
        for bname in bm_names:
            row = [bname] + [f"{main_table[bname].get(m, 0):.4f}" for m in methods]
            writer.writerow(row)
        row = ['Average'] + [f"{avg_row.get(m, {}).get('mean', 0):.4f}+/-{avg_row.get(m, {}).get('std', 0):.4f}" for m in methods]
        writer.writerow(row)
    print(f"  Saved {csv_path}")
    return eval_results


def generate_figures(all_interactions, decomp, selection_results, baseline_results,
                     benchmarks, opt_levels, transfer_results, ablation_order, ablation_budget_res):
    """Generate all figures."""
    print("\n=== Generating Figures ===")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'figure.dpi': 150,
                          'savefig.bbox': 'tight', 'savefig.dpi': 300})

    agg = aggregate_interactions(all_interactions)
    bm_names = list(benchmarks.keys())

    # Figure 1: Variance Decomposition
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bm_names))
    f1 = [decomp[b]['frac_order1'] for b in bm_names]
    f2 = [decomp[b]['frac_order2'] for b in bm_names]
    f3 = [decomp[b]['frac_order3'] for b in bm_names]
    ax.bar(x, f1, label='Order 1 (Individual)', color='#4C72B0')
    ax.bar(x, f2, bottom=f1, label='Order 2 (Pairwise)', color='#DD8452')
    ax.bar(x, f3, bottom=[a+b for a, b in zip(f1, f2)], label='Order 3 (Triple)', color='#C44E52')
    ax.set_xticks(x); ax.set_xticklabels(bm_names, rotation=45, ha='right')
    ax.set_ylabel('Fraction of Interaction Variance')
    ax.set_title('Variance Decomposition by Interaction Order')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'variance_decomposition.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'variance_decomposition.png')); plt.close()
    print("  variance_decomposition")

    # Figure 2: Interaction Heatmaps (2 benchmarks)
    for bname in bm_names[:2]:
        phi2 = agg[bname]['order2']
        n_p = len(PASS_LIST)
        hm = np.zeros((n_p, n_p))
        for key, v in phi2.items():
            parts = key.split('+')
            if len(parts) == 2:
                i = PASS_LIST.index(parts[0]) if parts[0] in PASS_LIST else None
                j = PASS_LIST.index(parts[1]) if parts[1] in PASS_LIST else None
                if i is not None and j is not None:
                    hm[i, j] = hm[j, i] = v['mean']
        fig, ax = plt.subplots(figsize=(12, 10))
        vmax = max(abs(hm.min()), abs(hm.max())) or 0.01
        sns.heatmap(hm, xticklabels=PASS_LIST, yticklabels=PASS_LIST, cmap='RdBu_r',
                     center=0, vmin=-vmax, vmax=vmax, ax=ax, square=True, linewidths=0.5)
        ax.set_title(f'Pairwise Shapley Interactions: {bname}')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'heatmap_{bname}.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, f'heatmap_{bname}.png')); plt.close()
    print("  heatmaps")

    # Figure 3: Top Order-3 Interactions
    all_o3 = []
    for bname in bm_names:
        for key, v in agg[bname]['order3'].items():
            all_o3.append((key, bname, v['mean'], v['std']))
    all_o3.sort(key=lambda x: abs(x[2]), reverse=True)
    top_o3 = all_o3[:15]
    if top_o3:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [f"{t[0]}\n({t[1]})" for t in top_o3]
        values = [t[2] for t in top_o3]
        stds = [t[3] for t in top_o3]
        colors = ['#C44E52' if v > 0 else '#4C72B0' for v in values]
        y = np.arange(len(top_o3))
        ax.barh(y, values, xerr=stds, color=colors, alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Shapley Interaction Index')
        ax.set_title('Top 15 Order-3 Interactions'); ax.axvline(x=0, color='black', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'top_order3_interactions.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, 'top_order3_interactions.png')); plt.close()
    print("  top_order3_interactions")

    # Figure 4: Selection Performance Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # 4a: Bar chart at k=10
    ax = axes[0]
    mtp = ['O3', 'random_search', 'genetic_algorithm', 'individual_greedy',
           'pairwise_greedy', 'interaction_greedy', 'synergy_seeded']
    mlabels = ['O3', 'Random\nSearch', 'GA', 'Indiv.\nGreedy', 'Pairwise\nGreedy', 'Interact.\nGreedy', 'Synergy\nSeeded']
    avgs, stds = [], []
    for m in mtp:
        vals = []
        for bname in bm_names:
            if m in ['O1', 'O2', 'O3', 'Os', 'Oz']:
                lvls = opt_levels[bname]
                if lvls.get(m): vals.append((lvls['O0'] - lvls[m]) / lvls['O0'])
            elif m in ['random_search', 'genetic_algorithm']:
                for k_key in [10, '10']:
                    if k_key in baseline_results.get(bname, {}):
                        vals.append(baseline_results[bname][k_key][m]['mean']); break
            else:
                for k_key in [10, '10']:
                    if k_key in selection_results.get(bname, {}):
                        vals.append(selection_results[bname][k_key][m]['value']); break
        avgs.append(np.mean(vals) if vals else 0)
        stds.append(np.std(vals) if vals else 0)
    colors = ['#55A868', '#CCB974', '#CCB974', '#4C72B0', '#DD8452', '#C44E52', '#8172B3']
    x = np.arange(len(mtp))
    ax.bar(x, avgs, yerr=stds, color=colors, alpha=0.8, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(mlabels, fontsize=9)
    ax.set_ylabel('IR Reduction Fraction'); ax.set_title('Method Comparison (k=10)')

    # 4b: Performance vs k
    ax = axes[1]
    for method, label, color, ls in [
        ('individual_greedy', 'Individual', '#4C72B0', '-'),
        ('pairwise_greedy', 'Pairwise', '#DD8452', '-'),
        ('interaction_greedy', 'Interaction', '#C44E52', '-'),
        ('synergy_seeded', 'Synergy Seeded', '#8172B3', '--'),
    ]:
        avg_by_k = []
        for k in K_BUDGETS:
            vals = []
            for bname in bm_names:
                for k_key in [k, str(k)]:
                    if k_key in selection_results.get(bname, {}):
                        vals.append(selection_results[bname][k_key][method]['value']); break
            avg_by_k.append(np.mean(vals) if vals else 0)
        ax.plot(K_BUDGETS, avg_by_k, marker='o', label=label, color=color, linestyle=ls)
    ax.set_xlabel('Number of Selected Passes (k)')
    ax.set_ylabel('IR Reduction Fraction'); ax.set_title('Selection Performance vs. Budget')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'selection_comparison.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'selection_comparison.png')); plt.close()
    print("  selection_comparison")

    # Figure 5: Interaction Network
    import networkx as nx
    for bname in bm_names[:1]:
        phi1 = agg[bname]['order1']
        phi2 = agg[bname]['order2']
        G = nx.Graph()
        for p in PASS_LIST:
            if p in phi1: G.add_node(p, weight=phi1[p]['mean'])
        for key, v in phi2.items():
            parts = key.split('+')
            if len(parts) == 2 and abs(v['mean']) > 0.001:
                G.add_edge(parts[0], parts[1], weight=v['mean'])
        fig, ax = plt.subplots(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42, k=2)
        nsizes = [abs(G.nodes[n].get('weight', 0)) * 5000 + 100 for n in G.nodes]
        ncols = ['#C44E52' if G.nodes[n].get('weight', 0) > 0 else '#4C72B0' for n in G.nodes]
        edges = list(G.edges())
        if edges:
            ew = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges,
                                    width=[abs(w) * 50 + 0.5 for w in ew],
                                    edge_color=['#C44E52' if w > 0 else '#4C72B0' for w in ew],
                                    alpha=0.5, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_size=nsizes, node_color=ncols, alpha=0.7, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        ax.set_title(f'Pass Interaction Network: {bname}'); ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'network_{bname}.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, f'network_{bname}.png')); plt.close()
    print("  network")

    # Figure 6: Transferability
    if '_similarity_matrix' in transfer_results:
        sim = np.array(transfer_results['_similarity_matrix']['matrix'])
        bns = transfer_results['_similarity_matrix']['benchmarks']
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(sim, xticklabels=bns, yticklabels=bns, cmap='YlOrRd',
                     vmin=0, vmax=1, annot=True, fmt='.2f', ax=ax, square=True)
        ax.set_title('Benchmark Similarity (Cosine of Interaction Vectors)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'transferability.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, 'transferability.png')); plt.close()
    print("  transferability")

    # Figure 7: Convergence
    if ablation_budget_res:
        budgets = [500, 1000, 2000, 3000]
        fig, ax = plt.subplots(figsize=(8, 5))
        for bname in list(ablation_budget_res.keys())[:5]:
            vals = []
            for b in budgets:
                for bkey in [b, str(b)]:
                    if bkey in ablation_budget_res[bname]:
                        vals.append(ablation_budget_res[bname][bkey]['selection_value']); break
            if vals: ax.plot(budgets[:len(vals)], vals, marker='o', label=bname)
        ax.set_xlabel('Evaluation Budget'); ax.set_ylabel('Selection Performance (k=10)')
        ax.set_title('Convergence: Selection Performance vs. Budget')
        ax.legend(fontsize=9); plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'convergence.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, 'convergence.png')); plt.close()
    print("  convergence")

    # Figure 8: Ablation - Interaction Order
    if ablation_order:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(bm_names)); width = 0.25
        for i, (okey, lbl, col) in enumerate([
            ('order1', 'Order 1 Only', '#4C72B0'),
            ('order1_2', 'Order 1+2', '#DD8452'),
            ('order1_2_3', 'Order 1+2+3', '#C44E52'),
        ]):
            vals = []
            for bname in bm_names:
                for k_key in [10, '10']:
                    if bname in ablation_order and k_key in ablation_order[bname]:
                        vals.append(ablation_order[bname][k_key][okey]); break
                else:
                    vals.append(0)
            ax.bar(x + i * width, vals, width, label=lbl, color=col, alpha=0.8)
        ax.set_xticks(x + width); ax.set_xticklabels(bm_names, rotation=45, ha='right')
        ax.set_ylabel('IR Reduction Fraction')
        ax.set_title('Ablation: Effect of Interaction Order (k=10)')
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'ablation_order.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, 'ablation_order.png')); plt.close()
    print("  ablation_order")

    print("  All figures done.")


def save_final_results(eval_results, decomp, selection_results, baseline_results,
                       benchmarks, opt_levels, ablation_order):
    """Save final results.json."""
    bm_names = list(benchmarks.keys())
    results = {
        "experiment": "ShapleyPass: Compiler Pass Interaction Analysis via Shapley Interaction Indices",
        "benchmarks": bm_names,
        "n_passes": len(PASS_LIST),
        "passes": PASS_LIST,
        "seeds": SEEDS,
        "shapley_budget": 2000,
        "hypothesis_evaluation": {
            "criterion1_significant_order3": eval_results['criterion1'],
            "criterion2_variance_order3": eval_results['criterion2'],
            "criterion3_selection_win_rate": eval_results['criterion3'],
            "overall_confirmed": eval_results['overall_confirmed'],
        },
        "variance_decomposition": {
            b: {"order1": decomp[b]['frac_order1'], "order2": decomp[b]['frac_order2'],
                "order3": decomp[b]['frac_order3']} for b in bm_names
        },
        "main_comparison": eval_results['main_table'],
        "ablation_interaction_order": {},
    }
    for bname in bm_names:
        if bname in ablation_order:
            for k_key in [10, '10']:
                if k_key in ablation_order[bname]:
                    results["ablation_interaction_order"][bname] = ablation_order[bname][k_key]
                    break

    results_path = os.path.join(os.path.dirname(__file__), '..', 'results.json')
    save_json(results, results_path)
    print(f"\n  Final results saved to {results_path}")


def main():
    t_start = time.time()
    print("ShapleyPass: Completing Remaining Experiments")
    print("=" * 60)

    benchmarks = get_benchmark_files()
    print(f"Benchmarks: {list(benchmarks.keys())}")

    # Load saved data
    all_interactions = load_json(os.path.join(RESULTS_DIR, 'data', 'all_interactions.json'))
    decomp = load_json(os.path.join(RESULTS_DIR, 'data', 'variance_decomposition.json'))
    selection_results = load_json(os.path.join(RESULTS_DIR, 'data', 'selection_results.json'))
    opt_levels = load_json(os.path.join(RESULTS_DIR, 'data', 'opt_levels.json'))
    agg = aggregate_interactions(all_interactions)

    print(f"Loaded: {len(all_interactions)} benchmarks with interactions")
    print(f"Variance decomp avg order-3: {decomp.get('_average', {}).get('frac_order3', 'N/A')}")

    # Run remaining steps
    baseline_results = run_baselines(benchmarks)
    ablation_order = run_ablation_order(benchmarks, agg)
    ablation_budget_res = run_ablation_budget(benchmarks)
    transfer_results = run_transferability(benchmarks, agg)

    # Statistical evaluation
    eval_results = statistical_evaluation(all_interactions, decomp, selection_results,
                                          baseline_results, benchmarks, opt_levels)

    # Figures
    generate_figures(all_interactions, decomp, selection_results, baseline_results,
                     benchmarks, opt_levels, transfer_results, ablation_order, ablation_budget_res)

    # Final results
    save_final_results(eval_results, decomp, selection_results, baseline_results,
                       benchmarks, opt_levels, ablation_order)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()

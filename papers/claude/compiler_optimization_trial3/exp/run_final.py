#!/usr/bin/env python3
"""
Complete all remaining experiments with optimized baselines.
Uses saved Shapley interaction data from prior run.
"""
import sys, os, json, time, itertools, csv
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, bool): return bool(obj)
        return super().default(obj)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder, default=str)


def aggregate_interactions(all_interactions):
    aggregated = {}
    for bname, seed_results in all_interactions.items():
        agg = {'order1': {}, 'order2': {}, 'order3': {}}
        for order_key in ['order1', 'order2', 'order3']:
            all_keys = set()
            for seed in SEEDS:
                s = str(seed)
                if s in seed_results:
                    all_keys.update(seed_results[s][order_key].keys())
            for key in all_keys:
                vals = [seed_results[str(seed)][order_key].get(key, 0.0) for seed in SEEDS if str(seed) in seed_results]
                agg[order_key][key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': vals}
        aggregated[bname] = agg
    return aggregated


# ---- Selection Algorithms ----
def _pair_key(a, b):
    return f"{min(a,b)}+{max(a,b)}"

def _triple_key(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}+{s[1]}+{s[2]}"

def greedy_individual(phi1, k):
    return [p for p, _ in sorted(phi1.items(), key=lambda x: x[1], reverse=True)[:k]]

def greedy_pairwise(phi1, phi2, k, pass_list):
    selected, remaining = [], set(range(len(pass_list)))
    for _ in range(k):
        best_s, best_i = -float('inf'), None
        for i in remaining:
            s = phi1.get(pass_list[i], 0)
            for j in selected:
                s += phi2.get(_pair_key(pass_list[i], pass_list[j]), 0)
            if s > best_s: best_s, best_i = s, i
        if best_i is not None: selected.append(best_i); remaining.remove(best_i)
    return [pass_list[i] for i in selected]

def greedy_interaction(phi1, phi2, phi3, k, pass_list):
    selected, remaining = [], set(range(len(pass_list)))
    for _ in range(k):
        best_s, best_i = -float('inf'), None
        for i in remaining:
            s = phi1.get(pass_list[i], 0)
            for j in selected: s += phi2.get(_pair_key(pass_list[i], pass_list[j]), 0)
            for j, m in itertools.combinations(selected, 2):
                s += phi3.get(_triple_key(pass_list[i], pass_list[j], pass_list[m]), 0)
            if s > best_s: best_s, best_i = s, i
        if best_i is not None: selected.append(best_i); remaining.remove(best_i)
    return [pass_list[i] for i in selected]

def synergy_seeded(phi1, phi2, phi3, k, pass_list):
    best_triple, best_val = None, -float('inf')
    for key, val in phi3.items():
        if val > best_val: best_val, best_triple = val, key
    selected = []
    if best_triple:
        for p in best_triple.split('+'):
            if p in pass_list:
                idx = pass_list.index(p)
                if idx not in selected: selected.append(idx)
    remaining = set(range(len(pass_list))) - set(selected)
    while len(selected) < k and remaining:
        best_s, best_i = -float('inf'), None
        for i in remaining:
            s = phi1.get(pass_list[i], 0)
            for j in selected: s += phi2.get(_pair_key(pass_list[i], pass_list[j]), 0)
            for j, m in itertools.combinations(selected, 2):
                s += phi3.get(_triple_key(pass_list[i], pass_list[j], pass_list[m]), 0)
            if s > best_s: best_s, best_i = s, i
        if best_i is not None: selected.append(best_i); remaining.remove(best_i)
    return [pass_list[i] for i in selected[:k]]

def evaluate_selection(game, selected_passes):
    """Evaluate using pre-created game object (avoids re-reading baseline)."""
    x = np.zeros(game.n_players)
    for p in selected_passes:
        if p in game.passes: x[game.passes.index(p)] = 1
    return float(game.value(x))


# ---- BASELINES (optimized) ----
def run_baselines(benchmarks):
    """Lean baselines: RS=100 samples, GA pop=15 gen=8, only k=10."""
    print("\n=== Baselines ===")
    results = {}
    for bname, bpath in benchmarks.items():
        t0 = time.time()
        game = CompilerGame(bpath)
        n = game.n_players
        bm = {}
        for k in [10]:  # Only k=10 for baselines (faster)
            # Random Search: 100 samples × 3 seeds
            rs_vals = []
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                best = -float('inf')
                for _ in range(100):
                    indices = rng.choice(n, k, replace=False)
                    x = np.zeros(n); x[indices] = 1
                    v = game.value(x)
                    if v > best: best = v
                rs_vals.append(best)
            bm[k] = {
                'random_search': {'mean': float(np.mean(rs_vals)), 'std': float(np.std(rs_vals)), 'values': [float(v) for v in rs_vals]},
            }

            # GA: pop=15, gen=8
            ga_vals = []
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                pop = np.zeros((15, n))
                for i in range(15):
                    idx = rng.choice(n, k, replace=False)
                    pop[i, idx] = 1
                fitness = np.array([game.value(ind) for ind in pop])
                for gen in range(8):
                    new_pop = np.zeros_like(pop)
                    for i in range(15):
                        cands = rng.choice(15, 3, replace=False)
                        new_pop[i] = pop[cands[np.argmax(fitness[cands])]].copy()
                    for i in range(0, 14, 2):
                        if rng.random() < 0.7:
                            pt = rng.randint(1, n)
                            new_pop[i], new_pop[i+1] = (
                                np.concatenate([new_pop[i,:pt], new_pop[i+1,pt:]]),
                                np.concatenate([new_pop[i+1,:pt], new_pop[i,pt:]])
                            )
                    for i in range(15):
                        for j in range(n):
                            if rng.random() < 0.1: new_pop[i,j] = 1 - new_pop[i,j]
                    pop = new_pop
                    fitness = np.array([game.value(ind) for ind in pop])
                ga_vals.append(float(np.max(fitness)))
            bm[k]['genetic_algorithm'] = {'mean': float(np.mean(ga_vals)), 'std': float(np.std(ga_vals)), 'values': [float(v) for v in ga_vals]}
        results[bname] = bm
        elapsed = time.time() - t0
        print(f"  {bname}: RS={bm[10]['random_search']['mean']:.4f}, GA={bm[10]['genetic_algorithm']['mean']:.4f} ({elapsed:.0f}s)")
    save_json(results, os.path.join(RESULTS_DIR, 'data', 'baseline_results.json'))
    return results


def run_selection(benchmarks, agg):
    """Run selection algorithms at all k budgets."""
    print("\n=== Selection Algorithms ===")
    results = {}
    for bname, bpath in benchmarks.items():
        game = CompilerGame(bpath)
        bm_agg = agg[bname]
        phi1 = {k: v['mean'] for k, v in bm_agg['order1'].items()}
        phi2 = {k: v['mean'] for k, v in bm_agg['order2'].items()}
        phi3 = {k: v['mean'] for k, v in bm_agg['order3'].items()}
        bm = {}
        for k in K_BUDGETS:
            methods = {}
            for method_name, func in [
                ('individual_greedy', lambda k: greedy_individual(phi1, k)),
                ('pairwise_greedy', lambda k: greedy_pairwise(phi1, phi2, k, PASS_LIST)),
                ('interaction_greedy', lambda k: greedy_interaction(phi1, phi2, phi3, k, PASS_LIST)),
                ('synergy_seeded', lambda k: synergy_seeded(phi1, phi2, phi3, k, PASS_LIST)),
            ]:
                sel = func(k)
                methods[method_name] = {'passes': sel, 'value': evaluate_selection(game, sel)}
            bm[k] = methods
        results[bname] = bm
        print(f"  {bname} (k=10): indiv={bm[10]['individual_greedy']['value']:.4f}, pair={bm[10]['pairwise_greedy']['value']:.4f}, inter={bm[10]['interaction_greedy']['value']:.4f}")
    save_json(results, os.path.join(RESULTS_DIR, 'data', 'selection_results.json'))
    return results


def run_ablation_order(benchmarks, agg):
    """Ablation: interaction order."""
    print("\n=== Ablation: Interaction Order ===")
    results = {}
    for bname, bpath in benchmarks.items():
        game = CompilerGame(bpath)
        bm_agg = agg[bname]
        phi1 = {k: v['mean'] for k, v in bm_agg['order1'].items()}
        phi2 = {k: v['mean'] for k, v in bm_agg['order2'].items()}
        phi3 = {k: v['mean'] for k, v in bm_agg['order3'].items()}
        bm = {}
        for k in K_BUDGETS:
            v1 = evaluate_selection(game, greedy_individual(phi1, k))
            v12 = evaluate_selection(game, greedy_pairwise(phi1, phi2, k, PASS_LIST))
            v123 = evaluate_selection(game, greedy_interaction(phi1, phi2, phi3, k, PASS_LIST))
            bm[k] = {'order1': v1, 'order1_2': v12, 'order1_2_3': v123,
                      'improvement_2_over_1': v12-v1, 'improvement_3_over_12': v123-v12}
        results[bname] = bm
        print(f"  {bname} (k=10): o1={bm[10]['order1']:.4f}, o1+2={bm[10]['order1_2']:.4f}, o1+2+3={bm[10]['order1_2_3']:.4f}")
    save_json(results, os.path.join(RESULTS_DIR, 'data', 'ablation_order.json'))
    return results


def run_ablation_budget(benchmarks):
    """Ablation: evaluation budget (3 benchmarks only)."""
    print("\n=== Ablation: Evaluation Budget ===")
    import shapiq
    bm_names = list(benchmarks.keys())[:3]
    budgets = [500, 1000, 2000]
    results = {}
    for bname in bm_names:
        bpath = benchmarks[bname]
        bm = {}
        for budget in budgets:
            game = CompilerGame(bpath)
            t0 = time.time()
            approx = shapiq.PermutationSamplingSII(n=game.n_players, max_order=3, random_state=42)
            iv = approx.approximate(budget=budget, game=game)
            elapsed = time.time() - t0
            phi1, phi2, phi3 = {}, {}, {}
            for idx, val in iv.dict_values.items():
                if len(idx) == 1: phi1[PASS_LIST[idx[0]]] = float(val)
                elif len(idx) == 2: phi2[f"{PASS_LIST[idx[0]]}+{PASS_LIST[idx[1]]}"] = float(val)
                elif len(idx) == 3: phi3[f"{PASS_LIST[idx[0]]}+{PASS_LIST[idx[1]]}+{PASS_LIST[idx[2]]}"] = float(val)
            sel = greedy_interaction(phi1, phi2, phi3, 10, PASS_LIST)
            game2 = CompilerGame(bpath)
            v = evaluate_selection(game2, sel)
            bm[budget] = {'selection_value': float(v), 'elapsed': float(elapsed)}
        results[bname] = bm
        print(f"  {bname}: " + ", ".join(f"B={b}: {bm[b]['selection_value']:.4f}" for b in budgets))
    save_json(results, os.path.join(RESULTS_DIR, 'data', 'ablation_budget.json'))
    return results


def run_transferability(benchmarks, agg):
    """Transferability analysis."""
    print("\n=== Transferability ===")
    bm_names = list(benchmarks.keys())
    n_bm = len(bm_names)

    vectors = {}
    for bname in bm_names:
        vec = []
        for k in sorted(agg[bname]['order1'].keys()): vec.append(agg[bname]['order1'][k]['mean'])
        for k in sorted(agg[bname]['order2'].keys()): vec.append(agg[bname]['order2'][k]['mean'])
        vectors[bname] = np.array(vec)

    sim_matrix = np.zeros((n_bm, n_bm))
    for i, b1 in enumerate(bm_names):
        for j, b2 in enumerate(bm_names):
            v1, v2 = vectors[b1], vectors[b2]
            ml = min(len(v1), len(v2))
            norm = np.linalg.norm(v1[:ml]) * np.linalg.norm(v2[:ml])
            sim_matrix[i,j] = np.dot(v1[:ml], v2[:ml]) / norm if norm > 0 else 0

    results = {}
    for i, bname in enumerate(bm_names):
        bpath = benchmarks[bname]
        game = CompilerGame(bpath)
        own = {o: {k: v['mean'] for k, v in agg[bname][o].items()} for o in ['order1', 'order2', 'order3']}
        v_oracle = evaluate_selection(game, greedy_interaction(own['order1'], own['order2'], own['order3'], 10, PASS_LIST))

        avg = {o: defaultdict(float) for o in ['order1', 'order2', 'order3']}
        for j, other in enumerate(bm_names):
            if other == bname: continue
            for o in ['order1', 'order2', 'order3']:
                for k, v in agg[other][o].items(): avg[o][k] += v['mean'] / (n_bm - 1)
        v_transfer = evaluate_selection(game, greedy_interaction(dict(avg['order1']), dict(avg['order2']), dict(avg['order3']), 10, PASS_LIST))

        sims = sim_matrix[i].copy(); sims[i] = -1
        ms_idx = np.argmax(sims); ms = bm_names[ms_idx]
        ms_phi = {o: {k: v['mean'] for k, v in agg[ms][o].items()} for o in ['order1', 'order2', 'order3']}
        v_sim = evaluate_selection(game, greedy_interaction(ms_phi['order1'], ms_phi['order2'], ms_phi['order3'], 10, PASS_LIST))

        results[bname] = {
            'oracle': float(v_oracle), 'transfer_avg': float(v_transfer), 'transfer_similar': float(v_sim),
            'most_similar': ms, 'similarity': float(sims[ms_idx]),
            'ratio_avg': float(v_transfer/v_oracle) if v_oracle > 0 else 0,
            'ratio_sim': float(v_sim/v_oracle) if v_oracle > 0 else 0,
        }
        print(f"  {bname}: oracle={v_oracle:.4f}, transfer={v_transfer:.4f}, similar({ms})={v_sim:.4f}")

    results['_similarity_matrix'] = {'benchmarks': bm_names, 'matrix': sim_matrix.tolist()}
    save_json(results, os.path.join(RESULTS_DIR, 'data', 'transferability.json'))
    return results


def statistical_evaluation(all_interactions, decomp, selection_results, baseline_results, benchmarks, opt_levels):
    """Evaluate against success criteria."""
    print("\n=== Statistical Evaluation ===")
    agg = aggregate_interactions(all_interactions)
    bm_names = list(benchmarks.keys())

    # Criterion 1: significant order-3 interactions (|mean| > 2*std)
    n_sig, n_total = 0, 0
    for bname in bm_names:
        for key, vals in agg[bname]['order3'].items():
            n_total += 1
            if vals['std'] > 0 and abs(vals['mean']) > 2 * vals['std']: n_sig += 1
            elif vals['std'] == 0 and abs(vals['mean']) > 1e-6: n_sig += 1
    frac_sig = n_sig / n_total if n_total > 0 else 0
    c1 = frac_sig >= 0.30
    print(f"  C1: {n_sig}/{n_total} ({frac_sig:.1%}) significant order-3 => {'PASS' if c1 else 'FAIL'} (thresh: 30%)")

    # Criterion 2: order-3 variance >=10%
    frac3 = [decomp[b]['frac_order3'] for b in bm_names]
    avg_f3 = np.mean(frac3)
    c2 = avg_f3 >= 0.10
    print(f"  C2: Order-3 variance = {avg_f3:.1%} => {'PASS' if c2 else 'FAIL'} (thresh: 10%)")

    # Criterion 3: interaction vs pairwise win rate
    wins, total = 0, 0
    for bname in bm_names:
        for kk in [10, '10']:
            if kk in selection_results.get(bname, {}):
                if selection_results[bname][kk]['interaction_greedy']['value'] >= selection_results[bname][kk]['pairwise_greedy']['value']:
                    wins += 1
                total += 1; break
    wr = wins / total if total > 0 else 0
    c3 = wr >= 0.60
    print(f"  C3: Win rate = {wins}/{total} ({wr:.1%}) => {'PASS' if c3 else 'FAIL'} (thresh: 60%)")

    # Main comparison table
    main_table = {}
    methods = ['O1', 'O2', 'O3', 'Os', 'Oz', 'random_search', 'genetic_algorithm',
               'individual_greedy', 'pairwise_greedy', 'interaction_greedy', 'synergy_seeded']
    for bname in bm_names:
        lvls = opt_levels[bname]; baseline = lvls['O0']; row = {}
        for lvl in ['O1', 'O2', 'O3', 'Os', 'Oz']:
            if lvls.get(lvl): row[lvl] = float((baseline - lvls[lvl]) / baseline)
        for kk in [10, '10']:
            if kk in selection_results.get(bname, {}):
                for m in ['individual_greedy', 'pairwise_greedy', 'interaction_greedy', 'synergy_seeded']:
                    row[m] = selection_results[bname][kk][m]['value']
                break
        for kk in [10, '10']:
            if kk in baseline_results.get(bname, {}):
                row['random_search'] = baseline_results[bname][kk]['random_search']['mean']
                row['genetic_algorithm'] = baseline_results[bname][kk]['genetic_algorithm']['mean']
                break
        main_table[bname] = row

    avg_row = {}
    for m in methods:
        vals = [main_table[b].get(m) for b in bm_names if main_table[b].get(m) is not None]
        if vals: avg_row[m] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    main_table['_average'] = avg_row

    eval_results = {
        'criterion1': {'n_sig': int(n_sig), 'n_total': int(n_total), 'frac': float(frac_sig), 'confirmed': bool(c1)},
        'criterion2': {'avg_frac_order3': float(avg_f3), 'per_bm': {b: float(v) for b, v in zip(bm_names, frac3)}, 'confirmed': bool(c2)},
        'criterion3': {'wins': int(wins), 'total': int(total), 'win_rate': float(wr), 'confirmed': bool(c3)},
        'main_table': main_table,
        'overall_confirmed': bool(c1 and c2 and c3),
    }
    save_json(eval_results, os.path.join(RESULTS_DIR, 'data', 'statistical_tests.json'))

    # CSV
    csv_path = os.path.join(RESULTS_DIR, 'tables', 'main_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Benchmark'] + methods)
        for bname in bm_names:
            w.writerow([bname] + [f"{main_table[bname].get(m, 0):.4f}" for m in methods])
        w.writerow(['Average'] + [f"{avg_row.get(m,{}).get('mean',0):.4f}+/-{avg_row.get(m,{}).get('std',0):.4f}" for m in methods])
    print(f"  Saved {csv_path}")
    return eval_results


def generate_all_figures(all_interactions, decomp, selection_results, baseline_results,
                          benchmarks, opt_levels, transfer_results, ablation_order, ablation_budget_res):
    """Generate all publication figures."""
    print("\n=== Generating Figures ===")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.dpi': 300})

    agg = aggregate_interactions(all_interactions)
    bm_names = list(benchmarks.keys())

    # Fig 1: Variance Decomposition (stacked bar)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bm_names))
    f1 = [decomp[b]['frac_order1'] for b in bm_names]
    f2 = [decomp[b]['frac_order2'] for b in bm_names]
    f3 = [decomp[b]['frac_order3'] for b in bm_names]
    ax.bar(x, f1, label='Order 1 (Individual)', color='#4C72B0')
    ax.bar(x, f2, bottom=f1, label='Order 2 (Pairwise)', color='#DD8452')
    ax.bar(x, f3, bottom=[a+b for a,b in zip(f1,f2)], label='Order 3 (Triple)', color='#C44E52')
    ax.set_xticks(x); ax.set_xticklabels(bm_names, rotation=45, ha='right')
    ax.set_ylabel('Fraction of Interaction Variance')
    ax.set_title('Variance Decomposition by Interaction Order')
    ax.legend(); plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(FIGURES_DIR, f'variance_decomposition.{ext}'))
    plt.close(); print("  fig: variance_decomposition")

    # Fig 2: Heatmaps
    for bname in bm_names[:2]:
        phi2 = agg[bname]['order2']
        n_p = len(PASS_LIST)
        hm = np.zeros((n_p, n_p))
        for key, v in phi2.items():
            parts = key.split('+')
            if len(parts) == 2:
                try:
                    i, j = PASS_LIST.index(parts[0]), PASS_LIST.index(parts[1])
                    hm[i,j] = hm[j,i] = v['mean']
                except ValueError:
                    pass
        fig, ax = plt.subplots(figsize=(12, 10))
        vmax = max(abs(hm.min()), abs(hm.max())) or 0.01
        sns.heatmap(hm, xticklabels=PASS_LIST, yticklabels=PASS_LIST, cmap='RdBu_r',
                     center=0, vmin=-vmax, vmax=vmax, ax=ax, square=True, linewidths=0.5)
        ax.set_title(f'Pairwise Shapley Interactions: {bname}'); plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(FIGURES_DIR, f'heatmap_{bname}.{ext}'))
        plt.close()
    print("  fig: heatmaps")

    # Fig 3: Top Order-3 Interactions
    all_o3 = []
    for bname in bm_names:
        for key, v in agg[bname]['order3'].items():
            all_o3.append((key, bname, v['mean'], v['std']))
    all_o3.sort(key=lambda x: abs(x[2]), reverse=True)
    top_o3 = all_o3[:15]
    if top_o3:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [f"{t[0]}\n({t[1]})" for t in top_o3]
        vals = [t[2] for t in top_o3]; errs = [t[3] for t in top_o3]
        cols = ['#C44E52' if v > 0 else '#4C72B0' for v in vals]
        y = np.arange(len(top_o3))
        ax.barh(y, vals, xerr=errs, color=cols, alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Shapley Interaction Index')
        ax.set_title('Top 15 Order-3 Interactions'); ax.axvline(x=0, color='black', lw=0.5)
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(FIGURES_DIR, f'top_order3_interactions.{ext}'))
        plt.close()
    print("  fig: top_order3")

    # Fig 4: Selection Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    mtp = ['O3', 'random_search', 'genetic_algorithm', 'individual_greedy', 'pairwise_greedy', 'interaction_greedy', 'synergy_seeded']
    mlabels = ['O3', 'Random\nSearch', 'GA', 'Indiv.\nGreedy', 'Pairwise\nGreedy', 'Interact.\nGreedy', 'Synergy\nSeeded']
    avgs, stds_v = [], []
    for m in mtp:
        vals = []
        for bname in bm_names:
            if m in ['O1','O2','O3','Os','Oz']:
                lvls = opt_levels[bname]
                if lvls.get(m): vals.append((lvls['O0'] - lvls[m]) / lvls['O0'])
            elif m in ['random_search','genetic_algorithm']:
                for kk in [10,'10']:
                    if kk in baseline_results.get(bname,{}):
                        vals.append(baseline_results[bname][kk][m]['mean']); break
            else:
                for kk in [10,'10']:
                    if kk in selection_results.get(bname,{}):
                        vals.append(selection_results[bname][kk][m]['value']); break
        avgs.append(np.mean(vals) if vals else 0); stds_v.append(np.std(vals) if vals else 0)
    colors = ['#55A868', '#CCB974', '#CCB974', '#4C72B0', '#DD8452', '#C44E52', '#8172B3']
    xp = np.arange(len(mtp))
    ax.bar(xp, avgs, yerr=stds_v, color=colors, alpha=0.8, capsize=3)
    ax.set_xticks(xp); ax.set_xticklabels(mlabels, fontsize=9)
    ax.set_ylabel('IR Reduction Fraction'); ax.set_title('Method Comparison (k=10)')

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
                for kk in [k, str(k)]:
                    if kk in selection_results.get(bname,{}):
                        vals.append(selection_results[bname][kk][method]['value']); break
            avg_by_k.append(np.mean(vals) if vals else 0)
        ax.plot(K_BUDGETS, avg_by_k, marker='o', label=label, color=color, linestyle=ls)
    ax.set_xlabel('Number of Selected Passes (k)'); ax.set_ylabel('IR Reduction Fraction')
    ax.set_title('Selection Performance vs. Budget'); ax.legend()
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(FIGURES_DIR, f'selection_comparison.{ext}'))
    plt.close(); print("  fig: selection_comparison")

    # Fig 5: Network
    import networkx as nx
    for bname in bm_names[:1]:
        G = nx.Graph()
        for p in PASS_LIST:
            if p in agg[bname]['order1']: G.add_node(p, weight=agg[bname]['order1'][p]['mean'])
        for key, v in agg[bname]['order2'].items():
            parts = key.split('+')
            if len(parts) == 2 and abs(v['mean']) > 0.001:
                try: G.add_edge(parts[0], parts[1], weight=v['mean'])
                except: pass
        fig, ax = plt.subplots(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42, k=2)
        ns = [abs(G.nodes[n].get('weight', 0)) * 5000 + 100 for n in G.nodes]
        nc = ['#C44E52' if G.nodes[n].get('weight', 0) > 0 else '#4C72B0' for n in G.nodes]
        edges = list(G.edges())
        if edges:
            ew = [G[u][v]['weight'] for u,v in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=[abs(w)*50+0.5 for w in ew],
                                    edge_color=['#C44E52' if w>0 else '#4C72B0' for w in ew], alpha=0.5, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_size=ns, node_color=nc, alpha=0.7, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        ax.set_title(f'Pass Interaction Network: {bname}'); ax.axis('off'); plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(FIGURES_DIR, f'network_{bname}.{ext}'))
        plt.close()
    print("  fig: network")

    # Fig 6: Transferability
    if '_similarity_matrix' in transfer_results:
        sim = np.array(transfer_results['_similarity_matrix']['matrix'])
        bns = transfer_results['_similarity_matrix']['benchmarks']
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(sim, xticklabels=bns, yticklabels=bns, cmap='YlOrRd',
                     vmin=0, vmax=1, annot=True, fmt='.2f', ax=ax, square=True)
        ax.set_title('Benchmark Similarity (Interaction Vectors)'); plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(FIGURES_DIR, f'transferability.{ext}'))
        plt.close()
    print("  fig: transferability")

    # Fig 7: Convergence
    if ablation_budget_res:
        budgets = sorted([int(b) for b in list(list(ablation_budget_res.values())[0].keys())])
        fig, ax = plt.subplots(figsize=(8, 5))
        for bname in ablation_budget_res:
            vals = [ablation_budget_res[bname][str(b) if str(b) in ablation_budget_res[bname] else b]['selection_value'] for b in budgets]
            ax.plot(budgets, vals, marker='o', label=bname)
        ax.set_xlabel('Evaluation Budget'); ax.set_ylabel('Selection Performance (k=10)')
        ax.set_title('Convergence vs. Budget'); ax.legend(fontsize=9); plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(FIGURES_DIR, f'convergence.{ext}'))
        plt.close()
    print("  fig: convergence")

    # Fig 8: Ablation Order
    if ablation_order:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(bm_names)); w = 0.25
        for i, (okey, lbl, col) in enumerate([
            ('order1', 'Order 1 Only', '#4C72B0'),
            ('order1_2', 'Order 1+2', '#DD8452'),
            ('order1_2_3', 'Order 1+2+3', '#C44E52'),
        ]):
            vals = []
            for bname in bm_names:
                for kk in [10, '10']:
                    if bname in ablation_order and kk in ablation_order[bname]:
                        vals.append(ablation_order[bname][kk][okey]); break
                else:
                    vals.append(0)
            ax.bar(x + i*w, vals, w, label=lbl, color=col, alpha=0.8)
        ax.set_xticks(x + w); ax.set_xticklabels(bm_names, rotation=45, ha='right')
        ax.set_ylabel('IR Reduction Fraction')
        ax.set_title('Ablation: Effect of Interaction Order (k=10)'); ax.legend(); plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(FIGURES_DIR, f'ablation_order.{ext}'))
        plt.close()
    print("  fig: ablation_order")
    print("  All figures generated.")


def save_final_results(eval_results, decomp, selection_results, baseline_results,
                       benchmarks, opt_levels, ablation_order):
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
            for kk in [10, '10']:
                if kk in ablation_order[bname]:
                    results["ablation_interaction_order"][bname] = ablation_order[bname][kk]; break

    results_path = os.path.join(os.path.dirname(__file__), '..', 'results.json')
    save_json(results, results_path)
    print(f"\nFinal results saved to {results_path}")


def main():
    t_start = time.time()
    print("ShapleyPass: Completing All Experiments")
    print("=" * 60)

    benchmarks = get_benchmark_files()
    all_interactions = load_json(os.path.join(RESULTS_DIR, 'data', 'all_interactions.json'))
    decomp = load_json(os.path.join(RESULTS_DIR, 'data', 'variance_decomposition.json'))
    opt_levels = load_json(os.path.join(RESULTS_DIR, 'data', 'opt_levels.json'))
    agg = aggregate_interactions(all_interactions)
    print(f"Loaded data for {len(benchmarks)} benchmarks")

    # Re-run selection (fast, uses cached games)
    selection_results = run_selection(benchmarks, agg)

    # Baselines (optimized)
    baseline_results = run_baselines(benchmarks)

    # Ablations
    ablation_order = run_ablation_order(benchmarks, agg)
    ablation_budget_res = run_ablation_budget(benchmarks)

    # Transferability
    transfer_results = run_transferability(benchmarks, agg)

    # Evaluation
    eval_results = statistical_evaluation(all_interactions, decomp, selection_results,
                                          baseline_results, benchmarks, opt_levels)

    # Figures
    generate_all_figures(all_interactions, decomp, selection_results, baseline_results,
                          benchmarks, opt_levels, transfer_results, ablation_order, ablation_budget_res)

    # Save final
    save_final_results(eval_results, decomp, selection_results, baseline_results,
                       benchmarks, opt_levels, ablation_order)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == '__main__':
    main()

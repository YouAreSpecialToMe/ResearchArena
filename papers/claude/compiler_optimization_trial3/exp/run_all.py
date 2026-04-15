#!/usr/bin/env python3
"""
ShapleyPass: Full experiment pipeline.
Computes Shapley Interaction Indices for compiler optimization passes
and evaluates interaction-guided pass selection algorithms.
"""
import sys, os, json, time, itertools
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
from game import CompilerGame, CANDIDATE_PASSES, count_ir_instructions, get_optimization_level_counts

# Configuration
BENCHMARKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmarks')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
SEEDS = [42, 123, 456]
SHAPLEY_BUDGET = 2000  # evaluations per benchmark per seed
PASS_LIST = CANDIDATE_PASSES  # 20 passes
K_BUDGETS = [5, 8, 10, 12, 15]

os.makedirs(os.path.join(RESULTS_DIR, 'data', 'interactions'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def get_benchmark_files():
    """Get all .bc benchmark files."""
    bc_files = {}
    for f in sorted(os.listdir(BENCHMARKS_DIR)):
        if f.endswith('.bc'):
            name = f.replace('.bc', '')
            bc_files[name] = os.path.join(BENCHMARKS_DIR, f)
    return bc_files


# ============================================================
# STEP 1: Pass Screening
# ============================================================
def screen_passes(benchmarks):
    """Screen individual pass effects on each benchmark."""
    print("\n" + "="*60)
    print("STEP 1: Pass Screening")
    print("="*60)

    screening = {}
    opt_levels = {}

    for bname, bpath in benchmarks.items():
        print(f"\n  Screening: {bname}")
        game = CompilerGame(bpath)
        baseline = game.baseline_count

        # Individual pass effects
        pass_effects = {}
        for i, p in enumerate(PASS_LIST):
            x = np.zeros(len(PASS_LIST))
            x[i] = 1
            v = game.value(x)
            pass_effects[p] = v
            if v > 0.01:
                print(f"    {p}: {v:.4f}")

        screening[bname] = {
            'baseline': baseline,
            'pass_effects': pass_effects,
            'full_value': float(game.value(np.ones(len(PASS_LIST)))),
            'o3_value': float(game.get_o3_value() or 0),
        }

        # Optimization levels
        levels = get_optimization_level_counts(bpath)
        opt_levels[bname] = {k: int(v) if v else None for k, v in levels.items()}

    # Save
    with open(os.path.join(RESULTS_DIR, 'data', 'pass_screening.json'), 'w') as f:
        json.dump(screening, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'data', 'opt_levels.json'), 'w') as f:
        json.dump(opt_levels, f, indent=2)

    print("\n  Pass screening complete.")
    return screening, opt_levels


# ============================================================
# STEP 2: Shapley Interaction Indices
# ============================================================
def compute_shapley_interactions(benchmarks):
    """Compute Shapley interaction indices up to order 3."""
    print("\n" + "="*60)
    print("STEP 2: Shapley Interaction Indices")
    print("="*60)

    import shapiq

    all_interactions = {}

    for bname, bpath in benchmarks.items():
        print(f"\n  Computing interactions for: {bname}")
        game = CompilerGame(bpath)
        n = game.n_players

        bm_results = {}
        for seed in SEEDS:
            print(f"    Seed {seed}...")
            t0 = time.time()

            # Use shapiq's PermutationSamplingSII approximator
            approximator = shapiq.PermutationSamplingSII(
                n=n,
                max_order=3,
                random_state=seed,
            )

            # Compute with budget
            interaction_values = approximator.approximate(
                budget=SHAPLEY_BUDGET,
                game=game,
            )

            elapsed = time.time() - t0
            n_evals = len(game.cache)
            print(f"    Done in {elapsed:.1f}s, {n_evals} cached evaluations")

            # Extract interaction values by order
            order1 = {}
            order2 = {}
            order3 = {}

            for idx, val in interaction_values.dict_values.items():
                if len(idx) == 1:
                    order1[PASS_LIST[idx[0]]] = float(val)
                elif len(idx) == 2:
                    order2[f"{PASS_LIST[idx[0]]}+{PASS_LIST[idx[1]]}"] = float(val)
                elif len(idx) == 3:
                    order3[f"{PASS_LIST[idx[0]]}+{PASS_LIST[idx[1]]}+{PASS_LIST[idx[2]]}"] = float(val)

            bm_results[seed] = {
                'order1': order1,
                'order2': order2,
                'order3': order3,
                'elapsed_seconds': elapsed,
                'n_evaluations': n_evals,
            }

        all_interactions[bname] = bm_results

        # Save per-benchmark
        with open(os.path.join(RESULTS_DIR, 'data', 'interactions', f'{bname}.json'), 'w') as f:
            json.dump(bm_results, f, indent=2)

    # Save all
    with open(os.path.join(RESULTS_DIR, 'data', 'all_interactions.json'), 'w') as f:
        json.dump(all_interactions, f, indent=2)

    print("\n  Shapley interaction computation complete.")
    return all_interactions


def aggregate_interactions(all_interactions):
    """Aggregate interaction values across seeds (mean +/- std)."""
    aggregated = {}
    for bname, seed_results in all_interactions.items():
        agg = {'order1': {}, 'order2': {}, 'order3': {}}
        for order_key in ['order1', 'order2', 'order3']:
            all_keys = set()
            for seed in SEEDS:
                all_keys.update(seed_results[seed][order_key].keys())
            for key in all_keys:
                vals = [seed_results[seed][order_key].get(key, 0.0) for seed in SEEDS]
                agg[order_key][key] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'values': vals,
                }
        aggregated[bname] = agg
    return aggregated


# ============================================================
# STEP 3: Variance Decomposition
# ============================================================
def variance_decomposition(all_interactions):
    """Decompose performance variance by interaction order."""
    print("\n" + "="*60)
    print("STEP 3: Variance Decomposition")
    print("="*60)

    agg = aggregate_interactions(all_interactions)
    decomp = {}

    for bname, bm_agg in agg.items():
        # Sum of squared interaction values at each order
        ss1 = sum(v['mean']**2 for v in bm_agg['order1'].values())
        ss2 = sum(v['mean']**2 for v in bm_agg['order2'].values())
        ss3 = sum(v['mean']**2 for v in bm_agg['order3'].values())
        total = ss1 + ss2 + ss3

        if total > 0:
            frac1 = ss1 / total
            frac2 = ss2 / total
            frac3 = ss3 / total
        else:
            frac1 = frac2 = frac3 = 0.0

        decomp[bname] = {
            'ss_order1': float(ss1),
            'ss_order2': float(ss2),
            'ss_order3': float(ss3),
            'frac_order1': float(frac1),
            'frac_order2': float(frac2),
            'frac_order3': float(frac3),
        }
        print(f"  {bname}: order1={frac1:.1%}, order2={frac2:.1%}, order3={frac3:.1%}")

    # Average across benchmarks
    avg_frac1 = np.mean([d['frac_order1'] for d in decomp.values()])
    avg_frac2 = np.mean([d['frac_order2'] for d in decomp.values()])
    avg_frac3 = np.mean([d['frac_order3'] for d in decomp.values()])
    decomp['_average'] = {
        'frac_order1': float(avg_frac1),
        'frac_order2': float(avg_frac2),
        'frac_order3': float(avg_frac3),
    }
    print(f"\n  Average: order1={avg_frac1:.1%}, order2={avg_frac2:.1%}, order3={avg_frac3:.1%}")

    with open(os.path.join(RESULTS_DIR, 'data', 'variance_decomposition.json'), 'w') as f:
        json.dump(decomp, f, indent=2)

    return decomp


# ============================================================
# STEP 4: Interaction Structure Analysis
# ============================================================
def interaction_structure_analysis(all_interactions):
    """Analyze top synergistic/redundant interactions and cross-program stability."""
    print("\n" + "="*60)
    print("STEP 4: Interaction Structure Analysis")
    print("="*60)

    agg = aggregate_interactions(all_interactions)
    structure = {}

    for bname, bm_agg in agg.items():
        # Top-10 order-2 interactions by magnitude
        o2_sorted = sorted(bm_agg['order2'].items(), key=lambda x: abs(x[1]['mean']), reverse=True)
        top_synergistic_2 = [(k, v['mean']) for k, v in o2_sorted if v['mean'] > 0][:10]
        top_redundant_2 = [(k, v['mean']) for k, v in o2_sorted if v['mean'] < 0][:10]

        # Top-10 order-3
        o3_sorted = sorted(bm_agg['order3'].items(), key=lambda x: abs(x[1]['mean']), reverse=True)
        top_synergistic_3 = [(k, v['mean']) for k, v in o3_sorted if v['mean'] > 0][:10]
        top_redundant_3 = [(k, v['mean']) for k, v in o3_sorted if v['mean'] < 0][:10]

        structure[bname] = {
            'top_synergistic_order2': top_synergistic_2,
            'top_redundant_order2': top_redundant_2,
            'top_synergistic_order3': top_synergistic_3,
            'top_redundant_order3': top_redundant_3,
        }

        if top_synergistic_2:
            print(f"\n  {bname} top synergy (order 2): {top_synergistic_2[0][0]} = {top_synergistic_2[0][1]:.4f}")
        if top_synergistic_3:
            print(f"  {bname} top synergy (order 3): {top_synergistic_3[0][0]} = {top_synergistic_3[0][1]:.4f}")

    # Cross-program stability for order-2 interactions
    all_o2_keys = set()
    for bname in agg:
        all_o2_keys.update(agg[bname]['order2'].keys())

    stability = {}
    for key in all_o2_keys:
        values = []
        for bname in agg:
            if key in agg[bname]['order2']:
                values.append(agg[bname]['order2'][key]['mean'])
        if len(values) >= 3:
            signs = [1 if v > 0 else -1 for v in values]
            sign_consistency = max(sum(1 for s in signs if s > 0), sum(1 for s in signs if s < 0)) / len(signs)
            stability[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'sign_consistency': float(sign_consistency),
                'n_benchmarks': len(values),
            }

    # Universal interactions (consistent across >80% benchmarks)
    universal = {k: v for k, v in stability.items() if v['sign_consistency'] >= 0.8}
    print(f"\n  Universal order-2 interactions (>80% sign consistency): {len(universal)}/{len(stability)}")

    structure['_cross_program'] = {
        'universal_order2': {k: v for k, v in sorted(universal.items(), key=lambda x: abs(x[1]['mean']), reverse=True)[:20]},
        'n_universal': len(universal),
        'n_total': len(stability),
    }

    with open(os.path.join(RESULTS_DIR, 'data', 'interaction_structure.json'), 'w') as f:
        json.dump(structure, f, indent=2, default=str)

    return structure


# ============================================================
# STEP 5: Pass Selection Algorithms
# ============================================================
def greedy_individual(phi1, k):
    """Select top-k passes by individual Shapley value."""
    sorted_passes = sorted(phi1.items(), key=lambda x: x[1], reverse=True)
    return [p for p, _ in sorted_passes[:k]]


def greedy_pairwise(phi1, phi2, k, pass_list):
    """Greedy selection using individual + pairwise interactions."""
    selected = []
    remaining = set(range(len(pass_list)))

    for _ in range(k):
        best_score = -float('inf')
        best_idx = None
        for i in remaining:
            score = phi1.get(pass_list[i], 0)
            for j_idx in selected:
                pair_key = _pair_key(pass_list[i], pass_list[j_idx])
                score += phi2.get(pair_key, 0)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
    return [pass_list[i] for i in selected]


def greedy_interaction(phi1, phi2, phi3, k, pass_list):
    """Greedy selection using individual + pairwise + triple interactions."""
    selected = []
    remaining = set(range(len(pass_list)))

    for _ in range(k):
        best_score = -float('inf')
        best_idx = None
        for i in remaining:
            score = phi1.get(pass_list[i], 0)
            # Add pairwise terms
            for j_idx in selected:
                pair_key = _pair_key(pass_list[i], pass_list[j_idx])
                score += phi2.get(pair_key, 0)
            # Add triple terms
            for j_idx, k_idx in itertools.combinations(selected, 2):
                triple_key = _triple_key(pass_list[i], pass_list[j_idx], pass_list[k_idx])
                score += phi3.get(triple_key, 0)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
    return [pass_list[i] for i in selected]


def synergy_seeded(phi1, phi2, phi3, k, pass_list):
    """Seed with top synergistic triple, then greedy interaction."""
    # Find top synergistic triple
    best_triple = None
    best_val = -float('inf')
    for key, val in phi3.items():
        if val > best_val:
            best_val = val
            best_triple = key

    selected = []
    if best_triple:
        parts = best_triple.split('+')
        for p in parts:
            if p in pass_list:
                idx = pass_list.index(p)
                if idx not in selected:
                    selected.append(idx)

    # Fill remaining with interaction greedy
    remaining = set(range(len(pass_list))) - set(selected)
    while len(selected) < k and remaining:
        best_score = -float('inf')
        best_idx = None
        for i in remaining:
            score = phi1.get(pass_list[i], 0)
            for j_idx in selected:
                pair_key = _pair_key(pass_list[i], pass_list[j_idx])
                score += phi2.get(pair_key, 0)
            for j_idx, k_idx in itertools.combinations(selected, 2):
                triple_key = _triple_key(pass_list[i], pass_list[j_idx], pass_list[k_idx])
                score += phi3.get(triple_key, 0)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [pass_list[i] for i in selected[:k]]


def _pair_key(a, b):
    return f"{min(a,b)}+{max(a,b)}"

def _triple_key(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}+{s[1]}+{s[2]}"


def evaluate_selection(benchmark_path, selected_passes, all_passes=PASS_LIST):
    """Evaluate a selected pass subset."""
    game = CompilerGame(benchmark_path, passes=all_passes)
    x = np.zeros(len(all_passes))
    for p in selected_passes:
        if p in all_passes:
            x[all_passes.index(p)] = 1
    return float(game.value(x))


def run_selection_experiments(benchmarks, all_interactions):
    """Run all pass selection algorithms."""
    print("\n" + "="*60)
    print("STEP 5: Pass Selection Algorithms")
    print("="*60)

    agg = aggregate_interactions(all_interactions)
    selection_results = {}

    for bname, bpath in benchmarks.items():
        print(f"\n  Selection for: {bname}")
        bm_agg = agg[bname]

        # Build phi dictionaries
        phi1 = {k: v['mean'] for k, v in bm_agg['order1'].items()}
        phi2 = {k: v['mean'] for k, v in bm_agg['order2'].items()}
        phi3 = {k: v['mean'] for k, v in bm_agg['order3'].items()}

        bm_results = {}
        for k in K_BUDGETS:
            methods = {}

            # Individual greedy
            sel = greedy_individual(phi1, k)
            methods['individual_greedy'] = {
                'passes': sel,
                'value': evaluate_selection(bpath, sel),
            }

            # Pairwise greedy
            sel = greedy_pairwise(phi1, phi2, k, PASS_LIST)
            methods['pairwise_greedy'] = {
                'passes': sel,
                'value': evaluate_selection(bpath, sel),
            }

            # Full interaction greedy
            sel = greedy_interaction(phi1, phi2, phi3, k, PASS_LIST)
            methods['interaction_greedy'] = {
                'passes': sel,
                'value': evaluate_selection(bpath, sel),
            }

            # Synergy-seeded
            sel = synergy_seeded(phi1, phi2, phi3, k, PASS_LIST)
            methods['synergy_seeded'] = {
                'passes': sel,
                'value': evaluate_selection(bpath, sel),
            }

            bm_results[k] = methods
            print(f"    k={k}: indiv={methods['individual_greedy']['value']:.4f}, "
                  f"pair={methods['pairwise_greedy']['value']:.4f}, "
                  f"interact={methods['interaction_greedy']['value']:.4f}, "
                  f"synergy={methods['synergy_seeded']['value']:.4f}")

        selection_results[bname] = bm_results

    with open(os.path.join(RESULTS_DIR, 'data', 'selection_results.json'), 'w') as f:
        json.dump(selection_results, f, indent=2)

    return selection_results


# ============================================================
# STEP 6: Baselines
# ============================================================
def run_baselines(benchmarks):
    """Run random search and genetic algorithm baselines."""
    print("\n" + "="*60)
    print("STEP 6: Baseline Comparisons")
    print("="*60)

    baseline_results = {}

    for bname, bpath in benchmarks.items():
        print(f"\n  Baselines for: {bname}")
        game = CompilerGame(bpath)
        n = game.n_players

        bm_baselines = {}

        for k in K_BUDGETS:
            k_results = {}

            # Random search (500 random subsets of size k, 3 seeds)
            rs_vals = []
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                best = -float('inf')
                for _ in range(500):
                    # Random subset of size k
                    indices = rng.choice(n, k, replace=False)
                    x = np.zeros(n)
                    x[indices] = 1
                    v = game(x)
                    if v > best:
                        best = v
                rs_vals.append(best)
            k_results['random_search'] = {
                'mean': float(np.mean(rs_vals)),
                'std': float(np.std(rs_vals)),
                'values': [float(v) for v in rs_vals],
            }

            # Genetic Algorithm (population=30, generations=15)
            ga_vals = []
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                pop_size = 30
                n_gen = 15
                mutation_rate = 0.1

                # Initialize population: random subsets of size ~k
                pop = np.zeros((pop_size, n))
                for i in range(pop_size):
                    indices = rng.choice(n, k, replace=False)
                    pop[i, indices] = 1

                # Evaluate
                fitness = np.array([game(ind) for ind in pop])

                for gen in range(n_gen):
                    # Tournament selection
                    new_pop = np.zeros_like(pop)
                    for i in range(pop_size):
                        # Tournament of 3
                        candidates = rng.choice(pop_size, 3, replace=False)
                        winner = candidates[np.argmax(fitness[candidates])]
                        new_pop[i] = pop[winner].copy()

                    # Crossover
                    for i in range(0, pop_size - 1, 2):
                        if rng.random() < 0.7:
                            point = rng.randint(1, n)
                            child1 = np.concatenate([new_pop[i, :point], new_pop[i+1, point:]])
                            child2 = np.concatenate([new_pop[i+1, :point], new_pop[i, point:]])
                            new_pop[i] = child1
                            new_pop[i+1] = child2

                    # Mutation
                    for i in range(pop_size):
                        for j in range(n):
                            if rng.random() < mutation_rate:
                                new_pop[i, j] = 1 - new_pop[i, j]

                    pop = new_pop
                    fitness = np.array([game(ind) for ind in pop])

                ga_vals.append(float(np.max(fitness)))

            k_results['genetic_algorithm'] = {
                'mean': float(np.mean(ga_vals)),
                'std': float(np.std(ga_vals)),
                'values': [float(v) for v in ga_vals],
            }

            bm_baselines[k] = k_results
            print(f"    k={k}: RS={k_results['random_search']['mean']:.4f}±{k_results['random_search']['std']:.4f}, "
                  f"GA={k_results['genetic_algorithm']['mean']:.4f}±{k_results['genetic_algorithm']['std']:.4f}")

        baseline_results[bname] = bm_baselines

    with open(os.path.join(RESULTS_DIR, 'data', 'baseline_results.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2)

    return baseline_results


# ============================================================
# STEP 7: Ablation Studies
# ============================================================
def ablation_interaction_order(benchmarks, all_interactions):
    """Ablation: vary interaction order used in selection."""
    print("\n" + "="*60)
    print("STEP 7a: Ablation - Interaction Order")
    print("="*60)

    agg = aggregate_interactions(all_interactions)
    ablation_results = {}

    for bname, bpath in benchmarks.items():
        bm_agg = agg[bname]
        phi1 = {k: v['mean'] for k, v in bm_agg['order1'].items()}
        phi2 = {k: v['mean'] for k, v in bm_agg['order2'].items()}
        phi3 = {k: v['mean'] for k, v in bm_agg['order3'].items()}

        bm_results = {}
        for k in K_BUDGETS:
            # Order 1 only
            sel1 = greedy_individual(phi1, k)
            v1 = evaluate_selection(bpath, sel1)

            # Order 1+2
            sel12 = greedy_pairwise(phi1, phi2, k, PASS_LIST)
            v12 = evaluate_selection(bpath, sel12)

            # Order 1+2+3
            sel123 = greedy_interaction(phi1, phi2, phi3, k, PASS_LIST)
            v123 = evaluate_selection(bpath, sel123)

            bm_results[k] = {
                'order1': float(v1),
                'order1_2': float(v12),
                'order1_2_3': float(v123),
                'improvement_2_over_1': float(v12 - v1),
                'improvement_3_over_12': float(v123 - v12),
            }

        ablation_results[bname] = bm_results
        print(f"  {bname} (k=10): o1={bm_results[10]['order1']:.4f}, o1+2={bm_results[10]['order1_2']:.4f}, o1+2+3={bm_results[10]['order1_2_3']:.4f}")

    with open(os.path.join(RESULTS_DIR, 'data', 'ablation_order.json'), 'w') as f:
        json.dump(ablation_results, f, indent=2)

    return ablation_results


def ablation_num_passes(benchmarks):
    """Ablation: vary number of candidate passes (K=10, 15, 20)."""
    print("\n" + "="*60)
    print("STEP 7b: Ablation - Number of Candidate Passes")
    print("="*60)

    import shapiq

    # Use 5 representative benchmarks
    bm_names = list(benchmarks.keys())[:5]
    ablation_results = {}

    for K in [10, 15, 20]:
        passes_subset = PASS_LIST[:K]
        print(f"\n  K={K} passes")
        k_results = {}

        for bname in bm_names:
            bpath = benchmarks[bname]
            game = CompilerGame(bpath, passes=passes_subset)
            n = game.n_players

            t0 = time.time()
            approximator = shapiq.PermutationSamplingSII(
                n=n, max_order=3, random_state=42,
            )
            iv = approximator.approximate(budget=SHAPLEY_BUDGET, game=game)
            elapsed = time.time() - t0

            # Extract phi1, phi2, phi3
            phi1 = {}
            phi2 = {}
            phi3 = {}
            for idx, val in iv.dict_values.items():
                if len(idx) == 1:
                    phi1[passes_subset[idx[0]]] = float(val)
                elif len(idx) == 2:
                    phi2[f"{passes_subset[idx[0]]}+{passes_subset[idx[1]]}"] = float(val)
                elif len(idx) == 3:
                    phi3[f"{passes_subset[idx[0]]}+{passes_subset[idx[1]]}+{passes_subset[idx[2]]}"] = float(val)

            # Selection at k=10 (or K if K<10)
            sel_k = min(10, K)
            sel = greedy_interaction(phi1, phi2, phi3, sel_k, passes_subset)
            v = evaluate_selection(bpath, sel, all_passes=passes_subset)

            # Variance decomposition
            ss1 = sum(v**2 for v in phi1.values())
            ss2 = sum(v**2 for v in phi2.values())
            ss3 = sum(v**2 for v in phi3.values())
            total = ss1 + ss2 + ss3

            k_results[bname] = {
                'selection_value': float(v),
                'elapsed': float(elapsed),
                'frac_order1': float(ss1/total) if total > 0 else 0,
                'frac_order2': float(ss2/total) if total > 0 else 0,
                'frac_order3': float(ss3/total) if total > 0 else 0,
            }
            print(f"    {bname}: val={v:.4f}, time={elapsed:.1f}s, o3frac={k_results[bname]['frac_order3']:.1%}")

        ablation_results[K] = k_results

    with open(os.path.join(RESULTS_DIR, 'data', 'ablation_num_passes.json'), 'w') as f:
        json.dump(ablation_results, f, indent=2)

    return ablation_results


def ablation_budget(benchmarks):
    """Ablation: vary evaluation budget for Shapley estimation."""
    print("\n" + "="*60)
    print("STEP 7c: Ablation - Evaluation Budget")
    print("="*60)

    import shapiq

    bm_names = list(benchmarks.keys())[:5]
    budgets = [500, 1000, 2000, 3000]
    ablation_results = {}

    for bname in bm_names:
        bpath = benchmarks[bname]
        bm_results = {}

        for budget in budgets:
            game = CompilerGame(bpath)
            n = game.n_players

            t0 = time.time()
            approximator = shapiq.PermutationSamplingSII(
                n=n, max_order=3, random_state=42,
            )
            iv = approximator.approximate(budget=budget, game=game)
            elapsed = time.time() - t0

            phi1 = {}
            phi2 = {}
            phi3 = {}
            for idx, val in iv.dict_values.items():
                if len(idx) == 1:
                    phi1[PASS_LIST[idx[0]]] = float(val)
                elif len(idx) == 2:
                    phi2[f"{PASS_LIST[idx[0]]}+{PASS_LIST[idx[1]]}"] = float(val)
                elif len(idx) == 3:
                    phi3[f"{PASS_LIST[idx[0]]}+{PASS_LIST[idx[1]]}+{PASS_LIST[idx[2]]}"] = float(val)

            sel = greedy_interaction(phi1, phi2, phi3, 10, PASS_LIST)
            v = evaluate_selection(bpath, sel)

            bm_results[budget] = {
                'selection_value': float(v),
                'elapsed': float(elapsed),
            }

        ablation_results[bname] = bm_results
        print(f"  {bname}: " + ", ".join(f"B={b}: {bm_results[b]['selection_value']:.4f}" for b in budgets))

    with open(os.path.join(RESULTS_DIR, 'data', 'ablation_budget.json'), 'w') as f:
        json.dump(ablation_results, f, indent=2)

    return ablation_results


# ============================================================
# STEP 8: Transferability Analysis
# ============================================================
def transferability_analysis(benchmarks, all_interactions):
    """Test if interactions from one benchmark help selection for another."""
    print("\n" + "="*60)
    print("STEP 8: Transferability Analysis")
    print("="*60)

    agg = aggregate_interactions(all_interactions)
    bm_names = list(benchmarks.keys())
    n_bm = len(bm_names)

    # Build interaction vectors for each benchmark
    vectors = {}
    for bname in bm_names:
        bm_agg = agg[bname]
        vec = []
        for k in sorted(bm_agg['order1'].keys()):
            vec.append(bm_agg['order1'][k]['mean'])
        for k in sorted(bm_agg['order2'].keys()):
            vec.append(bm_agg['order2'][k]['mean'])
        vectors[bname] = np.array(vec)

    # Cosine similarity matrix
    sim_matrix = np.zeros((n_bm, n_bm))
    for i, b1 in enumerate(bm_names):
        for j, b2 in enumerate(bm_names):
            v1, v2 = vectors[b1], vectors[b2]
            # Ensure same length
            min_len = min(len(v1), len(v2))
            v1, v2 = v1[:min_len], v2[:min_len]
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            sim_matrix[i, j] = np.dot(v1, v2) / norm if norm > 0 else 0

    # Leave-one-out transfer test
    transfer_results = {}
    for i, bname in enumerate(bm_names):
        bpath = benchmarks[bname]

        # Oracle: use own interactions
        own_phi1 = {k: v['mean'] for k, v in agg[bname]['order1'].items()}
        own_phi2 = {k: v['mean'] for k, v in agg[bname]['order2'].items()}
        own_phi3 = {k: v['mean'] for k, v in agg[bname]['order3'].items()}
        sel_oracle = greedy_interaction(own_phi1, own_phi2, own_phi3, 10, PASS_LIST)
        v_oracle = evaluate_selection(bpath, sel_oracle)

        # Transfer: average interactions from other benchmarks
        avg_phi1 = defaultdict(float)
        avg_phi2 = defaultdict(float)
        avg_phi3 = defaultdict(float)
        for j, other in enumerate(bm_names):
            if other == bname:
                continue
            for k, v in agg[other]['order1'].items():
                avg_phi1[k] += v['mean'] / (n_bm - 1)
            for k, v in agg[other]['order2'].items():
                avg_phi2[k] += v['mean'] / (n_bm - 1)
            for k, v in agg[other]['order3'].items():
                avg_phi3[k] += v['mean'] / (n_bm - 1)

        sel_transfer = greedy_interaction(dict(avg_phi1), dict(avg_phi2), dict(avg_phi3), 10, PASS_LIST)
        v_transfer = evaluate_selection(bpath, sel_transfer)

        # Most similar benchmark
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        most_similar_idx = np.argmax(sims)
        most_similar = bm_names[most_similar_idx]
        ms_phi1 = {k: v['mean'] for k, v in agg[most_similar]['order1'].items()}
        ms_phi2 = {k: v['mean'] for k, v in agg[most_similar]['order2'].items()}
        ms_phi3 = {k: v['mean'] for k, v in agg[most_similar]['order3'].items()}
        sel_similar = greedy_interaction(ms_phi1, ms_phi2, ms_phi3, 10, PASS_LIST)
        v_similar = evaluate_selection(bpath, sel_similar)

        transfer_results[bname] = {
            'oracle': float(v_oracle),
            'transfer_avg': float(v_transfer),
            'transfer_similar': float(v_similar),
            'most_similar_benchmark': most_similar,
            'similarity': float(sims[most_similar_idx]),
            'transfer_ratio_avg': float(v_transfer / v_oracle) if v_oracle > 0 else 0,
            'transfer_ratio_similar': float(v_similar / v_oracle) if v_oracle > 0 else 0,
        }
        print(f"  {bname}: oracle={v_oracle:.4f}, transfer_avg={v_transfer:.4f}, "
              f"similar({most_similar})={v_similar:.4f}")

    transfer_results['_similarity_matrix'] = {
        'benchmarks': bm_names,
        'matrix': sim_matrix.tolist(),
    }

    with open(os.path.join(RESULTS_DIR, 'data', 'transferability.json'), 'w') as f:
        json.dump(transfer_results, f, indent=2)

    return transfer_results


# ============================================================
# STEP 9: Statistical Evaluation
# ============================================================
def statistical_evaluation(all_interactions, decomp, selection_results, baseline_results, benchmarks, opt_levels):
    """Evaluate results against success criteria."""
    print("\n" + "="*60)
    print("STEP 9: Statistical Evaluation")
    print("="*60)

    agg = aggregate_interactions(all_interactions)
    bm_names = [b for b in benchmarks.keys()]

    # Criterion 1: Significant order-3 interactions
    n_significant_triples = 0
    n_total_triples = 0
    for bname in bm_names:
        for key, vals in agg[bname]['order3'].items():
            n_total_triples += 1
            if vals['std'] > 0 and abs(vals['mean']) > 2 * vals['std']:
                n_significant_triples += 1
            elif vals['std'] == 0 and abs(vals['mean']) > 1e-6:
                n_significant_triples += 1

    frac_significant = n_significant_triples / n_total_triples if n_total_triples > 0 else 0
    criterion1 = frac_significant >= 0.30
    print(f"\n  Criterion 1: {n_significant_triples}/{n_total_triples} ({frac_significant:.1%}) order-3 interactions significant")
    print(f"    Result: {'CONFIRMED' if criterion1 else 'NOT CONFIRMED'} (threshold: 30%)")

    # Criterion 2: Variance explained by order 3
    frac3_values = [decomp[b]['frac_order3'] for b in bm_names]
    avg_frac3 = np.mean(frac3_values)
    criterion2 = avg_frac3 >= 0.10
    print(f"\n  Criterion 2: Order-3 variance fraction = {avg_frac3:.1%}")
    print(f"    Result: {'CONFIRMED' if criterion2 else 'NOT CONFIRMED'} (threshold: 10%)")

    # Criterion 3: Interaction-guided vs pairwise win rate
    wins = 0
    total = 0
    for bname in bm_names:
        if str(10) in selection_results.get(bname, {}):
            v_pair = selection_results[bname][str(10)]['pairwise_greedy']['value']
            v_inter = selection_results[bname][str(10)]['interaction_greedy']['value']
        elif 10 in selection_results.get(bname, {}):
            v_pair = selection_results[bname][10]['pairwise_greedy']['value']
            v_inter = selection_results[bname][10]['interaction_greedy']['value']
        else:
            continue
        total += 1
        if v_inter >= v_pair:
            wins += 1

    win_rate = wins / total if total > 0 else 0
    criterion3 = win_rate >= 0.60
    print(f"\n  Criterion 3: Interaction vs pairwise win rate = {wins}/{total} ({win_rate:.1%})")
    print(f"    Result: {'CONFIRMED' if criterion3 else 'NOT CONFIRMED'} (threshold: 60%)")

    # Main comparison table
    main_table = {}
    for bname in bm_names:
        bpath = benchmarks[bname]
        levels = opt_levels[bname]
        baseline = levels['O0']
        row = {}

        # LLVM levels
        for lvl in ['O1', 'O2', 'O3', 'Os', 'Oz']:
            if levels.get(lvl):
                row[lvl] = float((baseline - levels[lvl]) / baseline)
            else:
                row[lvl] = None

        # Selection methods (k=10)
        k = 10
        if k in selection_results.get(bname, {}):
            for method in ['individual_greedy', 'pairwise_greedy', 'interaction_greedy', 'synergy_seeded']:
                row[method] = selection_results[bname][k][method]['value']
        elif str(k) in selection_results.get(bname, {}):
            for method in ['individual_greedy', 'pairwise_greedy', 'interaction_greedy', 'synergy_seeded']:
                row[method] = selection_results[bname][str(k)][method]['value']

        # Baselines (k=10)
        if k in baseline_results.get(bname, {}):
            row['random_search'] = baseline_results[bname][k]['random_search']['mean']
            row['genetic_algorithm'] = baseline_results[bname][k]['genetic_algorithm']['mean']
        elif str(k) in baseline_results.get(bname, {}):
            row['random_search'] = baseline_results[bname][str(k)]['random_search']['mean']
            row['genetic_algorithm'] = baseline_results[bname][str(k)]['genetic_algorithm']['mean']

        main_table[bname] = row

    # Compute averages
    methods = ['O1', 'O2', 'O3', 'Os', 'Oz', 'random_search', 'genetic_algorithm',
               'individual_greedy', 'pairwise_greedy', 'interaction_greedy', 'synergy_seeded']
    avg_row = {}
    for method in methods:
        vals = [main_table[b].get(method, None) for b in bm_names]
        vals = [v for v in vals if v is not None]
        if vals:
            avg_row[method] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    main_table['_average'] = avg_row

    eval_results = {
        'criterion1_significant_triples': {
            'n_significant': n_significant_triples,
            'n_total': n_total_triples,
            'fraction': float(frac_significant),
            'confirmed': criterion1,
            'threshold': 0.30,
        },
        'criterion2_variance_order3': {
            'avg_frac_order3': float(avg_frac3),
            'per_benchmark': {b: float(v) for b, v in zip(bm_names, frac3_values)},
            'confirmed': criterion2,
            'threshold': 0.10,
        },
        'criterion3_selection_win_rate': {
            'wins': wins,
            'total': total,
            'win_rate': float(win_rate),
            'confirmed': criterion3,
            'threshold': 0.60,
        },
        'main_table': main_table,
        'overall_confirmed': criterion1 and criterion2 and criterion3,
    }

    with open(os.path.join(RESULTS_DIR, 'data', 'statistical_tests.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)

    # Save table as CSV
    import csv
    csv_path = os.path.join(RESULTS_DIR, 'tables', 'main_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Benchmark'] + methods
        writer.writerow(header)
        for bname in bm_names:
            row = [bname]
            for m in methods:
                v = main_table[bname].get(m, None)
                row.append(f"{v:.4f}" if v is not None else "N/A")
            writer.writerow(row)
        # Average row
        row = ['Average']
        for m in methods:
            if m in avg_row:
                row.append(f"{avg_row[m]['mean']:.4f}±{avg_row[m]['std']:.4f}")
            else:
                row.append("N/A")
        writer.writerow(row)

    print(f"\n  Results saved to {csv_path}")
    return eval_results


# ============================================================
# STEP 10: Visualization
# ============================================================
def generate_figures(all_interactions, decomp, selection_results, baseline_results,
                     benchmarks, opt_levels, transfer_results, ablation_order, ablation_budget_res):
    """Generate all publication figures."""
    print("\n" + "="*60)
    print("STEP 10: Generating Figures")
    print("="*60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'figure.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
    })

    agg = aggregate_interactions(all_interactions)
    bm_names = [b for b in benchmarks.keys()]

    # ---- Figure 1: Variance Decomposition ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bm_names))
    frac1 = [decomp[b]['frac_order1'] for b in bm_names]
    frac2 = [decomp[b]['frac_order2'] for b in bm_names]
    frac3 = [decomp[b]['frac_order3'] for b in bm_names]

    ax.bar(x, frac1, label='Order 1 (Individual)', color='#4C72B0')
    ax.bar(x, frac2, bottom=frac1, label='Order 2 (Pairwise)', color='#DD8452')
    ax.bar(x, frac3, bottom=[f1+f2 for f1, f2 in zip(frac1, frac2)], label='Order 3 (Triple)', color='#C44E52')
    ax.set_xticks(x)
    ax.set_xticklabels(bm_names, rotation=45, ha='right')
    ax.set_ylabel('Fraction of Interaction Variance')
    ax.set_title('Variance Decomposition by Interaction Order')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'variance_decomposition.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'variance_decomposition.png'))
    plt.close()
    print("  Saved variance_decomposition figure")

    # ---- Figure 2: Interaction Heatmap (2 benchmarks) ----
    for bname in bm_names[:2]:
        phi2 = agg[bname]['order2']
        n_passes = len(PASS_LIST)
        heatmap = np.zeros((n_passes, n_passes))
        for key, v in phi2.items():
            parts = key.split('+')
            if len(parts) == 2:
                i = PASS_LIST.index(parts[0]) if parts[0] in PASS_LIST else None
                j = PASS_LIST.index(parts[1]) if parts[1] in PASS_LIST else None
                if i is not None and j is not None:
                    heatmap[i, j] = v['mean']
                    heatmap[j, i] = v['mean']

        fig, ax = plt.subplots(figsize=(12, 10))
        vmax = max(abs(heatmap.min()), abs(heatmap.max())) or 0.01
        sns.heatmap(heatmap, xticklabels=PASS_LIST, yticklabels=PASS_LIST,
                     cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax, ax=ax,
                     square=True, linewidths=0.5)
        ax.set_title(f'Pairwise Shapley Interactions: {bname}')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'heatmap_{bname}.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, f'heatmap_{bname}.png'))
        plt.close()
    print("  Saved heatmap figures")

    # ---- Figure 3: Top Order-3 Interactions ----
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
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Shapley Interaction Index')
        ax.set_title('Top 15 Order-3 Interactions')
        ax.axvline(x=0, color='black', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'top_order3_interactions.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, 'top_order3_interactions.png'))
        plt.close()
        print("  Saved top_order3_interactions figure")

    # ---- Figure 4: Selection Performance Comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 4a: Bar chart at k=10
    ax = axes[0]
    methods_to_plot = ['O3', 'random_search', 'genetic_algorithm',
                       'individual_greedy', 'pairwise_greedy', 'interaction_greedy', 'synergy_seeded']
    method_labels = ['O3', 'Random\nSearch', 'GA', 'Indiv.\nGreedy', 'Pairwise\nGreedy',
                     'Interaction\nGreedy', 'Synergy\nSeeded']
    avg_vals = []
    std_vals = []
    for m in methods_to_plot:
        vals = []
        for bname in bm_names:
            if m in ['O1', 'O2', 'O3', 'Os', 'Oz']:
                levels = opt_levels[bname]
                if levels.get(m):
                    vals.append((levels['O0'] - levels[m]) / levels['O0'])
            elif m in ['random_search', 'genetic_algorithm']:
                k = 10
                bres = baseline_results.get(bname, {})
                if k in bres:
                    vals.append(bres[k][m]['mean'])
                elif str(k) in bres:
                    vals.append(bres[str(k)][m]['mean'])
            else:
                k = 10
                sres = selection_results.get(bname, {})
                if k in sres:
                    vals.append(sres[k][m]['value'])
                elif str(k) in sres:
                    vals.append(sres[str(k)][m]['value'])
        avg_vals.append(np.mean(vals) if vals else 0)
        std_vals.append(np.std(vals) if vals else 0)

    colors = ['#55A868', '#CCB974', '#CCB974', '#4C72B0', '#DD8452', '#C44E52', '#8172B3']
    x = np.arange(len(methods_to_plot))
    ax.bar(x, avg_vals, yerr=std_vals, color=colors, alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylabel('IR Reduction Fraction')
    ax.set_title('Method Comparison (k=10 passes)')

    # 4b: Performance vs. pass budget k
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
                sres = selection_results.get(bname, {})
                if k in sres:
                    vals.append(sres[k][method]['value'])
                elif str(k) in sres:
                    vals.append(sres[str(k)][method]['value'])
            avg_by_k.append(np.mean(vals) if vals else 0)
        ax.plot(K_BUDGETS, avg_by_k, marker='o', label=label, color=color, linestyle=ls)

    ax.set_xlabel('Number of Selected Passes (k)')
    ax.set_ylabel('IR Reduction Fraction')
    ax.set_title('Selection Performance vs. Budget')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'selection_comparison.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'selection_comparison.png'))
    plt.close()
    print("  Saved selection_comparison figure")

    # ---- Figure 5: Interaction Network ----
    import networkx as nx
    for bname in bm_names[:1]:
        phi1 = agg[bname]['order1']
        phi2 = agg[bname]['order2']

        G = nx.Graph()
        for p in PASS_LIST:
            if p in phi1:
                G.add_node(p, weight=phi1[p]['mean'])

        for key, v in phi2.items():
            parts = key.split('+')
            if len(parts) == 2 and abs(v['mean']) > 0.001:
                G.add_edge(parts[0], parts[1], weight=v['mean'])

        fig, ax = plt.subplots(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42, k=2)

        # Node sizes proportional to Shapley value
        node_sizes = [abs(G.nodes[n].get('weight', 0)) * 5000 + 100 for n in G.nodes]
        node_colors = ['#C44E52' if G.nodes[n].get('weight', 0) > 0 else '#4C72B0' for n in G.nodes]

        # Edge widths/colors proportional to interaction
        edges = G.edges()
        if edges:
            edge_weights = [G[u][v]['weight'] for u, v in edges]
            edge_widths = [abs(w) * 50 + 0.5 for w in edge_weights]
            edge_colors = ['#C44E52' if w > 0 else '#4C72B0' for w in edge_weights]
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=edge_widths,
                                    edge_color=edge_colors, alpha=0.5, ax=ax)

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        ax.set_title(f'Pass Interaction Network: {bname}')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'network_{bname}.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, f'network_{bname}.png'))
        plt.close()
    print("  Saved network figure")

    # ---- Figure 6: Transferability Similarity Matrix ----
    if '_similarity_matrix' in transfer_results:
        sim = np.array(transfer_results['_similarity_matrix']['matrix'])
        bnames_t = transfer_results['_similarity_matrix']['benchmarks']
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(sim, xticklabels=bnames_t, yticklabels=bnames_t,
                     cmap='YlOrRd', vmin=0, vmax=1, annot=True, fmt='.2f', ax=ax, square=True)
        ax.set_title('Benchmark Similarity (Cosine of Interaction Vectors)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'transferability.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, 'transferability.png'))
        plt.close()
        print("  Saved transferability figure")

    # ---- Figure 7: Convergence (Ablation Budget) ----
    if ablation_budget_res:
        fig, ax = plt.subplots(figsize=(8, 5))
        budgets = [500, 1000, 2000, 3000]
        for bname in list(ablation_budget_res.keys())[:5]:
            vals = []
            for b in budgets:
                bkey = b if b in ablation_budget_res[bname] else str(b)
                vals.append(ablation_budget_res[bname][bkey]['selection_value'])
            ax.plot(budgets, vals, marker='o', label=bname)
        ax.set_xlabel('Evaluation Budget')
        ax.set_ylabel('Selection Performance (k=10)')
        ax.set_title('Convergence: Selection Performance vs. Budget')
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'convergence.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, 'convergence.png'))
        plt.close()
        print("  Saved convergence figure")

    # ---- Figure 8: Ablation - Interaction Order ----
    if ablation_order:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(bm_names))
        width = 0.25
        for i, (order_key, label, color) in enumerate([
            ('order1', 'Order 1 Only', '#4C72B0'),
            ('order1_2', 'Order 1+2', '#DD8452'),
            ('order1_2_3', 'Order 1+2+3', '#C44E52'),
        ]):
            vals = []
            for bname in bm_names:
                if bname in ablation_order:
                    k = 10
                    bkey = k if k in ablation_order[bname] else str(k)
                    vals.append(ablation_order[bname][bkey][order_key])
                else:
                    vals.append(0)
            ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(bm_names, rotation=45, ha='right')
        ax.set_ylabel('IR Reduction Fraction')
        ax.set_title('Ablation: Effect of Interaction Order (k=10)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'ablation_order.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, 'ablation_order.png'))
        plt.close()
        print("  Saved ablation_order figure")

    print("\n  All figures generated.")


# ============================================================
# STEP 11: Aggregate Results
# ============================================================
def save_final_results(eval_results, decomp, selection_results, baseline_results,
                       benchmarks, opt_levels, ablation_order):
    """Save the final aggregated results.json."""
    bm_names = list(benchmarks.keys())

    # Build comprehensive results
    results = {
        "experiment": "ShapleyPass: Compiler Pass Interaction Analysis",
        "benchmarks": bm_names,
        "n_passes": len(PASS_LIST),
        "passes": PASS_LIST,
        "seeds": SEEDS,
        "shapley_budget": SHAPLEY_BUDGET,

        "hypothesis_evaluation": {
            "criterion1_significant_order3": eval_results['criterion1_significant_triples'],
            "criterion2_variance_order3": eval_results['criterion2_variance_order3'],
            "criterion3_selection_win_rate": eval_results['criterion3_selection_win_rate'],
            "overall_confirmed": eval_results['overall_confirmed'],
        },

        "variance_decomposition": {
            b: {
                "order1": decomp[b]['frac_order1'],
                "order2": decomp[b]['frac_order2'],
                "order3": decomp[b]['frac_order3'],
            } for b in bm_names
        },

        "main_comparison": eval_results['main_table'],

        "ablation_interaction_order": {},
    }

    # Ablation summary
    for bname in bm_names:
        if bname in ablation_order:
            k = 10
            bkey = k if k in ablation_order[bname] else str(k)
            results["ablation_interaction_order"][bname] = ablation_order[bname][bkey]

    # Save
    results_path = os.path.join(os.path.dirname(__file__), '..', 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Final results saved to {results_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    print("ShapleyPass Experiment Pipeline")
    print("="*60)

    benchmarks = get_benchmark_files()
    print(f"Found {len(benchmarks)} benchmarks: {list(benchmarks.keys())}")

    # Step 1: Pass screening
    screening, opt_levels = screen_passes(benchmarks)

    # Step 2: Shapley interactions
    all_interactions = compute_shapley_interactions(benchmarks)

    # Step 3: Variance decomposition
    decomp = variance_decomposition(all_interactions)

    # Step 4: Interaction structure
    structure = interaction_structure_analysis(all_interactions)

    # Step 5: Selection algorithms
    selection_results = run_selection_experiments(benchmarks, all_interactions)

    # Step 6: Baselines
    baseline_results = run_baselines(benchmarks)

    # Step 7: Ablations
    ablation_order = ablation_interaction_order(benchmarks, all_interactions)
    ablation_np = ablation_num_passes(benchmarks)
    ablation_budget_res = ablation_budget(benchmarks)

    # Step 8: Transferability
    transfer_results = transferability_analysis(benchmarks, all_interactions)

    # Step 9: Statistical evaluation
    eval_results = statistical_evaluation(
        all_interactions, decomp, selection_results, baseline_results, benchmarks, opt_levels
    )

    # Step 10: Figures
    generate_figures(all_interactions, decomp, selection_results, baseline_results,
                     benchmarks, opt_levels, transfer_results, ablation_order, ablation_budget_res)

    # Step 11: Final results
    save_final_results(eval_results, decomp, selection_results, baseline_results,
                       benchmarks, opt_levels, ablation_order)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Total experiment time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

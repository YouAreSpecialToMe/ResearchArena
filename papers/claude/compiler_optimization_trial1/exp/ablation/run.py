"""Ablation study: contribution of each algebraic component.

Tests: full method, no synergy chaining, no anti-interference,
no idempotency pruning, top-K only.
"""
import sys
import csv
import json
import time
import random
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from exp.shared.utils import *

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Import categories from algebra ordering
CANONICALIZATION_PASSES = {
    'mem2reg', 'sroa', 'early-cse', 'simplifycfg', 'loop-simplify',
    'lcssa', 'loop-rotate', 'instsimplify',
}
CORE_OPTIMIZATION_PASSES = {
    'instcombine', 'gvn', 'newgvn', 'sccp', 'correlated-propagation',
    'jump-threading', 'licm', 'indvars', 'loop-idiom', 'loop-deletion',
    'loop-reduce', 'reassociate', 'aggressive-instcombine',
    'constraint-elimination', 'nary-reassociate', 'float2int',
    'div-rem-pairs', 'gvn-hoist', 'gvn-sink',
}
CLEANUP_PASSES = {
    'adce', 'bdce', 'dce', 'dse', 'deadargelim', 'globalopt',
    'constmerge', 'memcpyopt', 'tailcallelim', 'sink', 'mergereturn',
}


def load_interference_matrix():
    matrix = {}
    path = RESULTS_DIR / "interference_matrix.csv"
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        passes_list = header[1:]
        for row in reader:
            pi = row[0]
            for j, pj in enumerate(passes_list):
                matrix[(pi, pj)] = float(row[j+1])
    return matrix, passes_list


def load_idempotency_summary():
    summary = {}
    path = RESULTS_DIR / "idempotency_summary.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary[row['pass_name']] = {
                'structural_rate': float(row['structural_idempotency_rate']),
                'instcount_rate': float(row['instcount_idempotency_rate']),
                'classification': row['classification']
            }
    return summary


def load_single_pass_effects():
    effects = {}
    path = RESULTS_DIR / "interference_raw.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pi = row['pass_i']
            pj = row['pass_j']
            baseline = int(row['baseline_instcount'])
            for p, after_key in [(pi, 'after_i_instcount'), (pj, 'after_j_instcount')]:
                after = int(row[after_key])
                delta_pct = (baseline - after) / baseline * 100 if baseline > 0 else 0
                if p not in effects:
                    effects[p] = []
                effects[p].append(delta_pct)
    return {p: sum(v)/len(v) for p, v in effects.items()}


def apply_sequence(benchmark_path, sequence):
    ir_text = get_baseline_ir(benchmark_path)
    for pass_name in sequence:
        ir_new, _, _ = apply_pass_to_ir_text(ir_text, pass_name)
        if ir_new is None:
            continue
        ir_text = ir_new
    return count_instructions(ir_text)


def prune_passes(passes, idem_summary, pass_effects):
    """Prune useless passes based on idempotency and effect."""
    useful = [p for p in passes if pass_effects.get(p, 0) > 0.05 or
              idem_summary.get(p, {}).get('classification', '') != 'strongly_idempotent']
    return useful if useful else passes[:]


def synergy_sort(pass_list, interf_matrix, pass_effects, rng):
    """Sort passes by synergy chaining with anti-interference."""
    if len(pass_list) <= 1:
        return pass_list
    ranked = sorted(pass_list, key=lambda p: -pass_effects.get(p, 0))
    result = [ranked[0]]
    remaining = set(ranked[1:])
    while remaining:
        current = result[-1]
        best_p = None
        best_score = float('-inf')
        for p in remaining:
            synergy = interf_matrix.get((current, p), 0)
            benefit = pass_effects.get(p, 0)
            anti = 0
            for recent in result[-2:]:
                val = interf_matrix.get((recent, p), 0)
                if val < -3:
                    anti += abs(val)
            score = synergy * 1.5 + benefit * 0.5 - anti + rng.random() * 0.05
            if score > best_score:
                best_score = score
                best_p = p
        result.append(best_p)
        remaining.discard(best_p)
    return result


def build_full_sequence(passes, interf_matrix, idem_summary, pass_effects, seed, max_len=30):
    """Full algebra-guided: pruning + phase-based + synergy + anti-interference + repetition."""
    rng = random.Random(seed)
    useful = prune_passes(passes, idem_summary, pass_effects)

    canon = [p for p in useful if p in CANONICALIZATION_PASSES]
    core = [p for p in useful if p in CORE_OPTIMIZATION_PASSES]
    cleanup = [p for p in useful if p in CLEANUP_PASSES]

    canon_ordered = synergy_sort(canon, interf_matrix, pass_effects, rng)
    core_ordered = synergy_sort(core, interf_matrix, pass_effects, rng)
    cleanup_ordered = synergy_sort(cleanup, interf_matrix, pass_effects, rng)

    sequence = list(canon_ordered)
    if 'instcombine' in useful:
        sequence.append('instcombine')
    if 'simplifycfg' in useful:
        sequence.append('simplifycfg')
    sequence.extend(core_ordered)
    if 'simplifycfg' in useful:
        sequence.append('simplifycfg')
    if 'instcombine' in useful:
        sequence.append('instcombine')
    sequence.extend(cleanup_ordered)
    if 'simplifycfg' in useful:
        sequence.append('simplifycfg')

    return sequence[:max_len]


def build_no_synergy_sequence(passes, interf_matrix, idem_summary, pass_effects, seed, max_len=30):
    """Phase-based ordering but random within each phase (no synergy chaining)."""
    rng = random.Random(seed)
    useful = prune_passes(passes, idem_summary, pass_effects)

    canon = [p for p in useful if p in CANONICALIZATION_PASSES]
    core = [p for p in useful if p in CORE_OPTIMIZATION_PASSES]
    cleanup = [p for p in useful if p in CLEANUP_PASSES]

    rng.shuffle(canon)
    rng.shuffle(core)
    rng.shuffle(cleanup)

    sequence = list(canon)
    if 'instcombine' in useful:
        sequence.append('instcombine')
    if 'simplifycfg' in useful:
        sequence.append('simplifycfg')
    sequence.extend(core)
    if 'simplifycfg' in useful:
        sequence.append('simplifycfg')
    if 'instcombine' in useful:
        sequence.append('instcombine')
    sequence.extend(cleanup)
    if 'simplifycfg' in useful:
        sequence.append('simplifycfg')

    return sequence[:max_len]


def build_no_anti_sequence(passes, interf_matrix, idem_summary, pass_effects, seed, max_len=30):
    """Synergy chain but don't avoid destructive pairs."""
    rng = random.Random(seed)
    useful = prune_passes(passes, idem_summary, pass_effects)

    canon = [p for p in useful if p in CANONICALIZATION_PASSES]
    core = [p for p in useful if p in CORE_OPTIMIZATION_PASSES]
    cleanup = [p for p in useful if p in CLEANUP_PASSES]

    def synergy_only_sort(pass_list):
        if len(pass_list) <= 1:
            return pass_list
        ranked = sorted(pass_list, key=lambda p: -pass_effects.get(p, 0))
        result = [ranked[0]]
        remaining = set(ranked[1:])
        while remaining:
            current = result[-1]
            best_p = max(remaining, key=lambda p: interf_matrix.get((current, p), 0) +
                        pass_effects.get(p, 0) * 0.5 + rng.random() * 0.05)
            result.append(best_p)
            remaining.discard(best_p)
        return result

    sequence = synergy_only_sort(canon)
    if 'instcombine' in useful:
        sequence.append('instcombine')
    if 'simplifycfg' in useful:
        sequence.append('simplifycfg')
    sequence.extend(synergy_only_sort(core))
    if 'simplifycfg' in useful:
        sequence.append('simplifycfg')
    if 'instcombine' in useful:
        sequence.append('instcombine')
    sequence.extend(synergy_only_sort(cleanup))
    if 'simplifycfg' in useful:
        sequence.append('simplifycfg')

    return sequence[:max_len]


def build_no_pruning_sequence(passes, interf_matrix, idem_summary, pass_effects, seed, max_len=30):
    """Include ALL passes without idempotency-based pruning."""
    rng = random.Random(seed)

    canon = [p for p in passes if p in CANONICALIZATION_PASSES]
    core = [p for p in passes if p in CORE_OPTIMIZATION_PASSES]
    cleanup = [p for p in passes if p in CLEANUP_PASSES]
    other = [p for p in passes if p not in CANONICALIZATION_PASSES
             and p not in CORE_OPTIMIZATION_PASSES and p not in CLEANUP_PASSES]

    sequence = synergy_sort(canon, interf_matrix, pass_effects, rng)
    if 'instcombine' in passes:
        sequence.append('instcombine')
    if 'simplifycfg' in passes:
        sequence.append('simplifycfg')
    sequence.extend(synergy_sort(core, interf_matrix, pass_effects, rng))
    if 'simplifycfg' in passes:
        sequence.append('simplifycfg')
    if 'instcombine' in passes:
        sequence.append('instcombine')
    sequence.extend(synergy_sort(cleanup, interf_matrix, pass_effects, rng))
    sequence.extend(sorted(other, key=lambda p: -pass_effects.get(p, 0)))
    if 'simplifycfg' in passes:
        sequence.append('simplifycfg')

    return sequence[:max_len]


def build_no_phases_sequence(passes, interf_matrix, idem_summary, pass_effects, seed, max_len=30):
    """Pruning + synergy + anti-interference but NO phase-based ordering (flat synergy chain)."""
    rng = random.Random(seed)
    useful = prune_passes(passes, idem_summary, pass_effects)
    sequence = synergy_sort(useful, interf_matrix, pass_effects, rng)
    return sequence[:max_len]


def build_top_k_sequence(passes, interf_matrix, pass_effects, seed, k=10):
    """Use only the top-K most beneficial passes ordered by synergy."""
    rng = random.Random(seed)
    ranked = sorted(passes, key=lambda p: -pass_effects.get(p, 0))
    top_k = ranked[:k]
    return synergy_sort(top_k, interf_matrix, pass_effects, rng)


def main():
    benchmarks = get_benchmark_files()
    passes = get_pass_list()
    seeds = [42, 123, 456]

    interf_matrix, _ = load_interference_matrix()
    idem_summary = load_idempotency_summary()
    pass_effects = load_single_pass_effects()

    # Use 30 benchmarks: all PolyBench + some custom
    polybench = [b for b in benchmarks if b.stem.startswith("pb_")]
    custom = [b for b in benchmarks if not b.stem.startswith("pb_")]
    custom_indices = list(range(0, len(custom), max(1, len(custom) // 10)))[:10]
    benchmarks_subset = polybench + [custom[i] for i in custom_indices]

    variants = {
        'full': lambda seed: build_full_sequence(passes, interf_matrix, idem_summary, pass_effects, seed),
        'no_synergy': lambda seed: build_no_synergy_sequence(passes, interf_matrix, idem_summary, pass_effects, seed),
        'no_anti_interference': lambda seed: build_no_anti_sequence(passes, interf_matrix, idem_summary, pass_effects, seed),
        'no_pruning': lambda seed: build_no_pruning_sequence(passes, interf_matrix, idem_summary, pass_effects, seed),
        'no_phases': lambda seed: build_no_phases_sequence(passes, interf_matrix, idem_summary, pass_effects, seed),
        'top_k_only': lambda seed: build_top_k_sequence(passes, interf_matrix, pass_effects, seed),
    }

    start_time = time.time()
    all_results = []

    log_file = open(LOG_DIR / "ablation.log", 'w')

    for variant_name, builder in variants.items():
        print(f"Running ablation variant: {variant_name}")
        for seed in seeds:
            sequence = builder(seed)
            log_file.write(f"{variant_name} (seed={seed}): {sequence}\n")
            for bm in benchmarks_subset:
                ic = apply_sequence(str(bm), sequence)
                baseline_ic = count_instructions(get_baseline_ir(str(bm)))
                ratio = round(ic / baseline_ic, 4) if baseline_ic and baseline_ic > 0 and ic else None
                all_results.append({
                    'variant': variant_name,
                    'seed': seed,
                    'benchmark': bm.stem,
                    'instcount': ic,
                    'baseline_O0': baseline_ic,
                    'ratio': ratio,
                    'num_passes': len(sequence)
                })

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")
    log_file.write(f"\nRuntime: {elapsed:.1f}s\n")
    log_file.close()

    # Save raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "ablation_raw.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['variant', 'seed', 'benchmark',
                                                'instcount', 'baseline_O0', 'ratio', 'num_passes'])
        writer.writeheader()
        writer.writerows(all_results)

    # Summary
    print("\n=== Ablation Summary ===")
    summary = []
    for variant_name in variants:
        v_results = [r for r in all_results if r['variant'] == variant_name and r['ratio'] is not None]
        ratios = [r['ratio'] for r in v_results]
        geo_mean = math.exp(sum(math.log(max(r, 0.001)) for r in ratios) / len(ratios)) if ratios else 1.0
        mean_passes = sum(r['num_passes'] for r in v_results) / len(v_results) if v_results else 0

        seed_geos = []
        for seed in seeds:
            s_ratios = [r['ratio'] for r in v_results if r['seed'] == seed]
            if s_ratios:
                seed_geos.append(math.exp(sum(math.log(max(r, 0.001)) for r in s_ratios) / len(s_ratios)))
        std_geo = (sum((g - geo_mean)**2 for g in seed_geos) / len(seed_geos))**0.5 if len(seed_geos) > 1 else 0

        summary.append({
            'variant': variant_name,
            'geo_mean_ratio': round(geo_mean, 4),
            'std_ratio': round(std_geo, 4),
            'reduction_pct': round((1-geo_mean)*100, 2),
            'mean_num_passes': round(mean_passes, 1)
        })
        print(f"  {variant_name:25s}: {geo_mean:.4f} +/- {std_geo:.4f} ({(1-geo_mean)*100:.1f}% reduction, {mean_passes:.0f} passes)")

    with open(RESULTS_DIR / "ablation_summary.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['variant', 'geo_mean_ratio', 'std_ratio',
                                                'reduction_pct', 'mean_num_passes'])
        writer.writeheader()
        writer.writerows(summary)

    exp_results = {
        'experiment': 'ablation',
        'num_benchmarks': len(benchmarks_subset),
        'runtime_seconds': round(elapsed, 1),
        'variants': summary
    }
    with open(Path(__file__).parent / 'results.json', 'w') as f:
        json.dump(exp_results, f, indent=2)


if __name__ == '__main__':
    main()

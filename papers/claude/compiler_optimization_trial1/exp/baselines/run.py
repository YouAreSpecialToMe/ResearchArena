"""Baseline experiments: standard optimization levels, random ordering, greedy ordering."""
import sys
import csv
import json
import time
import random
import tempfile
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from exp.shared.utils import *


def run_standard_opts(benchmark_path):
    """Run standard optimization levels on a benchmark."""
    bm = Path(benchmark_path).stem
    baseline_ir = get_baseline_ir(benchmark_path)
    baseline_ic = count_instructions(baseline_ir)

    results = {'benchmark': bm, 'O0': baseline_ic}
    for level in ['-O1', '-O2', '-O3', '-Oz']:
        ir = run_opt_pipeline(str(benchmark_path), level, timeout=60)
        if ir is not None:
            results[level.replace('-', '')] = count_instructions(ir)
        else:
            results[level.replace('-', '')] = None
    return results


def run_random_ordering(benchmark_path, passes, seed, num_passes=30):
    """Apply a random sequence of passes."""
    bm = Path(benchmark_path).stem
    rng = random.Random(seed)
    sequence = [rng.choice(passes) for _ in range(num_passes)]

    ir_text = get_baseline_ir(benchmark_path)
    for pass_name in sequence:
        ir_new, _, _ = apply_pass_to_ir_text(ir_text, pass_name)
        if ir_new is None:
            break
        ir_text = ir_new

    ic = count_instructions(ir_text)
    return {'benchmark': bm, 'seed': seed, 'instcount': ic, 'num_passes': num_passes}


def run_greedy_ordering(benchmark_path, passes, max_steps=15):
    """Greedy ordering: at each step, pick the pass with greatest IC reduction."""
    bm = Path(benchmark_path).stem
    ir_text = get_baseline_ir(benchmark_path)
    baseline_ic = count_instructions(ir_text)
    current_ic = baseline_ic
    sequence = []

    for step in range(max_steps):
        best_pass = None
        best_ic = current_ic
        best_ir = ir_text

        for pass_name in passes:
            new_ir, new_ic, _ = apply_pass_to_ir_text(ir_text, pass_name)
            if new_ir is not None and new_ic is not None and new_ic < best_ic:
                best_pass = pass_name
                best_ic = new_ic
                best_ir = new_ir

        if best_pass is None or best_ic >= current_ic:
            break

        sequence.append(best_pass)
        ir_text = best_ir
        current_ic = best_ic

    return {
        'benchmark': bm, 'instcount': current_ic,
        'num_passes': len(sequence), 'sequence': ','.join(sequence)
    }


def main():
    benchmarks = get_benchmark_files()
    passes = get_pass_list()
    seeds = [42, 123, 456]

    start_time = time.time()

    # 1. Standard optimization levels
    print("Running standard optimization levels...")
    std_results = []
    for bm in benchmarks:
        result = run_standard_opts(str(bm))
        std_results.append(result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "baseline_opt_levels.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['benchmark', 'O0', 'O1', 'O2', 'O3', 'Oz'])
        writer.writeheader()
        writer.writerows(std_results)
    print(f"  Done ({time.time()-start_time:.1f}s)")

    # 2. Random ordering
    print("Running random ordering baseline...")
    random_results = []
    for seed in seeds:
        for bm in benchmarks:
            result = run_random_ordering(str(bm), passes, seed)
            random_results.append(result)

    with open(RESULTS_DIR / "baseline_random.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['benchmark', 'seed', 'instcount', 'num_passes'])
        writer.writeheader()
        writer.writerows(random_results)
    print(f"  Done ({time.time()-start_time:.1f}s)")

    # 3. Greedy ordering (on 20-benchmark subset due to cost)
    print("Running greedy ordering baseline...")
    subset_indices = list(range(0, len(benchmarks), max(1, len(benchmarks)//20)))[:20]
    benchmarks_subset = [benchmarks[i] for i in subset_indices]

    greedy_results = []
    for i, bm in enumerate(benchmarks_subset):
        result = run_greedy_ordering(str(bm), passes)
        greedy_results.append(result)
        if (i+1) % 5 == 0:
            print(f"  Greedy: {i+1}/{len(benchmarks_subset)}")

    with open(RESULTS_DIR / "baseline_greedy.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['benchmark', 'instcount', 'num_passes', 'sequence'])
        writer.writeheader()
        writer.writerows(greedy_results)

    elapsed = time.time() - start_time
    print(f"All baselines completed in {elapsed:.1f}s")

    # Compile comparison table
    comparison = []
    for sr in std_results:
        bm = sr['benchmark']
        # Get random results for this benchmark
        rand_ics = [r['instcount'] for r in random_results if r['benchmark'] == bm]
        rand_mean = sum(rand_ics) / len(rand_ics) if rand_ics else None
        rand_std = (sum((x - rand_mean)**2 for x in rand_ics) / len(rand_ics))**0.5 if rand_ics and len(rand_ics) > 1 else 0

        # Get greedy result
        greedy_r = [r for r in greedy_results if r['benchmark'] == bm]
        greedy_ic = greedy_r[0]['instcount'] if greedy_r else None

        comparison.append({
            'benchmark': bm,
            'O0': sr['O0'],
            'O1': sr['O1'],
            'O2': sr['O2'],
            'O3': sr['O3'],
            'Oz': sr['Oz'],
            'random_mean': round(rand_mean, 1) if rand_mean else None,
            'random_std': round(rand_std, 1) if rand_std else None,
            'greedy': greedy_ic
        })

    with open(RESULTS_DIR / "baseline_comparison.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)

    # Print summary
    print("\n=== Baseline Comparison (geometric mean IC ratio vs O0) ===")
    import math
    for method in ['O1', 'O2', 'O3', 'Oz']:
        ratios = [sr[method] / sr['O0'] for sr in std_results
                  if sr[method] is not None and sr['O0'] is not None and sr['O0'] > 0]
        if ratios:
            geo_mean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            print(f"  {method}: {geo_mean:.4f} ({(1-geo_mean)*100:.1f}% reduction)")

    # Random
    rand_ratios = []
    for sr in std_results:
        rand_ics = [r['instcount'] for r in random_results if r['benchmark'] == sr['benchmark']]
        if rand_ics and sr['O0'] and sr['O0'] > 0:
            rand_ratios.append(sum(rand_ics)/len(rand_ics) / sr['O0'])
    if rand_ratios:
        geo_mean = math.exp(sum(math.log(r) for r in rand_ratios) / len(rand_ratios))
        print(f"  Random: {geo_mean:.4f} ({(1-geo_mean)*100:.1f}% reduction)")

    # Greedy
    greedy_ratios = []
    for gr in greedy_results:
        sr = [s for s in std_results if s['benchmark'] == gr['benchmark']]
        if sr and sr[0]['O0'] and sr[0]['O0'] > 0:
            greedy_ratios.append(gr['instcount'] / sr[0]['O0'])
    if greedy_ratios:
        geo_mean = math.exp(sum(math.log(r) for r in greedy_ratios) / len(greedy_ratios))
        print(f"  Greedy: {geo_mean:.4f} ({(1-geo_mean)*100:.1f}% reduction)")

    exp_results = {
        'experiment': 'baselines',
        'num_benchmarks': len(benchmarks),
        'runtime_seconds': round(elapsed, 1)
    }
    with open(Path(__file__).parent / 'results.json', 'w') as f:
        json.dump(exp_results, f, indent=2)


if __name__ == '__main__':
    main()

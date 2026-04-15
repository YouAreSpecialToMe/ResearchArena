"""Experiment 3: Pairwise interference quantification."""
import sys
import csv
import json
import time
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from exp.shared.utils import *


def compute_single_pass_effect(args):
    """Compute instruction count after applying a single pass."""
    pass_name, benchmark_path = args
    bm = Path(benchmark_path).stem
    baseline_ir = get_baseline_ir(benchmark_path)
    baseline_ic = count_instructions(baseline_ir)
    _, after_ic, _ = apply_pass_to_ir(benchmark_path, pass_name)
    if after_ic is None:
        return None
    delta = baseline_ic - after_ic
    return (pass_name, bm, baseline_ic, after_ic, delta)


def compute_pair_effect(args):
    """Compute instruction count after applying two passes sequentially."""
    pass_i, pass_j, benchmark_path = args
    bm = Path(benchmark_path).stem
    ir1, _, _ = apply_pass_to_ir(benchmark_path, pass_i)
    if ir1 is None:
        return None
    _, ic_ij, _ = apply_pass_to_ir_text(ir1, pass_j)
    if ic_ij is None:
        return None
    return (pass_i, pass_j, bm, ic_ij)


def main():
    passes = get_pass_list()
    benchmarks = get_benchmark_files()

    # Use 20 representative benchmarks: 12 PolyBench + 8 custom
    polybench = [b for b in benchmarks if b.stem.startswith("pb_")]
    custom = [b for b in benchmarks if not b.stem.startswith("pb_")]
    pb_indices = list(range(0, len(polybench), max(1, len(polybench) // 12)))[:12]
    cu_indices = list(range(0, len(custom), max(1, len(custom) // 8)))[:8]
    benchmarks_subset = [polybench[i] for i in pb_indices] + [custom[i] for i in cu_indices]

    print(f"Interference: {len(passes)} passes, {len(benchmarks_subset)} benchmarks")

    # Step 1: Compute single-pass effects
    print("Computing single-pass effects...")
    single_tasks = [(p, str(b)) for p in passes for b in benchmarks_subset]
    single_effects = {}  # (pass, benchmark) -> (baseline_ic, after_ic, delta)
    baselines = {}  # benchmark -> baseline_ic

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(compute_single_pass_effect, t): t for t in single_tasks}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                p, bm, base_ic, after_ic, delta = result
                single_effects[(p, bm)] = (base_ic, after_ic, delta)
                baselines[bm] = base_ic

    print(f"  Single-pass effects computed in {time.time()-start_time:.1f}s")

    # Step 2: Compute pair effects
    print("Computing pair effects...")
    pass_pairs = list(combinations(passes, 2))
    pair_tasks = [(pi, pj, str(b)) for pi, pj in pass_pairs for b in benchmarks_subset]
    pair_effects = {}  # (pass_i, pass_j, benchmark) -> ic_ij

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(compute_pair_effect, t): t for t in pair_tasks}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                pi, pj, bm, ic_ij = result
                pair_effects[(pi, pj, bm)] = ic_ij
            done += 1
            if done % 2000 == 0:
                elapsed = time.time() - start_time
                print(f"  {done}/{len(pair_tasks)} ({elapsed:.0f}s)")

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")

    # Step 3: Compute interference scores
    results = []
    for pi, pj in pass_pairs:
        for bm in benchmarks_subset:
            bm_name = bm.stem
            if (pi, bm_name) not in single_effects or (pj, bm_name) not in single_effects:
                continue
            if (pi, pj, bm_name) not in pair_effects:
                continue

            base_ic = baselines[bm_name]
            _, after_i, delta_i = single_effects[(pi, bm_name)]
            _, after_j, delta_j = single_effects[(pj, bm_name)]
            ic_ij = pair_effects[(pi, pj, bm_name)]
            delta_ij = base_ic - ic_ij
            interference = delta_ij - (delta_i + delta_j)
            interference_pct = (interference / base_ic * 100) if base_ic > 0 else 0

            results.append({
                'pass_i': pi, 'pass_j': pj, 'benchmark': bm_name,
                'baseline_instcount': base_ic,
                'after_i_instcount': after_i,
                'after_j_instcount': after_j,
                'after_ij_instcount': ic_ij,
                'delta_i': delta_i, 'delta_j': delta_j, 'delta_ij': delta_ij,
                'interference': interference,
                'interference_pct': round(interference_pct, 4)
            })

    # Save raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "interference_raw.csv"
    with open(raw_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        writer.writeheader()
        writer.writerows(results)

    # Compute interference matrix (mean interference_pct across benchmarks)
    interference_matrix = {}
    for pi, pj in pass_pairs:
        pair_results = [r for r in results if r['pass_i'] == pi and r['pass_j'] == pj]
        if pair_results:
            mean_interf = sum(r['interference_pct'] for r in pair_results) / len(pair_results)
        else:
            mean_interf = 0.0
        interference_matrix[(pi, pj)] = round(mean_interf, 4)
        interference_matrix[(pj, pi)] = round(mean_interf, 4)

    # Save interference matrix
    matrix_path = RESULTS_DIR / "interference_matrix.csv"
    with open(matrix_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + passes)
        for pi in passes:
            row = [pi]
            for pj in passes:
                if pi == pj:
                    row.append(0.0)
                else:
                    row.append(interference_matrix.get((pi, pj), 0.0))
            writer.writerow(row)

    # Summary
    significant_pairs = []
    for pi, pj in pass_pairs:
        pair_results = [r for r in results if r['pass_i'] == pi and r['pass_j'] == pj]
        if not pair_results:
            continue
        mean_abs = sum(abs(r['interference_pct']) for r in pair_results) / len(pair_results)
        mean_val = sum(r['interference_pct'] for r in pair_results) / len(pair_results)
        if mean_abs > 5.0:
            significant_pairs.append((pi, pj, round(mean_val, 2), round(mean_abs, 2)))

    # Sort by interference magnitude
    constructive = sorted([s for s in significant_pairs if s[2] > 0], key=lambda x: -x[2])
    destructive = sorted([s for s in significant_pairs if s[2] < 0], key=lambda x: x[2])

    print(f"\n=== Interference Summary ===")
    print(f"Total pairs tested: {len(pass_pairs)}")
    print(f"Pairs with significant interference (|I| > 5%): {len(significant_pairs)} ({len(significant_pairs)/len(pass_pairs)*100:.1f}%)")
    print(f"  Constructive: {len(constructive)}")
    print(f"  Destructive: {len(destructive)}")

    print(f"\nTop-10 constructive (synergistic) pairs:")
    for pi, pj, mean_val, mean_abs in constructive[:10]:
        print(f"  {pi:25s} + {pj:25s}: +{mean_val:.2f}%")

    print(f"\nTop-10 destructive pairs:")
    for pi, pj, mean_val, mean_abs in destructive[:10]:
        print(f"  {pi:25s} + {pj:25s}: {mean_val:.2f}%")

    # Save summary
    summary_path = RESULTS_DIR / "interference_summary.csv"
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pass_i', 'pass_j', 'mean_interference_pct', 'mean_abs_interference_pct', 'type'])
        for pi, pj, mean_val, mean_abs in constructive:
            writer.writerow([pi, pj, mean_val, mean_abs, 'constructive'])
        for pi, pj, mean_val, mean_abs in destructive:
            writer.writerow([pi, pj, mean_val, mean_abs, 'destructive'])

    exp_results = {
        'experiment': 'interference',
        'num_passes': len(passes),
        'num_benchmarks': len(benchmarks_subset),
        'total_pairs': len(pass_pairs),
        'significant_pairs': len(significant_pairs),
        'significant_fraction': round(len(significant_pairs)/len(pass_pairs), 4) if pass_pairs else 0,
        'num_constructive': len(constructive),
        'num_destructive': len(destructive),
        'runtime_seconds': round(elapsed, 1),
        'top_constructive': [{'pass_i': pi, 'pass_j': pj, 'interference_pct': v}
                              for pi, pj, v, _ in constructive[:10]],
        'top_destructive': [{'pass_i': pi, 'pass_j': pj, 'interference_pct': v}
                             for pi, pj, v, _ in destructive[:10]]
    }
    with open(Path(__file__).parent / 'results.json', 'w') as f:
        json.dump(exp_results, f, indent=2)


if __name__ == '__main__':
    main()

"""Experiment 2: Pairwise commutativity analysis."""
import sys
import csv
import json
import time
import os
import tempfile
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from exp.shared.utils import *


def test_commutativity(args):
    """Test if two passes commute on a given benchmark."""
    pass_i, pass_j, benchmark_path = args
    benchmark = Path(benchmark_path).stem

    # Apply pass_i then pass_j
    ir_i, _, _ = apply_pass_to_ir(benchmark_path, pass_i)
    if ir_i is None:
        return None
    ir_ij, ic_ij, h_ij = apply_pass_to_ir_text(ir_i, pass_j)
    if ir_ij is None:
        return None

    # Apply pass_j then pass_i
    ir_j, _, _ = apply_pass_to_ir(benchmark_path, pass_j)
    if ir_j is None:
        return None
    ir_ji, ic_ji, h_ji = apply_pass_to_ir_text(ir_j, pass_i)
    if ir_ji is None:
        return None

    return {
        'pass_i': pass_i, 'pass_j': pass_j, 'benchmark': benchmark,
        'ir_ij_hash': h_ij, 'ir_ji_hash': h_ji,
        'ir_ij_instcount': ic_ij, 'ir_ji_instcount': ic_ji,
        'structurally_commutative': h_ij == h_ji,
        'instcount_commutative': ic_ij == ic_ji,
    }


def check_activity(args):
    """Check if a pass makes changes on a benchmark."""
    pass_name, bm_path = args
    bm = Path(bm_path).stem
    baseline_ir = get_baseline_ir(bm_path)
    h0 = structural_hash(baseline_ir)
    _, _, h1 = apply_pass_to_ir(bm_path, pass_name)
    return (pass_name, bm, h0 != h1)


def prefilter_inactive_pairs(passes, benchmarks):
    """Pre-compute which passes are active (make changes) on each benchmark."""
    print("Pre-filtering: computing per-pass activity...")
    active = {}

    tasks = [(p, str(b)) for p in passes for b in benchmarks]
    with ProcessPoolExecutor(max_workers=2) as executor:
        for result in executor.map(check_activity, tasks):
            p, bm, is_active = result
            active[(p, bm)] = is_active

    return active


def main():
    passes = get_pass_list()
    benchmarks = get_benchmark_files()

    # Use 25 representative benchmarks: 15 PolyBench + 10 custom
    polybench = [b for b in benchmarks if b.stem.startswith("pb_")]
    custom = [b for b in benchmarks if not b.stem.startswith("pb_")]
    pb_indices = list(range(0, len(polybench), max(1, len(polybench) // 15)))[:15]
    cu_indices = list(range(0, len(custom), max(1, len(custom) // 10)))[:10]
    benchmarks_subset = [polybench[i] for i in pb_indices] + [custom[i] for i in cu_indices]

    print(f"Commutativity: {len(passes)} passes, {len(benchmarks_subset)} benchmarks")

    # Pre-filter to skip trivially commutative pairs
    active = prefilter_inactive_pairs(passes, benchmarks_subset)

    # Generate all pairs
    pass_pairs = list(combinations(passes, 2))
    print(f"Total pass pairs: {len(pass_pairs)}")

    # Build task list, skipping pairs where both passes are inactive on a benchmark
    tasks = []
    skipped = 0
    for pi, pj in pass_pairs:
        for bm in benchmarks_subset:
            bm_name = bm.stem
            if not active.get((pi, bm_name), True) and not active.get((pj, bm_name), True):
                skipped += 1
                continue
            tasks.append((pi, pj, str(bm)))
    print(f"Tasks after filtering: {len(tasks)} (skipped {skipped} trivially commutative)")

    results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(test_commutativity, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            done += 1
            if done % 1000 == 0:
                elapsed = time.time() - start_time
                rate = done / elapsed
                remaining = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(tasks)} ({elapsed:.0f}s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"Completed {len(results)} tests in {elapsed:.1f}s")

    # Save raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "commutativity_raw.csv"
    fields = ['pass_i', 'pass_j', 'benchmark', 'ir_ij_hash', 'ir_ji_hash',
              'ir_ij_instcount', 'ir_ji_instcount', 'structurally_commutative', 'instcount_commutative']
    with open(raw_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    # Compute commutativity matrix
    comm_matrix = {}
    for pi, pj in pass_pairs:
        pair_results = [r for r in results if r['pass_i'] == pi and r['pass_j'] == pj]
        if not pair_results:
            comm_rate = 1.0  # Both inactive everywhere = trivially commutative
        else:
            comm_rate = sum(1 for r in pair_results if r['structurally_commutative']) / len(pair_results)
        comm_matrix[(pi, pj)] = round(comm_rate, 4)
        comm_matrix[(pj, pi)] = round(comm_rate, 4)

    # Save commutativity matrix
    matrix_path = RESULTS_DIR / "commutativity_matrix.csv"
    with open(matrix_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + passes)
        for pi in passes:
            row = [pi]
            for pj in passes:
                if pi == pj:
                    row.append(1.0)
                else:
                    row.append(comm_matrix.get((pi, pj), 1.0))
            writer.writerow(row)

    # Summary statistics
    non_comm_count = sum(1 for v in set((min(k), max(k)) for k, v in comm_matrix.items() if v < 0.5)
                         if True)
    # Actually compute properly
    non_comm_pairs = []
    for pi, pj in pass_pairs:
        rate = comm_matrix.get((pi, pj), 1.0)
        if rate < 0.5:  # Non-commutative on majority of benchmarks
            non_comm_pairs.append((pi, pj, rate))

    total_pairs = len(pass_pairs)
    non_comm_fraction = len(non_comm_pairs) / total_pairs if total_pairs > 0 else 0

    print(f"\n=== Commutativity Summary ===")
    print(f"Total unique pairs: {total_pairs}")
    print(f"Non-commutative pairs (< 50% commutativity): {len(non_comm_pairs)} ({non_comm_fraction*100:.1f}%)")

    # Top non-commutative pairs
    non_comm_pairs.sort(key=lambda x: x[2])
    print(f"\nTop-10 most non-commutative pairs:")
    for pi, pj, rate in non_comm_pairs[:10]:
        print(f"  {pi:25s} x {pj:25s}: {rate*100:.1f}% commutative")

    exp_results = {
        'experiment': 'commutativity',
        'num_passes': len(passes),
        'num_benchmarks': len(benchmarks_subset),
        'total_pairs': total_pairs,
        'non_commutative_pairs': len(non_comm_pairs),
        'non_commutative_fraction': round(non_comm_fraction, 4),
        'runtime_seconds': round(elapsed, 1),
        'top_non_commutative': [{'pass_i': pi, 'pass_j': pj, 'comm_rate': rate}
                                  for pi, pj, rate in non_comm_pairs[:20]]
    }
    with open(Path(__file__).parent / 'results.json', 'w') as f:
        json.dump(exp_results, f, indent=2)


if __name__ == '__main__':
    main()

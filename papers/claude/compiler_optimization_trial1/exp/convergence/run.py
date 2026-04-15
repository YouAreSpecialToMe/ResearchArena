"""Experiment 4: Convergence analysis of iterative pipeline application."""
import sys
import csv
import json
import time
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from exp.shared.utils import *


def analyze_convergence(benchmark_path, pipeline="-O2", max_iterations=20):
    """Apply a pipeline iteratively and track convergence."""
    bm = Path(benchmark_path).stem
    results = []

    ir_text = get_baseline_ir(benchmark_path)
    baseline_ic = count_instructions(ir_text)
    baseline_hash = structural_hash(ir_text)
    results.append({
        'benchmark': bm, 'pipeline': pipeline,
        'iteration': 0, 'instcount': baseline_ic, 'ir_hash': baseline_hash,
        'converged': False, 'oscillating': False
    })

    seen_hashes = {baseline_hash: 0}
    prev_hash = baseline_hash

    for i in range(1, max_iterations + 1):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
            f.write(ir_text)
            tmp_path = f.name
        try:
            new_ir = run_opt_pipeline(tmp_path, pipeline, timeout=60)
        finally:
            os.unlink(tmp_path)

        if new_ir is None:
            break

        ic = count_instructions(new_ir)
        h = structural_hash(new_ir)

        converged = (h == prev_hash)
        oscillating = (h in seen_hashes and h != prev_hash)

        results.append({
            'benchmark': bm, 'pipeline': pipeline,
            'iteration': i, 'instcount': ic, 'ir_hash': h,
            'converged': converged, 'oscillating': oscillating
        })

        if converged:
            break

        if oscillating:
            # Continue a few more iterations to confirm the cycle
            cycle_start = seen_hashes[h]
            if i - cycle_start >= 2:  # Confirmed cycle
                break

        seen_hashes[h] = i
        prev_hash = h
        ir_text = new_ir

    return results


def main():
    benchmarks = get_benchmark_files()
    start_time = time.time()

    all_results = []
    pipelines = ['-O2', '-O3', '-Oz']

    for pipeline in pipelines:
        print(f"Running convergence analysis with {pipeline}...")
        for i, bm in enumerate(benchmarks):
            results = analyze_convergence(str(bm), pipeline)
            all_results.extend(results)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(benchmarks)} benchmarks")

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")

    # Save raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "convergence_raw.csv"
    fields = ['benchmark', 'pipeline', 'iteration', 'instcount', 'ir_hash', 'converged', 'oscillating']
    with open(raw_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)

    # Compute summary per pipeline
    print("\n=== Convergence Summary ===")
    for pipeline in pipelines:
        pipe_results = [r for r in all_results if r['pipeline'] == pipeline]
        bm_names = sorted(set(r['benchmark'] for r in pipe_results))

        convergence_iters = []
        oscillating_count = 0
        for bm in bm_names:
            bm_data = sorted([r for r in pipe_results if r['benchmark'] == bm],
                            key=lambda x: x['iteration'])
            if any(r['converged'] for r in bm_data):
                conv_iter = min(r['iteration'] for r in bm_data if r['converged'])
                convergence_iters.append(conv_iter)
            if any(r['oscillating'] for r in bm_data):
                oscillating_count += 1

        conv_within_2 = sum(1 for c in convergence_iters if c <= 2) / len(bm_names) if bm_names else 0
        conv_within_5 = sum(1 for c in convergence_iters if c <= 5) / len(bm_names) if bm_names else 0
        avg_iters = sum(convergence_iters) / len(convergence_iters) if convergence_iters else float('inf')

        print(f"\n{pipeline}:")
        print(f"  Converged: {len(convergence_iters)}/{len(bm_names)}")
        print(f"  Within 2 iterations: {conv_within_2*100:.1f}%")
        print(f"  Within 5 iterations: {conv_within_5*100:.1f}%")
        print(f"  Average iterations to converge: {avg_iters:.1f}")
        print(f"  Oscillating: {oscillating_count}/{len(bm_names)}")

    # Save summary
    summary = {'experiment': 'convergence', 'runtime_seconds': round(elapsed, 1), 'pipelines': {}}
    for pipeline in pipelines:
        pipe_results = [r for r in all_results if r['pipeline'] == pipeline]
        bm_names = sorted(set(r['benchmark'] for r in pipe_results))
        convergence_iters = []
        oscillating_count = 0
        for bm in bm_names:
            bm_data = sorted([r for r in pipe_results if r['benchmark'] == bm],
                            key=lambda x: x['iteration'])
            if any(r['converged'] for r in bm_data):
                conv_iter = min(r['iteration'] for r in bm_data if r['converged'])
                convergence_iters.append(conv_iter)
            if any(r['oscillating'] for r in bm_data):
                oscillating_count += 1
        summary['pipelines'][pipeline] = {
            'num_benchmarks': len(bm_names),
            'converged': len(convergence_iters),
            'oscillating': oscillating_count,
            'conv_within_2_frac': round(sum(1 for c in convergence_iters if c <= 2) / len(bm_names), 4) if bm_names else 0,
            'avg_iters': round(sum(convergence_iters)/len(convergence_iters), 2) if convergence_iters else None
        }

    with open(Path(__file__).parent / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()

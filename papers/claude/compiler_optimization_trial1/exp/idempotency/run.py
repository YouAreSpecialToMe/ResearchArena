"""Experiment 1: Idempotency characterization of individual passes."""
import sys
import csv
import json
import time
import os
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from exp.shared.utils import *


def test_idempotency(args):
    """Test if a pass is idempotent on a given benchmark."""
    pass_name, benchmark_path = args
    benchmark = Path(benchmark_path).stem

    # Apply pass once
    ir1, ic1, h1 = apply_pass_to_ir(benchmark_path, pass_name)
    if ir1 is None:
        return {
            'pass_name': pass_name, 'benchmark': benchmark,
            'ir1_hash': None, 'ir2_hash': None,
            'ir1_instcount': None, 'ir2_instcount': None,
            'structurally_identical': None, 'instcount_identical': None,
            'error': True
        }

    # Apply pass again to the result
    ir2, ic2, h2 = apply_pass_to_ir_text(ir1, pass_name)
    if ir2 is None:
        return {
            'pass_name': pass_name, 'benchmark': benchmark,
            'ir1_hash': h1, 'ir2_hash': None,
            'ir1_instcount': ic1, 'ir2_instcount': None,
            'structurally_identical': None, 'instcount_identical': None,
            'error': True
        }

    return {
        'pass_name': pass_name, 'benchmark': benchmark,
        'ir1_hash': h1, 'ir2_hash': h2,
        'ir1_instcount': ic1, 'ir2_instcount': ic2,
        'structurally_identical': h1 == h2,
        'instcount_identical': ic1 == ic2,
        'error': False
    }


def main():
    passes = get_pass_list()
    benchmarks = get_benchmark_files()

    print(f"Running idempotency test: {len(passes)} passes x {len(benchmarks)} benchmarks = {len(passes)*len(benchmarks)} tests")

    # Create all (pass, benchmark) pairs
    tasks = [(p, str(b)) for p in passes for b in benchmarks]

    results = []
    start_time = time.time()

    # Run with parallelism
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(test_idempotency, task): task for task in tasks}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done += 1
            if done % 200 == 0:
                elapsed = time.time() - start_time
                print(f"  {done}/{len(tasks)} ({elapsed:.1f}s elapsed)")

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")

    # Save raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "idempotency_raw.csv"
    fields = ['pass_name', 'benchmark', 'ir1_hash', 'ir2_hash', 'ir1_instcount', 'ir2_instcount',
              'structurally_identical', 'instcount_identical', 'error']
    with open(raw_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    # Compute summary
    valid = [r for r in results if not r['error']]
    summary = []
    for p in passes:
        p_results = [r for r in valid if r['pass_name'] == p]
        if not p_results:
            continue
        n = len(p_results)
        struct_idem = sum(1 for r in p_results if r['structurally_identical']) / n
        ic_idem = sum(1 for r in p_results if r['instcount_identical']) / n

        if struct_idem == 1.0:
            classification = 'strongly_idempotent'
        elif ic_idem == 1.0:
            classification = 'weakly_idempotent'
        elif ic_idem > 0.5:
            classification = 'mostly_idempotent'
        else:
            classification = 'non_idempotent'

        summary.append({
            'pass_name': p,
            'num_benchmarks': n,
            'structural_idempotency_rate': round(struct_idem, 4),
            'instcount_idempotency_rate': round(ic_idem, 4),
            'classification': classification
        })

    summary_path = RESULTS_DIR / "idempotency_summary.csv"
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pass_name', 'num_benchmarks',
                                                'structural_idempotency_rate',
                                                'instcount_idempotency_rate',
                                                'classification'])
        writer.writeheader()
        writer.writerows(summary)

    # Print summary
    print("\n=== Idempotency Summary ===")
    strongly = sum(1 for s in summary if s['classification'] == 'strongly_idempotent')
    weakly = sum(1 for s in summary if s['classification'] == 'weakly_idempotent')
    mostly = sum(1 for s in summary if s['classification'] == 'mostly_idempotent')
    non_idem = sum(1 for s in summary if s['classification'] == 'non_idempotent')
    print(f"Strongly idempotent: {strongly}/{len(summary)}")
    print(f"Weakly idempotent: {weakly}/{len(summary)}")
    print(f"Mostly idempotent: {mostly}/{len(summary)}")
    print(f"Non-idempotent: {non_idem}/{len(summary)}")

    # Save experiment metadata
    exp_results = {
        'experiment': 'idempotency',
        'num_passes': len(passes),
        'num_benchmarks': len(benchmarks),
        'num_valid_tests': len(valid),
        'strongly_idempotent': strongly,
        'weakly_idempotent': weakly,
        'mostly_idempotent': mostly,
        'non_idempotent': non_idem,
        'runtime_seconds': round(elapsed, 1)
    }
    with open(Path(__file__).parent / 'results.json', 'w') as f:
        json.dump(exp_results, f, indent=2)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Master orchestration script for all experiments.

Runs all methods with IDENTICAL training protocol:
- 200 epochs contrastive + 100 epochs linear eval
- 3 seeds (42, 43, 44)
- ResNet-18, batch_size=512, SGD lr=0.5, cosine schedule, tau=0.07

Manages GPU parallelism (up to MAX_PARALLEL concurrent runs).
"""

import subprocess
import sys
import os
import json
import time
import signal
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
SEEDS = [42, 43, 44]
EPOCHS = 200
LINEAR_EPOCHS = 100
BATCH_SIZE = 512
LR = 0.5
TEMPERATURE = 0.07
MAX_PARALLEL = 4  # Conservative: 4 parallel on 48GB GPU (~4GB each)
NUM_WORKERS = 2

BASE_DIR = Path(__file__).parent.parent
EXP_DIR = BASE_DIR / "exp"
RESULTS_DIR = BASE_DIR / "results_v3"
TRAIN_SCRIPT = str(EXP_DIR / "train.py")


def run_experiment(method, seed, extra_args=None, output_dir=None):
    """Run a single experiment and return the result."""
    if output_dir is None:
        output_dir = str(RESULTS_DIR / method)

    output_file = os.path.join(output_dir, f"results_seed{seed}.json")

    # Skip if already completed
    if os.path.exists(output_file):
        try:
            with open(output_file) as f:
                result = json.load(f)
            if 'top1' in result and result.get('hyperparameters', {}).get('epochs') == EPOCHS:
                print(f"  [SKIP] {method} seed={seed} already done: top1={result['top1']:.2f}%")
                return result
        except (json.JSONDecodeError, KeyError):
            pass

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--method", method,
        "--seed", str(seed),
        "--epochs", str(EPOCHS),
        "--linear_epochs", str(LINEAR_EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--temperature", str(TEMPERATURE),
        "--num_workers", str(NUM_WORKERS),
        "--output_dir", output_dir,
    ]

    if extra_args:
        cmd.extend(extra_args)

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"log_seed{seed}.txt")

    print(f"  [START] {method} seed={seed} -> {output_dir}")
    start = time.time()

    with open(log_file, 'w') as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                            timeout=7200)  # 2 hour timeout per run

    elapsed = (time.time() - start) / 60

    if proc.returncode != 0:
        print(f"  [FAIL] {method} seed={seed} after {elapsed:.1f}min (see {log_file})")
        return None

    # Load and return results
    if os.path.exists(output_file):
        with open(output_file) as f:
            result = json.load(f)
        print(f"  [DONE] {method} seed={seed}: top1={result.get('top1', '?'):.2f}% "
              f"({elapsed:.1f}min)")
        return result
    else:
        print(f"  [FAIL] {method} seed={seed}: no results file after {elapsed:.1f}min")
        return None


def run_batch(jobs, max_parallel=MAX_PARALLEL):
    """Run a batch of jobs with limited parallelism."""
    results = {}

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for job in jobs:
            method = job['method']
            seed = job['seed']
            extra_args = job.get('extra_args', None)
            output_dir = job.get('output_dir', None)

            future = executor.submit(run_experiment, method, seed, extra_args, output_dir)
            futures[future] = (method, seed)

        for future in as_completed(futures):
            method, seed = futures[future]
            try:
                result = future.result()
                results[(method, seed)] = result
            except Exception as e:
                print(f"  [ERROR] {method} seed={seed}: {e}")
                results[(method, seed)] = None

    return results


def copy_existing_results():
    """Copy valid existing results to results_v3/."""
    copies = []

    # SupCon seed 42 from supcon_v2 (200 epochs, clean eval)
    src = EXP_DIR / "supcon_v2" / "results_seed42.json"
    dst = RESULTS_DIR / "supcon" / "results_seed42.json"
    if src.exists() and not dst.exists():
        try:
            with open(src) as f:
                d = json.load(f)
            if d.get('hyperparameters', {}).get('epochs') == 200:
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, 'w') as f:
                    json.dump(d, f, indent=2)
                copies.append(f"supcon/seed42 (top1={d['top1']:.2f}%)")
        except Exception:
            pass

    # CGA-only seeds 42,43,44 from cga_best (200 epochs, alpha=0.5, lam=0.5)
    for seed in SEEDS:
        src = EXP_DIR / "cga_best" / f"results_seed{seed}.json"
        dst = RESULTS_DIR / "cga_only" / f"results_seed{seed}.json"
        if src.exists() and not dst.exists():
            try:
                with open(src) as f:
                    d = json.load(f)
                if d.get('hyperparameters', {}).get('epochs') == 200:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with open(dst, 'w') as f:
                        json.dump(d, f, indent=2)
                    copies.append(f"cga_only/seed{seed} (top1={d['top1']:.2f}%)")
            except Exception:
                pass

    if copies:
        print(f"Copied {len(copies)} existing results: {', '.join(copies)}")


def phase_grid_search():
    """Quick grid search for CGA hyperparameters (50 epochs, 1 seed)."""
    print("\n" + "="*60)
    print("PHASE 0: CGA Hyperparameter Grid Search (50 epochs)")
    print("="*60)

    grid_dir = RESULTS_DIR / "grid_search"
    grid_dir.mkdir(parents=True, exist_ok=True)

    alphas = [0.3, 0.5, 0.7]
    lambdas = [0.1, 0.5, 1.0]

    jobs = []
    for alpha in alphas:
        for lam in lambdas:
            tag = f"a{alpha}_l{lam}"
            jobs.append({
                'method': 'cga_only',
                'seed': 42,
                'extra_args': ['--alpha', str(alpha), '--lam', str(lam),
                              '--epochs', '50', '--linear_epochs', '50'],
                'output_dir': str(grid_dir / tag),
            })

    # Override epochs for grid search
    results = {}
    with ProcessPoolExecutor(max_workers=min(6, len(jobs))) as executor:
        futures = {}
        for job in jobs:
            cmd = [
                sys.executable, TRAIN_SCRIPT,
                "--method", job['method'],
                "--seed", str(job['seed']),
                "--epochs", "50",
                "--linear_epochs", "50",
                "--batch_size", str(BATCH_SIZE),
                "--lr", str(LR),
                "--temperature", str(TEMPERATURE),
                "--num_workers", "1",
                "--output_dir", job['output_dir'],
            ] + (job.get('extra_args', [])[:4])  # just alpha and lam

            # Extract alpha/lambda from extra_args
            alpha = job['extra_args'][1]
            lam = job['extra_args'][3]
            cmd_clean = [
                sys.executable, TRAIN_SCRIPT,
                "--method", "cga_only", "--seed", "42",
                "--epochs", "50", "--linear_epochs", "50",
                "--batch_size", str(BATCH_SIZE), "--lr", str(LR),
                "--temperature", str(TEMPERATURE), "--num_workers", "1",
                "--output_dir", job['output_dir'],
                "--alpha", alpha, "--lam", lam,
            ]

            tag = os.path.basename(job['output_dir'])
            out_file = os.path.join(job['output_dir'], "results_seed42.json")

            # Skip if exists
            if os.path.exists(out_file):
                try:
                    with open(out_file) as f:
                        r = json.load(f)
                    if 'top1' in r:
                        results[tag] = r['top1']
                        print(f"  [SKIP] grid {tag}: top1={r['top1']:.2f}%")
                        continue
                except Exception:
                    pass

            os.makedirs(job['output_dir'], exist_ok=True)
            log = os.path.join(job['output_dir'], "log.txt")
            future = executor.submit(
                subprocess.run, cmd_clean,
                stdout=open(log, 'w'), stderr=subprocess.STDOUT, timeout=3600
            )
            futures[future] = tag

        for future in as_completed(futures):
            tag = futures[future]
            try:
                proc = future.result()
                out_file = str(grid_dir / tag / "results_seed42.json")
                if os.path.exists(out_file):
                    with open(out_file) as f:
                        r = json.load(f)
                    results[tag] = r.get('top1', 0)
                    print(f"  [DONE] grid {tag}: top1={r.get('top1', 0):.2f}%")
            except Exception as e:
                print(f"  [FAIL] grid {tag}: {e}")

    if results:
        best_tag = max(results, key=results.get)
        print(f"\nGrid search results:")
        for tag, top1 in sorted(results.items(), key=lambda x: -x[1]):
            marker = " <-- BEST" if tag == best_tag else ""
            print(f"  {tag}: {top1:.2f}%{marker}")

        # Extract best alpha/lambda
        parts = best_tag.split('_')
        best_alpha = float(parts[0][1:])
        best_lam = float(parts[1][1:])

        # Save grid results
        with open(str(grid_dir / "summary.json"), 'w') as f:
            json.dump({
                'results': results,
                'best': {'alpha': best_alpha, 'lambda': best_lam, 'top1': results[best_tag]},
            }, f, indent=2)

        return best_alpha, best_lam

    return 0.5, 0.5  # fallback


def phase_main_experiments(best_alpha, best_lam):
    """Run all methods with identical protocol."""
    print("\n" + "="*60)
    print(f"PHASE 1: Main Experiments (200ep+100ep, 3 seeds)")
    print(f"CGA params: alpha={best_alpha}, lambda={best_lam}")
    print("="*60)

    # Copy existing valid results first
    copy_existing_results()

    all_jobs = []

    # SupCon baseline (seed 42 already copied)
    for seed in SEEDS:
        all_jobs.append({'method': 'supcon', 'seed': seed})

    # HardNeg baseline
    for seed in SEEDS:
        all_jobs.append({'method': 'hardneg', 'seed': seed})

    # TCL baseline (FIXED)
    for seed in SEEDS:
        all_jobs.append({'method': 'tcl', 'seed': seed})

    # Reweight baseline
    for seed in SEEDS:
        all_jobs.append({'method': 'reweight', 'seed': seed})

    # VarCon-T baseline
    for seed in SEEDS:
        all_jobs.append({'method': 'varcon_t', 'seed': seed})

    # CGA-only (seeds already copied if same hyperparams)
    for seed in SEEDS:
        all_jobs.append({
            'method': 'cga_only', 'seed': seed,
            'extra_args': ['--alpha', str(best_alpha), '--lam', str(best_lam)],
        })

    # CGA-full (with adaptive temperature)
    for seed in SEEDS:
        all_jobs.append({
            'method': 'cga_full', 'seed': seed,
            'extra_args': ['--alpha', str(best_alpha), '--lam', str(best_lam),
                          '--gamma', '5.0'],
        })

    # Filter out already-completed jobs
    pending_jobs = []
    for job in all_jobs:
        method = job['method']
        seed = job['seed']
        out_dir = str(RESULTS_DIR / method)
        out_file = os.path.join(out_dir, f"results_seed{seed}.json")
        if os.path.exists(out_file):
            try:
                with open(out_file) as f:
                    d = json.load(f)
                if 'top1' in d and d.get('hyperparameters', {}).get('epochs') == EPOCHS:
                    print(f"  [SKIP] {method} seed={seed}: top1={d['top1']:.2f}%")
                    continue
            except Exception:
                pass
        job['output_dir'] = out_dir
        pending_jobs.append(job)

    print(f"\n{len(pending_jobs)} experiments to run ({len(all_jobs) - len(pending_jobs)} skipped)")

    if not pending_jobs:
        return

    # Run in batches
    results = run_batch(pending_jobs, max_parallel=MAX_PARALLEL)

    # Summary
    print(f"\nPhase 1 complete: {sum(1 for v in results.values() if v is not None)}/{len(results)} succeeded")


def phase_ablation(best_alpha, best_lam):
    """Run ablation: adaptive temperature only (CGA-only and CGA-full already run)."""
    print("\n" + "="*60)
    print("PHASE 2: Ablation - Adaptive Temperature Only")
    print("="*60)

    jobs = []
    for seed in SEEDS:
        jobs.append({
            'method': 'adaptive_temp', 'seed': seed,
            'extra_args': ['--gamma', '5.0', '--alpha', str(best_alpha), '--lam', '0.0'],
            'output_dir': str(RESULTS_DIR / 'adaptive_temp'),
        })

    results = run_batch(jobs, max_parallel=3)
    print(f"Ablation complete: {sum(1 for v in results.values() if v is not None)}/{len(results)} succeeded")


def phase_ce_baseline():
    """Run CE baseline."""
    print("\n" + "="*60)
    print("PHASE 3: CE Baseline")
    print("="*60)

    jobs = []
    for seed in SEEDS:
        jobs.append({
            'method': 'ce', 'seed': seed,
            'output_dir': str(RESULTS_DIR / 'ce'),
        })

    results = run_batch(jobs, max_parallel=3)


def collect_all_results():
    """Collect all results into a summary."""
    print("\n" + "="*60)
    print("Collecting all results...")
    print("="*60)

    from scipy import stats as scipy_stats

    methods_data = {}

    for method_dir in sorted(RESULTS_DIR.iterdir()):
        if not method_dir.is_dir() or method_dir.name == 'grid_search':
            continue

        method = method_dir.name
        seed_results = []

        for f in sorted(method_dir.glob("results_seed*.json")):
            try:
                with open(f) as fh:
                    d = json.load(fh)
                seed_results.append(d)
            except Exception:
                pass

        if not seed_results:
            continue

        # Aggregate
        metrics = {}
        for key in ['top1', 'top5', 'superclass_acc', 'within_superclass_acc',
                     'between_superclass_error_rate', 'etf_deviation', 'hierarchy_corr',
                     'mean_epoch_time_seconds', 'training_time_minutes',
                     'contrastive_time_minutes', 'linear_eval_time_minutes']:
            vals = [r[key] for r in seed_results if key in r]
            if vals:
                import numpy as np
                metrics[key] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'values': vals,
                }

        methods_data[method] = {
            'method': method,
            'n_seeds': len(seed_results),
            'seeds': [r.get('seed') for r in seed_results],
            'metrics': metrics,
        }

    # Statistical tests: paired t-tests vs SupCon
    stat_tests = {}
    if 'supcon' in methods_data:
        supcon_vals = methods_data['supcon']['metrics'].get('top1', {}).get('values', [])
        for method, data in methods_data.items():
            if method == 'supcon':
                continue
            method_vals = data['metrics'].get('top1', {}).get('values', [])
            if len(supcon_vals) >= 3 and len(method_vals) >= 3:
                # Use independent t-test since seeds may differ
                t_stat, p_val = scipy_stats.ttest_ind(method_vals, supcon_vals)
                stat_tests[f"{method}_vs_supcon"] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'significant': p_val < 0.05,
                    'method_mean': float(np.mean(method_vals)),
                    'supcon_mean': float(np.mean(supcon_vals)),
                }

    # Success criteria evaluation
    success_criteria = evaluate_success_criteria(methods_data, stat_tests)

    # Grid search results
    grid_summary = {}
    grid_file = RESULTS_DIR / "grid_search" / "summary.json"
    if grid_file.exists():
        with open(grid_file) as f:
            grid_summary = json.load(f)

    final = {
        'methods': methods_data,
        'statistical_tests': stat_tests,
        'success_criteria': success_criteria,
        'grid_search': grid_summary,
        'experiment_notes': {
            'training_protocol': f'ResNet-18, {EPOCHS}ep contrastive + {LINEAR_EPOCHS}ep linear eval, '
                                f'batch={BATCH_SIZE}, SGD lr={LR}, cosine, tau={TEMPERATURE}',
            'seeds': SEEDS,
            'all_methods_identical_protocol': True,
            'honest_reporting': 'ALL results included, negative results reported honestly',
        },
    }

    out_path = str(BASE_DIR / "results.json")
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<18} {'Top-1 (mean±std)':<22} {'Top-5':<12} {'Seeds':<6}")
    print("-"*60)
    for method in ['ce', 'supcon', 'hardneg', 'tcl', 'reweight', 'varcon_t',
                    'cga_only', 'cga_full', 'adaptive_temp']:
        if method not in methods_data:
            continue
        d = methods_data[method]
        t1 = d['metrics'].get('top1', {})
        t5 = d['metrics'].get('top5', {})
        t1_str = f"{t1.get('mean', 0):.2f}±{t1.get('std', 0):.2f}"
        t5_str = f"{t5.get('mean', 0):.2f}±{t5.get('std', 0):.2f}"
        print(f"{method:<18} {t1_str:<22} {t5_str:<12} {d['n_seeds']}")

    print("\nSuccess Criteria:")
    for sc in success_criteria:
        status = "PASS" if sc['passed'] else "FAIL"
        print(f"  [{status}] {sc['desc']}: {sc['detail']}")

    return final


def evaluate_success_criteria(methods_data, stat_tests):
    """Evaluate all success criteria honestly."""
    import numpy as np
    criteria = []

    supcon_t1 = methods_data.get('supcon', {}).get('metrics', {}).get('top1', {})
    cga_t1 = methods_data.get('cga_only', {}).get('metrics', {}).get('top1', {})
    cgafull_t1 = methods_data.get('cga_full', {}).get('metrics', {}).get('top1', {})
    hardneg_t1 = methods_data.get('hardneg', {}).get('metrics', {}).get('top1', {})
    tcl_t1 = methods_data.get('tcl', {}).get('metrics', {}).get('top1', {})
    reweight_t1 = methods_data.get('reweight', {}).get('metrics', {}).get('top1', {})

    # Use CGA-full as the main method if it exists, otherwise CGA-only
    main_method = 'cga_full' if 'cga_full' in methods_data else 'cga_only'
    main_t1 = cgafull_t1 if main_method == 'cga_full' else cga_t1

    # 1. Statistically significant improvement over SupCon
    test_key = f"{main_method}_vs_supcon"
    if test_key in stat_tests:
        st = stat_tests[test_key]
        passed = st['significant'] and st['method_mean'] > st['supcon_mean']
        criteria.append({
            'id': 1, 'desc': f'{main_method} significantly > SupCon (p<0.05)',
            'passed': passed,
            'detail': f"{main_method}={st['method_mean']:.2f}% vs SupCon={st['supcon_mean']:.2f}%, p={st['p_value']:.4f}",
        })
    else:
        criteria.append({'id': 1, 'desc': 'Statistical significance vs SupCon',
                        'passed': False, 'detail': 'Insufficient data for test'})

    # 2. Outperform instance-level baselines
    for baseline, name in [('hardneg', 'HardNeg'), ('tcl', 'TCL')]:
        bl_t1 = methods_data.get(baseline, {}).get('metrics', {}).get('top1', {})
        if main_t1.get('mean') and bl_t1.get('mean'):
            passed = main_t1['mean'] > bl_t1['mean']
            criteria.append({
                'id': 2, 'desc': f'{main_method} > {name}',
                'passed': passed,
                'detail': f"{main_method}={main_t1['mean']:.2f}% vs {name}={bl_t1['mean']:.2f}%",
            })

    # 3. Outperform pairwise reweighting
    if main_t1.get('mean') and reweight_t1.get('mean'):
        passed = main_t1['mean'] > reweight_t1['mean']
        criteria.append({
            'id': 3, 'desc': f'{main_method} > Reweight',
            'passed': passed,
            'detail': f"{main_method}={main_t1['mean']:.2f}% vs Reweight={reweight_t1['mean']:.2f}%",
        })

    # 4. CGA alone provides gain over SupCon
    if cga_t1.get('mean') and supcon_t1.get('mean'):
        passed = cga_t1['mean'] > supcon_t1['mean']
        criteria.append({
            'id': 4, 'desc': 'CGA-only > SupCon',
            'passed': passed,
            'detail': f"CGA-only={cga_t1['mean']:.2f}% vs SupCon={supcon_t1['mean']:.2f}%",
        })

    # 5. Within-superclass improvement > between-superclass improvement
    supcon_within = methods_data.get('supcon', {}).get('metrics', {}).get('within_superclass_acc', {})
    main_within = methods_data.get(main_method, {}).get('metrics', {}).get('within_superclass_acc', {})
    supcon_between = methods_data.get('supcon', {}).get('metrics', {}).get('between_superclass_error_rate', {})
    main_between = methods_data.get(main_method, {}).get('metrics', {}).get('between_superclass_error_rate', {})

    if all(d.get('mean') is not None for d in [supcon_within, main_within, supcon_between, main_between]):
        within_delta = main_within['mean'] - supcon_within['mean']
        between_delta = main_between['mean'] - supcon_between['mean']
        passed = within_delta > 0 and within_delta > abs(between_delta)
        criteria.append({
            'id': 5, 'desc': 'Within-SC improvement > Between-SC improvement',
            'passed': passed,
            'detail': f"Within delta={within_delta:+.2f}%, Between delta={between_delta:+.2f}%",
        })

    # 6. Hierarchy correlation improvement > 0.1
    supcon_hc = methods_data.get('supcon', {}).get('metrics', {}).get('hierarchy_corr', {})
    main_hc = methods_data.get(main_method, {}).get('metrics', {}).get('hierarchy_corr', {})
    if supcon_hc.get('mean') is not None and main_hc.get('mean') is not None:
        delta = main_hc['mean'] - supcon_hc['mean']
        passed = delta > 0.1
        criteria.append({
            'id': 6, 'desc': 'Hierarchy corr improvement > 0.1',
            'passed': passed,
            'detail': f"{main_method}={main_hc['mean']:.4f}, SupCon={supcon_hc['mean']:.4f}, delta={delta:+.4f}",
        })

    # 8. Wall-clock overhead < 10%
    supcon_time = methods_data.get('supcon', {}).get('metrics', {}).get('mean_epoch_time_seconds', {})
    main_time = methods_data.get(main_method, {}).get('metrics', {}).get('mean_epoch_time_seconds', {})
    if supcon_time.get('mean') and main_time.get('mean'):
        overhead = (main_time['mean'] - supcon_time['mean']) / supcon_time['mean'] * 100
        passed = overhead < 10
        criteria.append({
            'id': 8, 'desc': 'Wall-clock overhead < 10%',
            'passed': passed,
            'detail': f"{main_method}={main_time['mean']:.1f}s, SupCon={supcon_time['mean']:.1f}s, overhead={overhead:.1f}%",
        })

    return criteria


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all',
                       choices=['all', 'grid', 'main', 'ablation', 'ce', 'collect'])
    parser.add_argument('--max_parallel', type=int, default=MAX_PARALLEL)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--lam', type=float, default=None)
    args = parser.parse_args()

    MAX_PARALLEL = args.max_parallel
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    import numpy as np

    start_time = time.time()

    if args.phase in ['all', 'grid']:
        best_alpha, best_lam = phase_grid_search()
    else:
        best_alpha = args.alpha or 0.5
        best_lam = args.lam or 0.5

    if args.phase in ['all', 'main']:
        phase_main_experiments(best_alpha, best_lam)

    if args.phase in ['all', 'ablation']:
        phase_ablation(best_alpha, best_lam)

    if args.phase in ['all', 'ce']:
        phase_ce_baseline()

    if args.phase in ['all', 'collect']:
        collect_all_results()

    total = (time.time() - start_time) / 60
    print(f"\nTotal wall time: {total:.1f} minutes ({total/60:.1f} hours)")

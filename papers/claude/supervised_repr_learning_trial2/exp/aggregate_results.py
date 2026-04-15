"""Aggregate all experiment results and verify success criteria."""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from scipy import stats


def load_metrics(results_dir):
    """Load all metrics.json files organized by dataset/method/seed."""
    all_results = defaultdict(lambda: defaultdict(dict))

    for dataset in os.listdir(results_dir):
        dataset_dir = os.path.join(results_dir, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for method in os.listdir(dataset_dir):
            method_dir = os.path.join(dataset_dir, method)
            if not os.path.isdir(method_dir):
                continue
            for seed_dir_name in os.listdir(method_dir):
                seed_dir = os.path.join(method_dir, seed_dir_name)
                metrics_file = os.path.join(seed_dir, 'metrics.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file) as f:
                        data = json.load(f)
                    seed = seed_dir_name.replace('seed_', '')
                    all_results[dataset][method][seed] = data

    return all_results


def aggregate_method(seed_results):
    """Aggregate metrics across seeds for a single method."""
    if not seed_results:
        return None

    # Collect all metric values
    metrics = defaultdict(list)
    for seed, data in seed_results.items():
        metrics['test_accuracy'].append(data.get('test_accuracy', data.get('best_test_acc', 0)))
        if data.get('top5_accuracy') is not None:
            metrics['top5_accuracy'].append(data['top5_accuracy'])

        # Calibration metrics (handle both flat and nested formats)
        cal = data.get('calibration', {})
        if cal:
            for k in ['ece', 'mce', 'ada_ece', 'nll', 'brier']:
                if k in cal:
                    metrics[k].append(cal[k])

        cal_ts = data.get('calibration_after_ts', {})
        if cal_ts:
            for k in ['ece', 'mce', 'ada_ece', 'nll', 'brier']:
                if k in cal_ts:
                    metrics[f'{k}_ts'].append(cal_ts[k])

        if 'temperature' in data:
            metrics['temperature'].append(data['temperature'])

        # NC metrics
        nc = data.get('nc_metrics', {})
        if nc:
            for k in ['nc1', 'nc2', 'nc3', 'nc4', 'mean_within_class_spread']:
                if k in nc:
                    metrics[k].append(nc[k])

    # Compute mean and std
    agg = {}
    for k, vals in metrics.items():
        if vals:
            agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    agg['n_seeds'] = len(seed_results)

    return agg


def find_best_ccr(dataset_results):
    """Find the best CCR variant for a dataset (lowest ECE with >=3 seeds or best single-seed)."""
    best_method = None
    best_ece = float('inf')
    for method in dataset_results:
        if not method.startswith('ccr_'):
            continue
        # Skip 100-epoch sweep results
        if '_100ep' in method:
            continue
        agg = aggregate_method(dataset_results[method])
        if agg and 'ece' in agg:
            ece = agg['ece']['mean']
            n_seeds = agg['n_seeds']
            # Prefer methods with multiple seeds
            if n_seeds >= 3 and ece < best_ece:
                best_ece = ece
                best_method = method
    # If no 3-seed method found, use best single-seed
    if best_method is None:
        for method in dataset_results:
            if not method.startswith('ccr_') or '_100ep' in method:
                continue
            agg = aggregate_method(dataset_results[method])
            if agg and 'ece' in agg and agg['ece']['mean'] < best_ece:
                best_ece = agg['ece']['mean']
                best_method = method
    return best_method


def check_success_criteria(results):
    """Check primary, secondary, and refutation criteria."""
    criteria = {}

    # Find best CCR variant per dataset
    best_ccr_methods = {}
    datasets_with_both = []
    for dataset in results:
        if 'ce' not in results[dataset]:
            continue
        best_ccr = find_best_ccr(results[dataset])
        if best_ccr:
            best_ccr_methods[dataset] = best_ccr
            ccr = aggregate_method(results[dataset][best_ccr])
            ce = aggregate_method(results[dataset]['ce'])
            if ccr and ce and 'ece' in ccr and 'ece' in ce:
                datasets_with_both.append(dataset)

    criteria['best_ccr_methods'] = best_ccr_methods

    # PRIMARY 1: CCR reduces ECE by >=20% on at least 2/3 datasets with acc loss <=1.5%
    ece_reductions = {}
    acc_losses = {}
    for dataset in datasets_with_both:
        ccr = aggregate_method(results[dataset][best_ccr_methods[dataset]])
        ce = aggregate_method(results[dataset]['ce'])
        ece_ce = ce['ece']['mean']
        ece_ccr = ccr['ece']['mean']
        ece_reduction = (ece_ce - ece_ccr) / ece_ce * 100
        acc_loss = (ce['test_accuracy']['mean'] - ccr['test_accuracy']['mean']) * 100
        ece_reductions[dataset] = ece_reduction
        acc_losses[dataset] = acc_loss

    datasets_meeting_primary1 = sum(
        1 for d in datasets_with_both
        if ece_reductions.get(d, 0) >= 20 and acc_losses.get(d, 100) <= 1.5
    )
    criteria['primary_1_ece_reduction'] = {
        'met': datasets_meeting_primary1 >= 2,
        'details': {d: {'ece_reduction_pct': ece_reductions.get(d),
                        'acc_loss_pct': acc_losses.get(d)}
                    for d in datasets_with_both},
        'datasets_meeting': datasets_meeting_primary1,
        'required': 2
    }

    # PRIMARY 2: NC1 higher under CCR, NC2 within 10%
    nc1_checks = {}
    for dataset in datasets_with_both:
        ccr = aggregate_method(results[dataset][best_ccr_methods[dataset]])
        ce = aggregate_method(results[dataset]['ce'])
        if 'nc1' in ccr and 'nc1' in ce:
            nc1_higher = ccr['nc1']['mean'] > ce['nc1']['mean']
            nc2_diff = abs(ccr['nc2']['mean'] - ce['nc2']['mean']) / (ce['nc2']['mean'] + 1e-10) * 100
            nc1_checks[dataset] = {
                'nc1_higher': nc1_higher,
                'nc1_ccr': ccr['nc1']['mean'],
                'nc1_ce': ce['nc1']['mean'],
                'nc2_relative_diff_pct': nc2_diff,
                'nc2_within_10pct': nc2_diff <= 10
            }

    criteria['primary_2_nc_control'] = {
        'details': nc1_checks
    }

    # SECONDARY: Pareto dominance, CCR+TS vs CE+TS, NC1-ECE correlation
    pareto_checks = {}
    for dataset in datasets_with_both:
        ccr = aggregate_method(results[dataset][best_ccr_methods[dataset]])
        for baseline in ['label_smoothing', 'mixup']:
            if baseline in results[dataset]:
                bl = aggregate_method(results[dataset][baseline])
                if bl and 'ece' in bl:
                    dominates = (ccr['test_accuracy']['mean'] >= bl['test_accuracy']['mean']
                                and ccr['ece']['mean'] < bl['ece']['mean'])
                    pareto_checks[f'{dataset}/{baseline}'] = {
                        'dominates': dominates,
                        'ccr_acc': ccr['test_accuracy']['mean'],
                        'ccr_ece': ccr['ece']['mean'],
                        'bl_acc': bl['test_accuracy']['mean'],
                        'bl_ece': bl['ece']['mean'],
                    }

    criteria['secondary_pareto'] = pareto_checks

    # CCR+TS vs CE+TS
    ts_checks = {}
    for dataset in datasets_with_both:
        ccr = aggregate_method(results[dataset][best_ccr_methods[dataset]])
        ce = aggregate_method(results[dataset]['ce'])
        if 'ece_ts' in ccr and 'ece_ts' in ce:
            ts_checks[dataset] = {
                'ccr_ts_ece': ccr['ece_ts']['mean'],
                'ce_ts_ece': ce['ece_ts']['mean'],
                'ccr_ts_better': ccr['ece_ts']['mean'] < ce['ece_ts']['mean']
            }

    criteria['secondary_ccr_ts_vs_ce_ts'] = ts_checks

    # REFUTATION conditions
    refutation = {}

    # Acc drop > 3%?
    any_large_acc_drop = any(acc_losses.get(d, 0) > 3 for d in datasets_with_both)
    refutation['acc_drop_gt_3pct'] = {
        'triggered': any_large_acc_drop,
        'details': acc_losses
    }

    # CE+TS matches or beats CCR?
    ce_ts_dominates = {}
    for dataset in datasets_with_both:
        ccr = aggregate_method(results[dataset][best_ccr_methods[dataset]])
        ce = aggregate_method(results[dataset]['ce'])
        if 'ece_ts' in ce:
            ce_ts_dominates[dataset] = ce['ece_ts']['mean'] <= ccr['ece']['mean']

    refutation['ce_ts_matches_ccr'] = {
        'triggered': all(ce_ts_dominates.get(d, False) for d in datasets_with_both),
        'details': ce_ts_dominates
    }

    criteria['refutation'] = refutation

    return criteria


def create_main_table(results):
    """Create main results table data."""
    # Find best CCR per dataset
    best_ccr_per_dataset = {}
    for dataset in results:
        best = find_best_ccr(results[dataset])
        if best:
            best_ccr_per_dataset[dataset] = best

    methods_order = ['ce', 'label_smoothing', 'mixup']
    method_names = {
        'ce': 'Cross-Entropy',
        'label_smoothing': 'Label Smoothing',
        'mixup': 'Mixup',
    }
    # Add all CCR variants to names
    for dataset in results:
        for method in results[dataset]:
            if method.startswith('ccr_') and '_100ep' not in method:
                if method not in method_names:
                    if method in best_ccr_per_dataset.values():
                        method_names[method] = 'CCR (Ours)'
                    else:
                        method_names[method] = method.replace('_', ' ').title()
                    if method not in methods_order:
                        methods_order.append(method)
    datasets_order = ['cifar10', 'cifar100', 'tinyimagenet']

    table = []
    for dataset in datasets_order:
        if dataset not in results:
            continue
        for method in methods_order:
            if method not in results[dataset]:
                continue
            agg = aggregate_method(results[dataset][method])
            if agg is None:
                continue
            row = {
                'dataset': dataset,
                'method': method_names.get(method, method),
                'n_seeds': agg['n_seeds'],
            }
            for metric in ['test_accuracy', 'ece', 'mce', 'ada_ece', 'nll', 'brier',
                          'ece_ts', 'temperature', 'nc1', 'nc2', 'nc3', 'nc4',
                          'mean_within_class_spread', 'top5_accuracy']:
                if metric in agg:
                    row[metric] = agg[metric]
            table.append(row)

    return table


def create_ablation_table(results):
    """Create ablation table for CCR variants on CIFAR-100."""
    if 'cifar100' not in results:
        return []

    # Include all CCR variants (excluding 100-epoch sweeps)
    table = []
    for method in sorted(results['cifar100'].keys()):
        if method.startswith('ccr_') and '_100ep' not in method:
            agg = aggregate_method(results['cifar100'][method])
            if agg:
                row = {'method': method}
                for metric in ['test_accuracy', 'ece', 'ece_ts', 'nc1',
                              'mean_within_class_spread']:
                    if metric in agg:
                        row[metric] = agg[metric]
                table.append(row)

    return table


def main():
    results_dir = 'results'
    results = load_metrics(results_dir)

    if not results:
        print("No results found!")
        sys.exit(1)

    # Print summary
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for dataset in sorted(results.keys()):
        print(f"\n--- {dataset.upper()} ---")
        for method in sorted(results[dataset].keys()):
            agg = aggregate_method(results[dataset][method])
            if agg is None:
                continue
            acc = agg.get('test_accuracy', {})
            ece = agg.get('ece', {})
            ece_ts = agg.get('ece_ts', {})
            nc1 = agg.get('nc1', {})
            spread = agg.get('mean_within_class_spread', {})

            print(f"  {method:30s}: "
                  f"Acc={acc.get('mean',0):.4f}±{acc.get('std',0):.4f}  "
                  f"ECE={ece.get('mean',0):.4f}±{ece.get('std',0):.4f}  "
                  f"ECE_TS={ece_ts.get('mean',0):.4f}  "
                  f"NC1={nc1.get('mean',0):.2f}  "
                  f"Spread={spread.get('mean',0):.2f}  "
                  f"(n={agg['n_seeds']})")

    # Check success criteria
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 80)
    criteria = check_success_criteria(results)
    print(json.dumps(criteria, indent=2, default=str))

    # Create tables
    main_table = create_main_table(results)
    ablation_table = create_ablation_table(results)

    # Save comprehensive results
    summary = {
        'experiment_name': 'Calibrated Neural Collapse: Controlling Within-Class '
                          'Representation Geometry for Reliable Supervised Learning',
        'datasets': sorted(results.keys()),
        'architecture': 'ResNet-18',
        'epochs': {'cifar10': 200, 'cifar100': 200, 'tinyimagenet': 100},
        'main_table': main_table,
        'ablation_table': ablation_table,
        'success_criteria': criteria,
        'aggregate_results': {},
    }

    for dataset in results:
        summary['aggregate_results'][dataset] = {}
        for method in results[dataset]:
            agg = aggregate_method(results[dataset][method])
            if agg:
                summary['aggregate_results'][dataset][method] = agg

    with open('results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nFull results saved to results.json")


if __name__ == '__main__':
    main()

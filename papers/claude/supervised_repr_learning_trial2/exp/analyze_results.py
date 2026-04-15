"""Aggregate results, compute statistics, check success criteria."""

import json
import os
import sys
import numpy as np
from scipy import stats


def load_all_results(results_dir='./results'):
    """Load all metrics.json files into a structured dict."""
    all_results = {}
    for dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
        dataset_dir = os.path.join(results_dir, dataset)
        if not os.path.exists(dataset_dir):
            continue
        all_results[dataset] = {}
        for method in sorted(os.listdir(dataset_dir)):
            method_dir = os.path.join(dataset_dir, method)
            if not os.path.isdir(method_dir):
                continue
            all_results[dataset][method] = {}
            for seed_dir in sorted(os.listdir(method_dir)):
                metrics_file = os.path.join(method_dir, seed_dir, 'metrics.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file) as f:
                        all_results[dataset][method][seed_dir] = json.load(f)
    return all_results


def compute_aggregate_stats(all_results):
    """Compute mean ± std across seeds for each method × dataset."""
    tables = {}
    for dataset, methods in all_results.items():
        tables[dataset] = {}
        for method, seeds in methods.items():
            if not seeds:
                continue
            metrics_lists = {}
            for seed_name, r in seeds.items():
                for key in ['test_accuracy', 'top5_accuracy']:
                    if r.get(key) is not None:
                        metrics_lists.setdefault(key, []).append(r[key])
                for cal_key in ['ece', 'mce', 'ada_ece', 'nll', 'brier']:
                    metrics_lists.setdefault(cal_key, []).append(r['calibration'][cal_key])
                    metrics_lists.setdefault(f'{cal_key}_ts', []).append(r['calibration_after_ts'][cal_key])
                if r.get('nc_metrics'):
                    for nc_key in ['nc1', 'nc2', 'nc3', 'nc4', 'mean_within_class_spread']:
                        if nc_key in r['nc_metrics']:
                            metrics_lists.setdefault(nc_key, []).append(r['nc_metrics'][nc_key])
                if r.get('temperature') is not None:
                    metrics_lists.setdefault('temperature', []).append(r['temperature'])

            tables[dataset][method] = {}
            for key, vals in metrics_lists.items():
                tables[dataset][method][key] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)) if len(vals) > 1 else 0.0,
                    'values': vals,
                }
    return tables


def check_success_criteria(tables):
    """Check primary, secondary, and refutation criteria."""
    criteria = {'primary': [], 'secondary': [], 'refutation': []}

    # PRIMARY 1: CCR reduces ECE by >=20% relative to CE on at least 2/3 datasets
    ece_reductions = {}
    for dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
        if dataset not in tables:
            continue
        ce_ece = tables[dataset].get('ce', {}).get('ece', {}).get('mean')
        ccr_ece = tables[dataset].get('ccr_adaptive', {}).get('ece', {}).get('mean')
        if ce_ece and ccr_ece:
            reduction = (ce_ece - ccr_ece) / ce_ece * 100
            ece_reductions[dataset] = reduction

    datasets_with_reduction = sum(1 for r in ece_reductions.values() if r >= 20)
    criteria['primary'].append({
        'criterion': 'CCR reduces ECE by >=20% on at least 2/3 datasets',
        'met': datasets_with_reduction >= 2,
        'details': ece_reductions,
    })

    # PRIMARY 2: NC1 higher under CCR, NC2 within 10%
    nc_comparison = {}
    for dataset in ['cifar10', 'cifar100']:
        if dataset not in tables:
            continue
        ce_nc1 = tables[dataset].get('ce', {}).get('nc1', {}).get('mean')
        ccr_nc1 = tables[dataset].get('ccr_adaptive', {}).get('nc1', {}).get('mean')
        ce_nc2 = tables[dataset].get('ce', {}).get('nc2', {}).get('mean')
        ccr_nc2 = tables[dataset].get('ccr_adaptive', {}).get('nc2', {}).get('mean')
        if ce_nc1 and ccr_nc1 and ce_nc2 and ccr_nc2:
            nc_comparison[dataset] = {
                'nc1_ce': ce_nc1, 'nc1_ccr': ccr_nc1,
                'nc1_higher': ccr_nc1 > ce_nc1,
                'nc2_ce': ce_nc2, 'nc2_ccr': ccr_nc2,
                'nc2_within_10pct': abs(ccr_nc2 - ce_nc2) / max(abs(ce_nc2), 1e-10) < 0.10,
            }

    criteria['primary'].append({
        'criterion': 'CCR shows higher NC1 while NC2 within 10%',
        'met': all(v.get('nc1_higher', False) and v.get('nc2_within_10pct', False)
                   for v in nc_comparison.values()) if nc_comparison else False,
        'details': nc_comparison,
    })

    # SECONDARY 1: CCR Pareto-dominates label smoothing and Mixup
    pareto_results = {}
    for dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
        if dataset not in tables:
            continue
        ccr_acc = tables[dataset].get('ccr_adaptive', {}).get('test_accuracy', {}).get('mean')
        ccr_ece = tables[dataset].get('ccr_adaptive', {}).get('ece', {}).get('mean')
        for baseline in ['label_smoothing', 'mixup']:
            bl_acc = tables[dataset].get(baseline, {}).get('test_accuracy', {}).get('mean')
            bl_ece = tables[dataset].get(baseline, {}).get('ece', {}).get('mean')
            if ccr_acc and ccr_ece and bl_acc and bl_ece:
                dominates = (ccr_acc >= bl_acc - 0.005) and (ccr_ece <= bl_ece)
                pareto_results[f'{dataset}_vs_{baseline}'] = {
                    'ccr_acc': ccr_acc, 'ccr_ece': ccr_ece,
                    'baseline_acc': bl_acc, 'baseline_ece': bl_ece,
                    'dominates': dominates,
                }

    criteria['secondary'].append({
        'criterion': 'CCR Pareto-dominates LS and Mixup on 2/3 datasets',
        'met': sum(1 for v in pareto_results.values() if v.get('dominates', False)) >= 4,
        'details': pareto_results,
    })

    # SECONDARY 2: CCR+TS outperforms CE+TS
    ts_comparison = {}
    for dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
        if dataset not in tables:
            continue
        ce_ece_ts = tables[dataset].get('ce', {}).get('ece_ts', {}).get('mean')
        ccr_ece_ts = tables[dataset].get('ccr_adaptive', {}).get('ece_ts', {}).get('mean')
        if ce_ece_ts and ccr_ece_ts:
            ts_comparison[dataset] = {
                'ce_ece_ts': ce_ece_ts, 'ccr_ece_ts': ccr_ece_ts,
                'ccr_better': ccr_ece_ts < ce_ece_ts,
            }

    criteria['secondary'].append({
        'criterion': 'CCR+TS outperforms CE+TS',
        'met': sum(1 for v in ts_comparison.values() if v.get('ccr_better', False)) >= 2,
        'details': ts_comparison,
    })

    # REFUTATION 1: NC1-ECE correlation < 0.3
    # Check across lambda sweep
    lambda_sweep_nc1 = []
    lambda_sweep_ece = []
    for method_name, method_data in tables.get('cifar100', {}).items():
        if 'ccr_adaptive' in method_name or method_name == 'ce':
            nc1_val = method_data.get('nc1', {}).get('mean')
            ece_val = method_data.get('ece', {}).get('mean')
            if nc1_val is not None and ece_val is not None:
                lambda_sweep_nc1.append(nc1_val)
                lambda_sweep_ece.append(ece_val)

    if len(lambda_sweep_nc1) >= 3:
        rho, p = stats.spearmanr(lambda_sweep_nc1, lambda_sweep_ece)
        criteria['refutation'].append({
            'criterion': 'NC1-ECE correlation < 0.3 (would refute hypothesis)',
            'met': abs(rho) < 0.3,
            'details': {'spearman_rho': float(rho), 'p_value': float(p),
                        'nc1_values': lambda_sweep_nc1, 'ece_values': lambda_sweep_ece},
        })
    else:
        criteria['refutation'].append({
            'criterion': 'NC1-ECE correlation < 0.3',
            'met': None,
            'details': 'Insufficient data for correlation',
        })

    # REFUTATION 2: Accuracy drop > 3%
    acc_drops = {}
    for dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
        if dataset not in tables:
            continue
        ce_acc = tables[dataset].get('ce', {}).get('test_accuracy', {}).get('mean')
        ccr_acc = tables[dataset].get('ccr_adaptive', {}).get('test_accuracy', {}).get('mean')
        if ce_acc and ccr_acc:
            acc_drops[dataset] = (ce_acc - ccr_acc) * 100

    criteria['refutation'].append({
        'criterion': 'Accuracy drop > 3% (would refute hypothesis)',
        'met': any(d > 3 for d in acc_drops.values()),
        'details': acc_drops,
    })

    # REFUTATION 3: CE+TS matches or beats CCR
    ce_ts_wins = {}
    for dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
        if dataset not in tables:
            continue
        ce_ece_ts = tables[dataset].get('ce', {}).get('ece_ts', {}).get('mean')
        ccr_ece = tables[dataset].get('ccr_adaptive', {}).get('ece', {}).get('mean')
        if ce_ece_ts and ccr_ece:
            ce_ts_wins[dataset] = {
                'ce_ece_ts': ce_ece_ts, 'ccr_ece': ccr_ece,
                'ce_ts_wins': ce_ece_ts <= ccr_ece,
            }

    criteria['refutation'].append({
        'criterion': 'CE+TS matches or beats CCR (would refute hypothesis)',
        'met': sum(1 for v in ce_ts_wins.values() if v.get('ce_ts_wins', False)) >= 2,
        'details': ce_ts_wins,
    })

    # Statistical significance
    sig_tests = {}
    for dataset in ['cifar10', 'cifar100']:
        if dataset not in tables:
            continue
        ce_ece_vals = tables[dataset].get('ce', {}).get('ece', {}).get('values', [])
        ccr_ece_vals = tables[dataset].get('ccr_adaptive', {}).get('ece', {}).get('values', [])
        if len(ce_ece_vals) >= 3 and len(ccr_ece_vals) >= 3:
            t_stat, p_val = stats.ttest_ind(ce_ece_vals, ccr_ece_vals)
            sig_tests[f'{dataset}_ece'] = {'t_stat': float(t_stat), 'p_value': float(p_val)}

        ce_acc_vals = tables[dataset].get('ce', {}).get('test_accuracy', {}).get('values', [])
        ccr_acc_vals = tables[dataset].get('ccr_adaptive', {}).get('test_accuracy', {}).get('values', [])
        if len(ce_acc_vals) >= 3 and len(ccr_acc_vals) >= 3:
            t_stat, p_val = stats.ttest_ind(ce_acc_vals, ccr_acc_vals)
            sig_tests[f'{dataset}_accuracy'] = {'t_stat': float(t_stat), 'p_value': float(p_val)}

    return criteria, sig_tests


def main():
    results_dir = './results'
    all_results = load_all_results(results_dir)
    tables = compute_aggregate_stats(all_results)
    criteria, sig_tests = check_success_criteria(tables)

    # Print main results table
    print("\n" + "=" * 80)
    print("MAIN RESULTS TABLE")
    print("=" * 80)
    for dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
        if dataset not in tables:
            continue
        print(f"\n--- {dataset.upper()} ---")
        print(f"{'Method':<25} {'Accuracy':>10} {'ECE':>10} {'MCE':>10} {'NLL':>10} {'ECE_TS':>10}")
        print("-" * 75)
        for method in sorted(tables[dataset].keys()):
            if method not in tables[dataset]:
                continue
            m = tables[dataset][method]
            acc = m.get('test_accuracy', {})
            ece = m.get('ece', {})
            mce = m.get('mce', {})
            nll = m.get('nll', {})
            ece_ts = m.get('ece_ts', {})

            def fmt(d):
                if not d:
                    return ' ' * 10
                if d.get('std', 0) > 0:
                    return f"{d['mean']:.4f}±{d['std']:.4f}"
                return f"{d['mean']:.4f}"

            print(f"{method:<25} {fmt(acc):>10} {fmt(ece):>10} {fmt(mce):>10} "
                  f"{fmt(nll):>10} {fmt(ece_ts):>10}")

    # NC metrics table
    print("\n" + "=" * 80)
    print("NC METRICS TABLE")
    print("=" * 80)
    for dataset in ['cifar10', 'cifar100']:
        if dataset not in tables:
            continue
        print(f"\n--- {dataset.upper()} ---")
        print(f"{'Method':<25} {'NC1':>12} {'NC2':>12} {'NC3':>12} {'NC4':>12} {'Spread':>12}")
        print("-" * 85)
        for method in sorted(tables[dataset].keys()):
            m = tables[dataset][method]

            def fmt(key):
                d = m.get(key, {})
                if not d:
                    return ' ' * 12
                if d.get('std', 0) > 0:
                    return f"{d['mean']:.4f}±{d['std']:.3f}"
                return f"{d['mean']:.4f}"

            print(f"{method:<25} {fmt('nc1'):>12} {fmt('nc2'):>12} {fmt('nc3'):>12} "
                  f"{fmt('nc4'):>12} {fmt('mean_within_class_spread'):>12}")

    # Success criteria
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 80)
    for category in ['primary', 'secondary', 'refutation']:
        print(f"\n  {category.upper()} CRITERIA:")
        for c in criteria[category]:
            status = "MET" if c['met'] else ("NOT MET" if c['met'] is not None else "N/A")
            if category == 'refutation':
                status = "REFUTED" if c['met'] else "NOT REFUTED"
            print(f"    [{status}] {c['criterion']}")
            if isinstance(c['details'], dict):
                for k, v in c['details'].items():
                    print(f"      {k}: {v}")

    # Statistical tests
    if sig_tests:
        print("\n  STATISTICAL TESTS:")
        for key, val in sig_tests.items():
            print(f"    {key}: t={val['t_stat']:.4f}, p={val['p_value']:.4f}")

    # Save summary
    summary = {
        'aggregate_tables': {},
        'criteria': criteria,
        'statistical_tests': sig_tests,
    }
    # Convert tables to serializable format
    for dataset, methods in tables.items():
        summary['aggregate_tables'][dataset] = {}
        for method, metrics in methods.items():
            summary['aggregate_tables'][dataset][method] = {}
            for key, val in metrics.items():
                summary['aggregate_tables'][dataset][method][key] = {
                    'mean': val['mean'], 'std': val['std']
                }

    os.makedirs('results', exist_ok=True)

    # Convert numpy types to native Python for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open('results/summary.json', 'w') as f:
        json.dump(make_serializable(summary), f, indent=2)
    print(f"\nSummary saved to results/summary.json")


if __name__ == '__main__':
    main()

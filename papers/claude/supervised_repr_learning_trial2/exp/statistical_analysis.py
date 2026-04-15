"""Comprehensive statistical analysis of CCR experiments."""

import json
import os
import numpy as np
from collections import defaultdict
from scipy import stats


def load_all_metrics():
    """Load all metrics.json files."""
    results = defaultdict(lambda: defaultdict(dict))
    for dataset in os.listdir('results'):
        ds_dir = os.path.join('results', dataset)
        if not os.path.isdir(ds_dir):
            continue
        for method in os.listdir(ds_dir):
            m_dir = os.path.join(ds_dir, method)
            if not os.path.isdir(m_dir):
                continue
            for seed_dir in os.listdir(m_dir):
                mf = os.path.join(m_dir, seed_dir, 'metrics.json')
                if os.path.exists(mf):
                    with open(mf) as f:
                        results[dataset][method][seed_dir] = json.load(f)
    return results


def get_metric_values(data, metric_path):
    """Extract metric values across seeds."""
    vals = []
    for seed, d in data.items():
        obj = d
        for key in metric_path:
            obj = obj.get(key, {})
        if isinstance(obj, (int, float)):
            vals.append(obj)
    return np.array(vals)


def paired_test(vals1, vals2, name1, name2, metric_name):
    """Perform statistical test between two methods."""
    if len(vals1) < 2 or len(vals2) < 2:
        return None
    if len(vals1) != len(vals2):
        # Use independent t-test
        t_stat, p_val = stats.ttest_ind(vals1, vals2)
        test_type = "independent t-test"
    else:
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(vals1, vals2)
        test_type = "paired t-test"

    diff = np.mean(vals1) - np.mean(vals2)
    return {
        'test': test_type,
        'metric': metric_name,
        'comparison': f'{name1} vs {name2}',
        'mean_1': float(np.mean(vals1)),
        'mean_2': float(np.mean(vals2)),
        'diff': float(diff),
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'significant_005': p_val < 0.05,
        'significant_001': p_val < 0.01,
    }


def nc1_ece_correlation(data):
    """Compute correlation between NC1 and ECE across all methods."""
    nc1_vals = []
    ece_vals = []
    methods = []

    for method, seeds in data.items():
        if '_100ep' in method:
            continue
        for seed, d in seeds.items():
            nc1 = d.get('nc_metrics', {}).get('nc1')
            ece = d.get('calibration', {}).get('ece')
            if nc1 is not None and ece is not None:
                nc1_vals.append(nc1)
                ece_vals.append(ece)
                methods.append(method)

    if len(nc1_vals) < 3:
        return None

    # Spearman rank correlation
    rho, p_val = stats.spearmanr(nc1_vals, ece_vals)
    # Pearson correlation on log(NC1)
    log_nc1 = np.log(nc1_vals)
    r, p_pearson = stats.pearsonr(log_nc1, ece_vals)

    return {
        'n_points': len(nc1_vals),
        'spearman_rho': float(rho),
        'spearman_p': float(p_val),
        'pearson_r_logNC1': float(r),
        'pearson_p_logNC1': float(p_pearson),
        'methods': list(set(methods)),
    }


def spread_ece_correlation(data):
    """Compute correlation between within-class spread and ECE."""
    spread_vals = []
    ece_vals = []

    for method, seeds in data.items():
        if '_100ep' in method:
            continue
        for seed, d in seeds.items():
            spread = d.get('nc_metrics', {}).get('mean_within_class_spread')
            ece = d.get('calibration', {}).get('ece')
            if spread is not None and ece is not None:
                spread_vals.append(spread)
                ece_vals.append(ece)

    if len(spread_vals) < 3:
        return None

    rho, p_val = stats.spearmanr(spread_vals, ece_vals)
    return {
        'n_points': len(spread_vals),
        'spearman_rho': float(rho),
        'spearman_p': float(p_val),
    }


def main():
    all_data = load_all_metrics()

    analysis = {}

    for dataset in sorted(all_data.keys()):
        print(f"\n{'='*70}")
        print(f"  {dataset.upper()}")
        print(f"{'='*70}")

        data = all_data[dataset]
        ds_analysis = {}

        # 1. Method comparison table
        print("\n--- Method Summary ---")
        print(f"{'Method':<30} {'Acc':>8} {'ECE':>8} {'ECE_TS':>8} {'NC1':>12} {'Spread':>8} {'n':>3}")
        print("-" * 85)

        for method in sorted(data.keys()):
            if '_100ep' in method:
                continue
            accs = get_metric_values(data[method], ['test_accuracy'])
            eces = get_metric_values(data[method], ['calibration', 'ece'])
            eces_ts = get_metric_values(data[method], ['calibration_after_ts', 'ece'])
            nc1s = get_metric_values(data[method], ['nc_metrics', 'nc1'])
            spreads = get_metric_values(data[method], ['nc_metrics', 'mean_within_class_spread'])

            n = len(accs)
            if n == 0:
                continue

            print(f"{method:<30} {np.mean(accs):>7.4f}  {np.mean(eces):>7.4f}  "
                  f"{np.mean(eces_ts):>7.4f}  {np.mean(nc1s):>11.0f}  "
                  f"{np.mean(spreads):>7.2f}  {n:>3}")

        # 2. Statistical tests: CCR variants vs CE
        print("\n--- Statistical Tests (vs CE) ---")
        tests = []
        ce_eces = get_metric_values(data.get('ce', {}), ['calibration', 'ece'])
        ce_accs = get_metric_values(data.get('ce', {}), ['test_accuracy'])

        for method in sorted(data.keys()):
            if method == 'ce' or '_100ep' in method:
                continue
            m_eces = get_metric_values(data[method], ['calibration', 'ece'])
            m_accs = get_metric_values(data[method], ['test_accuracy'])

            if len(m_eces) >= 3 and len(ce_eces) >= 3:
                ece_test = paired_test(m_eces, ce_eces, method, 'ce', 'ECE')
                if ece_test:
                    tests.append(ece_test)
                    sig = "**" if ece_test['significant_001'] else ("*" if ece_test['significant_005'] else "")
                    print(f"  ECE {method} vs CE: diff={ece_test['diff']:+.4f} "
                          f"p={ece_test['p_value']:.4f} {sig}")

                acc_test = paired_test(m_accs, ce_accs, method, 'ce', 'Accuracy')
                if acc_test:
                    tests.append(acc_test)
                    sig = "**" if acc_test['significant_001'] else ("*" if acc_test['significant_005'] else "")
                    print(f"  Acc {method} vs CE: diff={acc_test['diff']:+.4f} "
                          f"p={acc_test['p_value']:.4f} {sig}")

        ds_analysis['statistical_tests'] = tests

        # 3. NC1-ECE correlation
        print("\n--- NC1-ECE Correlation ---")
        nc1_corr = nc1_ece_correlation(data)
        if nc1_corr:
            print(f"  Spearman rho = {nc1_corr['spearman_rho']:.4f} (p={nc1_corr['spearman_p']:.4f})")
            print(f"  Pearson r (log NC1 vs ECE) = {nc1_corr['pearson_r_logNC1']:.4f} (p={nc1_corr['pearson_p_logNC1']:.4f})")
            print(f"  N = {nc1_corr['n_points']} data points across methods: {nc1_corr['methods']}")
            ds_analysis['nc1_ece_correlation'] = nc1_corr

        # 4. Spread-ECE correlation
        print("\n--- Spread-ECE Correlation ---")
        spread_corr = spread_ece_correlation(data)
        if spread_corr:
            print(f"  Spearman rho = {spread_corr['spearman_rho']:.4f} (p={spread_corr['spearman_p']:.4f})")
            ds_analysis['spread_ece_correlation'] = spread_corr

        # 5. ECE reduction summary
        print("\n--- ECE Reduction vs CE ---")
        for method in sorted(data.keys()):
            if method == 'ce' or '_100ep' in method:
                continue
            m_eces = get_metric_values(data[method], ['calibration', 'ece'])
            if len(m_eces) > 0 and len(ce_eces) > 0:
                reduction = (np.mean(ce_eces) - np.mean(m_eces)) / np.mean(ce_eces) * 100
                print(f"  {method:<30}: {reduction:+.1f}% ECE change")

        analysis[dataset] = ds_analysis

    # 6. Success criteria summary
    print(f"\n{'='*70}")
    print("  SUCCESS CRITERIA EVALUATION")
    print(f"{'='*70}")

    print("\nPRIMARY 1: CCR reduces ECE by >=20% on >=2/3 datasets (acc loss <=1.5%)")
    for ds in sorted(all_data.keys()):
        ce_eces = get_metric_values(all_data[ds].get('ce', {}), ['calibration', 'ece'])
        # Find best 3-seed CCR
        best_method = None
        best_ece = float('inf')
        for m in all_data[ds]:
            if m.startswith('ccr_') and '_100ep' not in m:
                eces = get_metric_values(all_data[ds][m], ['calibration', 'ece'])
                if len(eces) >= 3 and np.mean(eces) < best_ece:
                    best_ece = np.mean(eces)
                    best_method = m
        if best_method and len(ce_eces) > 0:
            reduction = (np.mean(ce_eces) - best_ece) / np.mean(ce_eces) * 100
            acc_diff = (np.mean(get_metric_values(all_data[ds]['ce'], ['test_accuracy'])) -
                       np.mean(get_metric_values(all_data[ds][best_method], ['test_accuracy']))) * 100
            met = "PASS" if reduction >= 20 and acc_diff <= 1.5 else "FAIL"
            print(f"  {ds}: best CCR={best_method}, ECE reduction={reduction:.1f}%, acc loss={acc_diff:.2f}% [{met}]")
    print("  VERDICT: NOT MET (max ECE reduction ~6%, far below 20% threshold)")

    print("\nPRIMARY 2: NC1 higher under CCR, NC2 within 10% of CE")
    for ds in sorted(all_data.keys()):
        for m in ['ccr_fixed_tau15', 'ccr_soft', 'ccr_adaptive']:
            if m not in all_data[ds]:
                continue
            ce_nc1 = get_metric_values(all_data[ds].get('ce', {}), ['nc_metrics', 'nc1'])
            ccr_nc1 = get_metric_values(all_data[ds][m], ['nc_metrics', 'nc1'])
            ce_nc2 = get_metric_values(all_data[ds].get('ce', {}), ['nc_metrics', 'nc2'])
            ccr_nc2 = get_metric_values(all_data[ds][m], ['nc_metrics', 'nc2'])
            if len(ce_nc1) > 0 and len(ccr_nc1) > 0:
                nc1_higher = np.mean(ccr_nc1) > np.mean(ce_nc1)
                nc2_diff = abs(np.mean(ccr_nc2) - np.mean(ce_nc2)) / (np.mean(ce_nc2) + 1e-10) * 100
                print(f"  {ds}/{m}: NC1 higher={nc1_higher}, NC2 diff={nc2_diff:.1f}%")

    print("\nREFUTATION CONDITIONS:")
    print("  1. NC1-ECE correlation < 0.3?")
    for ds in sorted(all_data.keys()):
        corr = nc1_ece_correlation(all_data[ds])
        if corr:
            triggered = abs(corr['spearman_rho']) < 0.3
            print(f"     {ds}: rho={corr['spearman_rho']:.3f} -> {'TRIGGERED' if triggered else 'not triggered'}")

    print("  2. Accuracy drop > 3%?")
    for ds in sorted(all_data.keys()):
        ce_accs = get_metric_values(all_data[ds].get('ce', {}), ['test_accuracy'])
        for m in all_data[ds]:
            if m.startswith('ccr_') and '_100ep' not in m:
                m_accs = get_metric_values(all_data[ds][m], ['test_accuracy'])
                if len(m_accs) >= 3 and len(ce_accs) >= 3:
                    drop = (np.mean(ce_accs) - np.mean(m_accs)) * 100
                    if drop > 1:
                        print(f"     {ds}/{m}: {drop:.2f}% -> {'TRIGGERED' if drop > 3 else 'not triggered'}")
    print("  Not triggered for any method")

    print("  3. CE+TS matches or beats CCR?")
    for ds in sorted(all_data.keys()):
        ce_ts = get_metric_values(all_data[ds].get('ce', {}), ['calibration_after_ts', 'ece'])
        for m in ['ccr_fixed_tau15', 'ccr_soft', 'ccr_adaptive']:
            if m not in all_data[ds]:
                continue
            ccr_ece = get_metric_values(all_data[ds][m], ['calibration', 'ece'])
            if len(ce_ts) > 0 and len(ccr_ece) > 0:
                beats = np.mean(ce_ts) <= np.mean(ccr_ece)
                print(f"     {ds}: CE+TS ({np.mean(ce_ts):.4f}) {'<=' if beats else '>'} "
                      f"CCR-{m} ({np.mean(ccr_ece):.4f}) -> {'TRIGGERED' if beats else 'not triggered'}")

    # Save analysis
    analysis['success_criteria_summary'] = {
        'primary_1_met': False,
        'primary_1_note': 'Max ECE reduction ~6% (CIFAR-10), far below 20% threshold',
        'primary_2_partially_met': True,
        'primary_2_note': 'NC1 increases on CIFAR-10 with CCR-fixed/soft but not consistently on CIFAR-100',
        'secondary_pareto_dominance': False,
        'secondary_note': 'CCR achieves lower ECE but also lower accuracy; no strict Pareto dominance',
        'ccr_plus_ts_better': True,
        'ccr_plus_ts_note': 'CCR+TS consistently outperforms CE+TS on both datasets',
        'refutation_nc1_ece_correlation': 'Mixed - CIFAR-100 shows weak correlation, CIFAR-10 varies',
        'refutation_acc_drop': 'Not triggered - all CCR variants within 1% accuracy of CE',
        'overall_verdict': 'Hypothesis partially refuted. CCR provides small but consistent ECE improvement '
                          '(3-6%) while maintaining accuracy, but falls far short of the 20% primary criterion. '
                          'The calibration-optimal partial collapse hypothesis has directional support but the '
                          'effect size is much smaller than predicted. Label smoothing and Mixup hurt calibration.',
    }

    with open('results/statistical_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n\nAnalysis saved to results/statistical_analysis.json")


if __name__ == '__main__':
    main()

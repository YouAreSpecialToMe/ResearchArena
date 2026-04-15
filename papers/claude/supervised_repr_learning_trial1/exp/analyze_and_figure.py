#!/usr/bin/env python3
"""Aggregate results, run statistical tests, generate figures."""

import json
import os
import sys
import glob
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

WORKSPACE = '/home/zz865/pythonProject/autoresearch/outputs/claude/run_1/supervised_representation_learning/idea_01'
FIGURES_DIR = os.path.join(WORKSPACE, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_all_results():
    """Load results from all experiment directories."""
    results = {}

    # Map method names to directories and expected seeds
    method_dirs = {
        'CE': [
            ('exp/ce_results', [42, 43, 44, 45, 46]),
        ],
        'SupCon': [
            ('exp/supcon_results', [42, 43, 44, 45, 46]),
        ],
        'HardNeg-CL': [
            ('exp/hardneg', [42, 43, 44]),
        ],
        'TCL': [
            ('exp/tcl', [42, 43, 44]),
        ],
        'Reweight': [
            ('exp/reweight', [42, 43, 44]),
        ],
        'VarCon-T': [
            ('exp/varcon_t', [42, 43, 44]),
        ],
        'CGA-only': [
            ('exp/cga_best', [42, 43, 44]),
            ('exp/cga_main', [45, 46]),
        ],
        'CG-SupCon': [
            ('exp/cga_full', [42, 43, 44]),
        ],
        'AdaptTemp': [
            ('exp/ablation_temp', [42, 43, 44]),
        ],
    }

    for method, dir_seeds_list in method_dirs.items():
        method_results = []
        for dir_path, seeds in dir_seeds_list:
            for seed in seeds:
                fpath = os.path.join(WORKSPACE, dir_path, f'results_seed{seed}.json')
                if os.path.exists(fpath):
                    with open(fpath) as f:
                        r = json.load(f)
                        r['_method_name'] = method
                        method_results.append(r)
        if method_results:
            results[method] = method_results

    return results


def load_grid_results():
    """Load grid search results."""
    grid_results = []
    for pattern in ['exp/grid_results/results_*.json', 'exp/grid_search/results_*.json']:
        for fpath in glob.glob(os.path.join(WORKSPACE, pattern)):
            with open(fpath) as f:
                grid_results.append(json.load(f))
    # Also include CGA-best seed 42 as a grid point
    fpath = os.path.join(WORKSPACE, 'exp/cga_best/results_seed42.json')
    if os.path.exists(fpath):
        with open(fpath) as f:
            grid_results.append(json.load(f))
    return grid_results


def aggregate_metrics(method_results):
    """Compute mean and std for all metrics across seeds."""
    metrics = {}
    keys = ['top1', 'top5', 'superclass_acc', 'within_superclass_acc',
            'between_superclass_error_rate', 'etf_deviation', 'hierarchy_corr',
            'mean_epoch_time_seconds', 'training_time_minutes']

    for key in keys:
        vals = [r[key] for r in method_results if key in r]
        if vals:
            metrics[key] = {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'values': vals,
                'n': len(vals)
            }
    return metrics


def statistical_tests(results):
    """Run pairwise t-tests between CGA-only and all baselines."""
    tests = {}
    cga_results = results.get('CGA-only', [])
    if not cga_results:
        return tests

    cga_top1 = [r['top1'] for r in cga_results]

    for method, method_results in results.items():
        if method == 'CGA-only':
            continue
        top1s = [r['top1'] for r in method_results]
        if len(top1s) >= 2 and len(cga_top1) >= 2:
            # Use Welch's t-test (unequal variances)
            t_stat, p_val = stats.ttest_ind(cga_top1, top1s, equal_var=False)
            tests[method] = {
                't_stat': t_stat,
                'p_value': p_val,
                'cga_mean': np.mean(cga_top1),
                'other_mean': np.mean(top1s),
                'delta': np.mean(cga_top1) - np.mean(top1s),
                'significant': p_val < 0.05
            }
    return tests


def generate_main_table(results, tests):
    """Generate Table 1: Main results comparison."""
    print("\n" + "="*100)
    print("TABLE 1: Main Results on CIFAR-100 (ResNet-18, 200 epochs)")
    print("="*100)

    order = ['CE', 'SupCon', 'HardNeg-CL', 'TCL', 'Reweight', 'VarCon-T',
             'AdaptTemp', 'CGA-only', 'CG-SupCon']

    header = f"{'Method':<15} {'Top-1 (%)':<15} {'Top-5 (%)':<15} {'SC Acc (%)':<15} {'W-SC (%)':<12} {'ETF Dev':<12} {'Hier Corr':<12} {'Seeds':>5}"
    print(header)
    print("-"*100)

    rows = []
    for method in order:
        if method not in results:
            continue
        m = aggregate_metrics(results[method])
        if 'top1' not in m:
            continue

        row = {
            'method': method,
            'top1': m['top1'],
            'top5': m.get('top5', {'mean': 0, 'std': 0}),
            'sc_acc': m.get('superclass_acc', {'mean': 0, 'std': 0}),
            'wsc': m.get('within_superclass_acc', {'mean': 0, 'std': 0}),
            'etf': m.get('etf_deviation', {'mean': 0, 'std': 0}),
            'hier': m.get('hierarchy_corr', {'mean': 0, 'std': 0}),
            'n': m['top1']['n']
        }
        rows.append(row)

        p_str = ''
        if method in tests:
            p = tests[method]['p_value']
            if p < 0.01:
                p_str = '**'
            elif p < 0.05:
                p_str = '*'

        print(f"{method:<15} "
              f"{row['top1']['mean']:>5.2f}±{row['top1']['std']:.2f}{p_str:<4} "
              f"{row['top5']['mean']:>5.2f}±{row['top5']['std']:.2f}   "
              f"{row['sc_acc']['mean']:>5.2f}±{row['sc_acc']['std']:.2f}   "
              f"{row['wsc']['mean']:>5.2f}±{row['wsc']['std']:.2f} "
              f"{row['etf']['mean']:>8.5f}   "
              f"{row['hier']['mean']:>8.4f}   "
              f"{row['n']:>5}")

    print("-"*100)
    if tests:
        print("* p<0.05, ** p<0.01 (Welch's t-test vs CGA-only)")

    return rows


def figure1_bar_comparison(results):
    """Figure 1: Bar chart comparing all methods on Top-1 accuracy."""
    order = ['CE', 'SupCon', 'HardNeg-CL', 'TCL', 'Reweight', 'VarCon-T',
             'AdaptTemp', 'CGA-only', 'CG-SupCon']

    methods = []
    means = []
    stds = []
    colors = []

    color_map = {
        'CE': '#95a5a6',
        'SupCon': '#3498db',
        'HardNeg-CL': '#e67e22',
        'TCL': '#e74c3c',
        'Reweight': '#9b59b6',
        'VarCon-T': '#1abc9c',
        'AdaptTemp': '#f39c12',
        'CGA-only': '#2ecc71',
        'CG-SupCon': '#27ae60',
    }

    for method in order:
        if method not in results:
            continue
        m = aggregate_metrics(results[method])
        if 'top1' not in m:
            continue
        methods.append(method)
        means.append(m['top1']['mean'])
        stds.append(m['top1']['std'])
        colors.append(color_map.get(method, '#333333'))

    if not methods:
        print("No results for bar chart")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor='black',
                  linewidth=0.5, alpha=0.85)

    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('CIFAR-100 Classification Accuracy (ResNet-18, 200 epochs)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.2,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=8)

    # Set y-axis to show differences clearly
    min_val = min(means) - max(stds) - 2
    max_val = max(means) + max(stds) + 2
    ax.set_ylim(min_val, max_val)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'figure1_main_results.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'figure1_main_results.png'))
    plt.close()
    print("Saved figure1_main_results.pdf/png")


def figure2_ablation(results):
    """Figure 2: Ablation bar chart (SupCon, +CGA, +AdaptTemp, +Both)."""
    variants = ['SupCon', 'CGA-only', 'AdaptTemp', 'CG-SupCon']
    labels = ['SupCon\n(baseline)', '+CGA only', '+Adapt. Temp\nonly', 'CG-SupCon\n(full)']

    means = []
    stds = []
    found = []

    for v in variants:
        if v in results:
            m = aggregate_metrics(results[v])
            if 'top1' in m:
                means.append(m['top1']['mean'])
                stds.append(m['top1']['std'])
                found.append(True)
            else:
                means.append(0)
                stds.append(0)
                found.append(False)
        else:
            means.append(0)
            stds.append(0)
            found.append(False)

    if sum(found) < 2:
        print("Not enough ablation results for figure2")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#3498db', '#2ecc71', '#f39c12', '#27ae60']
    x = np.arange(len(labels))

    for i in range(len(labels)):
        if found[i]:
            ax.bar(x[i], means[i], yerr=stds[i], capsize=5, color=colors[i],
                   edgecolor='black', linewidth=0.5, alpha=0.85)
            ax.text(x[i], means[i] + stds[i] + 0.15, f'{means[i]:.2f}',
                    ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Ablation Study: Component Contributions')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    min_val = min([m for m, f in zip(means, found) if f]) - 2
    max_val = max([m for m, f in zip(means, found) if f]) + 2
    ax.set_ylim(min_val, max_val)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'figure2_ablation.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'figure2_ablation.png'))
    plt.close()
    print("Saved figure2_ablation.pdf/png")


def figure3_sensitivity(grid_results):
    """Figure 3: Sensitivity heatmap for alpha and lambda."""
    if not grid_results:
        print("No grid search results for sensitivity figure")
        return

    # Collect (alpha, lambda, top1) triples
    points = []
    for r in grid_results:
        hp = r.get('hyperparameters', {})
        alpha = hp.get('alpha', None)
        lam = hp.get('lambda', None)
        top1 = r.get('top1', None)
        if alpha is not None and lam is not None and top1 is not None:
            points.append((alpha, lam, top1))

    if len(points) < 2:
        print("Not enough grid points for sensitivity figure")
        return

    # 1D sensitivity plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    alphas = sorted(set(p[0] for p in points))
    lambdas = sorted(set(p[1] for p in points))

    # Alpha sensitivity (for each lambda)
    for lam in lambdas:
        pts = [(a, t) for a, l, t in points if l == lam]
        if len(pts) >= 2:
            pts.sort()
            ax1.plot([p[0] for p in pts], [p[1] for p in pts],
                    'o-', label=f'λ={lam}')
    ax1.set_xlabel('α (CGA strength)')
    ax1.set_ylabel('Top-1 Accuracy (%)')
    ax1.set_title('Sensitivity to α')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Lambda sensitivity (for each alpha)
    for alpha in alphas:
        pts = [(l, t) for a, l, t in points if a == alpha]
        if len(pts) >= 2:
            pts.sort()
            ax2.plot([p[0] for p in pts], [p[1] for p in pts],
                    's-', label=f'α={alpha}')
    ax2.set_xlabel('λ (CGA weight)')
    ax2.set_ylabel('Top-1 Accuracy (%)')
    ax2.set_title('Sensitivity to λ')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'figure3_sensitivity.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'figure3_sensitivity.png'))
    plt.close()
    print("Saved figure3_sensitivity.pdf/png")


def figure4_superclass_analysis(results):
    """Figure 4: Within-superclass vs between-superclass improvement."""
    methods = ['SupCon', 'CGA-only']
    labels_method = []
    within_accs = []
    between_errs = []

    for method in methods:
        if method in results:
            m = aggregate_metrics(results[method])
            if 'within_superclass_acc' in m and 'between_superclass_error_rate' in m:
                labels_method.append(method)
                within_accs.append(m['within_superclass_acc'])
                between_errs.append(m['between_superclass_error_rate'])

    if len(labels_method) < 2:
        print("Not enough data for superclass analysis figure")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(labels_method))
    w_means = [m['mean'] for m in within_accs]
    w_stds = [m['std'] for m in within_accs]
    b_means = [m['mean'] for m in between_errs]
    b_stds = [m['std'] for m in between_errs]

    colors = ['#3498db', '#2ecc71']
    ax1.bar(x, w_means, yerr=w_stds, capsize=5, color=colors, edgecolor='black',
            linewidth=0.5, alpha=0.85)
    ax1.set_ylabel('Within-Superclass Accuracy (%)')
    ax1.set_title('Within-Superclass Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_method)
    ax1.set_ylim(min(w_means)-1, max(w_means)+1)
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x, b_means, yerr=b_stds, capsize=5, color=colors, edgecolor='black',
            linewidth=0.5, alpha=0.85)
    ax2.set_ylabel('Between-Superclass Error Rate (%)')
    ax2.set_title('Between-Superclass Error Rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_method)
    ax2.set_ylim(min(b_means)-1, max(b_means)+1)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'figure4_superclass.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'figure4_superclass.png'))
    plt.close()
    print("Saved figure4_superclass.pdf/png")


def figure5_overhead(results):
    """Figure 5: Training time overhead comparison."""
    order = ['CE', 'SupCon', 'HardNeg-CL', 'TCL', 'Reweight', 'VarCon-T',
             'CGA-only', 'CG-SupCon']

    methods = []
    times = []

    for method in order:
        if method in results:
            m = aggregate_metrics(results[method])
            if 'mean_epoch_time_seconds' in m:
                methods.append(method)
                times.append(m['mean_epoch_time_seconds']['mean'])

    if len(methods) < 2:
        print("Not enough timing data for overhead figure")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(methods))
    colors = ['#95a5a6' if m in ['CE', 'SupCon'] else '#e67e22' if m in ['HardNeg-CL', 'TCL', 'Reweight', 'VarCon-T'] else '#2ecc71' for m in methods]

    ax.bar(x, times, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    # Add overhead % relative to SupCon
    supcon_time = None
    for m, t in zip(methods, times):
        if m == 'SupCon':
            supcon_time = t
            break

    if supcon_time:
        for i, (m, t) in enumerate(zip(methods, times)):
            overhead = (t - supcon_time) / supcon_time * 100
            if m != 'SupCon':
                ax.text(i, t + 0.5, f'{overhead:+.0f}%', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Time per Epoch (seconds)')
    ax.set_title('Training Time per Epoch')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'figure5_overhead.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'figure5_overhead.png'))
    plt.close()
    print("Saved figure5_overhead.pdf/png")


def evaluate_success_criteria(results, tests):
    """Evaluate the 8 success criteria from idea.json."""
    print("\n" + "="*80)
    print("SUCCESS CRITERIA EVALUATION")
    print("="*80)

    criteria = []

    # 1. CGA > SupCon (p < 0.05)
    if 'SupCon' in tests:
        t = tests['SupCon']
        passed = t['delta'] > 0 and t['p_value'] < 0.05
        criteria.append({
            'id': 1,
            'description': 'CGA > SupCon (p < 0.05)',
            'passed': passed,
            'detail': f"delta={t['delta']:.2f}%, p={t['p_value']:.4f}"
        })
    else:
        criteria.append({'id': 1, 'description': 'CGA > SupCon (p < 0.05)',
                        'passed': False, 'detail': 'Insufficient data'})

    # 2. CGA > HardNeg-CL and TCL
    for baseline in ['HardNeg-CL', 'TCL']:
        if baseline in tests:
            t = tests[baseline]
            passed = t['delta'] > 0
            criteria.append({
                'id': 2,
                'description': f'CGA > {baseline}',
                'passed': passed,
                'detail': f"delta={t['delta']:.2f}%"
            })

    # 3. CGA > Reweight
    if 'Reweight' in tests:
        t = tests['Reweight']
        passed = t['delta'] > 0
        criteria.append({
            'id': 3,
            'description': 'CGA > Reweight',
            'passed': passed,
            'detail': f"delta={t['delta']:.2f}%"
        })

    # 4. CGA-only > SupCon (primary contribution)
    cga_results = results.get('CGA-only', [])
    sc_results = results.get('SupCon', [])
    if cga_results and sc_results:
        cga_mean = np.mean([r['top1'] for r in cga_results])
        sc_mean = np.mean([r['top1'] for r in sc_results])
        passed = cga_mean > sc_mean
        criteria.append({
            'id': 4,
            'description': 'CGA-only provides gain over SupCon',
            'passed': passed,
            'detail': f"CGA={cga_mean:.2f}% vs SupCon={sc_mean:.2f}%, delta={cga_mean-sc_mean:.2f}%"
        })

    # 5. Within-SC improvement > Between-SC improvement
    if cga_results and sc_results:
        cga_wsc = np.mean([r.get('within_superclass_acc', 0) for r in cga_results])
        sc_wsc = np.mean([r.get('within_superclass_acc', 0) for r in sc_results])
        cga_bsc = np.mean([r.get('between_superclass_error_rate', 0) for r in cga_results])
        sc_bsc = np.mean([r.get('between_superclass_error_rate', 0) for r in sc_results])
        wsc_delta = cga_wsc - sc_wsc
        bsc_delta = sc_bsc - cga_bsc  # lower error rate is better
        passed = wsc_delta > bsc_delta
        criteria.append({
            'id': 5,
            'description': 'Within-SC improvement > Between-SC improvement',
            'passed': passed,
            'detail': f"Within-SC delta={wsc_delta:.2f}%, Between-SC delta={bsc_delta:.2f}%"
        })

    # 6. Hierarchy correlation improvement > 0.1
    if cga_results and sc_results:
        cga_hier = np.mean([r.get('hierarchy_corr', 0) for r in cga_results])
        sc_hier = np.mean([r.get('hierarchy_corr', 0) for r in sc_results])
        delta = cga_hier - sc_hier
        passed = delta > 0.1
        criteria.append({
            'id': 6,
            'description': 'Hierarchy correlation improvement > 0.1',
            'passed': passed,
            'detail': f"CGA={cga_hier:.4f} vs SupCon={sc_hier:.4f}, delta={delta:.4f}"
        })

    # 8. Overhead < 10%
    if cga_results and sc_results:
        cga_time = np.mean([r.get('mean_epoch_time_seconds', 0) for r in cga_results])
        sc_time = np.mean([r.get('mean_epoch_time_seconds', 0) for r in sc_results])
        if sc_time > 0:
            overhead = (cga_time - sc_time) / sc_time * 100
            passed = overhead < 10
            criteria.append({
                'id': 8,
                'description': 'Wall-clock overhead < 10%',
                'passed': passed,
                'detail': f"Overhead={overhead:.1f}%"
            })

    for c in criteria:
        status = "PASS" if c['passed'] else "FAIL"
        print(f"  [{status}] Criterion {c['id']}: {c['description']}")
        print(f"         {c['detail']}")

    return criteria


def save_aggregated_results(results, tests, criteria):
    """Save comprehensive results.json at workspace root."""
    output = {
        'methods': {},
        'statistical_tests': {},
        'success_criteria': criteria,
    }

    for method, method_results in results.items():
        m = aggregate_metrics(method_results)
        output['methods'][method] = {
            'n_seeds': len(method_results),
            'seeds': [r['seed'] for r in method_results],
            'metrics': {k: {'mean': v['mean'], 'std': v['std']} for k, v in m.items()},
        }

    for method, test in tests.items():
        output['statistical_tests'][f'CGA-only_vs_{method}'] = {
            't_statistic': test['t_stat'],
            'p_value': test['p_value'],
            'cga_mean': test['cga_mean'],
            'other_mean': test['other_mean'],
            'delta': test['delta'],
            'significant_at_0.05': test['significant'],
        }

    out_path = os.path.join(WORKSPACE, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved aggregated results to {out_path}")


def main():
    print("Loading results...")
    results = load_all_results()
    grid_results = load_grid_results()

    print(f"Found results for {len(results)} methods:")
    for method, r in results.items():
        seeds = [x['seed'] for x in r]
        top1s = [x['top1'] for x in r]
        print(f"  {method}: {len(r)} seeds {seeds}, "
              f"top1={np.mean(top1s):.2f}±{np.std(top1s):.2f}")

    print(f"\nGrid search: {len(grid_results)} configs")

    # Statistical tests
    tests = statistical_tests(results)

    # Table 1
    rows = generate_main_table(results, tests)

    # Success criteria
    criteria = evaluate_success_criteria(results, tests)

    # Figures
    print("\nGenerating figures...")
    figure1_bar_comparison(results)
    figure2_ablation(results)
    figure3_sensitivity(grid_results)
    figure4_superclass_analysis(results)
    figure5_overhead(results)

    # Save aggregated results
    save_aggregated_results(results, tests, criteria)

    print("\nDone!")


if __name__ == '__main__':
    main()

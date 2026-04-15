#!/usr/bin/env python3
"""Final collection and analysis script.

Run this after all experiments complete to:
1. Collect all results
2. Run statistical tests
3. Evaluate success criteria
4. Generate all figures
5. Save aggregated results.json
"""

import json
import os
import sys
import glob
import numpy as np
from scipy import stats
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

WORKSPACE = '/home/zz865/pythonProject/autoresearch/outputs/claude/run_1/supervised_representation_learning/idea_01'
FIGURES_DIR = os.path.join(WORKSPACE, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})


def load_all():
    """Load all available results."""
    results = defaultdict(list)

    # Method -> (directory, display_name)
    search_configs = [
        ('exp/ce_results', 'CE'),
        ('exp/supcon_results', 'SupCon'),
        ('exp/hardneg', 'HardNeg-CL'),
        ('exp/tcl', 'TCL'),
        ('exp/reweight', 'Reweight'),
        ('exp/varcon_t', 'VarCon-T'),
        ('exp/cga_best', 'CGA-only'),
        ('exp/cga_main', 'CGA-only'),
        ('exp/cga_full', 'CG-SupCon'),
        ('exp/ablation_temp', 'AdaptTemp'),
    ]

    for dirpath, method_name in search_configs:
        full_dir = os.path.join(WORKSPACE, dirpath)
        if not os.path.exists(full_dir):
            continue
        for fpath in sorted(glob.glob(os.path.join(full_dir, 'results_seed*.json'))):
            try:
                with open(fpath) as f:
                    r = json.load(f)
                # Sanity check: valid result
                if 'top1' in r and r['top1'] > 20:
                    r['_source'] = fpath
                    r['_method'] = method_name
                    # Deduplicate by seed
                    existing_seeds = [x['seed'] for x in results[method_name]]
                    if r['seed'] not in existing_seeds:
                        results[method_name].append(r)
            except Exception as e:
                print(f"  Warning: Could not load {fpath}: {e}")

    return dict(results)


def agg(method_results, key):
    """Get mean and std of a metric across seeds."""
    vals = [r[key] for r in method_results if key in r]
    if not vals:
        return None
    return {'mean': np.mean(vals), 'std': np.std(vals), 'n': len(vals), 'values': vals}


def print_table(results):
    """Print main results table."""
    order = ['CE', 'SupCon', 'HardNeg-CL', 'TCL', 'Reweight', 'VarCon-T',
             'AdaptTemp', 'CGA-only', 'CG-SupCon']

    print("\n" + "="*110)
    print("TABLE 1: Main Results on CIFAR-100 (ResNet-18, 200 epochs)")
    print("="*110)
    print(f"{'Method':<15} {'Top-1 (%)':<16} {'Top-5 (%)':<16} {'SC Acc (%)':<14} "
          f"{'ETF Dev':<12} {'Hier Corr':<12} {'Time(s)':<10} {'N':>3}")
    print("-"*110)

    for method in order:
        if method not in results:
            continue
        r = results[method]
        t1 = agg(r, 'top1')
        t5 = agg(r, 'top5')
        sc = agg(r, 'superclass_acc')
        etf = agg(r, 'etf_deviation')
        hier = agg(r, 'hierarchy_corr')
        time_ep = agg(r, 'mean_epoch_time_seconds')

        if t1 is None:
            continue

        print(f"{method:<15} "
              f"{t1['mean']:>6.2f} ± {t1['std']:.2f}   "
              f"{t5['mean'] if t5 else 0:>6.2f} ± {t5['std'] if t5 else 0:.2f}   "
              f"{sc['mean'] if sc else 0:>6.2f} ± {sc['std'] if sc else 0:.2f} "
              f"{etf['mean'] if etf else 0:>10.5f} "
              f"{hier['mean'] if hier else 0:>10.4f} "
              f"{time_ep['mean'] if time_ep else 0:>8.1f} "
              f"{t1['n']:>3}")

    print("-"*110)


def run_tests(results):
    """Statistical tests: CGA-only vs each baseline."""
    tests = {}
    cga = results.get('CGA-only', [])
    if not cga:
        return tests

    cga_top1 = [r['top1'] for r in cga]

    for method in ['CE', 'SupCon', 'HardNeg-CL', 'TCL', 'Reweight', 'VarCon-T']:
        if method not in results:
            continue
        other_top1 = [r['top1'] for r in results[method]]
        if len(other_top1) < 2 or len(cga_top1) < 2:
            continue

        t_stat, p_val = stats.ttest_ind(cga_top1, other_top1, equal_var=False)
        tests[method] = {
            'cga_mean': np.mean(cga_top1),
            'other_mean': np.mean(other_top1),
            'delta': np.mean(cga_top1) - np.mean(other_top1),
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'significant': p_val < 0.05,
        }

    print("\n" + "="*80)
    print("STATISTICAL TESTS (Welch's t-test: CGA-only vs baseline)")
    print("="*80)
    for method, t in sorted(tests.items()):
        sig = "YES" if t['significant'] else "no"
        print(f"  vs {method:<15}: delta={t['delta']:+.2f}%, "
              f"t={t['t_stat']:.3f}, p={t['p_value']:.4f}, sig={sig}")

    return tests


def eval_criteria(results, tests):
    """Evaluate success criteria from idea.json."""
    print("\n" + "="*80)
    print("SUCCESS CRITERIA EVALUATION")
    print("="*80)

    cga = results.get('CGA-only', [])
    sc = results.get('SupCon', [])

    criteria = []

    # 1. CGA > SupCon (p<0.05)
    if 'SupCon' in tests:
        t = tests['SupCon']
        passed = t['delta'] > 0 and t['p_value'] < 0.05
        criteria.append({'id': 1, 'desc': 'CGA > SupCon (p<0.05)',
                        'passed': passed, 'detail': f"Δ={t['delta']:+.2f}%, p={t['p_value']:.4f}"})
    else:
        criteria.append({'id': 1, 'desc': 'CGA > SupCon (p<0.05)',
                        'passed': False, 'detail': 'Insufficient data'})

    # 2. CGA > instance-level baselines
    for bl in ['HardNeg-CL', 'TCL']:
        if bl in tests:
            t = tests[bl]
            criteria.append({'id': 2, 'desc': f'CGA > {bl}',
                            'passed': t['delta'] > 0, 'detail': f"Δ={t['delta']:+.2f}%"})

    # 3. CGA > Reweight
    if 'Reweight' in tests:
        t = tests['Reweight']
        criteria.append({'id': 3, 'desc': 'CGA > Reweight',
                        'passed': t['delta'] > 0, 'detail': f"Δ={t['delta']:+.2f}%"})

    # 4. CGA-only > SupCon (primary contribution)
    if cga and sc:
        cga_mean = np.mean([r['top1'] for r in cga])
        sc_mean = np.mean([r['top1'] for r in sc])
        criteria.append({'id': 4, 'desc': 'CGA-only provides gain over SupCon',
                        'passed': cga_mean > sc_mean,
                        'detail': f"CGA={cga_mean:.2f}% vs SC={sc_mean:.2f}%"})

    # 5. Within-SC improvement > Between-SC improvement
    if cga and sc:
        cga_wsc = np.mean([r.get('within_superclass_acc', 0) for r in cga])
        sc_wsc = np.mean([r.get('within_superclass_acc', 0) for r in sc])
        wsc_delta = cga_wsc - sc_wsc
        cga_bsc = np.mean([r.get('between_superclass_error_rate', 100) for r in cga])
        sc_bsc = np.mean([r.get('between_superclass_error_rate', 100) for r in sc])
        bsc_delta = sc_bsc - cga_bsc  # positive = improvement
        criteria.append({'id': 5, 'desc': 'Within-SC impr > Between-SC impr',
                        'passed': wsc_delta > bsc_delta,
                        'detail': f"W-SC Δ={wsc_delta:.2f}%, B-SC Δ={bsc_delta:.2f}%"})

    # 6. Hierarchy correlation improvement > 0.1
    if cga and sc:
        cga_h = np.mean([r.get('hierarchy_corr', 0) for r in cga])
        sc_h = np.mean([r.get('hierarchy_corr', 0) for r in sc])
        delta = cga_h - sc_h
        criteria.append({'id': 6, 'desc': 'Hierarchy corr improvement > 0.1',
                        'passed': delta > 0.1,
                        'detail': f"CGA={cga_h:.4f}, SC={sc_h:.4f}, Δ={delta:.4f}"})

    # 8. Overhead < 10%
    if cga and sc:
        cga_t = np.mean([r.get('mean_epoch_time_seconds', 0) for r in cga])
        sc_t = np.mean([r.get('mean_epoch_time_seconds', 0) for r in sc])
        if sc_t > 0:
            ovhd = (cga_t - sc_t) / sc_t * 100
            criteria.append({'id': 8, 'desc': 'Overhead < 10%',
                            'passed': ovhd < 10,
                            'detail': f"CGA={cga_t:.1f}s, SC={sc_t:.1f}s, overhead={ovhd:.1f}%"})

    for c in criteria:
        status = "PASS" if c['passed'] else "FAIL"
        print(f"  [{status}] Criterion {c['id']}: {c['desc']}")
        print(f"         {c['detail']}")

    return criteria


def make_figures(results, tests):
    """Generate all paper figures."""
    # Figure 1: Main comparison bar chart
    order = ['CE', 'SupCon', 'HardNeg-CL', 'TCL', 'Reweight', 'VarCon-T',
             'AdaptTemp', 'CGA-only', 'CG-SupCon']
    color_map = {
        'CE': '#95a5a6', 'SupCon': '#3498db', 'HardNeg-CL': '#e67e22',
        'TCL': '#e74c3c', 'Reweight': '#9b59b6', 'VarCon-T': '#1abc9c',
        'AdaptTemp': '#f39c12', 'CGA-only': '#2ecc71', 'CG-SupCon': '#27ae60',
    }

    methods_found = [m for m in order if m in results and agg(results[m], 'top1')]
    means = [agg(results[m], 'top1')['mean'] for m in methods_found]
    stds = [agg(results[m], 'top1')['std'] for m in methods_found]
    colors = [color_map.get(m, '#333') for m in methods_found]

    if len(methods_found) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(methods_found))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                      edgecolor='black', linewidth=0.5, alpha=0.85)
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.15,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=8)
        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.set_title('CIFAR-100 Classification (ResNet-18, 200 epochs)')
        ax.set_xticks(x)
        ax.set_xticklabels(methods_found, rotation=30, ha='right')
        ymin = min(means) - max(stds) - 2
        ymax = max(means) + max(stds) + 2
        ax.set_ylim(ymin, ymax)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'figure1_main_results.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, 'figure1_main_results.png'))
        plt.close()
        print("Saved figure1_main_results")

    # Figure 2: Ablation
    abl_methods = ['SupCon', 'CGA-only', 'AdaptTemp', 'CG-SupCon']
    abl_labels = ['SupCon\n(base)', '+CGA\nonly', '+Adapt\nTemp', 'Full\nCG-SupCon']
    abl_colors = ['#3498db', '#2ecc71', '#f39c12', '#27ae60']
    abl_found = [(m, l, c) for m, l, c in zip(abl_methods, abl_labels, abl_colors)
                 if m in results and agg(results[m], 'top1')]

    if len(abl_found) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(abl_found))
        for i, (m, l, c) in enumerate(abl_found):
            a = agg(results[m], 'top1')
            ax.bar(i, a['mean'], yerr=a['std'], capsize=5, color=c,
                   edgecolor='black', linewidth=0.5, alpha=0.85)
            ax.text(i, a['mean'] + a['std'] + 0.15, f"{a['mean']:.2f}",
                    ha='center', va='bottom', fontsize=9)
        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.set_title('Ablation Study')
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l, _ in abl_found])
        vals = [agg(results[m], 'top1')['mean'] for m, _, _ in abl_found]
        ax.set_ylim(min(vals) - 2, max(vals) + 2)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'figure2_ablation.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, 'figure2_ablation.png'))
        plt.close()
        print("Saved figure2_ablation")

    # Figure 3: Grid search sensitivity
    grid_results = []
    for pattern in ['exp/grid_results/results_*.json', 'exp/grid_search/results_*.json']:
        for fpath in glob.glob(os.path.join(WORKSPACE, pattern)):
            try:
                with open(fpath) as f:
                    r = json.load(f)
                    hp = r.get('hyperparameters', {})
                    if 'alpha' in hp and 'lambda' in hp and 'top1' in r:
                        grid_results.append((hp['alpha'], hp['lambda'], r['top1']))
            except:
                pass
    # Add CGA-best seed 42
    cga_best_path = os.path.join(WORKSPACE, 'exp/cga_best/results_seed42.json')
    if os.path.exists(cga_best_path):
        with open(cga_best_path) as f:
            r = json.load(f)
            hp = r.get('hyperparameters', {})
            if 'alpha' in hp and 'lambda' in hp:
                grid_results.append((hp['alpha'], hp['lambda'], r['top1']))

    if len(grid_results) >= 3:
        fig, ax = plt.subplots(figsize=(8, 5))
        alphas = [g[0] for g in grid_results]
        lambdas = [g[1] for g in grid_results]
        top1s = [g[2] for g in grid_results]
        scatter = ax.scatter(alphas, lambdas, c=top1s, cmap='viridis', s=200,
                           edgecolors='black', linewidths=1)
        for a, l, t in grid_results:
            ax.annotate(f'{t:.1f}', (a, l), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=8)
        plt.colorbar(scatter, label='Top-1 Accuracy (%)')
        ax.set_xlabel('α (CGA strength)')
        ax.set_ylabel('λ (CGA weight)')
        ax.set_title('Hyperparameter Sensitivity')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'figure3_sensitivity.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, 'figure3_sensitivity.png'))
        plt.close()
        print("Saved figure3_sensitivity")

    # Figure 4: Per-method error bars with significance markers
    if len(methods_found) >= 3 and 'CGA-only' in results:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(methods_found))
        for i, m in enumerate(methods_found):
            a = agg(results[m], 'top1')
            color = color_map.get(m, '#333')
            ax.errorbar(i, a['mean'], yerr=a['std'], fmt='o', color=color,
                       markersize=8, capsize=6, capthick=2, linewidth=2)
            # Mark significance
            if m in tests:
                t = tests[m]
                if t['p_value'] < 0.01:
                    ax.annotate('**', (i, a['mean'] + a['std'] + 0.3),
                              ha='center', fontsize=12, color='red')
                elif t['p_value'] < 0.05:
                    ax.annotate('*', (i, a['mean'] + a['std'] + 0.3),
                              ha='center', fontsize=12, color='red')

        # Add horizontal line for SupCon mean
        if 'SupCon' in results:
            sc_mean = agg(results['SupCon'], 'top1')['mean']
            ax.axhline(y=sc_mean, color='#3498db', linestyle='--', alpha=0.5,
                      label=f'SupCon ({sc_mean:.1f}%)')
            ax.legend()

        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.set_title('Method Comparison with Error Bars')
        ax.set_xticks(x)
        ax.set_xticklabels(methods_found, rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'figure4_comparison.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, 'figure4_comparison.png'))
        plt.close()
        print("Saved figure4_comparison")

    # Figure 5: Training overhead
    overhead_methods = [m for m in methods_found
                       if m in results and agg(results[m], 'mean_epoch_time_seconds')]
    if len(overhead_methods) >= 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        times = [agg(results[m], 'mean_epoch_time_seconds')['mean'] for m in overhead_methods]
        colors = [color_map.get(m, '#333') for m in overhead_methods]
        x = np.arange(len(overhead_methods))
        ax.bar(x, times, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

        sc_time = agg(results.get('SupCon', [{}]), 'mean_epoch_time_seconds')
        if sc_time:
            for i, (m, t) in enumerate(zip(overhead_methods, times)):
                if m != 'SupCon':
                    ovhd = (t - sc_time['mean']) / sc_time['mean'] * 100
                    ax.text(i, t + 0.5, f'{ovhd:+.0f}%', ha='center', fontsize=8)

        ax.set_ylabel('Time per Epoch (seconds)')
        ax.set_title('Training Time Overhead')
        ax.set_xticks(x)
        ax.set_xticklabels(overhead_methods, rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'figure5_overhead.pdf'))
        plt.savefig(os.path.join(FIGURES_DIR, 'figure5_overhead.png'))
        plt.close()
        print("Saved figure5_overhead")


def save_results_json(results, tests, criteria):
    """Save comprehensive results.json."""
    output = {'methods': {}, 'statistical_tests': {}, 'success_criteria': criteria}

    for method, method_results in results.items():
        metrics = {}
        for key in ['top1', 'top5', 'superclass_acc', 'within_superclass_acc',
                     'between_superclass_error_rate', 'etf_deviation',
                     'hierarchy_corr', 'mean_epoch_time_seconds', 'training_time_minutes']:
            a = agg(method_results, key)
            if a:
                metrics[key] = {'mean': round(a['mean'], 4), 'std': round(a['std'], 4)}

        output['methods'][method] = {
            'n_seeds': len(method_results),
            'seeds': [r['seed'] for r in method_results],
            'metrics': metrics,
        }

    for method, t in tests.items():
        output['statistical_tests'][f'CGA-only_vs_{method}'] = {
            'delta': round(t['delta'], 4),
            'p_value': round(t['p_value'], 6),
            'significant': t['significant'],
        }

    out_path = os.path.join(WORKSPACE, 'results.json')

    # Custom encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved results.json to {out_path}")
    return output


def main():
    print("=" * 60)
    print("FINAL RESULTS COLLECTION AND ANALYSIS")
    print("=" * 60)

    results = load_all()

    print(f"\nLoaded {len(results)} methods:")
    for method, r in sorted(results.items()):
        seeds = sorted([x['seed'] for x in r])
        t1 = agg(r, 'top1')
        print(f"  {method}: {len(r)} seeds {seeds} → "
              f"top1={t1['mean']:.2f}±{t1['std']:.2f}" if t1 else f"  {method}: no valid results")

    print_table(results)
    tests = run_tests(results)
    criteria = eval_criteria(results, tests)
    make_figures(results, tests)
    output = save_results_json(results, tests, criteria)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Figures saved to {FIGURES_DIR}")
    print(f"Results saved to {os.path.join(WORKSPACE, 'results.json')}")
    print("="*60)


if __name__ == '__main__':
    main()

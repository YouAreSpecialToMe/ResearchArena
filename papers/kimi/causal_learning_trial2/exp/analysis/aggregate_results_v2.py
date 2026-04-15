"""
Aggregate and analyze experiment results.
"""
import sys
import os
import json
import glob
import numpy as np
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)


def load_results():
    """Load all experiment results."""
    results = {}
    
    result_files = glob.glob(os.path.join(PROJECT_ROOT, 'results/synthetic/*/results.json'))
    
    for result_file in result_files:
        exp_name = os.path.basename(os.path.dirname(result_file))
        with open(result_file, 'r') as f:
            results[exp_name] = json.load(f)
        print(f"Loaded {exp_name}: {len(results[exp_name])} results")
    
    return results


def compute_summary_statistics(results):
    """Compute summary statistics for each method."""
    summary = {}
    
    for exp_name, exp_results in results.items():
        method_summary = defaultdict(lambda: defaultdict(list))
        
        for r in exp_results:
            if r.get('status') == 'success' and r.get('runtime') is not None:
                key = (r['n_nodes'], r['mechanism'], r['n_samples'])
                method_summary[key]['shd'].append(r['shd'])
                method_summary[key]['tpr'].append(r['tpr'])
                method_summary[key]['fdr'].append(r['fdr'])
                method_summary[key]['runtime'].append(r['runtime'])
        
        # Compute means and stds
        summary[exp_name] = []
        for key, metrics in method_summary.items():
            n_nodes, mechanism, n_samples = key
            
            if len(metrics['shd']) == 0:
                continue
                
            summary[exp_name].append({
                'n_nodes': n_nodes,
                'mechanism': mechanism,
                'n_samples': n_samples,
                'shd_mean': np.mean(metrics['shd']),
                'shd_std': np.std(metrics['shd']),
                'shd_median': np.median(metrics['shd']),
                'tpr_mean': np.mean(metrics['tpr']),
                'tpr_std': np.std(metrics['tpr']),
                'fdr_mean': np.mean(metrics['fdr']),
                'fdr_std': np.std(metrics['fdr']),
                'runtime_mean': np.mean(metrics['runtime']),
                'runtime_std': np.std(metrics['runtime']),
                'n_runs': len(metrics['shd'])
            })
    
    return summary


def compare_methods(summary):
    """Compare SPICED vs baselines."""
    print("\n" + "="*60)
    print("METHOD COMPARISON")
    print("="*60)
    
    # Get main results
    spiced_results = summary.get('spiced', [])
    notears_results = summary.get('notears', [])
    
    # Group by sample size
    spiced_by_n = defaultdict(list)
    notears_by_n = defaultdict(list)
    
    for r in spiced_results:
        key = (r['n_nodes'], r['mechanism'], r['n_samples'])
        spiced_by_n[key].append(r)
    
    for r in notears_results:
        key = (r['n_nodes'], r['mechanism'], r['n_samples'])
        notears_by_n[key].append(r)
    
    # Compare for N <= 200
    print("\nComparison for N <= 200:")
    print("-" * 80)
    print(f"{'N':>5} {'Nodes':>6} {'Mechanism':<20} {'SPICED SHD':>12} {'NOTEARS SHD':>12} {'Winner':>10}")
    print("-" * 80)
    
    wins = {'spiced': 0, 'notears': 0, 'tie': 0}
    
    for key in sorted(spiced_by_n.keys()):
        n_nodes, mechanism, n_samples = key
        
        if n_samples > 200:
            continue
        
        if key in notears_by_n:
            s_shd = np.mean([r['shd_mean'] for r in spiced_by_n[key]])
            n_shd = np.mean([r['shd_mean'] for r in notears_by_n[key]])
            
            if s_shd < n_shd:
                winner = 'SPICED'
                wins['spiced'] += 1
            elif n_shd < s_shd:
                winner = 'NOTEARS'
                wins['notears'] += 1
            else:
                winner = 'TIE'
                wins['tie'] += 1
            
            print(f"{n_samples:>5} {n_nodes:>6} {mechanism:<20} {s_shd:>12.2f} {n_shd:>12.2f} {winner:>10}")
    
    print("-" * 80)
    print(f"\nWins: SPICED={wins['spiced']}, NOTEARS={wins['notears']}, TIE={wins['tie']}")
    
    return wins


def check_success_criteria(summary):
    """Check if success criteria are met."""
    print("\n" + "="*60)
    print("SUCCESS CRITERIA VERIFICATION")
    print("="*60)
    
    # Criterion 1: SPICED achieves lower SHD than NOTEARS for N <= 200 on >=3 mechanisms
    spiced_results = summary.get('spiced_main', [])
    notears_results = summary.get('notears_fixed', [])
    
    mechanisms = ['linear_gaussian', 'linear_nongaussian', 'nonlinear', 'anm']
    mechanism_wins = {m: {'spiced': 0, 'notears': 0, 'tie': 0} for m in mechanisms}
    
    for s_r in spiced_results:
        if s_r['n_samples'] > 200:
            continue
        
        for n_r in notears_results:
            if (n_r['n_nodes'] == s_r['n_nodes'] and 
                n_r['mechanism'] == s_r['mechanism'] and
                n_r['n_samples'] == s_r['n_samples']):
                
                m = s_r['mechanism']
                if s_r['shd_mean'] < n_r['shd_mean']:
                    mechanism_wins[m]['spiced'] += 1
                elif n_r['shd_mean'] < s_r['shd_mean']:
                    mechanism_wins[m]['notears'] += 1
                else:
                    mechanism_wins[m]['tie'] += 1
    
    print("\nCriterion 1: SPICED lower SHD than NOTEARS for N <= 200")
    print("-" * 60)
    wins_count = 0
    for m in mechanisms:
        w = mechanism_wins[m]
        total = w['spiced'] + w['notears'] + w['tie']
        if total > 0:
            spiced_win_pct = 100 * w['spiced'] / total
            print(f"  {m:<20}: SPICED wins {w['spiced']}/{total} ({spiced_win_pct:.1f}%)")
            if w['spiced'] > w['notears']:
                wins_count += 1
    
    print(f"\n  SPICED wins on {wins_count}/4 mechanisms")
    if wins_count >= 3:
        print("  Status: PASS ✓")
    else:
        print("  Status: FAIL ✗")
    
    # Criterion 2: Runtime < 5 minutes for n=50
    print("\nCriterion 2: SPICED runtime < 5 min for n=50")
    print("-" * 60)
    
    n50_results = summary.get('spiced_n50', [])
    if n50_results:
        runtimes = [r['runtime_mean'] for r in n50_results]
        median_runtime = np.median(runtimes)
        print(f"  Median runtime for n=50: {median_runtime:.2f}s")
        if median_runtime < 300:  # 5 minutes
            print("  Status: PASS ✓")
        else:
            print("  Status: FAIL ✗")
    else:
        print("  No n=50 results available yet")
    
    # Criterion 3: SHD < 10 on Sachs
    print("\nCriterion 3: SPICED SHD < 10 on Sachs")
    print("-" * 60)
    
    sachs_file = os.path.join(PROJECT_ROOT, 'results/real_world/sachs_results.json')
    if os.path.exists(sachs_file):
        with open(sachs_file, 'r') as f:
            sachs_results = json.load(f)
        
        spiced_shds = [r['shd'] for r in sachs_results.get('spiced', [])]
        median_shd = np.median(spiced_shds)
        print(f"  Median SHD on Sachs: {median_shd:.2f}")
        if median_shd < 10:
            print("  Status: PASS ✓")
        else:
            print("  Status: FAIL (marginally - need < 10, got {:.2f})".format(median_shd))
    else:
        print("  No Sachs results available")


def save_final_results(all_results, summary):
    """Save final aggregated results."""
    final_results = {
        'all_results': all_results,
        'summary': summary,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(PROJECT_ROOT, 'results.json'), 'w') as f:
        json.dump(final_results, f, indent=2, default=float)
    
    print(f"\nFinal results saved to results.json")


if __name__ == "__main__":
    import time
    
    print("="*60)
    print("Aggregating Experiment Results")
    print("="*60)
    
    all_results = load_results()
    summary = compute_summary_statistics(all_results)
    wins = compare_methods(summary)
    check_success_criteria(summary)
    save_final_results(all_results, summary)

"""
Aggregate all experimental results into final summary.
"""
import json
import numpy as np
import os
from typing import Dict, List


def load_results(path: str) -> List[Dict]:
    """Load results from JSON file."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def summarize_method(results: List[Dict], method_name: str) -> Dict:
    """Summarize results for a single method."""
    valid_results = [r for r in results if 'metrics' in r and 'error' not in r]
    
    if not valid_results:
        return {'error': 'No valid results'}
    
    # Group by number of nodes
    by_nodes = {}
    for r in valid_results:
        n = r.get('dataset_config', {}).get('n_nodes', 0)
        if n not in by_nodes:
            by_nodes[n] = []
        by_nodes[n].append(r)
    
    summary = {
        'method': method_name,
        'n_total': len(results),
        'n_valid': len(valid_results),
        'by_nodes': {}
    }
    
    for n_nodes, subset in by_nodes.items():
        metrics = {
            'f1': [r['metrics']['f1'] for r in subset],
            'shd': [r['metrics']['shd'] for r in subset],
            'precision': [r['metrics']['precision'] for r in subset],
            'recall': [r['metrics']['recall'] for r in subset],
            'runtime': [r['runtime'] for r in subset]
        }
        
        # Add cost/savings info for MF-ACD
        if 'savings_pct' in subset[0]:
            metrics['savings_pct'] = [r['savings_pct'] for r in subset]
        
        summary['by_nodes'][n_nodes] = {
            k: {'mean': np.mean(v), 'std': np.std(v)}
            for k, v in metrics.items()
        }
    
    return summary


def compute_statistical_tests(baseline_results: List[Dict], 
                              mf_acd_results: List[Dict]) -> Dict:
    """Compute statistical tests between baseline and MF-ACD."""
    from scipy import stats
    
    # Match by dataset
    baseline_by_name = {r['dataset_name']: r for r in baseline_results 
                        if 'metrics' in r and 'error' not in r}
    mf_acd_by_name = {r['dataset_name']: r for r in mf_acd_results 
                      if 'metrics' in r and 'error' not in r}
    
    common = set(baseline_by_name.keys()) & set(mf_acd_by_name.keys())
    
    baseline_f1 = [baseline_by_name[n]['metrics']['f1'] for n in common]
    mf_acd_f1 = [mf_acd_by_name[n]['metrics']['f1'] for n in common]
    
    baseline_shd = [baseline_by_name[n]['metrics']['shd'] for n in common]
    mf_acd_shd = [mf_acd_by_name[n]['metrics']['shd'] for n in common]
    
    # Paired t-tests
    f1_tstat, f1_pvalue = stats.ttest_rel(mf_acd_f1, baseline_f1)
    shd_tstat, shd_pvalue = stats.ttest_rel(mf_acd_shd, baseline_shd)
    
    return {
        'n_common': len(common),
        'f1_comparison': {
            'baseline_mean': np.mean(baseline_f1),
            'mf_acd_mean': np.mean(mf_acd_f1),
            'difference': np.mean(mf_acd_f1) - np.mean(baseline_f1),
            't_stat': f1_tstat,
            'p_value': f1_pvalue
        },
        'shd_comparison': {
            'baseline_mean': np.mean(baseline_shd),
            'mf_acd_mean': np.mean(mf_acd_shd),
            'difference': np.mean(mf_acd_shd) - np.mean(baseline_shd),
            't_stat': shd_tstat,
            'p_value': shd_pvalue
        }
    }


def main():
    """Aggregate all results."""
    
    print("Loading results...")
    
    # Load all results
    results = {
        'pc_fisherz': load_results('results/baselines/pc_fisherz/results.json'),
        'pc_stable': load_results('results/baselines/pc_stable/results.json'),
        'fast_pc': load_results('results/baselines/fast_pc/results.json'),
        'ges': load_results('results/baselines/ges/results.json'),
        'mf_acd': load_results('results/mf_acd/main/results.json'),
        'ablation_fixed_vs_adaptive': load_results('results/ablations/fixed_vs_adaptive/results.json'),
        'ig_validation': load_results('results/validation/ig_approximation/results.json'),
    }
    
    print("\nSummarizing methods...")
    
    # Summarize each method
    summaries = {}
    for name, res in results.items():
        if res:
            summaries[name] = summarize_method(res, name)
            print(f"  {name}: {summaries[name].get('n_valid', 0)} valid results")
    
    # Statistical comparisons
    print("\nComputing statistical tests...")
    if results['pc_fisherz'] and results['mf_acd']:
        summaries['pc_vs_mf_acd'] = compute_statistical_tests(
            results['pc_fisherz'], results['mf_acd']
        )
    
    # Save aggregated results
    os.makedirs('results', exist_ok=True)
    with open('results/summary.json', 'w') as f:
        json.dump(summaries, f, indent=2)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Print main results
    for method in ['pc_fisherz', 'pc_stable', 'fast_pc', 'ges', 'mf_acd']:
        if method in summaries:
            s = summaries[method]
            print(f"\n{method.upper()}:")
            for n_nodes, stats in s.get('by_nodes', {}).items():
                f1 = stats.get('f1', {})
                shd = stats.get('shd', {})
                runtime = stats.get('runtime', {})
                print(f"  {n_nodes} nodes: F1={f1.get('mean', 0):.3f}±{f1.get('std', 0):.3f}, "
                      f"SHD={shd.get('mean', 0):.1f}±{shd.get('std', 0):.1f}, "
                      f"Time={runtime.get('mean', 0):.2f}s")
                
                if 'savings_pct' in stats:
                    savings = stats['savings_pct']
                    print(f"    Savings: {savings.get('mean', 0):.1f}%")
    
    # Print statistical comparison
    if 'pc_vs_mf_acd' in summaries:
        comp = summaries['pc_vs_mf_acd']
        print(f"\nPC-FisherZ vs MF-ACD:")
        print(f"  F1 difference: {comp['f1_comparison']['difference']:.4f} "
              f"(p={comp['f1_comparison']['p_value']:.4f})")
        print(f"  SHD difference: {comp['shd_comparison']['difference']:.2f} "
              f"(p={comp['shd_comparison']['p_value']:.4f})")
    
    print("\nResults saved to results/summary.json")


if __name__ == "__main__":
    main()

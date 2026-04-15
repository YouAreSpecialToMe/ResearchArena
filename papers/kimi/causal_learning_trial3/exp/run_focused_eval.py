"""
Focused evaluation runner - runs all experiments on a representative subset.
Prioritizes completing all experiment types over exhaustive coverage.
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from baselines.pc_fisherz.run import run_experiment as run_pc_fisherz
from baselines.pc_stable.run import run_experiment as run_pc_stable
from baselines.fast_pc.run import run_experiment as run_fast_pc
from baselines.ges.run import run_experiment as run_ges
from mf_acd.mf_acd import MFACD
from shared.metrics import compute_metrics
from shared.utils import load_dataset, save_results, Timer


def run_mf_acd_experiment(dataset_path, use_adaptive=True, budget_allocation=(0.34, 0.20, 0.46)):
    """Run MF-ACD on a dataset."""
    dataset = load_dataset(dataset_path)
    data = dataset['data']
    true_adj = dataset['adjacency']
    
    with Timer() as timer:
        mf_acd = MFACD(
            budget_allocation=budget_allocation,
            use_adaptive=use_adaptive,
            alpha1=0.10,
            alpha2=0.05,
            alpha3=0.01,
            cost_weights=(1.0, 1.1, 15.0)
        )
        result = mf_acd.fit(data)
    
    metrics = compute_metrics(true_adj, result['adjacency'])
    
    return {
        'metrics': metrics,
        'runtime': float(timer.elapsed),
        'phase_costs': result['phase_costs'],
        'n_tests': result['n_tests'],
        'total_cost': result['total_cost'],
        'baseline_cost': result['baseline_cost'],
        'savings_pct': result['savings_pct'],
        'config': {'use_adaptive': use_adaptive, 'budget_allocation': budget_allocation}
    }


def main():
    """Run focused evaluation."""
    print("="*70)
    print("FOCUSED EVALUATION - MF-ACD")
    print("="*70)
    
    # Load dataset manifest
    with open('data/synthetic/manifest.json') as f:
        manifest = json.load(f)
    
    # Select focused subset: 20, 50 nodes x 2 densities x 10 seeds = 40 configs each
    # Plus some 100-node for scalability
    focused = []
    for ds in manifest['datasets']:
        n = ds['config']['n_nodes']
        ep = ds['config']['edge_param']
        seed = ds['config']['seed']
        # Include: 20 nodes (all), 50 nodes (all), 100 nodes (first 30)
        if n == 20 or n == 50:
            focused.append(ds)
        elif n == 100 and len([d for d in focused if d['config']['n_nodes'] == 100]) < 30:
            focused.append(ds)
    
    print(f"Selected {len(focused)} datasets for focused evaluation")
    print(f"  - 20 nodes: {len([d for d in focused if d['config']['n_nodes'] == 20])}")
    print(f"  - 50 nodes: {len([d for d in focused if d['config']['n_nodes'] == 50])}")
    print(f"  - 100 nodes: {len([d for d in focused if d['config']['n_nodes'] == 100])}")
    
    # Create output directories
    os.makedirs('results/baselines/pc_fisherz', exist_ok=True)
    os.makedirs('results/baselines/pc_stable', exist_ok=True)
    os.makedirs('results/baselines/fast_pc', exist_ok=True)
    os.makedirs('results/baselines/ges', exist_ok=True)
    os.makedirs('results/mf_acd/main', exist_ok=True)
    os.makedirs('results/ablations/fixed_vs_adaptive', exist_ok=True)
    
    # Run all experiments
    results = {
        'pc_fisherz': [],
        'pc_stable': [],
        'fast_pc': [],
        'ges': [],
        'mf_acd_adaptive': [],
        'mf_acd_fixed': []
    }
    
    total_start = time.time()
    
    for i, ds in enumerate(focused):
        print(f"\n[{i+1}/{len(focused)}] Processing {ds['name']}...")
        dataset_path = ds['path']
        if not dataset_path.startswith('data/synthetic/'):
            dataset_path = os.path.join('data/synthetic', os.path.basename(dataset_path))
        
        try:
            # Baseline 1: PC-FisherZ
            r = run_pc_fisherz(dataset_path)
            r['dataset_name'] = ds['name']
            r['dataset_config'] = ds['config']
            results['pc_fisherz'].append(r)
            print(f"  PC-FisherZ: F1={r['metrics']['f1']:.3f}, Time={r['runtime']:.2f}s")
            
            # Baseline 2: PC-Stable
            r = run_pc_stable(dataset_path)
            r['dataset_name'] = ds['name']
            r['dataset_config'] = ds['config']
            results['pc_stable'].append(r)
            print(f"  PC-Stable:  F1={r['metrics']['f1']:.3f}, Time={r['runtime']:.2f}s")
            
            # Baseline 3: Fast-PC
            r = run_fast_pc(dataset_path)
            r['dataset_name'] = ds['name']
            r['dataset_config'] = ds['config']
            results['fast_pc'].append(r)
            print(f"  Fast-PC:    F1={r['metrics']['f1']:.3f}, Time={r['runtime']:.2f}s")
            
            # Baseline 4: GES
            r = run_ges(dataset_path)
            r['dataset_name'] = ds['name']
            r['dataset_config'] = ds['config']
            results['ges'].append(r)
            print(f"  GES:        F1={r['metrics']['f1']:.3f}, Time={r['runtime']:.2f}s")
            
            # MF-ACD Adaptive
            r = run_mf_acd_experiment(dataset_path, use_adaptive=True)
            r['dataset_name'] = ds['name']
            r['dataset_config'] = ds['config']
            results['mf_acd_adaptive'].append(r)
            print(f"  MF-ACD (A): F1={r['metrics']['f1']:.3f}, Savings={r['savings_pct']:.1f}%, Time={r['runtime']:.2f}s")
            
            # MF-ACD Fixed (for ablation)
            r = run_mf_acd_experiment(dataset_path, use_adaptive=False)
            r['dataset_name'] = ds['name']
            r['dataset_config'] = ds['config']
            results['mf_acd_fixed'].append(r)
            print(f"  MF-ACD (F): F1={r['metrics']['f1']:.3f}, Savings={r['savings_pct']:.1f}%, Time={r['runtime']:.2f}s")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Save individual results
    for name, res in results.items():
        if 'mf_acd' in name:
            if 'fixed' in name:
                save_results(res, 'results/ablations/fixed_vs_adaptive/fixed_results.json')
            else:
                save_results(res, 'results/mf_acd/main/results.json')
        else:
            save_results(res, f'results/baselines/{name}/results.json')
    
    # Statistical analysis
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    # Compare MF-ACD vs baselines
    for n_nodes in [20, 50, 100]:
        subset = [r for r in results['mf_acd_adaptive'] 
                  if r.get('dataset_config', {}).get('n_nodes') == n_nodes]
        pc_subset = [r for r in results['pc_fisherz']
                     if r.get('dataset_config', {}).get('n_nodes') == n_nodes]
        
        if subset and pc_subset:
            mf_f1 = [r['metrics']['f1'] for r in subset]
            pc_f1 = [r['metrics']['f1'] for r in pc_subset]
            mf_time = [r['runtime'] for r in subset]
            pc_time = [r['runtime'] for r in pc_subset]
            savings = [r['savings_pct'] for r in subset]
            
            # Paired t-test for F1
            t_stat, p_val = stats.ttest_rel(mf_f1, pc_f1)
            
            print(f"\n{n_nodes} nodes:")
            print(f"  MF-ACD:     F1={np.mean(mf_f1):.3f}±{np.std(mf_f1):.3f}, Time={np.mean(mf_time):.2f}s")
            print(f"  PC-FisherZ: F1={np.mean(pc_f1):.3f}±{np.std(pc_f1):.3f}, Time={np.mean(pc_time):.2f}s")
            print(f"  Savings:    {np.mean(savings):.1f}%±{np.std(savings):.1f}%")
            print(f"  t-test:     t={t_stat:.3f}, p={p_val:.4f}")
    
    # Fixed vs Adaptive ablation
    print("\n" + "-"*70)
    print("ABLATION: Fixed vs Adaptive Budget Allocation")
    print("-"*70)
    
    adaptive_50 = [r for r in results['mf_acd_adaptive'] 
                   if r.get('dataset_config', {}).get('n_nodes') == 50]
    fixed_50 = [r for r in results['mf_acd_fixed']
                if r.get('dataset_config', {}).get('n_nodes') == 50]
    
    if adaptive_50 and fixed_50:
        adaptive_f1 = [r['metrics']['f1'] for r in adaptive_50]
        fixed_f1 = [r['metrics']['f1'] for r in fixed_50]
        adaptive_savings = [r['savings_pct'] for r in adaptive_50]
        fixed_savings = [r['savings_pct'] for r in fixed_50]
        
        t_stat, p_val = stats.ttest_rel(adaptive_f1, fixed_f1)
        
        print(f"  Adaptive: F1={np.mean(adaptive_f1):.3f}±{np.std(adaptive_f1):.3f}, "
              f"Savings={np.mean(adaptive_savings):.1f}%")
        print(f"  Fixed:    F1={np.mean(fixed_f1):.3f}±{np.std(fixed_f1):.3f}, "
              f"Savings={np.mean(fixed_savings):.1f}%")
        print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"  Significant: {'Yes' if p_val < 0.05 else 'No'}")
    
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    
    # Save summary
    summary = {
        'n_datasets': len(focused),
        'total_runtime_minutes': total_time / 60,
        'methods': list(results.keys()),
        'ablation_p_value': float(p_val) if adaptive_50 and fixed_50 else None
    }
    save_results(summary, 'results/focused_eval_summary.json')
    
    print("\nResults saved to results/ directory")


if __name__ == "__main__":
    main()

"""
Quick validation of tuned MF-ACD parameters.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from baselines.pc_fisherz.run import run_experiment as run_pc_fisherz
from mf_acd.mf_acd import MFACD
from shared.metrics import compute_metrics
from shared.utils import load_dataset, Timer


def run_mf_acd_tuned(dataset_path, config_name, **kwargs):
    """Run MF-ACD with tuned parameters."""
    dataset = load_dataset(dataset_path)
    data = dataset['data']
    true_adj = dataset['adjacency']
    
    with Timer() as timer:
        mf_acd = MFACD(**kwargs)
        result = mf_acd.fit(data)
    
    metrics = compute_metrics(true_adj, result['adjacency'])
    
    return {
        'metrics': metrics,
        'runtime': float(timer.elapsed),
        'savings_pct': result['savings_pct'],
        'config': config_name
    }


def main():
    # Load a representative subset
    with open('data/synthetic/manifest.json') as f:
        manifest = json.load(f)
    
    # Select 6 representative datasets: 2 graph types x 3 sizes
    test_datasets = []
    for ds in manifest['datasets']:
        n = ds['config']['n_nodes']
        ep = ds['config']['edge_param']
        seed = ds['config']['seed']
        graph_type = ds['config']['graph_type']
        
        # Pick one sparse (e=0.1) and one dense (e=0.3) for each size
        if seed == 42 and ((n == 20 and ep == 0.1) or (n == 20 and ep == 0.3) or 
                           (n == 50 and ep == 0.1) or (n == 50 and ep == 0.2)):
            test_datasets.append(ds)
        
        if len(test_datasets) >= 6:
            break
    
    print("="*70)
    print("PARAMETER TUNING VALIDATION")
    print("="*70)
    print(f"Testing {len(test_datasets)} datasets")
    
    # Configurations to test
    configs = [
        # Original config
        ("Original", {
            'budget_allocation': (0.34, 0.20, 0.46),
            'alpha1': 0.10, 'alpha2': 0.05, 'alpha3': 0.01,
            'cost_weights': (1.0, 1.1, 15.0),
            'use_adaptive': True
        }),
        # Tuned: more budget to medium fidelity, less conservative thresholds
        ("Tuned-v1", {
            'budget_allocation': (0.25, 0.35, 0.40),
            'alpha1': 0.15, 'alpha2': 0.08, 'alpha3': 0.02,
            'cost_weights': (1.0, 2.0, 10.0),
            'use_adaptive': True
        }),
        # Tuned: even more aggressive medium fidelity, higher alphas
        ("Tuned-v2", {
            'budget_allocation': (0.20, 0.40, 0.40),
            'alpha1': 0.20, 'alpha2': 0.10, 'alpha3': 0.03,
            'cost_weights': (1.0, 3.0, 8.0),
            'use_adaptive': True
        }),
    ]
    
    results = {name: [] for name, _ in configs}
    results['pc_fisherz'] = []
    
    for ds in test_datasets:
        print(f"\n{ds['name']} ({ds['config']['n_nodes']} nodes, {ds['config']['edge_param']} density)")
        
        # PC-FisherZ baseline
        r = run_pc_fisherz(ds['path'])
        results['pc_fisherz'].append(r)
        pc_f1 = r['metrics']['f1']
        print(f"  PC-FisherZ: F1={pc_f1:.3f}")
        
        # Test each config
        for name, params in configs:
            r = run_mf_acd_tuned(ds['path'], name, **params)
            results[name].append(r)
            f1 = r['metrics']['f1']
            savings = r['savings_pct']
            drop = (pc_f1 - f1) / pc_f1 * 100
            print(f"  {name:12s}: F1={f1:.3f} (drop: {drop:5.1f}%), Savings={savings:.1f}%")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    pc_f1s = [r['metrics']['f1'] for r in results['pc_fisherz']]
    print(f"PC-FisherZ:  F1={np.mean(pc_f1s):.3f}±{np.std(pc_f1s):.3f}")
    
    for name, _ in configs:
        f1s = [r['metrics']['f1'] for r in results[name]]
        savings = [r['savings_pct'] for r in results[name]]
        avg_drop = (np.mean(pc_f1s) - np.mean(f1s)) / np.mean(pc_f1s) * 100
        print(f"{name:12s}: F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}, "
              f"Drop={avg_drop:.1f}%, Savings={np.mean(savings):.1f}%")
    
    # Save results
    os.makedirs('results/validation', exist_ok=True)
    with open('results/validation/tuning_results.json', 'w') as f:
        json.dump({k: [{
            'f1': r['metrics']['f1'],
            'savings_pct': r['savings_pct'],
            'config': r.get('config', 'baseline')
        } for r in v] for k, v in results.items()}, f, indent=2)
    
    print("\nResults saved to results/validation/tuning_results.json")


if __name__ == "__main__":
    main()

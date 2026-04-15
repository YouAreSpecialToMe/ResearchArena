"""
Ablation Studies

Systematically remove each novel component to measure its contribution:
1. Full AIT-LCD: All components enabled
2. No Bias Correction: Use standard MI estimates
3. Fixed Threshold: Use constant threshold instead of adaptive
4. No Orientation: Skip edge orientation phase
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from shared.data_loader import load_dataset, load_ground_truth
from ait_lcd.ait_lcd import ait_lcd_learn
from shared.metrics import evaluate_mb_discovery, evaluate_pc_discovery


# Ablation variants
ABLATION_VARIANTS = [
    {'name': 'full', 'use_bias_correction': True, 'use_adaptive_threshold': True},
    {'name': 'no_bias_correction', 'use_bias_correction': False, 'use_adaptive_threshold': True},
    {'name': 'fixed_threshold', 'use_bias_correction': True, 'use_adaptive_threshold': False},
    {'name': 'no_adaptive_no_bias', 'use_bias_correction': False, 'use_adaptive_threshold': False},
]

NETWORKS = ['asia', 'child', 'insurance']
SAMPLE_SIZES = [100, 200, 500]
SEEDS = [1, 2, 3]


def run_single_ablation(network, n_samples, seed, variant, alpha=0.1, beta=10):
    """Run a single ablation experiment."""
    data = load_dataset(network, n_samples, seed)
    ground_truth = load_ground_truth(network)
    
    results_per_target = []
    
    targets = ground_truth['nodes'][:6]  # Sample for efficiency
    
    for target in targets:
        true_mb = ground_truth['mb_sets'][target]
        true_pc = ground_truth['pc_sets'][target]
        
        try:
            result = ait_lcd_learn(
                data, target,
                alpha=alpha, beta=beta,
                use_bias_correction=variant['use_bias_correction'],
                use_adaptive_threshold=variant['use_adaptive_threshold']
            )
            
            mb_metrics = evaluate_mb_discovery(result['mb'], true_mb)
            pc_metrics = evaluate_pc_discovery(result['pc'], true_pc)
            
            results_per_target.append({
                'target': target,
                'mb_f1': mb_metrics['f1'],
                'pc_f1': pc_metrics['f1'],
                'runtime': result['runtime']
            })
        except Exception as e:
            print(f"Error: {e}")
            results_per_target.append({
                'target': target,
                'mb_f1': 0.0,
                'pc_f1': 0.0,
                'runtime': 0.0
            })
    
    return {
        'network': network,
        'n_samples': n_samples,
        'seed': seed,
        'variant': variant['name'],
        'mb_f1': np.mean([r['mb_f1'] for r in results_per_target]),
        'pc_f1': np.mean([r['pc_f1'] for r in results_per_target]),
        'runtime': np.sum([r['runtime'] for r in results_per_target])
    }


def run_ablation_study(alpha=0.1, beta=10):
    """Run all ablation experiments."""
    results = []
    
    total_configs = len(NETWORKS) * len(SAMPLE_SIZES) * len(SEEDS) * len(ABLATION_VARIANTS)
    print(f"Total ablation configurations: {total_configs}")
    
    with tqdm(total=total_configs, desc="Ablation study") as pbar:
        for network in NETWORKS:
            for n_samples in SAMPLE_SIZES:
                for seed in SEEDS:
                    for variant in ABLATION_VARIANTS:
                        result = run_single_ablation(
                            network, n_samples, seed, variant,
                            alpha=alpha, beta=beta
                        )
                        results.append(result)
                        pbar.update(1)
    
    return results


def main():
    print("="*60)
    print("AIT-LCD Ablation Study")
    print("="*60)
    print()
    
    # Load parameters
    params_file = Path('results/selected_parameters.json')
    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
        alpha = params['alpha']
        beta = params['beta']
    else:
        alpha, beta = 0.1, 10
    
    print(f"Using parameters: alpha={alpha}, beta={beta}")
    print(f"Variants: {[v['name'] for v in ABLATION_VARIANTS]}")
    print()
    
    results = run_ablation_study(alpha=alpha, beta=beta)
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'ablation_study.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {output_dir / 'ablation_study.json'}")
    
    # Print summary
    print("\n" + "="*60)
    print("Ablation Study Summary (PC F1)")
    print("="*60)
    
    for variant in ABLATION_VARIANTS:
        variant_results = [r for r in results if r['variant'] == variant['name']]
        mean_f1 = np.mean([r['pc_f1'] for r in variant_results])
        print(f"{variant['name']:20s}: {mean_f1:.4f}")


if __name__ == '__main__':
    main()

"""
Phase 0: Pilot Study for Parameter Calibration

Grid search over alpha and beta parameters for the adaptive threshold function
tau(n,k) = alpha * sqrt(k/n) * log(1 + n/(k*beta))
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shared.data_loader import load_dataset, load_ground_truth
from ait_lcd.ait_lcd import ait_lcd_learn
from shared.metrics import evaluate_pc_discovery


# Reduced parameter grid for faster calibration
ALPHAS = [0.05, 0.1, 0.2]
BETAS = [5, 10, 20]

# Pilot networks and sample sizes
NETWORKS = ['asia', 'child']
SAMPLE_SIZES = [100, 200, 500]
SEEDS = [1, 2, 3]


def run_pilot_calibration():
    """Run parameter calibration pilot study."""
    results = []
    
    total_configs = len(ALPHAS) * len(BETAS) * len(NETWORKS) * len(SAMPLE_SIZES) * len(SEEDS)
    print(f"Total configurations: {total_configs}")
    print(f"Parameter grid: alpha={ALPHAS}, beta={BETAS}")
    print(f"Networks: {NETWORKS}")
    print(f"Sample sizes: {SAMPLE_SIZES}")
    print(f"Seeds: {SEEDS}")
    print()
    
    config_idx = 0
    for alpha in ALPHAS:
        for beta in BETAS:
            config_results = []
            
            for network in NETWORKS:
                ground_truth = load_ground_truth(network)
                
                for n_samples in SAMPLE_SIZES:
                    for seed in SEEDS:
                        config_idx += 1
                        print(f"[{config_idx}/{total_configs}] alpha={alpha}, beta={beta}, "
                              f"network={network}, n={n_samples}, seed={seed}")
                        
                        # Load data
                        data = load_dataset(network, n_samples, seed)
                        
                        # Run AIT-LCD for each target variable
                        pc_f1_scores = []
                        
                        # Use first 2 variables as targets (to save time)
                        targets = ground_truth['nodes'][:2]
                        
                        for target in targets:
                            try:
                                result = ait_lcd_learn(
                                    data, target,
                                    alpha=alpha, beta=beta,
                                    use_bias_correction=True,
                                    use_adaptive_threshold=True,
                                    verbose=False
                                )
                                
                                true_pc = ground_truth['pc_sets'][target]
                                learned_pc = result['pc']
                                
                                metrics = evaluate_pc_discovery(learned_pc, true_pc)
                                pc_f1_scores.append(metrics['f1'])
                                
                            except Exception as e:
                                print(f"  Error on {network}/{target}: {e}")
                                pc_f1_scores.append(0.0)
                        
                        avg_f1 = np.mean(pc_f1_scores) if pc_f1_scores else 0.0
                        
                        config_results.append({
                            'network': network,
                            'n_samples': n_samples,
                            'seed': seed,
                            'pc_f1': avg_f1
                        })
            
            # Aggregate across all runs for this config
            all_f1 = [r['pc_f1'] for r in config_results]
            results.append({
                'alpha': alpha,
                'beta': beta,
                'mean_f1': np.mean(all_f1),
                'std_f1': np.std(all_f1),
                'details': config_results
            })
            
            print(f"  -> alpha={alpha}, beta={beta}: mean F1={np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
            print()
    
    return results


def select_best_parameters(results):
    """Select the best (alpha, beta) pair."""
    # Sort by mean F1
    results_sorted = sorted(results, key=lambda x: x['mean_f1'], reverse=True)
    
    print("\n" + "="*60)
    print("Parameter Calibration Results (sorted by F1)")
    print("="*60)
    
    for i, r in enumerate(results_sorted):
        marker = " <-- BEST" if i == 0 else ""
        print(f"{i+1}. alpha={r['alpha']}, beta={r['beta']}: F1={r['mean_f1']:.4f} ± {r['std_f1']:.4f}{marker}")
    
    best = results_sorted[0]
    return best['alpha'], best['beta'], best['mean_f1']


def main():
    print("="*60)
    print("AIT-LCD Parameter Calibration Pilot Study")
    print("="*60)
    print()
    
    # Run calibration
    results = run_pilot_calibration()
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'pilot_calibration.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved calibration results to {output_dir / 'pilot_calibration.json'}")
    
    # Select best parameters
    best_alpha, best_beta, best_f1 = select_best_parameters(results)
    
    # Save selected parameters
    selected = {
        'alpha': best_alpha,
        'beta': best_beta,
        'mean_f1': best_f1,
        'calibration_method': 'grid_search',
        'networks': NETWORKS,
        'sample_sizes': SAMPLE_SIZES,
        'seeds': SEEDS
    }
    
    with open(output_dir / 'selected_parameters.json', 'w') as f:
        json.dump(selected, f, indent=2)
    print(f"\nSaved selected parameters to {output_dir / 'selected_parameters.json'}")
    
    print("\n" + "="*60)
    print(f"SELECTED PARAMETERS: alpha={best_alpha}, beta={best_beta}")
    print("="*60)


if __name__ == '__main__':
    main()

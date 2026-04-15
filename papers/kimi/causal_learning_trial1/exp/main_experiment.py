"""
Main Comparative Evaluation

Run all algorithms on all benchmark networks with calibrated parameters.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).parent))
from shared.data_loader import load_dataset, load_ground_truth
from ait_lcd.ait_lcd import ait_lcd_learn
from baselines.baseline_wrappers import run_baseline
from shared.metrics import evaluate_mb_discovery, evaluate_pc_discovery


NETWORKS = ['asia', 'child', 'insurance', 'alarm', 'hailfinder']
SAMPLE_SIZES = [100, 200, 500, 1000]
SEEDS = [1, 2, 3]
ALGORITHMS = ['ait-lcd', 'iamb', 'hiton-mb', 'pcmb', 'eamb-inspired']


def run_single_experiment(network, n_samples, seed, algorithm, alpha=0.1, beta=10):
    """Run a single experiment configuration."""
    # Load data and ground truth
    data = load_dataset(network, n_samples, seed)
    ground_truth = load_ground_truth(network)
    
    results_per_target = []
    
    # Run for each target variable (limit to first 8 for larger networks)
    targets = ground_truth['nodes']
    if len(targets) > 8:
        targets = targets[:8]  # Sample for efficiency
    
    for target in targets:
        true_mb = ground_truth['mb_sets'][target]
        true_pc = ground_truth['pc_sets'][target]
        
        try:
            if algorithm == 'ait-lcd':
                result = ait_lcd_learn(
                    data, target,
                    alpha=alpha, beta=beta,
                    use_bias_correction=True,
                    use_adaptive_threshold=True
                )
            else:
                result = run_baseline(algorithm, data, target, alpha=0.05)
            
            # Evaluate
            mb_metrics = evaluate_mb_discovery(result['mb'], true_mb)
            pc_metrics = evaluate_pc_discovery(result['pc'], true_pc)
            
            results_per_target.append({
                'target': target,
                'mb_precision': mb_metrics['precision'],
                'mb_recall': mb_metrics['recall'],
                'mb_f1': mb_metrics['f1'],
                'pc_precision': pc_metrics['precision'],
                'pc_recall': pc_metrics['recall'],
                'pc_f1': pc_metrics['f1'],
                'runtime': result['runtime'],
                'ci_tests': result.get('ci_tests', 0)
            })
            
        except Exception as e:
            print(f"Error on {network}/{target}/{algorithm}: {e}")
            results_per_target.append({
                'target': target,
                'mb_precision': 0.0,
                'mb_recall': 0.0,
                'mb_f1': 0.0,
                'pc_precision': 0.0,
                'pc_recall': 0.0,
                'pc_f1': 0.0,
                'runtime': 0.0,
                'ci_tests': 0
            })
    
    # Aggregate across targets
    return {
        'network': network,
        'n_samples': n_samples,
        'seed': seed,
        'algorithm': algorithm,
        'mb_precision': np.mean([r['mb_precision'] for r in results_per_target]),
        'mb_recall': np.mean([r['mb_recall'] for r in results_per_target]),
        'mb_f1': np.mean([r['mb_f1'] for r in results_per_target]),
        'pc_precision': np.mean([r['pc_precision'] for r in results_per_target]),
        'pc_recall': np.mean([r['pc_recall'] for r in results_per_target]),
        'pc_f1': np.mean([r['pc_f1'] for r in results_per_target]),
        'runtime': np.sum([r['runtime'] for r in results_per_target]),
        'ci_tests': np.sum([r['ci_tests'] for r in results_per_target]),
        'per_target': results_per_target
    }


def run_main_experiment(alpha=0.1, beta=10):
    """Run the main comparative evaluation."""
    results = []
    
    total_configs = len(NETWORKS) * len(SAMPLE_SIZES) * len(SEEDS) * len(ALGORITHMS)
    print(f"Total configurations: {total_configs}")
    print(f"Networks: {NETWORKS}")
    print(f"Sample sizes: {SAMPLE_SIZES}")
    print(f"Seeds: {SEEDS}")
    print(f"Algorithms: {ALGORITHMS}")
    print()
    
    with tqdm(total=total_configs, desc="Main experiment") as pbar:
        for network in NETWORKS:
            for n_samples in SAMPLE_SIZES:
                for seed in SEEDS:
                    for algorithm in ALGORITHMS:
                        result = run_single_experiment(
                            network, n_samples, seed, algorithm,
                            alpha=alpha, beta=beta
                        )
                        results.append(result)
                        pbar.update(1)
                        
                        # Print progress
                        if algorithm == 'ait-lcd':
                            print(f"\n{network}, n={n_samples}, seed={seed}: PC F1={result['pc_f1']:.3f}")
    
    return results


def main():
    print("="*60)
    print("AIT-LCD Main Comparative Evaluation")
    print("="*60)
    print()
    
    # Load selected parameters from pilot study if available
    params_file = Path('results/selected_parameters.json')
    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
        alpha = params['alpha']
        beta = params['beta']
        print(f"Using calibrated parameters: alpha={alpha}, beta={beta}")
    else:
        alpha, beta = 0.1, 10
        print(f"Using default parameters: alpha={alpha}, beta={beta}")
    print()
    
    # Run experiment
    results = run_main_experiment(alpha=alpha, beta=beta)
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'main_experiment.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {output_dir / 'main_experiment.json'}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary (PC F1 scores)")
    print("="*60)
    
    for algorithm in ALGORITHMS:
        alg_results = [r for r in results if r['algorithm'] == algorithm and r['n_samples'] <= 500]
        if alg_results:
            mean_f1 = np.mean([r['pc_f1'] for r in alg_results])
            print(f"{algorithm:15s}: {mean_f1:.4f}")


if __name__ == '__main__':
    main()

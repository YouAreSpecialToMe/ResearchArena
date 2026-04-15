"""
All-in-one experiment script for AIT-LCD.

Runs:
1. Quick pilot study (reduced parameter grid)
2. Main experiments with all baselines
3. Ablation studies
4. Analysis and figure generation
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


# Experiment configuration
NETWORKS = ['asia', 'child', 'insurance', 'alarm', 'hailfinder']
SAMPLE_SIZES = [100, 200, 500, 1000]
SEEDS = [1, 2, 3]
ALGORITHMS = ['ait-lcd', 'iamb', 'adaptive-iamb', 'simple-pc']


def get_targets(ground_truth, max_targets=5):
    """Get target variables for evaluation."""
    nodes = ground_truth['nodes']
    # Prefer nodes with non-empty PC sets
    nodes_with_pc = [n for n in nodes if ground_truth['pc_sets'][n]]
    
    if len(nodes_with_pc) >= max_targets:
        return nodes_with_pc[:max_targets]
    return nodes[:max_targets]


def run_single_experiment(network, n_samples, seed, algorithm, alpha=0.05, beta=10):
    """Run a single experiment configuration."""
    data = load_dataset(network, n_samples, seed)
    ground_truth = load_ground_truth(network)
    
    results_per_target = []
    targets = get_targets(ground_truth)
    
    for target in targets:
        true_mb = ground_truth['mb_sets'][target]
        true_pc = ground_truth['pc_sets'][target]
        
        try:
            if algorithm == 'ait-lcd':
                result = ait_lcd_learn(
                    data, target,
                    alpha=alpha, beta=beta,
                    use_bias_correction=True,
                    use_adaptive_threshold=True,
                    verbose=False
                )
            else:
                result = run_baseline(algorithm, data, target, alpha=0.05)
            
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
            print(f"Error: {network}/{target}/{algorithm}: {e}")
            results_per_target.append({
                'target': target,
                'mb_precision': 0.0, 'mb_recall': 0.0, 'mb_f1': 0.0,
                'pc_precision': 0.0, 'pc_recall': 0.0, 'pc_f1': 0.0,
                'runtime': 0.0, 'ci_tests': 0
            })
    
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
    }


def run_pilot_study():
    """Quick pilot study to select alpha parameter."""
    print("\n" + "="*60)
    print("Phase 0: Quick Parameter Calibration")
    print("="*60)
    
    # Reduced grid
    alphas = [0.01, 0.03, 0.05, 0.1]
    beta = 10
    
    pilot_networks = ['asia', 'child']
    pilot_sample_sizes = [100, 500]
    pilot_seeds = [1, 2]
    
    results = []
    
    for alpha in alphas:
        f1_scores = []
        
        for network in pilot_networks:
            for n_samples in pilot_sample_sizes:
                for seed in pilot_seeds:
                    result = run_single_experiment(network, n_samples, seed, 'ait-lcd', alpha=alpha, beta=beta)
                    f1_scores.append(result['pc_f1'])
        
        mean_f1 = np.mean(f1_scores)
        results.append({'alpha': alpha, 'mean_f1': mean_f1})
        print(f"alpha={alpha:.2f}: mean PC F1={mean_f1:.4f}")
    
    # Select best alpha
    best = max(results, key=lambda x: x['mean_f1'])
    print(f"\nSelected: alpha={best['alpha']:.2f} with F1={best['mean_f1']:.4f}")
    
    return best['alpha'], beta


def run_main_experiment(alpha, beta):
    """Run main comparative evaluation."""
    print("\n" + "="*60)
    print("Phase 1: Main Comparative Evaluation")
    print("="*60)
    
    results = []
    total = len(NETWORKS) * len(SAMPLE_SIZES) * len(SEEDS) * len(ALGORITHMS)
    
    with tqdm(total=total, desc="Main experiments") as pbar:
        for network in NETWORKS:
            for n_samples in SAMPLE_SIZES:
                for seed in SEEDS:
                    for algorithm in ALGORITHMS:
                        result = run_single_experiment(network, n_samples, seed, algorithm, alpha, beta)
                        results.append(result)
                        pbar.update(1)
    
    return results


def run_ablation_study(alpha, beta):
    """Run ablation studies."""
    print("\n" + "="*60)
    print("Phase 2: Ablation Studies")
    print("="*60)
    
    variants = [
        {'name': 'full', 'use_bias_correction': True, 'use_adaptive_threshold': True},
        {'name': 'no_bias_correction', 'use_bias_correction': False, 'use_adaptive_threshold': True},
        {'name': 'fixed_threshold', 'use_bias_correction': True, 'use_adaptive_threshold': False},
        {'name': 'neither', 'use_bias_correction': False, 'use_adaptive_threshold': False},
    ]
    
    ablation_networks = ['asia', 'child', 'insurance']
    ablation_sample_sizes = [100, 200, 500]
    
    results = []
    total = len(ablation_networks) * len(ablation_sample_sizes) * len(SEEDS) * len(variants)
    
    with tqdm(total=total, desc="Ablation study") as pbar:
        for network in ablation_networks:
            for n_samples in ablation_sample_sizes:
                for seed in SEEDS:
                    data = load_dataset(network, n_samples, seed)
                    ground_truth = load_ground_truth(network)
                    
                    for variant in variants:
                        targets = get_targets(ground_truth, max_targets=3)
                        f1_scores = []
                        
                        for target in targets:
                            true_pc = ground_truth['pc_sets'][target]
                            
                            try:
                                result = ait_lcd_learn(
                                    data, target,
                                    alpha=alpha, beta=beta,
                                    use_bias_correction=variant['use_bias_correction'],
                                    use_adaptive_threshold=variant['use_adaptive_threshold'],
                                    verbose=False
                                )
                                
                                pc_metrics = evaluate_pc_discovery(result['pc'], true_pc)
                                f1_scores.append(pc_metrics['f1'])
                            except:
                                f1_scores.append(0.0)
                        
                        results.append({
                            'network': network,
                            'n_samples': n_samples,
                            'seed': seed,
                            'variant': variant['name'],
                            'pc_f1': np.mean(f1_scores)
                        })
                        pbar.update(1)
    
    return results


def analyze_results(main_results, ablation_results, alpha, beta):
    """Analyze results and generate summary."""
    print("\n" + "="*60)
    print("Phase 3: Analysis")
    print("="*60)
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Save raw results
    with open(results_dir / 'main_experiment.json', 'w') as f:
        json.dump(main_results, f, indent=2)
    
    with open(results_dir / 'ablation_study.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    # Save parameters
    with open(results_dir / 'selected_parameters.json', 'w') as f:
        json.dump({'alpha': alpha, 'beta': beta}, f, indent=2)
    
    # Generate summary table
    print("\nSummary Table (PC F1 scores):")
    print("-" * 80)
    print(f"{'Algorithm':<20} {'n≤500':<15} {'n>500':<15} {'Overall':<15}")
    print("-" * 80)
    
    for alg in ALGORITHMS:
        small = [r['pc_f1'] for r in main_results if r['algorithm'] == alg and r['n_samples'] <= 500]
        large = [r['pc_f1'] for r in main_results if r['algorithm'] == alg and r['n_samples'] > 500]
        all_scores = [r['pc_f1'] for r in main_results if r['algorithm'] == alg]
        
        print(f"{alg:<20} {np.mean(small):.3f}±{np.std(small):.3f}   {np.mean(large):.3f}±{np.std(large):.3f}   {np.mean(all_scores):.3f}±{np.std(all_scores):.3f}")
    
    print("-" * 80)
    
    # Ablation summary
    print("\nAblation Study Summary:")
    print("-" * 40)
    for variant in ['full', 'no_bias_correction', 'fixed_threshold', 'neither']:
        scores = [r['pc_f1'] for r in ablation_results if r['variant'] == variant]
        if scores:
            print(f"{variant:<20}: {np.mean(scores):.3f}±{np.std(scores):.3f}")
    print("-" * 40)


def main():
    start_time = time.time()
    
    print("="*60)
    print("AIT-LCD Complete Experiment Suite")
    print("="*60)
    
    # Phase 0: Pilot study
    alpha, beta = run_pilot_study()
    
    # Phase 1: Main experiment
    main_results = run_main_experiment(alpha, beta)
    
    # Phase 2: Ablation study
    ablation_results = run_ablation_study(alpha, beta)
    
    # Phase 3: Analysis
    analyze_results(main_results, ablation_results, alpha, beta)
    
    # Generate figures
    print("\n" + "="*60)
    print("Phase 4: Generating Figures")
    print("="*60)
    
    try:
        import visualize
        visualize.main()
    except Exception as e:
        print(f"Figure generation error: {e}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed/60:.1f} minutes")
    print("\nExperiments complete!")


if __name__ == '__main__':
    main()

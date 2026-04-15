"""
Final AIT-LCD Experiments - Complete evaluation.
"""
import json
import sys
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.data_loader import load_dataset, load_ground_truth
from shared.metrics import evaluate_pc_discovery, evaluate_mb_discovery
from ait_lcd.ait_lcd_fast import ait_lcd_learn_fast
from baselines.baseline_wrappers import SimpleIAMB, AdaptiveIAMB
from baselines.hiton_mb import HITONMB
from baselines.pcmb import PCMB
from datetime import datetime
import time

ALGORITHMS = ['AIT-LCD', 'IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']
NETWORKS = ['asia', 'child', 'insurance', 'alarm', 'hailfinder']
SAMPLE_SIZES = [100, 200, 500, 1000]
SEEDS = [1, 2, 3]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def check_data_available(network, n_samples, seed):
    """Check if data file exists."""
    path = f'data/{network}/n{n_samples}/seed{seed}.csv'
    return os.path.exists(path)

def run_algorithm(algo_name, data, target):
    """Run a single algorithm."""
    t0 = time.time()
    
    if algo_name == 'AIT-LCD':
        result = ait_lcd_learn_fast(data, target, alpha=0.2, beta=10)
    elif algo_name == 'IAMB':
        algo = SimpleIAMB(alpha=0.05)
        result = algo.fit(data, target)
    elif algo_name == 'HITON-MB':
        algo = HITONMB(alpha=0.05, max_k=2)
        result = algo.fit(data, target)
    elif algo_name == 'PCMB':
        algo = PCMB(alpha=0.05, max_k=2)
        result = algo.fit(data, target)
    elif algo_name == 'EAMB-inspired':
        algo = AdaptiveIAMB(alpha=0.05)
        result = algo.fit(data, target)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    result['runtime'] = time.time() - t0
    return result

def run_ablations(network, n_samples, seed, target, data, true_pc):
    """Run ablation variants of AIT-LCD."""
    ablations = []
    
    variants = [
        ('Full AIT-LCD', True, True),
        ('No Bias Correction', False, True),
        ('Fixed Threshold', True, False),
        ('No Adaptations', False, False)
    ]
    
    for variant_name, use_bias, use_adaptive in variants:
        try:
            result = ait_lcd_learn_fast(
                data, target, alpha=0.2, beta=10,
                use_bias_correction=use_bias,
                use_adaptive_threshold=use_adaptive
            )
            metrics = evaluate_pc_discovery(result['pc'], true_pc)
            ablations.append({
                'variant': variant_name,
                'network': network,
                'n_samples': n_samples,
                'seed': seed,
                'target': target,
                'pc_f1': metrics['f1'],
                'pc_precision': metrics['precision'],
                'pc_recall': metrics['recall']
            })
        except Exception as e:
            log(f"  Ablation error: {variant_name}: {e}")
    
    return ablations

def main():
    log("="*70)
    log("AIT-LCD Final Experiments")
    log("="*70)
    
    main_results = []
    ablation_results = []
    
    total_configs = 0
    for network in NETWORKS:
        for n in SAMPLE_SIZES:
            for seed in SEEDS:
                if check_data_available(network, n, seed):
                    total_configs += 1
    
    log(f"Total data configurations available: {total_configs}")
    log(f"Algorithms: {ALGORITHMS}")
    log("")
    
    processed = 0
    
    for network in NETWORKS:
        log(f"Processing network: {network}")
        
        try:
            gt = load_ground_truth(network)
            targets = gt['nodes'][:3]  # First 3 targets
        except Exception as e:
            log(f"  Error loading ground truth: {e}")
            continue
        
        for n_samples in SAMPLE_SIZES:
            for seed in SEEDS:
                if not check_data_available(network, n_samples, seed):
                    continue
                
                try:
                    data = load_dataset(network, n_samples, seed)
                except Exception as e:
                    log(f"  Error loading data: {e}")
                    continue
                
                processed += 1
                log(f"  [{processed}/{total_configs}] {network}/n{n_samples}/seed{seed}")
                
                for target in targets:
                    true_pc = gt['pc_sets'][target]
                    true_mb = gt['mb_sets'][target]
                    
                    # Run main algorithms
                    for algo in ALGORITHMS:
                        try:
                            result = run_algorithm(algo, data, target)
                            pc_m = evaluate_pc_discovery(result['pc'], true_pc)
                            mb_m = evaluate_mb_discovery(result['mb'], true_mb)
                            
                            main_results.append({
                                'network': network,
                                'n_samples': n_samples,
                                'seed': seed,
                                'target': target,
                                'algorithm': algo,
                                'pc_f1': pc_m['f1'],
                                'pc_precision': pc_m['precision'],
                                'pc_recall': pc_m['recall'],
                                'mb_f1': mb_m['f1'],
                                'runtime': result['runtime'],
                                'ci_tests': result.get('ci_tests', 0)
                            })
                        except Exception as e:
                            log(f"    {algo} error: {e}")
                    
                    # Run ablations (only for subset)
                    if network in ['asia', 'child', 'insurance'] and n_samples <= 500:
                        abls = run_ablations(network, n_samples, seed, target, data, true_pc)
                        ablation_results.extend(abls)
    
    # Save raw results
    os.makedirs('exp/results', exist_ok=True)
    with open('exp/results/main_experiment.json', 'w') as f:
        json.dump(main_results, f, indent=2)
    
    with open('exp/results/ablation_study.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    # Aggregate results
    log("")
    log("="*70)
    log("AGGREGATING RESULTS")
    log("="*70)
    
    algo_summary = {}
    for algo in ALGORITHMS:
        algo_data = [r for r in main_results if r['algorithm'] == algo]
        if algo_data:
            algo_summary[algo] = {
                'pc_f1': {
                    'mean': float(np.mean([r['pc_f1'] for r in algo_data])),
                    'std': float(np.std([r['pc_f1'] for r in algo_data]))
                },
                'mb_f1': {
                    'mean': float(np.mean([r['mb_f1'] for r in algo_data])),
                    'std': float(np.std([r['mb_f1'] for r in algo_data]))
                },
                'runtime': {
                    'mean': float(np.mean([r['runtime'] for r in algo_data])),
                    'std': float(np.std([r['runtime'] for r in algo_data]))
                }
            }
    
    ablation_summary = {}
    for variant in ['Full AIT-LCD', 'No Bias Correction', 'Fixed Threshold', 'No Adaptations']:
        variant_data = [r for r in ablation_results if r['variant'] == variant]
        if variant_data:
            ablation_summary[variant] = {
                'pc_f1': {
                    'mean': float(np.mean([r['pc_f1'] for r in variant_data])),
                    'std': float(np.std([r['pc_f1'] for r in variant_data]))
                }
            }
    
    # Sample size analysis
    sample_size_analysis = {}
    for n in SAMPLE_SIZES:
        n_data = [r for r in main_results if r['n_samples'] == n and r['algorithm'] == 'AIT-LCD']
        if n_data:
            sample_size_analysis[str(n)] = {
                'pc_f1_mean': float(np.mean([r['pc_f1'] for r in n_data])),
                'pc_f1_std': float(np.std([r['pc_f1'] for r in n_data]))
            }
    
    # Final results
    final_results = {
        'experiment_info': {
            'title': 'AIT-LCD: Adaptive Information-Theoretic Local Causal Discovery',
            'date': datetime.now().isoformat(),
            'networks': NETWORKS,
            'sample_sizes': SAMPLE_SIZES,
            'seeds': SEEDS,
            'algorithms': ALGORITHMS,
            'total_experiments': len(main_results)
        },
        'main_results': algo_summary,
        'ablation_results': ablation_summary,
        'sample_size_analysis': sample_size_analysis
    }
    
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    log("")
    log("="*70)
    log("RESULTS SUMMARY")
    log("="*70)
    
    log("\nMain Results (PC F1):")
    for algo, metrics in algo_summary.items():
        log(f"  {algo:20s}: {metrics['pc_f1']['mean']:.3f} ± {metrics['pc_f1']['std']:.3f}")
    
    if ablation_summary:
        log("\nAblation Results (PC F1):")
        for variant, metrics in ablation_summary.items():
            log(f"  {variant:25s}: {metrics['pc_f1']['mean']:.3f} ± {metrics['pc_f1']['std']:.3f}")
    
    log("\nAIT-LCD by Sample Size:")
    for n, metrics in sample_size_analysis.items():
        log(f"  n={n:4s}: F1={metrics['pc_f1_mean']:.3f} ± {metrics['pc_f1_std']:.3f}")
    
    log("")
    log("="*70)
    log(f"Total experiments: {len(main_results)}")
    log("Results saved to: results.json")
    log("="*70)

if __name__ == '__main__':
    main()

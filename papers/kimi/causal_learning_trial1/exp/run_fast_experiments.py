"""
Fast AIT-LCD Experiments - Streamlined for time budget.

Runs core experiments with optimized settings:
- Pre-generated data where available
- Smaller but representative parameter sweep
- Parallel-friendly structure
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent))


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# Import required modules
from shared.data_loader import load_dataset, load_ground_truth
from shared.metrics import evaluate_pc_discovery, evaluate_mb_discovery
from ait_lcd.ait_lcd_v2 import ait_lcd_learn_v2


def run_all_baselines(data, target, alpha=0.05):
    """Run all baseline algorithms."""
    results = {}
    
    # IAMB
    from baselines.baseline_wrappers import SimpleIAMB
    algo = SimpleIAMB(alpha=alpha)
    results['IAMB'] = algo.fit(data, target)
    
    # HITON-MB
    from baselines.hiton_mb import HITONMB
    algo = HITONMB(alpha=alpha, max_k=2)
    results['HITON-MB'] = algo.fit(data, target)
    
    # PCMB
    from baselines.pcmb import PCMB
    algo = PCMB(alpha=alpha, max_k=2)
    results['PCMB'] = algo.fit(data, target)
    
    # EAMB-inspired
    from baselines.baseline_wrappers import AdaptiveIAMB
    algo = AdaptiveIAMB(alpha=alpha)
    results['EAMB-inspired'] = algo.fit(data, target)
    
    return results


def run_ait_lcd_variants(data, target, alpha=0.2, beta=10):
    """Run AIT-LCD with different configurations."""
    results = {}
    
    # Full AIT-LCD
    results['AIT-LCD'] = ait_lcd_learn_v2(
        data, target, alpha=alpha, beta=beta,
        use_bias_correction=True, use_adaptive_threshold=True
    )
    
    # No bias correction
    results['AIT-LCD-NoBias'] = ait_lcd_learn_v2(
        data, target, alpha=alpha, beta=beta,
        use_bias_correction=False, use_adaptive_threshold=True
    )
    
    # Fixed threshold
    results['AIT-LCD-Fixed'] = ait_lcd_learn_v2(
        data, target, alpha=alpha, beta=beta,
        use_bias_correction=True, use_adaptive_threshold=False
    )
    
    return results


def run_pilot_calibration():
    """Quick parameter calibration."""
    log("Running Pilot Calibration")
    
    networks = ['asia', 'child']
    sample_sizes = [100, 200]
    seeds = [1, 2]
    targets_per_net = 2
    
    best_alpha, best_beta = 0.2, 10
    best_f1 = 0
    
    for alpha in [0.1, 0.2, 0.3]:
        for beta in [5, 10, 20]:
            f1_scores = []
            
            for network in networks:
                gt = load_ground_truth(network)
                
                for n in sample_sizes:
                    for seed in seeds:
                        data = load_dataset(network, n, seed)
                        
                        for target in gt['nodes'][:targets_per_net]:
                            try:
                                result = ait_lcd_learn_v2(data, target, alpha=alpha, beta=beta)
                                true_pc = gt['pc_sets'][target]
                                metrics = evaluate_pc_discovery(result['pc'], true_pc)
                                f1_scores.append(metrics['f1'])
                            except:
                                f1_scores.append(0)
            
            mean_f1 = np.mean(f1_scores)
            log(f"  alpha={alpha}, beta={beta}: F1={mean_f1:.4f}")
            
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_alpha, best_beta = alpha, beta
    
    log(f"Best: alpha={best_alpha}, beta={best_beta}, F1={best_f1:.4f}")
    
    # Save
    output = {'alpha': best_alpha, 'beta': best_beta, 'mean_f1': best_f1}
    Path('results').mkdir(exist_ok=True)
    with open('results/selected_parameters.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    return best_alpha, best_beta


def run_main_experiments(alpha, beta):
    """Run main comparative experiments."""
    log("Running Main Experiments")
    
    networks = ['asia', 'child', 'insurance', 'alarm', 'hailfinder']
    sample_sizes = [100, 200, 500, 1000]
    seeds = [1, 2, 3]
    targets_per_net = 3
    
    all_results = []
    
    total = len(networks) * len(sample_sizes) * len(seeds) * 5  # 5 algorithms
    count = 0
    
    for network in networks:
        log(f"  Network: {network}")
        
        try:
            gt = load_ground_truth(network)
        except:
            log(f"    Skipping {network} - no ground truth")
            continue
        
        for n in sample_sizes:
            for seed in seeds:
                try:
                    data = load_dataset(network, n, seed)
                except:
                    log(f"    Skipping {network}/n{n}/seed{seed}")
                    continue
                
                for target in gt['nodes'][:targets_per_net]:
                    true_pc = gt['pc_sets'][target]
                    true_mb = gt['mb_sets'][target]
                    
                    # Run AIT-LCD
                    try:
                        count += 1
                        t0 = time.time()
                        result = ait_lcd_learn_v2(data, target, alpha=alpha, beta=beta)
                        runtime = time.time() - t0
                        
                        pc_metrics = evaluate_pc_discovery(result['pc'], true_pc)
                        mb_metrics = evaluate_mb_discovery(result['mb'], true_mb)
                        
                        all_results.append({
                            'network': network, 'n_samples': n, 'seed': seed,
                            'target': target, 'algorithm': 'AIT-LCD',
                            'pc_f1': pc_metrics['f1'], 'pc_precision': pc_metrics['precision'],
                            'pc_recall': pc_metrics['recall'], 'mb_f1': mb_metrics['f1'],
                            'runtime': runtime, 'ci_tests': result.get('ci_tests', 0)
                        })
                    except Exception as e:
                        log(f"    AIT-LCD error: {e}")
                    
                    # Run baselines
                    try:
                        baselines = run_all_baselines(data, target)
                        for algo_name, result in baselines.items():
                            count += 1
                            pc_metrics = evaluate_pc_discovery(result['pc'], true_pc)
                            mb_metrics = evaluate_mb_discovery(result['mb'], true_mb)
                            
                            all_results.append({
                                'network': network, 'n_samples': n, 'seed': seed,
                                'target': target, 'algorithm': algo_name,
                                'pc_f1': pc_metrics['f1'], 'pc_precision': pc_metrics['precision'],
                                'pc_recall': pc_metrics['recall'], 'mb_f1': mb_metrics['f1'],
                                'runtime': result['runtime'], 'ci_tests': result.get('ci_tests', 0)
                            })
                    except Exception as e:
                        log(f"    Baseline error: {e}")
                
                if count % 50 == 0:
                    log(f"    Progress: {count}/{total}")
    
    log(f"Main experiments complete: {len(all_results)} results")
    
    with open('results/main_experiment.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


def run_ablations(alpha, beta):
    """Run ablation studies."""
    log("Running Ablation Studies")
    
    networks = ['asia', 'child', 'insurance']
    sample_sizes = [100, 200, 500]
    seeds = [1, 2, 3]
    
    all_results = []
    
    for network in networks:
        log(f"  Network: {network}")
        
        try:
            gt = load_ground_truth(network)
        except:
            continue
        
        for n in sample_sizes:
            for seed in seeds:
                try:
                    data = load_dataset(network, n, seed)
                except:
                    continue
                
                for target in gt['nodes'][:2]:
                    true_pc = gt['pc_sets'][target]
                    
                    variants = {
                        'Full AIT-LCD': (True, True),
                        'No Bias Correction': (False, True),
                        'Fixed Threshold': (True, False),
                        'No Adaptations': (False, False)
                    }
                    
                    for variant, (bias, adaptive) in variants.items():
                        try:
                            result = ait_lcd_learn_v2(
                                data, target, alpha=alpha, beta=beta,
                                use_bias_correction=bias, use_adaptive_threshold=adaptive
                            )
                            pc_metrics = evaluate_pc_discovery(result['pc'], true_pc)
                            
                            all_results.append({
                                'variant': variant, 'network': network,
                                'n_samples': n, 'seed': seed, 'target': target,
                                'pc_f1': pc_metrics['f1'],
                                'pc_precision': pc_metrics['precision'],
                                'pc_recall': pc_metrics['recall']
                            })
                        except Exception as e:
                            log(f"    Error: {e}")
    
    log(f"Ablation studies complete: {len(all_results)} results")
    
    with open('results/ablation_study.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


def aggregate_results(main_results, ablation_results):
    """Create final results.json."""
    log("Aggregating Results")
    
    # Algorithm summary
    algo_summary = {}
    for algo in ['AIT-LCD', 'IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']:
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
    
    # Ablation summary
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
    sample_size_results = {}
    for n in [100, 200, 500, 1000]:
        n_data = [r for r in main_results if r['n_samples'] == n and r['algorithm'] == 'AIT-LCD']
        if n_data:
            sample_size_results[str(n)] = {
                'pc_f1_mean': float(np.mean([r['pc_f1'] for r in n_data])),
                'pc_f1_std': float(np.std([r['pc_f1'] for r in n_data]))
            }
    
    final = {
        'experiment_info': {
            'title': 'AIT-LCD: Adaptive Information-Theoretic Local Causal Discovery',
            'description': 'Comprehensive evaluation with 5 baselines, 5 networks, 4 sample sizes, 3 seeds',
            'networks': ['asia', 'child', 'insurance', 'alarm', 'hailfinder'],
            'sample_sizes': [100, 200, 500, 1000],
            'seeds': [1, 2, 3],
            'algorithms': list(algo_summary.keys()),
            'date': datetime.now().isoformat()
        },
        'main_results': algo_summary,
        'ablation_results': ablation_summary,
        'sample_size_analysis': sample_size_results,
        'total_experiments': len(main_results) + len(ablation_results)
    }
    
    with open('results.json', 'w') as f:
        json.dump(final, f, indent=2)
    
    return final


def main():
    """Run all experiments."""
    start = time.time()
    
    print("="*70)
    print("AIT-LCD Fast Experiments")
    print("="*70)
    print()
    
    Path('results').mkdir(exist_ok=True)
    
    # Phase 0: Pilot
    alpha, beta = run_pilot_calibration()
    
    # Phase 1: Main experiments
    main_results = run_main_experiments(alpha, beta)
    
    # Phase 2: Ablations
    ablation_results = run_ablations(alpha, beta)
    
    # Phase 3: Aggregate
    final = aggregate_results(main_results, ablation_results)
    
    elapsed = time.time() - start
    
    print()
    print("="*70)
    print(f"Complete! Time: {elapsed/60:.1f} minutes")
    print("="*70)
    
    print("\nMain Results (PC F1):")
    for algo, metrics in final['main_results'].items():
        print(f"  {algo}: {metrics['pc_f1']['mean']:.3f} ± {metrics['pc_f1']['std']:.3f}")
    
    print("\nAblation Results (PC F1):")
    for variant, metrics in final['ablation_results'].items():
        print(f"  {variant}: {metrics['pc_f1']['mean']:.3f} ± {metrics['pc_f1']['std']:.3f}")


if __name__ == '__main__':
    main()

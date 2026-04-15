"""
Efficient Experiment Runner for AIT-LCD
Optimized for completion within time budget while addressing all feedback points.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent))

from shared.data_loader import load_dataset, load_ground_truth
from ait_lcd.ait_lcd_v2 import AITLCDv2
from baselines.mi_iamb import MIBasedIAMB, HITONMB, PCMBAlgorithm

# Configuration - focused on smaller networks for faster completion
NETWORKS = ['asia', 'child']  # Focus on networks with complete data
SAMPLE_SIZES = [100, 200, 500, 1000]
SEEDS = [1, 2, 3]
ALPHA = 0.15
BETA = 10

def log_message(msg, log_file):
    """Log message to both console and file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(log_file, 'a') as f:
        f.write(full_msg + '\n')


def compute_metrics(predicted, true):
    """Compute precision, recall, F1."""
    predicted_set = set(predicted)
    true_set = set(true)
    
    tp = len(predicted_set & true_set)
    fp = len(predicted_set - true_set)
    fn = len(true_set - predicted_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def run_algorithm(algo_name, data, target, config=None):
    """Run specified algorithm."""
    if algo_name == 'AIT-LCD':
        cfg = config or {'use_bias_correction': True, 'use_adaptive_threshold': True}
        algo = AITLCDv2(alpha=ALPHA, beta=BETA, **cfg)
        return algo.fit(data, target)
    elif algo_name == 'IAMB':
        algo = MIBasedIAMB(threshold=0.05, use_mi_threshold=True)
        return algo.fit(data, target)
    elif algo_name == 'HITON-MB':
        algo = HITONMB(threshold=0.05, use_mi_threshold=True, max_k=2)
        return algo.fit(data, target)
    elif algo_name == 'PCMB':
        algo = PCMBAlgorithm(threshold=0.05, use_mi_threshold=True, max_k=2)
        return algo.fit(data, target)
    elif algo_name == 'EAMB-inspired':
        n = len(data)
        threshold = 0.05 * np.sqrt(100 / max(n, 100))
        algo = MIBasedIAMB(threshold=threshold, use_mi_threshold=True)
        return algo.fit(data, target)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def run_experiments(log_file):
    """Run main experiments."""
    log_message("="*60, log_file)
    log_message("MAIN EXPERIMENTS", log_file)
    log_message("="*60, log_file)
    
    algorithms = ['AIT-LCD', 'IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']
    results = []
    
    total = len(NETWORKS) * len(SAMPLE_SIZES) * len(SEEDS) * len(algorithms)
    completed = 0
    
    for network in NETWORKS:
        for n_samples in SAMPLE_SIZES:
            for seed in SEEDS:
                for algo in algorithms:
                    try:
                        data = load_dataset(network, n_samples, seed)
                        ground_truth = load_ground_truth(network)
                        
                        # Sample targets for efficiency
                        targets = ground_truth['nodes'][:8] if len(ground_truth['nodes']) > 8 else ground_truth['nodes']
                        
                        pc_f1s = []
                        mb_f1s = []
                        runtimes = []
                        
                        for target in targets:
                            true_pc = ground_truth['pc_sets'][target]
                            true_mb = ground_truth['mb_sets'][target]
                            
                            result = run_algorithm(algo, data, target)
                            
                            pc_metrics = compute_metrics(result['pc'], true_pc)
                            mb_metrics = compute_metrics(result['mb'], true_mb)
                            
                            pc_f1s.append(pc_metrics['f1'])
                            mb_f1s.append(mb_metrics['f1'])
                            runtimes.append(result['runtime'])
                        
                        results.append({
                            'network': network,
                            'n_samples': n_samples,
                            'seed': seed,
                            'algorithm': algo,
                            'pc_f1': float(np.mean(pc_f1s)),
                            'mb_f1': float(np.mean(mb_f1s)),
                            'runtime': float(np.sum(runtimes))
                        })
                        
                        completed += 1
                        log_message(f"✓ {completed}/{total}: {network} n={n_samples} s={seed} {algo}: PC F1={np.mean(pc_f1s):.3f}", log_file)
                        
                    except Exception as e:
                        completed += 1
                        log_message(f"✗ {completed}/{total}: {network} n={n_samples} s={seed} {algo}: {e}", log_file)
                        results.append({
                            'network': network,
                            'n_samples': n_samples,
                            'seed': seed,
                            'algorithm': algo,
                            'pc_f1': 0.0, 'mb_f1': 0.0, 'runtime': 0.0,
                            'error': str(e)
                        })
    
    return results


def run_ablations(log_file):
    """Run ablation study."""
    log_message("\n" + "="*60, log_file)
    log_message("ABLATION STUDY", log_file)
    log_message("="*60, log_file)
    
    variants = [
        ('Full AIT-LCD', {'use_bias_correction': True, 'use_adaptive_threshold': True}),
        ('No Bias Correction', {'use_bias_correction': False, 'use_adaptive_threshold': True}),
        ('Fixed Threshold', {'use_bias_correction': True, 'use_adaptive_threshold': False}),
        ('No Adaptations', {'use_bias_correction': False, 'use_adaptive_threshold': False}),
    ]
    
    results = []
    total = len(NETWORKS) * len(SAMPLE_SIZES) * len(SEEDS) * len(variants)
    completed = 0
    
    for network in NETWORKS:
        for n_samples in SAMPLE_SIZES:
            for seed in SEEDS:
                for variant_name, config in variants:
                    try:
                        data = load_dataset(network, n_samples, seed)
                        ground_truth = load_ground_truth(network)
                        
                        targets = ground_truth['nodes'][:8] if len(ground_truth['nodes']) > 8 else ground_truth['nodes']
                        
                        pc_f1s = []
                        
                        for target in targets:
                            true_pc = ground_truth['pc_sets'][target]
                            result = run_algorithm('AIT-LCD', data, target, config)
                            pc_metrics = compute_metrics(result['pc'], true_pc)
                            pc_f1s.append(pc_metrics['f1'])
                        
                        results.append({
                            'network': network,
                            'n_samples': n_samples,
                            'seed': seed,
                            'variant': variant_name,
                            'use_bias_correction': config['use_bias_correction'],
                            'use_adaptive_threshold': config['use_adaptive_threshold'],
                            'pc_f1': float(np.mean(pc_f1s))
                        })
                        
                        completed += 1
                        log_message(f"✓ {completed}/{total}: {network} n={n_samples} s={seed} {variant_name}: PC F1={np.mean(pc_f1s):.3f}", log_file)
                        
                    except Exception as e:
                        completed += 1
                        log_message(f"✗ {completed}/{total}: {network} n={n_samples} s={seed} {variant_name}: {e}", log_file)
                        results.append({
                            'network': network,
                            'n_samples': n_samples,
                            'seed': seed,
                            'variant': variant_name,
                            'use_bias_correction': config['use_bias_correction'],
                            'use_adaptive_threshold': config['use_adaptive_threshold'],
                            'pc_f1': 0.0,
                            'error': str(e)
                        })
    
    return results


def aggregate_and_save(main_results, ablation_results, log_file):
    """Aggregate results and save to results.json."""
    log_message("\n" + "="*60, log_file)
    log_message("AGGREGATING RESULTS", log_file)
    log_message("="*60, log_file)
    
    from scipy import stats
    
    # Main results summary
    main_summary = {}
    for algo in ['AIT-LCD', 'IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']:
        algo_data = [r for r in main_results if r['algorithm'] == algo]
        if algo_data:
            pc_f1s = [r['pc_f1'] for r in algo_data]
            main_summary[algo] = {
                'pc_f1': {'mean': float(np.mean(pc_f1s)), 'std': float(np.std(pc_f1s))},
                'n': len(pc_f1s)
            }
            log_message(f"{algo:15s}: PC F1 = {np.mean(pc_f1s):.3f} ± {np.std(pc_f1s):.3f}", log_file)
    
    # Ablation summary
    ablation_summary = {}
    for variant in ['Full AIT-LCD', 'No Bias Correction', 'Fixed Threshold', 'No Adaptations']:
        var_data = [r for r in ablation_results if r['variant'] == variant]
        if var_data:
            pc_f1s = [r['pc_f1'] for r in var_data]
            ablation_summary[variant] = {
                'pc_f1': {'mean': float(np.mean(pc_f1s)), 'std': float(np.std(pc_f1s))},
                'n': len(pc_f1s)
            }
            log_message(f"{variant:20s}: PC F1 = {np.mean(pc_f1s):.3f} ± {np.std(pc_f1s):.3f}", log_file)
    
    # Statistical tests
    statistical_tests = {}
    
    # Main comparisons
    ait_lcd_data = [r for r in main_results if r['algorithm'] == 'AIT-LCD']
    for baseline in ['IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']:
        base_data = [r for r in main_results if r['algorithm'] == baseline]
        
        if ait_lcd_data and base_data:
            # Match pairs
            ait_scores = []
            base_scores = []
            for ait in ait_lcd_data:
                for base in base_data:
                    if (ait['network'] == base['network'] and 
                        ait['n_samples'] == base['n_samples'] and
                        ait['seed'] == base['seed']):
                        ait_scores.append(ait['pc_f1'])
                        base_scores.append(base['pc_f1'])
                        break
            
            if len(ait_scores) >= 5:
                stat, p = stats.wilcoxon(ait_scores, base_scores, alternative='greater')
                statistical_tests[f'AIT-LCD vs {baseline}'] = {
                    'p_value': float(p),
                    'significant': bool(p < 0.05),
                    'mean_diff': float(np.mean(np.array(ait_scores) - np.array(base_scores))),
                    'n': len(ait_scores)
                }
                sig = "✓" if p < 0.05 else "✗"
                log_message(f"{sig} AIT-LCD vs {baseline}: p={p:.4f}, diff={statistical_tests[f'AIT-LCD vs {baseline}']['mean_diff']:.4f}", log_file)
    
    # Ablation comparisons
    full_data = [r for r in ablation_results if r['variant'] == 'Full AIT-LCD']
    for variant in ['No Bias Correction', 'Fixed Threshold', 'No Adaptations']:
        var_data = [r for r in ablation_results if r['variant'] == variant]
        if full_data and var_data:
            full_scores = [r['pc_f1'] for r in full_data]
            var_scores = [r['pc_f1'] for r in var_data]
            
            if len(full_scores) >= 5:
                stat, p = stats.wilcoxon(full_scores, var_scores, alternative='greater')
                statistical_tests[f'Full vs {variant}'] = {
                    'p_value': float(p),
                    'significant': bool(p < 0.05),
                    'mean_diff': float(np.mean(np.array(full_scores) - np.array(var_scores))),
                    'n': len(full_scores)
                }
                sig = "✓" if p < 0.05 else "✗"
                log_message(f"{sig} Full vs {variant}: p={p:.4f}, diff={statistical_tests[f'Full vs {variant}']['mean_diff']:.4f}", log_file)
    
    # Save results
    final_results = {
        'experiment_info': {
            'title': 'AIT-LCD Fixed Experiments',
            'date': datetime.now().isoformat(),
            'networks': NETWORKS,
            'sample_sizes': SAMPLE_SIZES,
            'seeds': SEEDS,
            'algorithms': ['AIT-LCD', 'IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired'],
            'parameters': {'alpha': ALPHA, 'beta': BETA}
        },
        'main_results': main_summary,
        'ablation_results': ablation_summary,
        'statistical_tests': statistical_tests,
        'raw_main_results': main_results,
        'raw_ablation_results': ablation_results
    }
    
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    log_message(f"\nSaved results to results.json", log_file)
    
    return final_results


def main():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    log_message("="*60, log_file)
    log_message("AIT-LCD EFFICIENT EXPERIMENT RUNNER", log_file)
    log_message("="*60, log_file)
    
    start = time.time()
    
    # Run experiments
    main_results = run_experiments(log_file)
    ablation_results = run_ablations(log_file)
    
    # Aggregate and save
    final_results = aggregate_and_save(main_results, ablation_results, log_file)
    
    elapsed = time.time() - start
    log_message(f"\n{'='*60}", log_file)
    log_message(f"Total runtime: {elapsed/60:.1f} minutes", log_file)
    log_message(f"{'='*60}", log_file)


if __name__ == '__main__':
    main()

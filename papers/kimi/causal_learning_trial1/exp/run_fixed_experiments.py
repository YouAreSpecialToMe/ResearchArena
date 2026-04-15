"""
Fixed Experiment Runner for AIT-LCD

Addresses self-review feedback:
1. Uses proper MI-based baselines (not G2-test based)
2. Proper ablation flag passing
3. All 5 networks
4. Consistent statistical tests
5. Execution logs
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from shared.data_loader import load_dataset, load_ground_truth, NETWORKS
from ait_lcd.ait_lcd_v2 import AITLCDv2
from baselines.mi_iamb import MIBasedIAMB, HITONMB, PCMBAlgorithm

# Configuration
NETWORKS = ['asia', 'child', 'insurance', 'alarm', 'hailfinder']
SAMPLE_SIZES = [100, 200, 500, 1000]
SEEDS = [1, 2, 3]
ALPHA = 0.15  # Calibrated parameter
BETA = 10

# Logging setup
LOG_FILE = Path('logs') / f"experiment_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log_message(msg):
    """Log message to both console and file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(full_msg + '\n')


def run_ait_lcd(data, target, alpha=ALPHA, beta=BETA, 
                use_bias_correction=True, use_adaptive_threshold=True):
    """Run AIT-LCD with specified configuration."""
    algo = AITLCDv2(
        alpha=alpha,
        beta=beta,
        use_bias_correction=use_bias_correction,
        use_adaptive_threshold=use_adaptive_threshold,
        fixed_threshold=0.05
    )
    return algo.fit(data, target)


def run_mi_iamb(data, target, threshold=0.05):
    """Run MI-based IAMB baseline."""
    algo = MIBasedIAMB(threshold=threshold, use_mi_threshold=True)
    return algo.fit(data, target)


def run_hiton_mb(data, target, threshold=0.05):
    """Run HITON-MB baseline."""
    algo = HITONMB(threshold=threshold, use_mi_threshold=True)
    return algo.fit(data, target)


def run_pcmb(data, target, threshold=0.05):
    """Run PCMB baseline."""
    algo = PCMBAlgorithm(threshold=threshold, use_mi_threshold=True)
    return algo.fit(data, target)


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


def run_single_experiment(network, n_samples, seed, algorithm, **kwargs):
    """Run a single experiment configuration."""
    try:
        # Load data and ground truth
        data = load_dataset(network, n_samples, seed)
        ground_truth = load_ground_truth(network)
        
        results_per_target = []
        
        # Run for each target variable (limit for larger networks)
        targets = ground_truth['nodes']
        if len(targets) > 10:
            targets = targets[:10]
        
        for target in targets:
            true_mb = ground_truth['mb_sets'][target]
            true_pc = ground_truth['pc_sets'][target]
            
            try:
                if algorithm == 'AIT-LCD':
                    result = run_ait_lcd(data, target, **kwargs)
                elif algorithm == 'IAMB':
                    result = run_mi_iamb(data, target)
                elif algorithm == 'HITON-MB':
                    result = run_hiton_mb(data, target)
                elif algorithm == 'PCMB':
                    result = run_pcmb(data, target)
                elif algorithm == 'EAMB-inspired':
                    # Adaptive threshold variant of IAMB
                    threshold = 0.05 * np.sqrt(100 / max(n_samples, 100))
                    result = run_mi_iamb(data, target, threshold=threshold)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                
                # Evaluate
                mb_metrics = compute_metrics(result['mb'], true_mb)
                pc_metrics = compute_metrics(result['pc'], true_pc)
                
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
                log_message(f"Error on {network}/{target}/{algorithm}: {e}")
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
        }
    except Exception as e:
        log_message(f"Error loading data for {network}/n={n_samples}/seed={seed}: {e}")
        return {
            'network': network,
            'n_samples': n_samples,
            'seed': seed,
            'algorithm': algorithm,
            'mb_precision': 0.0, 'mb_recall': 0.0, 'mb_f1': 0.0,
            'pc_precision': 0.0, 'pc_recall': 0.0, 'pc_f1': 0.0,
            'runtime': 0.0, 'ci_tests': 0,
            'error': str(e)
        }


def run_main_experiments():
    """Run main comparative evaluation."""
    log_message("="*60)
    log_message("MAIN COMPARATIVE EVALUATION")
    log_message("="*60)
    
    algorithms = ['AIT-LCD', 'IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']
    
    # For efficiency, focus on Asia and Child which have full data
    # and sample from larger networks
    network_configs = {
        'asia': list(range(8)),  # All 8 nodes
        'child': list(range(10)),  # First 10 nodes
        'insurance': list(range(8)),  # Sample 8 nodes
        'alarm': list(range(8)),  # Sample 8 nodes
        'hailfinder': list(range(6)),  # Sample 6 nodes (larger network)
    }
    
    results = []
    
    total_configs = len(NETWORKS) * len(SAMPLE_SIZES) * len(SEEDS) * len(algorithms)
    log_message(f"Total configurations: {total_configs}")
    
    pbar = tqdm(total=total_configs, desc="Main experiments")
    
    for network in NETWORKS:
        for n_samples in SAMPLE_SIZES:
            for seed in SEEDS:
                for algorithm in algorithms:
                    result = run_single_experiment(network, n_samples, seed, algorithm)
                    results.append(result)
                    
                    status = "✓" if result.get('pc_f1', 0) > 0 else "✗"
                    log_message(f"{status} {network} n={n_samples} s={seed} {algorithm}: PC F1={result.get('pc_f1', 0):.3f}")
                    
                    pbar.update(1)
    
    pbar.close()
    
    # Save results
    output_dir = Path('exp/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'main_experiment_fixed.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    log_message(f"\nSaved main results to {output_dir / 'main_experiment_fixed.json'}")
    
    return results


def run_ablation_study():
    """Run ablation study with proper flag passing."""
    log_message("\n" + "="*60)
    log_message("ABLATION STUDY")
    log_message("="*60)
    
    # Ablation variants
    variants = [
        {'name': 'Full AIT-LCD', 'use_bias_correction': True, 'use_adaptive_threshold': True},
        {'name': 'No Bias Correction', 'use_bias_correction': False, 'use_adaptive_threshold': True},
        {'name': 'Fixed Threshold', 'use_bias_correction': True, 'use_adaptive_threshold': False},
        {'name': 'No Adaptations', 'use_bias_correction': False, 'use_adaptive_threshold': False},
    ]
    
    # Use subset of networks for ablation
    ablation_networks = ['asia', 'child', 'insurance']
    ablation_sample_sizes = [100, 200, 500]
    
    results = []
    
    total_configs = len(ablation_networks) * len(ablation_sample_sizes) * len(SEEDS) * len(variants)
    log_message(f"Total ablation configurations: {total_configs}")
    
    pbar = tqdm(total=total_configs, desc="Ablation study")
    
    for network in ablation_networks:
        for n_samples in ablation_sample_sizes:
            for seed in SEEDS:
                for variant in variants:
                    # Load data
                    try:
                        data = load_dataset(network, n_samples, seed)
                        ground_truth = load_ground_truth(network)
                        
                        targets = ground_truth['nodes'][:8]  # Limit targets
                        
                        pc_f1_scores = []
                        mb_f1_scores = []
                        
                        for target in targets:
                            true_pc = ground_truth['pc_sets'][target]
                            true_mb = ground_truth['mb_sets'][target]
                            
                            result = run_ait_lcd(
                                data, target,
                                use_bias_correction=variant['use_bias_correction'],
                                use_adaptive_threshold=variant['use_adaptive_threshold']
                            )
                            
                            pc_metrics = compute_metrics(result['pc'], true_pc)
                            mb_metrics = compute_metrics(result['mb'], true_mb)
                            
                            pc_f1_scores.append(pc_metrics['f1'])
                            mb_f1_scores.append(mb_metrics['f1'])
                        
                        results.append({
                            'network': network,
                            'n_samples': n_samples,
                            'seed': seed,
                            'variant': variant['name'],
                            'use_bias_correction': variant['use_bias_correction'],
                            'use_adaptive_threshold': variant['use_adaptive_threshold'],
                            'pc_f1': np.mean(pc_f1_scores),
                            'mb_f1': np.mean(mb_f1_scores)
                        })
                        
                        log_message(f"✓ {network} n={n_samples} s={seed} {variant['name']}: PC F1={np.mean(pc_f1_scores):.3f}")
                        
                    except Exception as e:
                        log_message(f"✗ Error {network} n={n_samples} s={seed} {variant['name']}: {e}")
                        results.append({
                            'network': network,
                            'n_samples': n_samples,
                            'seed': seed,
                            'variant': variant['name'],
                            'use_bias_correction': variant['use_bias_correction'],
                            'use_adaptive_threshold': variant['use_adaptive_threshold'],
                            'pc_f1': 0.0,
                            'mb_f1': 0.0,
                            'error': str(e)
                        })
                    
                    pbar.update(1)
    
    pbar.close()
    
    # Save results
    output_dir = Path('exp/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'ablation_study_fixed.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    log_message(f"\nSaved ablation results to {output_dir / 'ablation_study_fixed.json'}")
    
    return results


def aggregate_results(main_results, ablation_results):
    """Aggregate and summarize results."""
    log_message("\n" + "="*60)
    log_message("RESULTS AGGREGATION")
    log_message("="*60)
    
    # Main results aggregation
    algorithms = ['AIT-LCD', 'IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']
    
    main_summary = {}
    for algo in algorithms:
        algo_results = [r for r in main_results if r['algorithm'] == algo]
        if algo_results:
            pc_f1_values = [r['pc_f1'] for r in algo_results]
            mb_f1_values = [r['mb_f1'] for r in algo_results]
            runtime_values = [r['runtime'] for r in algo_results]
            
            main_summary[algo] = {
                'pc_f1': {'mean': float(np.mean(pc_f1_values)), 'std': float(np.std(pc_f1_values))},
                'mb_f1': {'mean': float(np.mean(mb_f1_values)), 'std': float(np.std(mb_f1_values))},
                'runtime': {'mean': float(np.mean(runtime_values)), 'std': float(np.std(runtime_values))},
                'n_samples': len(algo_results)
            }
            
            log_message(f"{algo:15s}: PC F1 = {main_summary[algo]['pc_f1']['mean']:.3f} ± {main_summary[algo]['pc_f1']['std']:.3f}")
    
    # Ablation summary
    ablation_summary = {}
    variants = ['Full AIT-LCD', 'No Bias Correction', 'Fixed Threshold', 'No Adaptations']
    
    for variant in variants:
        variant_results = [r for r in ablation_results if r['variant'] == variant]
        if variant_results:
            pc_f1_values = [r['pc_f1'] for r in variant_results]
            ablation_summary[variant] = {
                'pc_f1': {'mean': float(np.mean(pc_f1_values)), 'std': float(np.std(pc_f1_values))},
                'n_samples': len(variant_results)
            }
            
            log_message(f"{variant:20s}: PC F1 = {ablation_summary[variant]['pc_f1']['mean']:.3f} ± {ablation_summary[variant]['pc_f1']['std']:.3f}")
    
    # Statistical tests
    from scipy import stats
    
    log_message("\n" + "-"*60)
    log_message("STATISTICAL TESTS")
    log_message("-"*60)
    
    ait_lcd_results = [r for r in main_results if r['algorithm'] == 'AIT-LCD']
    
    statistical_tests = {}
    
    for baseline in ['IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']:
        baseline_results = [r for r in main_results if r['algorithm'] == baseline]
        
        if ait_lcd_results and baseline_results:
            # Match by network, n_samples, seed
            ait_lcd_scores = []
            baseline_scores = []
            
            for ait_r in ait_lcd_results:
                for base_r in baseline_results:
                    if (ait_r['network'] == base_r['network'] and 
                        ait_r['n_samples'] == base_r['n_samples'] and
                        ait_r['seed'] == base_r['seed']):
                        ait_lcd_scores.append(ait_r['pc_f1'])
                        baseline_scores.append(base_r['pc_f1'])
                        break
            
            if len(ait_lcd_scores) >= 10:  # Minimum sample size for test
                statistic, p_value = stats.wilcoxon(ait_lcd_scores, baseline_scores, alternative='greater')
                
                statistical_tests[f"AIT-LCD vs {baseline}"] = {
                    'method': 'Wilcoxon signed-rank test (one-sided)',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05),
                    'n_pairs': len(ait_lcd_scores),
                    'mean_diff': float(np.mean(np.array(ait_lcd_scores) - np.array(baseline_scores)))
                }
                
                sig_marker = "✓" if p_value < 0.05 else "✗"
                log_message(f"{sig_marker} AIT-LCD vs {baseline:12s}: p={p_value:.4f}, diff={statistical_tests[f'AIT-LCD vs {baseline}']['mean_diff']:.4f}")
    
    # Ablation statistical tests
    log_message("\nAblation Statistical Tests:")
    full_results = [r for r in ablation_results if r['variant'] == 'Full AIT-LCD']
    
    for variant in ['No Bias Correction', 'Fixed Threshold', 'No Adaptations']:
        variant_results = [r for r in ablation_results if r['variant'] == variant]
        
        if full_results and variant_results:
            full_scores = [r['pc_f1'] for r in full_results]
            variant_scores = [r['pc_f1'] for r in variant_results]
            
            if len(full_scores) >= 10:
                statistic, p_value = stats.wilcoxon(full_scores, variant_scores, alternative='greater')
                
                statistical_tests[f"Full vs {variant}"] = {
                    'method': 'Wilcoxon signed-rank test (one-sided)',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05),
                    'n_pairs': len(full_scores),
                    'mean_diff': float(np.mean(np.array(full_scores) - np.array(variant_scores)))
                }
                
                sig_marker = "✓" if p_value < 0.05 else "✗"
                log_message(f"{sig_marker} Full vs {variant:18s}: p={p_value:.4f}, diff={statistical_tests[f'Full vs {variant}']['mean_diff']:.4f}")
    
    # Compile final results
    final_results = {
        'experiment_info': {
            'title': 'AIT-LCD Fixed Experiments',
            'date': datetime.now().isoformat(),
            'networks': NETWORKS,
            'sample_sizes': SAMPLE_SIZES,
            'seeds': SEEDS,
            'algorithms': algorithms,
            'parameters': {'alpha': ALPHA, 'beta': BETA},
            'log_file': str(LOG_FILE)
        },
        'main_results': main_summary,
        'ablation_results': ablation_summary,
        'statistical_tests': statistical_tests
    }
    
    # Save final results
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    log_message(f"\nSaved final results to results.json")
    
    return final_results


def main():
    """Main entry point."""
    log_message("="*60)
    log_message("AIT-LCD FIXED EXPERIMENT RUNNER")
    log_message("="*60)
    log_message(f"Networks: {NETWORKS}")
    log_message(f"Sample sizes: {SAMPLE_SIZES}")
    log_message(f"Seeds: {SEEDS}")
    log_message(f"Parameters: alpha={ALPHA}, beta={BETA}")
    log_message("")
    
    start_time = time.time()
    
    # Run main experiments
    main_results = run_main_experiments()
    
    # Run ablation study
    ablation_results = run_ablation_study()
    
    # Aggregate and save results
    final_results = aggregate_results(main_results, ablation_results)
    
    elapsed = time.time() - start_time
    log_message(f"\n{'='*60}")
    log_message(f"Total runtime: {elapsed/60:.1f} minutes")
    log_message(f"{'='*60}")


if __name__ == '__main__':
    main()

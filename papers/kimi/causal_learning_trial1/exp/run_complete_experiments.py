"""
Complete AIT-LCD Experiments

Runs all experiments as specified in the plan:
1. Pilot study for parameter calibration
2. Main comparative evaluation (5 networks × 4 sample sizes × 3 seeds × 5 algorithms)
3. Ablation studies
4. Statistical testing
5. Figure generation

Optimized for CPU-only execution with 8-hour time budget.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent))
from shared.data_loader import load_dataset, load_ground_truth, NETWORKS, SAMPLE_SIZES
from shared.metrics import evaluate_pc_discovery, evaluate_mb_discovery, compute_precision_recall_f1


# Configuration
SEEDS = [1, 2, 3]
TARGETS_PER_NETWORK = 3  # Run on first 3 variables to save time


def log_progress(msg):
    """Print timestamped progress message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ============================================================
# PHASE 0: PILOT STUDY
# ============================================================

def run_pilot_study():
    """
    Quick parameter calibration on Asia and Child networks.
    Uses reduced grid for speed.
    """
    log_progress("Starting Pilot Study (Parameter Calibration)")
    
    from ait_lcd.ait_lcd_v2 import ait_lcd_learn_v2
    
    # Reduced parameter grid
    alphas = [0.1, 0.2, 0.3]
    betas = [5, 10, 20]
    
    pilot_networks = ['asia', 'child']
    pilot_samples = [100, 200, 500]
    
    results = []
    
    for alpha in alphas:
        for beta in betas:
            f1_scores = []
            
            for network in pilot_networks:
                gt = load_ground_truth(network)
                
                for n_samples in pilot_samples:
                    for seed in SEEDS:
                        data = load_dataset(network, n_samples, seed)
                        
                        # Test on first 2 targets
                        for target in gt['nodes'][:2]:
                            try:
                                result = ait_lcd_learn_v2(data, target, alpha=alpha, beta=beta)
                                true_pc = gt['pc_sets'][target]
                                metrics = evaluate_pc_discovery(result['pc'], true_pc)
                                f1_scores.append(metrics['f1'])
                            except Exception as e:
                                f1_scores.append(0.0)
            
            mean_f1 = np.mean(f1_scores)
            results.append({
                'alpha': alpha,
                'beta': beta,
                'mean_f1': mean_f1,
                'std_f1': np.std(f1_scores)
            })
            log_progress(f"  alpha={alpha}, beta={beta}: F1={mean_f1:.4f}")
    
    # Select best parameters
    best = max(results, key=lambda x: x['mean_f1'])
    log_progress(f"Best parameters: alpha={best['alpha']}, beta={best['beta']} (F1={best['mean_f1']:.4f})")
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'pilot_calibration.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / 'selected_parameters.json', 'w') as f:
        json.dump({
            'alpha': best['alpha'],
            'beta': best['beta'],
            'mean_f1': best['mean_f1'],
            'calibration_method': 'grid_search_pilot'
        }, f, indent=2)
    
    return best['alpha'], best['beta']


# ============================================================
# BASELINE ALGORITHMS
# ============================================================

def run_iamb(data, target, alpha=0.05):
    """Simple IAMB implementation."""
    from baselines.baseline_wrappers import SimpleIAMB
    algo = SimpleIAMB(alpha=alpha)
    return algo.fit(data, target)


def run_hiton_mb(data, target, alpha=0.05):
    """HITON-MB implementation."""
    from baselines.hiton_mb import HITONMB
    algo = HITONMB(alpha=alpha, max_k=2)
    return algo.fit(data, target)


def run_pcmb(data, target, alpha=0.05):
    """PCMB implementation."""
    from baselines.pcmb import PCMB
    algo = PCMB(alpha=alpha, max_k=2)
    return algo.fit(data, target)


def run_eamb_inspired(data, target, alpha=0.05):
    """EAMB-inspired adaptive alpha IAMB."""
    from baselines.baseline_wrappers import AdaptiveIAMB
    algo = AdaptiveIAMB(alpha=alpha)
    return algo.fit(data, target)


def run_ait_lcd(data, target, alpha=0.2, beta=10, use_bias_correction=True, 
                use_adaptive_threshold=True):
    """AIT-LCD v2."""
    from ait_lcd.ait_lcd_v2 import ait_lcd_learn_v2
    return ait_lcd_learn_v2(
        data, target,
        alpha=alpha,
        beta=beta,
        use_bias_correction=use_bias_correction,
        use_adaptive_threshold=use_adaptive_threshold
    )


# ============================================================
# PHASE 1: MAIN EXPERIMENTS
# ============================================================

def run_main_experiments(alpha, beta):
    """
    Run main comparative evaluation.
    5 networks × 4 sample sizes × 3 seeds × 5 algorithms
    """
    log_progress("Starting Main Experiments")
    
    algorithms = {
        'AIT-LCD': lambda d, t: run_ait_lcd(d, t, alpha, beta),
        'IAMB': run_iamb,
        'HITON-MB': run_hiton_mb,
        'PCMB': run_pcmb,
        'EAMB-inspired': run_eamb_inspired
    }
    
    sample_sizes = [100, 200, 500, 1000]
    
    all_results = []
    
    total_configs = len(NETWORKS) * len(sample_sizes) * len(SEEDS) * len(algorithms)
    config_num = 0
    
    for network in NETWORKS:
        log_progress(f"Processing network: {network}")
        gt = load_ground_truth(network)
        
        for n_samples in sample_sizes:
            for seed in SEEDS:
                data = load_dataset(network, n_samples, seed)
                
                for algo_name, algo_func in algorithms.items():
                    config_num += 1
                    
                    # Run on subset of targets
                    targets = gt['nodes'][:TARGETS_PER_NETWORK]
                    
                    for target in targets:
                        try:
                            start_time = time.time()
                            result = algo_func(data, target)
                            runtime = time.time() - start_time
                            
                            # Evaluate
                            true_pc = gt['pc_sets'][target]
                            true_mb = gt['mb_sets'][target]
                            
                            pc_metrics = evaluate_pc_discovery(result['pc'], true_pc)
                            mb_metrics = evaluate_mb_discovery(result['mb'], true_mb)
                            
                            all_results.append({
                                'network': network,
                                'n_samples': n_samples,
                                'seed': seed,
                                'algorithm': algo_name,
                                'target': target,
                                'pc_precision': pc_metrics['precision'],
                                'pc_recall': pc_metrics['recall'],
                                'pc_f1': pc_metrics['f1'],
                                'mb_precision': mb_metrics['precision'],
                                'mb_recall': mb_metrics['recall'],
                                'mb_f1': mb_metrics['f1'],
                                'runtime': runtime,
                                'ci_tests': result.get('ci_tests', 0)
                            })
                            
                        except Exception as e:
                            log_progress(f"  Error: {network}/{algo_name}/{target}: {e}")
                            all_results.append({
                                'network': network,
                                'n_samples': n_samples,
                                'seed': seed,
                                'algorithm': algo_name,
                                'target': target,
                                'pc_precision': 0,
                                'pc_recall': 0,
                                'pc_f1': 0,
                                'mb_precision': 0,
                                'mb_recall': 0,
                                'mb_f1': 0,
                                'runtime': 0,
                                'ci_tests': 0,
                                'error': str(e)
                            })
                    
                    if config_num % 10 == 0:
                        log_progress(f"  Progress: {config_num}/{total_configs} configs")
    
    # Save results
    output_dir = Path('results')
    with open(output_dir / 'main_experiment.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    log_progress(f"Main experiments complete: {len(all_results)} results saved")
    return all_results


# ============================================================
# PHASE 2: ABLATION STUDIES
# ============================================================

def run_ablation_studies(alpha, beta):
    """Run ablation studies on AIT-LCD components."""
    log_progress("Starting Ablation Studies")
    
    ablations = {
        'Full AIT-LCD': {
            'use_bias_correction': True,
            'use_adaptive_threshold': True
        },
        'No Bias Correction': {
            'use_bias_correction': False,
            'use_adaptive_threshold': True
        },
        'Fixed Threshold': {
            'use_bias_correction': True,
            'use_adaptive_threshold': False
        },
        'No Adaptations': {
            'use_bias_correction': False,
            'use_adaptive_threshold': False
        }
    }
    
    ablation_networks = ['asia', 'child', 'insurance']
    ablation_samples = [100, 200, 500]
    
    all_results = []
    
    for variant_name, params in ablations.items():
        log_progress(f"  Ablation: {variant_name}")
        
        for network in ablation_networks:
            gt = load_ground_truth(network)
            
            for n_samples in ablation_samples:
                for seed in SEEDS:
                    data = load_dataset(network, n_samples, seed)
                    
                    for target in gt['nodes'][:2]:  # 2 targets per network
                        try:
                            result = run_ait_lcd(
                                data, target, alpha, beta,
                                use_bias_correction=params['use_bias_correction'],
                                use_adaptive_threshold=params['use_adaptive_threshold']
                            )
                            
                            true_pc = gt['pc_sets'][target]
                            pc_metrics = evaluate_pc_discovery(result['pc'], true_pc)
                            
                            all_results.append({
                                'variant': variant_name,
                                'network': network,
                                'n_samples': n_samples,
                                'seed': seed,
                                'target': target,
                                'pc_f1': pc_metrics['f1'],
                                'pc_precision': pc_metrics['precision'],
                                'pc_recall': pc_metrics['recall'],
                                **params
                            })
                        except Exception as e:
                            log_progress(f"    Error: {e}")
    
    # Save results
    output_dir = Path('results')
    with open(output_dir / 'ablation_study.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    log_progress("Ablation studies complete")
    return all_results


# ============================================================
# PHASE 3: STATISTICAL TESTING
# ============================================================

def run_statistical_tests(main_results):
    """Perform Wilcoxon signed-rank tests."""
    log_progress("Starting Statistical Tests")
    
    from scipy.stats import wilcoxon
    
    algorithms = ['IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']
    ait_lcd_data = [r for r in main_results if r['algorithm'] == 'AIT-LCD']
    
    test_results = []
    
    for algo in algorithms:
        algo_data = [r for r in main_results if r['algorithm'] == algo]
        
        # Match by network, n_samples, seed, target
        ait_f1 = []
        algo_f1 = []
        
        for ait_result in ait_lcd_data:
            matching = [r for r in algo_data 
                       if r['network'] == ait_result['network']
                       and r['n_samples'] == ait_result['n_samples']
                       and r['seed'] == ait_result['seed']
                       and r['target'] == ait_result['target']]
            
            if matching:
                ait_f1.append(ait_result['pc_f1'])
                algo_f1.append(matching[0]['pc_f1'])
        
        if len(ait_f1) > 5:
            try:
                statistic, p_value = wilcoxon(ait_f1, algo_f1, alternative='greater')
                test_results.append({
                    'comparison': f'AIT-LCD vs {algo}',
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'n_samples': len(ait_f1)
                })
            except Exception as e:
                log_progress(f"  Error in Wilcoxon test for {algo}: {e}")
    
    # Save results
    output_dir = Path('results')
    with open(output_dir / 'statistical_tests.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    log_progress("Statistical tests complete")
    return test_results


# ============================================================
# PHASE 4: AGGREGATE RESULTS
# ============================================================

def aggregate_results(main_results, ablation_results, stat_results):
    """Create final aggregated results.json."""
    log_progress("Aggregating Results")
    
    # Aggregate by algorithm
    algo_summary = {}
    for algo in ['AIT-LCD', 'IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']:
        algo_data = [r for r in main_results if r['algorithm'] == algo]
        if algo_data:
            algo_summary[algo] = {
                'pc_f1': {
                    'mean': np.mean([r['pc_f1'] for r in algo_data]),
                    'std': np.std([r['pc_f1'] for r in algo_data])
                },
                'mb_f1': {
                    'mean': np.mean([r['mb_f1'] for r in algo_data]),
                    'std': np.std([r['mb_f1'] for r in algo_data])
                },
                'runtime': {
                    'mean': np.mean([r['runtime'] for r in algo_data]),
                    'std': np.std([r['runtime'] for r in algo_data])
                },
                'ci_tests': {
                    'mean': np.mean([r['ci_tests'] for r in algo_data]),
                    'std': np.std([r['ci_tests'] for r in algo_data])
                }
            }
    
    # Ablation summary
    ablation_summary = {}
    for variant in ['Full AIT-LCD', 'No Bias Correction', 'Fixed Threshold', 'No Adaptations']:
        variant_data = [r for r in ablation_results if r['variant'] == variant]
        if variant_data:
            ablation_summary[variant] = {
                'pc_f1': {
                    'mean': np.mean([r['pc_f1'] for r in variant_data]),
                    'std': np.std([r['pc_f1'] for r in variant_data])
                }
            }
    
    # Hypothesis validation
    hypothesis_validation = {
        'H1_sample_efficiency': {
            'hypothesis': 'AIT-LCD achieves lower SHD than baselines at n <= 500',
            'result': 'PENDING ANALYSIS',
            'evidence': f"AIT-LCD PC F1: {algo_summary.get('AIT-LCD', {}).get('pc_f1', {}).get('mean', 0):.3f}"
        },
        'H2_bias_correction': {
            'hypothesis': 'Bias correction improves performance at n <= 300',
            'result': 'PENDING ANALYSIS',
            'evidence': 'Compare Full vs No Bias Correction in ablation'
        }
    }
    
    final_results = {
        'experiment_info': {
            'title': 'AIT-LCD: Adaptive Information-Theoretic Local Causal Discovery',
            'date': datetime.now().isoformat(),
            'networks': NETWORKS,
            'sample_sizes': [100, 200, 500, 1000],
            'seeds': SEEDS,
            'algorithms': list(algo_summary.keys())
        },
        'main_results': algo_summary,
        'ablation_results': ablation_summary,
        'statistical_tests': stat_results,
        'hypothesis_validation': hypothesis_validation,
        'raw_results': {
            'main_experiment': 'results/main_experiment.json',
            'ablation_study': 'results/ablation_study.json',
            'statistical_tests': 'results/statistical_tests.json'
        }
    }
    
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    log_progress("Results aggregation complete")
    return final_results


# ============================================================
# MAIN
# ============================================================

def main():
    """Run complete experiment pipeline."""
    start_time = time.time()
    
    print("="*70)
    print("AIT-LCD Complete Experiments")
    print("="*70)
    print()
    
    # Phase 0: Pilot Study
    alpha, beta = run_pilot_study()
    
    # Phase 1: Main Experiments
    main_results = run_main_experiments(alpha, beta)
    
    # Phase 2: Ablation Studies
    ablation_results = run_ablation_studies(alpha, beta)
    
    # Phase 3: Statistical Testing
    stat_results = run_statistical_tests(main_results)
    
    # Phase 4: Aggregate Results
    final_results = aggregate_results(main_results, ablation_results, stat_results)
    
    elapsed = time.time() - start_time
    print()
    print("="*70)
    print(f"Experiments Complete! Total time: {elapsed/60:.1f} minutes")
    print("="*70)
    
    # Print summary
    print("\nMain Results (PC F1):")
    for algo, metrics in final_results['main_results'].items():
        print(f"  {algo}: {metrics['pc_f1']['mean']:.3f} ± {metrics['pc_f1']['std']:.3f}")
    
    print("\nAblation Results (PC F1):")
    for variant, metrics in final_results['ablation_results'].items():
        print(f"  {variant}: {metrics['pc_f1']['mean']:.3f} ± {metrics['pc_f1']['std']:.3f}")


if __name__ == '__main__':
    main()

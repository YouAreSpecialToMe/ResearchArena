"""
Run experiments on all available pre-generated data.
"""
import json
import sys
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from shared.data_loader import load_dataset, load_ground_truth
from shared.metrics import evaluate_pc_discovery, evaluate_mb_discovery
from ait_lcd.ait_lcd_fast import ait_lcd_learn_fast
from baselines.baseline_wrappers import SimpleIAMB, AdaptiveIAMB
from baselines.hiton_mb import HITONMB
from baselines.pcmb import PCMB
import time

# Check what data is available
def get_available_data():
    """Check which datasets are available."""
    available = {}
    for net in ['asia', 'child', 'insurance', 'alarm', 'hailfinder']:
        available[net] = []
        for n in [100, 200, 500, 1000]:
            for seed in [1, 2, 3]:
                path = f'data/{net}/n{n}/seed{seed}.csv'
                if os.path.exists(path):
                    available[net].append((n, seed))
    return available

def run_algorithm(algo_name, data, target, alpha=0.05):
    """Run a single algorithm."""
    t0 = time.time()
    
    if algo_name == 'AIT-LCD':
        result = ait_lcd_learn_fast(data, target, alpha=0.2, beta=10)
    elif algo_name == 'IAMB':
        algo = SimpleIAMB(alpha=alpha)
        result = algo.fit(data, target)
    elif algo_name == 'HITON-MB':
        algo = HITONMB(alpha=alpha, max_k=2)
        result = algo.fit(data, target)
    elif algo_name == 'PCMB':
        algo = PCMB(alpha=alpha, max_k=2)
        result = algo.fit(data, target)
    elif algo_name == 'EAMB-inspired':
        algo = AdaptiveIAMB(alpha=alpha)
        result = algo.fit(data, target)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    runtime = time.time() - t0
    result['runtime'] = runtime
    return result

def main():
    print("="*70)
    print("AIT-LCD Experiments - Available Data Only")
    print("="*70)
    print()
    
    available = get_available_data()
    
    print("Available datasets:")
    for net, configs in available.items():
        if configs:
            print(f"  {net}: {len(configs)} configurations")
    print()
    
    algorithms = ['AIT-LCD', 'IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']
    all_results = []
    
    for network, configs in available.items():
        if not configs:
            continue
        
        print(f"Processing {network}...")
        
        try:
            gt = load_ground_truth(network)
        except:
            print(f"  Skipping {network} - no ground truth")
            continue
        
        # Use first 3 targets (or fewer for small networks)
        targets = gt['nodes'][:min(3, len(gt['nodes']))]
        
        for n_samples, seed in configs:
            try:
                data = load_dataset(network, n_samples, seed)
            except:
                print(f"  Error loading {network}/n{n_samples}/seed{seed}")
                continue
            
            for target in targets:
                true_pc = gt['pc_sets'][target]
                true_mb = gt['mb_sets'][target]
                
                for algo in algorithms:
                    try:
                        result = run_algorithm(algo, data, target)
                        pc_m = evaluate_pc_discovery(result['pc'], true_pc)
                        mb_m = evaluate_mb_discovery(result['mb'], true_mb)
                        
                        all_results.append({
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
                        print(f"  {algo} error on {target}: {e}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/main_experiment.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Aggregate results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    algo_summary = {}
    for algo in algorithms:
        algo_data = [r for r in all_results if r['algorithm'] == algo]
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
    
    print("\nMain Results (PC F1):")
    for algo, metrics in algo_summary.items():
        print(f"  {algo:20s}: {metrics['pc_f1']['mean']:.3f} ± {metrics['pc_f1']['std']:.3f}")
    
    # Sample size analysis
    print("\nAIT-LCD Performance by Sample Size:")
    for n in [100, 200, 500, 1000]:
        n_data = [r for r in all_results if r['n_samples'] == n and r['algorithm'] == 'AIT-LCD']
        if n_data:
            mean_f1 = np.mean([r['pc_f1'] for r in n_data])
            print(f"  n={n:4d}: F1={mean_f1:.3f}")
    
    # Save aggregated results
    final_results = {
        'experiment_info': {
            'title': 'AIT-LCD: Adaptive Information-Theoretic Local Causal Discovery',
            'description': 'Evaluation on available benchmark networks',
            'networks': list(available.keys()),
            'algorithms': algorithms,
            'total_runs': len(all_results)
        },
        'main_results': algo_summary
    }
    
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "="*70)
    print(f"Total experiments: {len(all_results)}")
    print("Results saved to: results.json, results/main_experiment.json")
    print("="*70)

if __name__ == '__main__':
    main()

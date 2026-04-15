"""
Compile all experimental results into a comprehensive summary with statistics.

Aggregates results across multiple seeds and computes mean ± std and confidence intervals.
"""
import os
import json
import numpy as np
from collections import defaultdict
import glob


def compute_statistics(values):
    """Compute mean, std, and standard error for a list of values."""
    if not values:
        return None
    
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    sem = std / np.sqrt(len(arr))  # Standard error of mean
    
    # 95% confidence interval
    from scipy import stats
    if len(arr) > 1:
        ci = stats.t.interval(0.95, len(arr)-1, loc=mean, scale=sem)
    else:
        ci = (mean, mean)
    
    return {
        'mean': float(mean),
        'std': float(std),
        'sem': float(sem),
        'ci_95_low': float(ci[0]),
        'ci_95_high': float(ci[1]),
        'n': len(arr),
        'values': [float(v) for v in values]
    }


def compile_results():
    """Compile all experimental results."""
    
    print("="*80)
    print("COMPILING EXPERIMENTAL RESULTS")
    print("="*80)
    
    results_dir = 'results/metrics'
    
    # Collect all result files
    lgsa_files = glob.glob(os.path.join(results_dir, 'lgsa_*.json'))
    truvrf_files = glob.glob(os.path.join(results_dir, 'truvrf_*.json'))
    lira_files = glob.glob(os.path.join(results_dir, 'lira_*.json'))
    ablation_files = glob.glob(os.path.join(results_dir, 'ablation_*.json'))
    
    print(f"\nFound result files:")
    print(f"  LGSA: {len(lgsa_files)}")
    print(f"  TruVRF: {len(truvrf_files)}")
    print(f"  LiRA: {len(lira_files)}")
    print(f"  Ablation: {len(ablation_files)}")
    
    # Parse LGSA results
    lgsa_results = defaultdict(lambda: defaultdict(list))
    for f in lgsa_files:
        with open(f, 'r') as fp:
            data = json.load(fp)
        key = (data.get('dataset'), data.get('model'), data.get('unlearn_method'))
        lgsa_results[key]['auc'].append(data.get('auc'))
        lgsa_results[key]['tpr_at_1fpr'].append(data.get('tpr_at_1fpr'))
        lgsa_results[key]['verify_time'].append(data.get('verify_time'))
        lgsa_results[key]['precision'].append(data.get('precision'))
    
    # Parse TruVRF results
    truvrf_results = defaultdict(lambda: defaultdict(list))
    for f in truvrf_files:
        with open(f, 'r') as fp:
            data = json.load(fp)
        key = (data.get('dataset'), data.get('model'))
        truvrf_results[key]['auc'].append(data.get('auc'))
        truvrf_results[key]['tpr_at_1fpr'].append(data.get('tpr_at_1fpr'))
        truvrf_results[key]['verify_time'].append(data.get('verify_time'))
    
    # Parse LiRA results
    lira_results = defaultdict(lambda: defaultdict(list))
    for f in lira_files:
        with open(f, 'r') as fp:
            data = json.load(fp)
        key = (data.get('dataset'), data.get('model'))
        lira_results[key]['auc'].append(data.get('auc'))
        lira_results[key]['tpr_at_1fpr'].append(data.get('tpr_at_1fpr'))
        lira_results[key]['verify_time'].append(data.get('verify_time'))
        lira_results[key]['total_time'].append(data.get('total_time'))
        lira_results[key]['shadow_train_time'].append(data.get('shadow_train_time'))
    
    # Parse ablation results
    ablation_results = defaultdict(lambda: defaultdict(list))
    for f in ablation_files:
        with open(f, 'r') as fp:
            data = json.load(fp)
        config = data.get('config')
        ablation_results[config]['auc'].append(data.get('auc'))
        ablation_results[config]['tpr_at_1fpr'].append(data.get('tpr_at_1fpr'))
    
    # Compile summary
    summary = {
        'lgsa': {},
        'truvrf': {},
        'lira': {},
        'ablation': {},
        'comparisons': {},
        'success_criteria': {}
    }
    
    # LGSA summary
    print("\n" + "-"*80)
    print("LGSA Results Summary")
    print("-"*80)
    for key, metrics in lgsa_results.items():
        key_str = f"{key[0]}_{key[1]}_{key[2]}"
        summary['lgsa'][key_str] = {
            'auc': compute_statistics(metrics['auc']),
            'tpr_at_1fpr': compute_statistics(metrics['tpr_at_1fpr']),
            'verify_time': compute_statistics(metrics['verify_time']),
            'precision': compute_statistics(metrics['precision'])
        }
        print(f"\n{key_str}:")
        if metrics['auc']:
            stats = compute_statistics(metrics['auc'])
            print(f"  AUC: {stats['mean']:.4f} ± {stats['std']:.4f} (95% CI: {stats['ci_95_low']:.4f} - {stats['ci_95_high']:.4f}), n={stats['n']}")
    
    # TruVRF summary
    print("\n" + "-"*80)
    print("TruVRF Results Summary")
    print("-"*80)
    for key, metrics in truvrf_results.items():
        key_str = f"{key[0]}_{key[1]}"
        summary['truvrf'][key_str] = {
            'auc': compute_statistics(metrics['auc']),
            'tpr_at_1fpr': compute_statistics(metrics['tpr_at_1fpr']),
            'verify_time': compute_statistics(metrics['verify_time'])
        }
        print(f"\n{key_str}:")
        if metrics['auc']:
            stats = compute_statistics(metrics['auc'])
            print(f"  AUC: {stats['mean']:.4f} ± {stats['std']:.4f} (95% CI: {stats['ci_95_low']:.4f} - {stats['ci_95_high']:.4f}), n={stats['n']}")
    
    # LiRA summary
    print("\n" + "-"*80)
    print("LiRA Results Summary")
    print("-"*80)
    for key, metrics in lira_results.items():
        key_str = f"{key[0]}_{key[1]}"
        summary['lira'][key_str] = {
            'auc': compute_statistics(metrics['auc']),
            'tpr_at_1fpr': compute_statistics(metrics['tpr_at_1fpr']),
            'verify_time': compute_statistics(metrics['verify_time']),
            'total_time': compute_statistics(metrics['total_time']),
            'shadow_train_time': compute_statistics(metrics['shadow_train_time'])
        }
        print(f"\n{key_str}:")
        if metrics['auc']:
            stats = compute_statistics(metrics['auc'])
            print(f"  AUC: {stats['mean']:.4f} ± {stats['std']:.4f} (95% CI: {stats['ci_95_low']:.4f} - {stats['ci_95_high']:.4f}), n={stats['n']}")
            if metrics['total_time']:
                time_stats = compute_statistics(metrics['total_time'])
                print(f"  Total time: {time_stats['mean']:.1f}s ± {time_stats['std']:.1f}s")
    
    # Ablation summary
    print("\n" + "-"*80)
    print("Ablation Study Summary")
    print("-"*80)
    for config, metrics in ablation_results.items():
        summary['ablation'][config] = {
            'auc': compute_statistics(metrics['auc']),
            'tpr_at_1fpr': compute_statistics(metrics['tpr_at_1fpr'])
        }
        print(f"\n{config}:")
        if metrics['auc']:
            stats = compute_statistics(metrics['auc'])
            print(f"  AUC: {stats['mean']:.4f} ± {stats['std']:.4f} (95% CI: {stats['ci_95_low']:.4f} - {stats['ci_95_high']:.4f}), n={stats['n']}")
    
    # Speedup comparisons
    print("\n" + "-"*80)
    print("Speedup Comparisons")
    print("-"*80)
    
    # Compare LGSA vs TruVRF
    if summary['lgsa'] and summary['truvrf']:
        for key in summary['lgsa']:
            if 'simplecnn_gold' in key or 'simplecnn_finetuning' in key:
                lgsa_time = summary['lgsa'][key].get('verify_time', {}).get('mean', 0)
                truvrf_key = key.rsplit('_', 1)[0]  # Remove unlearn method
                for tk in summary['truvrf']:
                    if tk in key:
                        truvrf_time = summary['truvrf'][tk].get('verify_time', {}).get('mean', 0)
                        if lgsa_time > 0 and truvrf_time > 0:
                            speedup = truvrf_time / lgsa_time
                            summary['comparisons'][f'lgsa_vs_truvrf_{key}'] = {
                                'lgsa_time': lgsa_time,
                                'truvrf_time': truvrf_time,
                                'speedup': speedup
                            }
                            print(f"\n{key}: LGSA vs TruVRF speedup = {speedup:.2f}x")
    
    # Compare LGSA vs LiRA
    if summary['lgsa'] and summary['lira']:
        for key in summary['lgsa']:
            if 'simplecnn' in key:
                lgsa_time = summary['lgsa'][key].get('verify_time', {}).get('mean', 0)
                for lk in summary['lira']:
                    lira_time = summary['lira'][lk].get('total_time', {}).get('mean', 0)
                    if lgsa_time > 0 and lira_time > 0:
                        speedup = lira_time / lgsa_time
                        summary['comparisons'][f'lgsa_vs_lira_{key}'] = {
                            'lgsa_time': lgsa_time,
                            'lira_time': lira_time,
                            'speedup': speedup
                        }
                        print(f"\n{key}: LGSA vs LiRA speedup = {speedup:.2f}x")
    
    # Success criteria evaluation
    print("\n" + "-"*80)
    print("Success Criteria Evaluation")
    print("-"*80)
    
    # Criterion 1: AUC > 0.85
    all_lgsa_aucs = []
    for key, metrics in lgsa_results.items():
        all_lgsa_aucs.extend(metrics['auc'])
    
    if all_lgsa_aucs:
        mean_auc = np.mean(all_lgsa_aucs)
        summary['success_criteria']['criterion_1_auc_above_0.85'] = {
            'target': '> 0.85',
            'actual_mean': float(mean_auc),
            'achieved': mean_auc > 0.85,
            'note': 'NOT ACHIEVED - AUC remains near random' if mean_auc <= 0.85 else 'ACHIEVED'
        }
        print(f"\n1. AUC > 0.85: {mean_auc:.4f} - {'PASS' if mean_auc > 0.85 else 'FAIL'}")
    
    # Criterion 2: Speedup vs TruVRF > 10x
    speedups = [v['speedup'] for k, v in summary['comparisons'].items() if 'truvrf' in k]
    if speedups:
        mean_speedup = np.mean(speedups)
        summary['success_criteria']['criterion_2_speedup_vs_truvrf'] = {
            'target': '> 10x',
            'actual': float(mean_speedup),
            'achieved': mean_speedup > 10,
            'note': 'PASS' if mean_speedup > 10 else 'FAIL'
        }
        print(f"2. Speedup vs TruVRF > 10x: {mean_speedup:.2f}x - {'PASS' if mean_speedup > 10 else 'FAIL'}")
    
    # Criterion 3: Speedup vs LiRA > 50x
    speedups = [v['speedup'] for k, v in summary['comparisons'].items() if 'lira' in k]
    if speedups:
        mean_speedup = np.mean(speedups)
        summary['success_criteria']['criterion_3_speedup_vs_lira'] = {
            'target': '> 50x',
            'actual': float(mean_speedup),
            'achieved': mean_speedup > 50,
            'note': 'PASS' if mean_speedup > 50 else 'FAIL'
        }
        print(f"3. Speedup vs LiRA > 50x: {mean_speedup:.2f}x - {'PASS' if mean_speedup > 50 else 'FAIL'}")
    
    # Save compiled results
    output_path = 'results.json'
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    
    summary = convert_to_native(summary)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Results compiled and saved to {output_path}")
    print("="*80)
    
    return summary


if __name__ == '__main__':
    compile_results()

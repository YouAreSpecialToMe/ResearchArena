#!/usr/bin/env python3
"""
Analyze LASER-SCL experiment results and generate summary report.
"""
import json
import glob
import os
import numpy as np
from collections import defaultdict

def load_results():
    """Load all result JSON files."""
    results = []
    for f in glob.glob('results/*.json'):
        try:
            with open(f) as fp:
                data = json.load(fp)
                data['filename'] = os.path.basename(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return results

def organize_results(results):
    """Organize results by method and dataset."""
    organized = defaultdict(lambda: defaultdict(list))
    for r in results:
        method = r.get('method', 'unknown')
        dataset = r.get('dataset', 'unknown')
        noise = r.get('noise_rate', 0)
        key = f"{dataset}_n{int(noise*100)}"
        organized[method][key].append(r)
    return organized

def compute_stats(results_list):
    """Compute mean and std of accuracies."""
    accs = [r['final_accuracy'] for r in results_list if 'final_accuracy' in r]
    if not accs:
        return None, None
    return np.mean(accs), np.std(accs)

def print_summary_table(organized):
    """Print summary table of results."""
    print("\n" + "="*80)
    print("LASER-SCL EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    methods = ['supcon', 'supcon_lr', 'supcon_il', 'laser_scl', 
               'ablation_no_curriculum', 'ablation_no_elp', 'ablation_static']
    method_names = {
        'supcon': 'SupCon (vanilla)',
        'supcon_lr': 'SupCon + Loss Reweighting',
        'supcon_il': 'SupCon + Inverse Loss',
        'laser_scl': 'LASER-SCL (Full)',
        'ablation_no_curriculum': 'Ablation: No Curriculum',
        'ablation_no_elp': 'Ablation: No ELP',
        'ablation_static': 'Ablation: Static'
    }
    
    # Get all dataset keys
    all_keys = set()
    for method in organized:
        all_keys.update(organized[method].keys())
    
    for key in sorted(all_keys):
        print(f"\n{key.upper().replace('_', ' ')}:")
        print("-" * 60)
        print(f"{'Method':<30} {'Accuracy':>20}")
        print("-" * 60)
        
        for method in methods:
            if key in organized[method]:
                mean, std = compute_stats(organized[method][key])
                if mean is not None:
                    name = method_names.get(method, method)
                    print(f"{name:<30} {mean:>8.2f}% ± {std:>5.2f}%")
        print("-" * 60)

def check_success_criteria(organized):
    """Check if success criteria are met."""
    print("\n" + "="*80)
    print("SUCCESS CRITERIA VERIFICATION")
    print("="*80)
    
    # Criterion 1: LASER-SCL vs SupCon+LR on CIFAR-100 40% noise
    cifar100_key = 'cifar100_n40'
    laser_mean = None
    lr_mean = None
    
    if cifar100_key in organized.get('laser_scl', {}):
        laser_mean, laser_std = compute_stats(organized['laser_scl'][cifar100_key])
    if cifar100_key in organized.get('supcon_lr', {}):
        lr_mean, lr_std = compute_stats(organized['supcon_lr'][cifar100_key])
    
    print("\n1. Primary Success Criterion:")
    print(f"   LASER-SCL should outperform SupCon+LR by ≥2% on CIFAR-100 40% noise")
    if laser_mean and lr_mean:
        diff = laser_mean - lr_mean
        status = "✓ PASS" if diff >= 2.0 else "✗ FAIL"
        print(f"   LASER-SCL: {laser_mean:.2f}%")
        print(f"   SupCon+LR: {lr_mean:.2f}%")
        print(f"   Difference: {diff:+.2f}% {status}")
    else:
        print("   Missing results - cannot evaluate")
    
    # Criterion 2: Component contributions
    print("\n2. Component Validation (≥0.5% contribution per component):")
    if laser_mean:
        for ablation in ['ablation_no_curriculum', 'ablation_no_elp', 'ablation_static']:
            if cifar100_key in organized.get(ablation, {}):
                abl_mean, _ = compute_stats(organized[ablation][cifar100_key])
                if abl_mean:
                    drop = laser_mean - abl_mean
                    status = "✓ PASS" if drop >= 0.5 else "✗ FAIL"
                    print(f"   {ablation}: {drop:+.2f}% drop {status}")
    
    # Criterion 3: ELP validation
    print("\n3. ELP Validation (AUC-ROC):")
    elp_files = glob.glob('results/elp_validation*.json')
    for f in elp_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
                elp_auc = data.get('elp_auc', 0)
                loss_auc = data.get('loss_auc', 0)
                delta = elp_auc - loss_auc
                status = "✓ PASS" if delta > 0.05 else "✗ FAIL"
                dataset = data.get('dataset', 'unknown')
                print(f"   {dataset}: ELP AUC={elp_auc:.4f}, Loss AUC={loss_auc:.4f}, Δ={delta:+.4f} {status}")
        except:
            pass

def save_aggregated_results(organized):
    """Save aggregated results to JSON."""
    summary = {}
    for method in organized:
        summary[method] = {}
        for key in organized[method]:
            mean, std = compute_stats(organized[method][key])
            if mean is not None:
                summary[method][key] = {
                    'mean_accuracy': float(mean),
                    'std_accuracy': float(std),
                    'n_runs': len(organized[method][key])
                }
    
    with open('results/aggregated_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAggregated results saved to results/aggregated_results.json")

def main():
    print("Analyzing LASER-SCL experiment results...")
    
    results = load_results()
    if not results:
        print("No results found!")
        return
    
    print(f"Loaded {len(results)} result files")
    
    organized = organize_results(results)
    print_summary_table(organized)
    check_success_criteria(organized)
    save_aggregated_results(organized)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    main()

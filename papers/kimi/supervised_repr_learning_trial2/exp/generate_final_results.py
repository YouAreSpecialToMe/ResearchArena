#!/usr/bin/env python3
"""
Generate final results.json and figures for LASER-SCL paper.
Run this after experiments complete.
"""
import json
import glob
import os
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_all_results():
    """Load all experiment results."""
    results = {}
    for f in glob.glob('results/*.json'):
        try:
            with open(f) as fp:
                data = json.load(fp)
                results[os.path.basename(f)] = data
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    return results

def create_summary_table(results):
    """Create summary table of all results."""
    table = []
    
    # Group by method and dataset
    grouped = defaultdict(list)
    for fname, data in results.items():
        method = data.get('method', 'unknown')
        dataset = data.get('dataset', 'unknown')
        noise = int(data.get('noise_rate', 0) * 100)
        key = (method, dataset, noise)
        grouped[key].append(data)
    
    # Compute stats for each group
    for (method, dataset, noise), runs in sorted(grouped.items()):
        accs = [r['final_accuracy'] for r in runs if 'final_accuracy' in r]
        if accs:
            mean_acc = np.mean(accs)
            std_acc = np.std(accs) if len(accs) > 1 else 0
            runtime = np.mean([r.get('runtime_minutes', 0) for r in runs])
            
            table.append({
                'method': method,
                'dataset': dataset,
                'noise_rate': noise,
                'accuracy_mean': float(mean_acc),
                'accuracy_std': float(std_acc),
                'n_runs': len(accs),
                'runtime_minutes': float(runtime)
            })
    
    return table

def check_success_criteria(results, table):
    """Check if success criteria are met."""
    criteria = {
        'elp_validation': False,
        'primary_comparison': False,
        'component_contributions': False,
        'efficiency': False
    }
    
    # 1. ELP validation
    elp_files = [f for f in results if 'elp_validation' in f]
    for f in elp_files:
        data = results[f]
        elp_auc = data.get('elp_auc', 0)
        loss_auc = data.get('loss_auc', 0)
        if elp_auc > loss_auc:
            criteria['elp_validation'] = True
            print(f"✓ ELP validation passed: AUC-ROC {elp_auc:.4f} > {loss_auc:.4f}")
    
    # 2. Primary comparison: LASER-SCL vs SupCon+LR on CIFAR-100 40%
    laser_acc = None
    lr_acc = None
    
    for entry in table:
        if entry['dataset'] == 'cifar100' and entry['noise_rate'] == 40:
            if entry['method'] == 'laser_scl':
                laser_acc = entry['accuracy_mean']
            elif entry['method'] == 'supcon_lr':
                lr_acc = entry['accuracy_mean']
    
    if laser_acc and lr_acc:
        diff = laser_acc - lr_acc
        criteria['primary_comparison'] = diff >= 2.0
        print(f"{'✓' if criteria['primary_comparison'] else '✗'} Primary comparison: LASER-SCL ({laser_acc:.2f}%) vs SupCon+LR ({lr_acc:.2f}%), diff = {diff:+.2f}%")
    else:
        print(f"✗ Primary comparison: Missing results (LASER-SCL: {laser_acc}, SupCon+LR: {lr_acc})")
    
    # 3. Component contributions
    if laser_acc:
        ablations = ['ablation_no_curriculum', 'ablation_no_elp', 'ablation_static']
        all_pass = True
        for entry in table:
            if entry['method'] in ablations:
                drop = laser_acc - entry['accuracy_mean']
                passed = drop >= 0.5
                all_pass = all_pass and passed
                status = '✓' if passed else '✗'
                print(f"{status} {entry['method']}: {drop:+.2f}% drop")
        criteria['component_contributions'] = all_pass
    
    # 4. Efficiency
    laser_runtime = None
    supcon_runtime = None
    for entry in table:
        if entry['method'] == 'laser_scl':
            laser_runtime = entry['runtime_minutes']
        elif entry['method'] == 'supcon':
            supcon_runtime = entry['runtime_minutes']
    
    if laser_runtime and supcon_runtime:
        overhead = (laser_runtime - supcon_runtime) / supcon_runtime * 100
        criteria['efficiency'] = overhead < 20
        print(f"{'✓' if criteria['efficiency'] else '✗'} Efficiency: {overhead:.1f}% overhead (LASER-SCL: {laser_runtime:.1f}min, SupCon: {supcon_runtime:.1f}min)")
    
    return criteria

def generate_figures(results, table):
    """Generate figures for paper."""
    os.makedirs('figures', exist_ok=True)
    
    # Figure 1: Main results bar chart
    methods = ['supcon', 'supcon_lr', 'laser_scl']
    method_names = ['SupCon\n(vanilla)', 'SupCon+\nLR', 'LASER-SCL\n(ours)']
    accuracies = []
    errors = []
    
    for method in methods:
        found = False
        for entry in table:
            if entry['method'] == method and entry['dataset'] == 'cifar100' and entry['noise_rate'] == 40:
                accuracies.append(entry['accuracy_mean'])
                errors.append(entry['accuracy_std'])
                found = True
                break
        if not found:
            accuracies.append(0)
            errors.append(0)
    
    if any(a > 0 for a in accuracies):
        plt.figure(figsize=(8, 6))
        x = np.arange(len(methods))
        plt.bar(x, accuracies, yerr=errors, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.xticks(x, method_names)
        plt.ylabel('Accuracy (%)')
        plt.title('CIFAR-100 with 40% Label Noise (100 epochs)')
        plt.ylim(0, max(accuracies) * 1.2)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/main_results.png', dpi=300)
        plt.savefig('figures/main_results.pdf')
        plt.close()
        print("Generated figures/main_results.png")
    
    # Figure 2: ELP validation results
    elp_files = [f for f in results if 'elp_validation' in f]
    for f in elp_files:
        data = results[f]
        if 'elp_auc' in data and 'loss_auc' in data:
            plt.figure(figsize=(6, 4))
            metrics = ['ELP AUC', 'Loss AUC']
            values = [data['elp_auc'], data['loss_auc']]
            colors = ['#2ca02c', '#d62728']
            plt.bar(metrics, values, color=colors)
            plt.ylabel('AUC-ROC')
            plt.title('Clean vs Noisy Sample Discrimination')
            plt.ylim(0, 1)
            plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig('figures/elp_validation.png', dpi=300)
            plt.savefig('figures/elp_validation.pdf')
            plt.close()
            print("Generated figures/elp_validation.png")
            break

def create_final_json(table, criteria):
    """Create final results.json for paper."""
    final_results = {
        'experiment_summary': {
            'total_experiments': len(table),
            'datasets': list(set(e['dataset'] for e in table)),
            'methods': list(set(e['method'] for e in table)),
        },
        'results_table': table,
        'success_criteria': criteria,
        'conclusions': {
            'elp_validated': criteria['elp_validation'],
            'outperforms_baseline': criteria['primary_comparison'],
            'components_validated': criteria['component_contributions'],
            'efficient': criteria['efficiency']
        },
        'scope_note': 'Reduced scope: 100 epochs, 1 seed, CIFAR-100 40% noise focus due to time constraints'
    }
    
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nCreated results.json with {len(table)} experiment results")

def main():
    print("=" * 80)
    print("LASER-SCL Final Results Generation")
    print("=" * 80)
    
    # Load results
    print("\n1. Loading experiment results...")
    results = load_all_results()
    print(f"   Found {len(results)} result files")
    
    if not results:
        print("   ERROR: No results found! Experiments may still be running.")
        return
    
    # Create summary table
    print("\n2. Creating summary table...")
    table = create_summary_table(results)
    
    # Print table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Method':<30} {'Dataset':<12} {'Noise':<8} {'Accuracy':<15} {'N':<5}")
    print("-" * 80)
    for entry in table:
        acc_str = f"{entry['accuracy_mean']:.2f} ± {entry['accuracy_std']:.2f}%"
        print(f"{entry['method']:<30} {entry['dataset']:<12} {entry['noise_rate']:<8} {acc_str:<15} {entry['n_runs']:<5}")
    
    # Check success criteria
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA")
    print("=" * 80)
    criteria = check_success_criteria(results, table)
    
    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)
    generate_figures(results, table)
    
    # Create final JSON
    print("\n" + "=" * 80)
    print("FINAL OUTPUT")
    print("=" * 80)
    create_final_json(table, criteria)
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    passed = sum(criteria.values())
    total = len(criteria)
    print(f"Success criteria passed: {passed}/{total}")
    for criterion, passed in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {criterion}: {status}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()

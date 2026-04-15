#!/usr/bin/env python3
"""
Main experiment execution script.
Runs all baselines, MF-ACD, ablations, and validation experiments.
"""
import os
import sys
import json
import pickle
import numpy as np
import time
from pathlib import Path

# Add exp directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.metrics import compute_metrics, summarize_metrics
from shared.data_loader import load_real_world_data

# Import baseline implementations
from comprehensive_experiments import (
    PCBaseline, GESBaseline, FastPCBaseline, HCCDBaseline, DCILPBaseline,
    MFACD, load_dataset, get_all_datasets
)


# ============== CONFIGURATION ==============

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
GRAPH_SIZES = [20, 50, 100]
DENSITIES = [0.1, 0.2]
SAMPLE_SIZES = [500, 1000, 2000]

RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def ensure_dirs():
    """Ensure all result directories exist."""
    dirs = [
        f"{RESULTS_DIR}/baselines/pc_fisherz",
        f"{RESULTS_DIR}/baselines/pc_stable",
        f"{RESULTS_DIR}/baselines/fast_pc",
        f"{RESULTS_DIR}/baselines/ges",
        f"{RESULTS_DIR}/baselines/hccd",
        f"{RESULTS_DIR}/baselines/dcilp",
        f"{RESULTS_DIR}/mf_acd/main",
        f"{RESULTS_DIR}/mf_acd/real_world",
        f"{RESULTS_DIR}/ablations/fixed_vs_adaptive",
        f"{RESULTS_DIR}/ablations/allocation_sensitivity",
        f"{RESULTS_DIR}/ablations/ugfs_components",
        f"{RESULTS_DIR}/ablations/mtc_comparison",
        f"{RESULTS_DIR}/validation/ig_approximation",
        f"{RESULTS_DIR}/validation/ugfs_overhead",
        f"{RESULTS_DIR}/validation/failure_modes",
        FIGURES_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def filter_datasets(datasets: list, n_nodes: int = None, density: float = None, 
                    n_samples: int = None, seed: int = None) -> list:
    """Filter datasets by criteria."""
    filtered = datasets
    if n_nodes is not None:
        filtered = [d for d in filtered if d['config'].get('n_nodes') == n_nodes]
    if density is not None:
        # Match approximate density
        filtered = [d for d in filtered if abs(d['config'].get('density', 0) - density) < 0.05]
    if n_samples is not None:
        filtered = [d for d in filtered if d['config'].get('n_samples') == n_samples]
    if seed is not None:
        filtered = [d for d in filtered if d['config'].get('seed') == seed]
    return filtered


def run_baseline(baseline_name: str, datasets: list, output_file: str):
    """Run a baseline method on datasets."""
    print(f"\n{'='*60}")
    print(f"Running {baseline_name} on {len(datasets)} datasets")
    print(f"{'='*60}")
    
    results = []
    
    # Initialize method
    if baseline_name == 'pc_fisherz':
        method = PCBaseline(alpha=0.05, stable=False)
    elif baseline_name == 'pc_stable':
        method = PCBaseline(alpha=0.05, stable=True)
    elif baseline_name == 'fast_pc':
        method = FastPCBaseline(alpha=0.05)
    elif baseline_name == 'ges':
        method = GESBaseline(score_type='bic')
    elif baseline_name == 'hccd':
        method = HCCDBaseline(alpha=0.05)
    elif baseline_name == 'dcilp':
        method = DCILPBaseline(alpha=0.05)
    else:
        print(f"Unknown baseline: {baseline_name}")
        return []
    
    for i, dataset_info in enumerate(datasets):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(datasets)}")
        
        try:
            dataset = load_dataset(dataset_info['path'])
            result = method.fit(dataset['data'])
            metrics = compute_metrics(result['adjacency'], dataset['adjacency'])
            
            results.append({
                'dataset': dataset_info['name'],
                'config': dataset_info['config'],
                'metrics': metrics,
                'runtime': result.get('runtime', 0),
                'n_tests': result.get('n_tests', 0)
            })
        except Exception as e:
            print(f"  Error on {dataset_info['name']}: {e}")
            results.append({
                'dataset': dataset_info['name'],
                'config': dataset_info['config'],
                'error': str(e)
            })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{baseline_name} Summary:")
    for n_nodes in GRAPH_SIZES:
        subset = [r for r in results if 'metrics' in r and 
                  r['config'].get('n_nodes') == n_nodes]
        if subset:
            avg_f1 = np.mean([r['metrics']['f1'] for r in subset])
            avg_time = np.mean([r['runtime'] for r in subset])
            print(f"  {n_nodes} nodes: F1={avg_f1:.3f}, Time={avg_time:.2f}s")
    
    return results


def run_mf_acd_main(datasets: list, output_file: str, 
                    budget_allocation=(0.34, 0.20, 0.46),
                    use_adaptive=True, use_ugfs=True,
                    variant_name="mf_acd"):
    """Run MF-ACD main experiments."""
    print(f"\n{'='*60}")
    print(f"Running {variant_name} on {len(datasets)} datasets")
    print(f"Budget: {budget_allocation}, Adaptive: {use_adaptive}, UGFS: {use_ugfs}")
    print(f"{'='*60}")
    
    results = []
    
    for i, dataset_info in enumerate(datasets):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(datasets)}")
        
        try:
            dataset = load_dataset(dataset_info['path'])
            
            method = MFACD(
                budget_allocation=budget_allocation,
                use_adaptive=use_adaptive,
                use_ugfs=use_ugfs
            )
            
            result = method.fit(dataset['data'])
            metrics = compute_metrics(result['adjacency'], dataset['adjacency'])
            
            results.append({
                'dataset': dataset_info['name'],
                'config': dataset_info['config'],
                'metrics': metrics,
                'runtime': result['runtime'],
                'phase_costs': result['phase_costs'],
                'n_tests': result['n_tests'],
                'total_cost': result['total_cost'],
                'baseline_cost': result['baseline_cost'],
                'savings_pct': result['savings_pct'],
                'variant': variant_name
            })
        except Exception as e:
            print(f"  Error on {dataset_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'dataset': dataset_info['name'],
                'config': dataset_info['config'],
                'error': str(e),
                'variant': variant_name
            })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{variant_name} Summary:")
    for n_nodes in GRAPH_SIZES:
        subset = [r for r in results if 'metrics' in r and 
                  r['config'].get('n_nodes') == n_nodes]
        if subset:
            avg_f1 = np.mean([r['metrics']['f1'] for r in subset])
            avg_savings = np.mean([r['savings_pct'] for r in subset])
            avg_time = np.mean([r['runtime'] for r in subset])
            print(f"  {n_nodes} nodes: F1={avg_f1:.3f}, Savings={avg_savings:.1f}%, Time={avg_time:.2f}s")
    
    return results


def run_ablation_studies(datasets_50: list):
    """Run all ablation studies."""
    print(f"\n{'='*60}")
    print("Running Ablation Studies")
    print(f"{'='*60}")
    
    # Ablation 1: Fixed vs Adaptive
    print("\n--- Ablation 1: Fixed vs Adaptive ---")
    run_mf_acd_main(datasets_50, 
                   f"{RESULTS_DIR}/ablations/fixed_vs_adaptive/fixed.json",
                   budget_allocation=(0.34, 0.20, 0.46),
                   use_adaptive=False, use_ugfs=True,
                   variant_name="mf_acd_fixed")
    
    run_mf_acd_main(datasets_50,
                   f"{RESULTS_DIR}/ablations/fixed_vs_adaptive/adaptive.json",
                   budget_allocation=(0.34, 0.20, 0.46),
                   use_adaptive=True, use_ugfs=True,
                   variant_name="mf_acd_adaptive")
    
    # Ablation 2: Allocation Sensitivity
    print("\n--- Ablation 2: Allocation Sensitivity ---")
    allocations = [
        ((0.40, 0.30, 0.30), "conservative"),
        ((0.25, 0.15, 0.60), "aggressive"),
        ((0.35, 0.20, 0.45), "balanced")
    ]
    
    for alloc, name in allocations:
        run_mf_acd_main(datasets_50,
                       f"{RESULTS_DIR}/ablations/allocation_sensitivity/{name}.json",
                       budget_allocation=alloc,
                       use_adaptive=True, use_ugfs=True,
                       variant_name=f"mf_acd_{name}")
    
    # Ablation 3: UGFS Components
    print("\n--- Ablation 3: UGFS Components ---")
    run_mf_acd_main(datasets_50,
                   f"{RESULTS_DIR}/ablations/ugfs_components/nougfs.json",
                   budget_allocation=(0.34, 0.20, 0.46),
                   use_adaptive=True, use_ugfs=False,
                   variant_name="mf_acd_nougfs")
    
    run_mf_acd_main(datasets_50,
                   f"{RESULTS_DIR}/ablations/ugfs_components/full.json",
                   budget_allocation=(0.34, 0.20, 0.46),
                   use_adaptive=True, use_ugfs=True,
                   variant_name="mf_acd_full")


def run_validation_experiments(datasets_50: list):
    """Run validation experiments."""
    print(f"\n{'='*60}")
    print("Running Validation Experiments")
    print(f"{'='*60}")
    
    # Validation: UGFS Overhead
    print("\n--- UGFS Overhead Quantification ---")
    overhead_results = []
    
    for dataset_info in datasets_50[:10]:  # Sample 10 datasets
        dataset = load_dataset(dataset_info['path'])
        n_vars = dataset['data'].shape[1]
        
        # Measure UGFS overhead
        start = time.time()
        method = MFACD(use_ugfs=True)
        _ = method.fit(dataset['data'])
        time_with_ugfs = time.time() - start
        
        start = time.time()
        method = MFACD(use_ugfs=False)
        _ = method.fit(dataset['data'])
        time_without_ugfs = time.time() - start
        
        overhead = (time_with_ugfs - time_without_ugfs) / time_without_ugfs * 100
        
        overhead_results.append({
            'dataset': dataset_info['name'],
            'n_vars': n_vars,
            'time_with_ugfs': time_with_ugfs,
            'time_without_ugfs': time_without_ugfs,
            'overhead_pct': overhead
        })
    
    with open(f"{RESULTS_DIR}/validation/ugfs_overhead/results.json", 'w') as f:
        json.dump(overhead_results, f, indent=2)
    
    avg_overhead = np.mean([r['overhead_pct'] for r in overhead_results])
    print(f"Average UGFS overhead: {avg_overhead:.1f}%")
    
    # Validation: Failure Modes
    print("\n--- Failure Mode Validation ---")
    
    # Get dense graphs (simulate with high edge density)
    dense_results = []
    for dataset_info in datasets_50[:5]:
        dataset = load_dataset(dataset_info['path'])
        method = MFACD(budget_allocation=(0.25, 0.15, 0.60))  # More conservative for dense
        result = method.fit(dataset['data'])
        metrics = compute_metrics(result['adjacency'], dataset['adjacency'])
        dense_results.append({
            'dataset': dataset_info['name'],
            'metrics': metrics,
            'savings_pct': result['savings_pct']
        })
    
    with open(f"{RESULTS_DIR}/validation/failure_modes/dense.json", 'w') as f:
        json.dump(dense_results, f, indent=2)


def run_real_world_experiments():
    """Run experiments on real-world datasets."""
    print(f"\n{'='*60}")
    print("Running Real-World Experiments")
    print(f"{'='*60}")
    
    results = []
    
    for dataset_name in ['sachs', 'child']:
        print(f"\n--- {dataset_name.upper()} Dataset ---")
        
        try:
            data, true_adj = load_real_world_data(dataset_name)
            
            # Run baselines
            for baseline_name, method_class in [
                ('pc_fisherz', PCBaseline),
                ('pc_stable', PCBaseline),
                ('fast_pc', FastPCBaseline)
            ]:
                if baseline_name == 'pc_stable':
                    method = method_class(alpha=0.05, stable=True)
                elif baseline_name == 'pc_fisherz':
                    method = method_class(alpha=0.05, stable=False)
                else:
                    method = method_class(alpha=0.05)
                
                result = method.fit(data)
                metrics = compute_metrics(result['adjacency'], true_adj)
                
                results.append({
                    'dataset': dataset_name,
                    'method': baseline_name,
                    'metrics': metrics,
                    'runtime': result.get('runtime', 0)
                })
            
            # Run MF-ACD
            method = MFACD()
            result = method.fit(data)
            metrics = compute_metrics(result['adjacency'], true_adj)
            
            results.append({
                'dataset': dataset_name,
                'method': 'mf_acd',
                'metrics': metrics,
                'runtime': result['runtime'],
                'savings_pct': result['savings_pct'],
                'phase_costs': result['phase_costs']
            })
            
        except Exception as e:
            print(f"  Error on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    with open(f"{RESULTS_DIR}/mf_acd/real_world/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nReal-World Results Summary:")
    for dataset_name in ['sachs', 'child']:
        subset = [r for r in results if r['dataset'] == dataset_name and 'metrics' in r]
        if subset:
            print(f"\n  {dataset_name.upper()}:")
            for r in subset:
                print(f"    {r['method']}: F1={r['metrics']['f1']:.3f}, Time={r.get('runtime', 0):.2f}s")


def aggregate_all_results():
    """Aggregate all results into final summary."""
    print(f"\n{'='*60}")
    print("Aggregating All Results")
    print(f"{'='*60}")
    
    all_results = {
        'baselines': {},
        'mf_acd': {},
        'ablations': {},
        'validation': {}
    }
    
    # Load baseline results
    for baseline in ['pc_fisherz', 'pc_stable', 'fast_pc', 'ges', 'hccd', 'dcilp']:
        path = f"{RESULTS_DIR}/baselines/{baseline}/results.json"
        if os.path.exists(path):
            with open(path) as f:
                all_results['baselines'][baseline] = json.load(f)
    
    # Load MF-ACD results
    path = f"{RESULTS_DIR}/mf_acd/main/results.json"
    if os.path.exists(path):
        with open(path) as f:
            all_results['mf_acd']['main'] = json.load(f)
    
    # Load real-world results
    path = f"{RESULTS_DIR}/mf_acd/real_world/results.json"
    if os.path.exists(path):
        with open(path) as f:
            all_results['mf_acd']['real_world'] = json.load(f)
    
    # Load ablation results
    for ablation in ['fixed_vs_adaptive', 'allocation_sensitivity', 'ugfs_components']:
        ablation_path = f"{RESULTS_DIR}/ablations/{ablation}"
        if os.path.exists(ablation_path):
            all_results['ablations'][ablation] = {}
            for filename in os.listdir(ablation_path):
                if filename.endswith('.json'):
                    with open(os.path.join(ablation_path, filename)) as f:
                        all_results['ablations'][ablation][filename.replace('.json', '')] = json.load(f)
    
    # Compute summary statistics
    summary = {}
    
    # Baseline comparison by graph size
    for n_nodes in GRAPH_SIZES:
        summary[f'{n_nodes}_nodes'] = {}
        
        for method_name, results in all_results['baselines'].items():
            subset = [r for r in results if 'metrics' in r and 
                     r['config'].get('n_nodes') == n_nodes]
            if subset:
                summary[f'{n_nodes}_nodes'][method_name] = {
                    'f1_mean': float(np.mean([r['metrics']['f1'] for r in subset])),
                    'f1_std': float(np.std([r['metrics']['f1'] for r in subset])),
                    'shd_mean': float(np.mean([r['metrics']['shd'] for r in subset])),
                    'time_mean': float(np.mean([r['runtime'] for r in subset]))
                }
        
        # MF-ACD results
        if 'main' in all_results['mf_acd']:
            subset = [r for r in all_results['mf_acd']['main'] if 'metrics' in r and
                     r['config'].get('n_nodes') == n_nodes]
            if subset:
                summary[f'{n_nodes}_nodes']['mf_acd'] = {
                    'f1_mean': float(np.mean([r['metrics']['f1'] for r in subset])),
                    'f1_std': float(np.std([r['metrics']['f1'] for r in subset])),
                    'shd_mean': float(np.mean([r['metrics']['shd'] for r in subset])),
                    'savings_mean': float(np.mean([r['savings_pct'] for r in subset])),
                    'time_mean': float(np.mean([r['runtime'] for r in subset]))
                }
    
    # Statistical tests
    from scipy import stats
    
    # Compare adaptive vs fixed
    if 'fixed_vs_adaptive' in all_results['ablations']:
        fixed_results = all_results['ablations']['fixed_vs_adaptive'].get('fixed', [])
        adaptive_results = all_results['ablations']['fixed_vs_adaptive'].get('adaptive', [])
        
        if fixed_results and adaptive_results:
            fixed_f1 = [r['metrics']['f1'] for r in fixed_results if 'metrics' in r]
            adaptive_f1 = [r['metrics']['f1'] for r in adaptive_results if 'metrics' in r]
            
            if len(fixed_f1) == len(adaptive_f1) and len(fixed_f1) > 0:
                t_stat, p_value = stats.ttest_rel(adaptive_f1, fixed_f1)
                summary['adaptive_vs_fixed'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
    
    all_results['summary'] = summary
    
    # Save final results
    with open('results_final.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nFinal Summary:")
    for n_nodes in GRAPH_SIZES:
        key = f'{n_nodes}_nodes'
        if key in summary:
            print(f"\n{n_nodes} nodes:")
            for method, stats in summary[key].items():
                if 'f1_mean' in stats:
                    print(f"  {method}: F1={stats['f1_mean']:.3f}±{stats.get('f1_std', 0):.3f}, "
                          f"SHD={stats.get('shd_mean', 0):.1f}")
    
    return all_results


def main():
    """Main experiment execution."""
    start_time = time.time()
    
    print("="*70)
    print("COMPREHENSIVE MF-ACD EXPERIMENTS")
    print("="*70)
    
    # Ensure directories exist
    ensure_dirs()
    
    # Get all datasets
    all_datasets = get_all_datasets("data/synthetic")
    print(f"\nFound {len(all_datasets)} datasets")
    
    # Filter datasets by size
    datasets_20 = filter_datasets(all_datasets, n_nodes=20)
    datasets_50 = filter_datasets(all_datasets, n_nodes=50)
    datasets_100 = filter_datasets(all_datasets, n_nodes=100)
    
    print(f"  20-node datasets: {len(datasets_20)}")
    print(f"  50-node datasets: {len(datasets_50)}")
    print(f"  100-node datasets: {len(datasets_100)}")
    
    # Run baselines
    print("\n" + "="*70)
    print("PHASE 1: BASELINES")
    print("="*70)
    
    for baseline in ['pc_fisherz', 'pc_stable', 'fast_pc', 'ges', 'hccd', 'dcilp']:
        output_file = f"{RESULTS_DIR}/baselines/{baseline}/results.json"
        if not os.path.exists(output_file) or os.path.getsize(output_file) < 100:
            # Run on all datasets
            all_data = datasets_20 + datasets_50 + datasets_100
            if baseline in ['hccd', 'dcilp']:
                # Run on subset for slower methods
                all_data = datasets_20 + datasets_50[:60]
            run_baseline(baseline, all_data, output_file)
        else:
            print(f"\n{baseline} results already exist, skipping...")
    
    # Run MF-ACD main experiments
    print("\n" + "="*70)
    print("PHASE 2: MF-ACD MAIN EXPERIMENTS")
    print("="*70)
    
    output_file = f"{RESULTS_DIR}/mf_acd/main/results.json"
    if not os.path.exists(output_file) or os.path.getsize(output_file) < 100:
        all_data = datasets_20 + datasets_50 + datasets_100
        run_mf_acd_main(all_data, output_file)
    else:
        print("\nMF-ACD main results already exist, skipping...")
    
    # Run ablation studies
    print("\n" + "="*70)
    print("PHASE 3: ABLATION STUDIES")
    print("="*70)
    
    run_ablation_studies(datasets_50)
    
    # Run validation experiments
    print("\n" + "="*70)
    print("PHASE 4: VALIDATION EXPERIMENTS")
    print("="*70)
    
    run_validation_experiments(datasets_50)
    
    # Run real-world experiments
    print("\n" + "="*70)
    print("PHASE 5: REAL-WORLD EXPERIMENTS")
    print("="*70)
    
    run_real_world_experiments()
    
    # Aggregate results
    print("\n" + "="*70)
    print("PHASE 6: RESULTS AGGREGATION")
    print("="*70)
    
    aggregate_all_results()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

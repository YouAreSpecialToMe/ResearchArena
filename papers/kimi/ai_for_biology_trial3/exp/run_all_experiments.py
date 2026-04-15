"""
Master script to run all CellStratCP experiments.
Follows the plan.json workflow.
"""

import os
import sys
import subprocess
import time
import json


def run_script(script_path, args=None, description=""):
    """Run a Python script and capture output."""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70)
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    runtime = time.time() - start_time
    
    print(f"Completed in {runtime:.1f}s with return code {result.returncode}")
    
    return result.returncode == 0


def main():
    print("="*70)
    print("CELLSTRATCP EXPERIMENT PIPELINE")
    print("="*70)
    
    overall_start = time.time()
    
    # Set seeds for reproducibility
    seeds = [42, 123, 456]
    
    # Step 1: Data Preparation
    print("\n\n" + "#"*70)
    print("STEP 1: DATA PREPARATION")
    print("#"*70)
    
    success = run_script(
        'exp/data_prep/prepare_datasets.py',
        ['--seed', '42'],
        "Preparing all datasets (PBMC, Jurkat-293T, Synthetic)"
    )
    
    if not success:
        print("WARNING: Data preparation had issues, continuing...")
    
    # Step 2: Train scVI Models
    print("\n\n" + "#"*70)
    print("STEP 2: TRAIN SCVI MODELS")
    print("#"*70)
    
    for seed in seeds:
        print(f"\n--- Training with seed {seed} ---")
        success = run_script(
            'exp/scvi_training/train_scvi.py',
            ['--seed', str(seed)],
            f"Training scVI models (seed={seed})"
        )
        if not success:
            print(f"WARNING: scVI training failed for seed {seed}")
    
    # Step 3: Run Baselines
    print("\n\n" + "#"*70)
    print("STEP 3: RUNNING BASELINES")
    print("#"*70)
    
    for seed in seeds:
        print(f"\n--- Baselines with seed {seed} ---")
        run_script(
            'exp/cp_baselines/standard_cp/run.py',
            ['--seed', str(seed)],
            f"Standard Split CP (seed={seed})"
        )
    
    # Step 4: Run CellStratCP Main Experiments
    print("\n\n" + "#"*70)
    print("STEP 4: CELLSTRATCP MAIN EXPERIMENTS")
    print("#"*70)
    
    for seed in seeds:
        print(f"\n--- CellStratCP with seed {seed} ---")
        run_script(
            'exp/cellstratcp/run.py',
            ['--seed', str(seed)],
            f"CellStratCP main experiments (seed={seed})"
        )
    
    # Step 5: Ablation Studies
    print("\n\n" + "#"*70)
    print("STEP 5: ABLATION STUDIES")
    print("#"*70)
    
    for seed in seeds:
        print(f"\n--- Ablations with seed {seed} ---")
        run_script(
            'exp/ablations/no_mondrian/run.py',
            ['--seed', str(seed)],
            f"Ablation: Mondrian stratification (seed={seed})"
        )
    
    # Step 6: Aggregate Results
    print("\n\n" + "#"*70)
    print("STEP 6: AGGREGATING RESULTS")
    print("#"*70)
    
    aggregate_results()
    
    overall_time = time.time() - overall_start
    print("\n" + "="*70)
    print(f"ALL EXPERIMENTS COMPLETE in {overall_time/60:.1f} minutes")
    print("="*70)


def aggregate_results():
    """Aggregate results from all experiments into a single JSON."""
    import glob
    import numpy as np
    
    print("\nAggregating results...")
    
    all_experiments = []
    
    # Collect all results
    result_files = glob.glob('exp/**/results.json', recursive=True)
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            if 'experiments' in data:
                for exp in data['experiments']:
                    exp['source_file'] = result_file
                    all_experiments.append(exp)
            elif 'models' in data:
                # scVI training results
                pass
            elif 'datasets' in data:
                # Data prep results
                pass
        except Exception as e:
            print(f"  Error loading {result_file}: {e}")
    
    # Compute summary statistics across seeds
    summary = {}
    
    # Group by dataset and method
    key_groups = {}
    for exp in all_experiments:
        if 'method' in exp and 'dataset' in exp:
            key = (exp['dataset'], exp['method'])
            if key not in key_groups:
                key_groups[key] = []
            key_groups[key].append(exp)
    
    # Compute mean ± std for each group
    for (dataset, method), exps in key_groups.items():
        if len(exps) > 0:
            summary_key = f"{dataset}_{method}"
            
            # Collect metrics
            metrics = {}
            for metric_name in ['marginal_coverage', 'mean_interval_width', 'max_coverage_discrepancy', 'runtime']:
                values = [exp.get(metric_name) for exp in exps if metric_name in exp and exp.get(metric_name) is not None]
                if values:
                    metrics[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'n': len(values)
                    }
            
            summary[summary_key] = {
                'dataset': dataset,
                'method': method,
                'metrics': metrics
            }
    
    final_results = {
        'experiments': all_experiments,
        'summary': summary,
        'n_total_experiments': len(all_experiments)
    }
    
    # Save aggregated results
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"  Saved aggregated results to results.json")
    print(f"  Total experiments: {len(all_experiments)}")
    print(f"  Summary entries: {len(summary)}")


if __name__ == '__main__':
    main()

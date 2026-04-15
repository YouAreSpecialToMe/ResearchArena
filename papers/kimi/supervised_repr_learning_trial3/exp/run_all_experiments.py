"""
Master script to run all CAG-HNM experiments.
Runs experiments sequentially on single GPU.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime

def run_experiment(script_path, args, description):
    """Run an experiment and return results."""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70)
    
    cmd = ['python', script_path] + args
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    runtime = (time.time() - start_time) / 60
    
    if result.returncode != 0:
        print(f"ERROR: Experiment failed with return code {result.returncode}")
        return False, runtime
    
    print(f"Completed in {runtime:.1f} minutes")
    return True, runtime


def main():
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Log file
    log_file = f'logs/experiment_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    print("="*70)
    print("CAG-HNM Experiment Suite")
    print("="*70)
    print(f"Start time: {datetime.now()}")
    
    total_start = time.time()
    experiment_results = []
    
    # Experiment 1: Cross-Entropy baseline (seed 42)
    print("\n[1/8] CIFAR-100 Cross-Entropy Baseline (seed=42)")
    success, runtime = run_experiment(
        'exp/cifar100_crossentropy/run.py',
        ['--seed', '42', '--epochs', '100', '--save_dir', './results'],
        "Cross-Entropy Baseline (seed=42)"
    )
    experiment_results.append({"name": "CE-seed42", "success": success, "runtime": runtime})
    
    # Experiment 2: Cross-Entropy baseline (seed 123)
    print("\n[2/8] CIFAR-100 Cross-Entropy Baseline (seed=123)")
    success, runtime = run_experiment(
        'exp/cifar100_crossentropy/run.py',
        ['--seed', '123', '--epochs', '100', '--save_dir', './results'],
        "Cross-Entropy Baseline (seed=123)"
    )
    experiment_results.append({"name": "CE-seed123", "success": success, "runtime": runtime})
    
    # Experiment 3: SupCon baseline (seed 42)
    print("\n[3/8] CIFAR-100 SupCon Baseline (seed=42)")
    success, runtime = run_experiment(
        'exp/cifar100_supcon/run.py',
        ['--seed', '42', '--epochs', '100', '--save_dir', './results'],
        "SupCon Baseline (seed=42)"
    )
    experiment_results.append({"name": "SupCon-seed42", "success": success, "runtime": runtime})
    
    # Experiment 4: SupCon baseline (seed 123)
    print("\n[4/8] CIFAR-100 SupCon Baseline (seed=123)")
    success, runtime = run_experiment(
        'exp/cifar100_supcon/run.py',
        ['--seed', '123', '--epochs', '100', '--save_dir', './results'],
        "SupCon Baseline (seed=123)"
    )
    experiment_results.append({"name": "SupCon-seed123", "success": success, "runtime": runtime})
    
    # Experiment 5: JD-CCL fixed (seed 42)
    print("\n[5/8] CIFAR-100 JD-CCL Fixed (seed=42)")
    success, runtime = run_experiment(
        'exp/cifar100_jdccl/run.py',
        ['--seed', '42', '--epochs', '100', '--save_dir', './results'],
        "JD-CCL Fixed (seed=42)"
    )
    experiment_results.append({"name": "JD-CCL-seed42", "success": success, "runtime": runtime})
    
    # Experiment 6: JD-CCL fixed (seed 123)
    print("\n[6/8] CIFAR-100 JD-CCL Fixed (seed=123)")
    success, runtime = run_experiment(
        'exp/cifar100_jdccl/run.py',
        ['--seed', '123', '--epochs', '100', '--save_dir', './results'],
        "JD-CCL Fixed (seed=123)"
    )
    experiment_results.append({"name": "JD-CCL-seed123", "success": success, "runtime": runtime})
    
    # Experiment 7: CAG-HNM (seed 42)
    print("\n[7/8] CIFAR-100 CAG-HNM (seed=42)")
    success, runtime = run_experiment(
        'exp/cifar100_caghnm/run.py',
        ['--seed', '42', '--epochs', '100', '--save_dir', './results'],
        "CAG-HNM (seed=42)"
    )
    experiment_results.append({"name": "CAG-HNM-seed42", "success": success, "runtime": runtime})
    
    # Experiment 8: CAG-HNM (seed 123)
    print("\n[8/8] CIFAR-100 CAG-HNM (seed=123)")
    success, runtime = run_experiment(
        'exp/cifar100_caghnm/run.py',
        ['--seed', '123', '--epochs', '100', '--save_dir', './results'],
        "CAG-HNM (seed=123)"
    )
    experiment_results.append({"name": "CAG-HNM-seed123", "success": success, "runtime": runtime})
    
    # Optional: CAG-HNM (seed 456) for statistical significance
    print("\n[9/9] CIFAR-100 CAG-HNM (seed=456) - extra seed for significance")
    success, runtime = run_experiment(
        'exp/cifar100_caghnm/run.py',
        ['--seed', '456', '--epochs', '100', '--save_dir', './results'],
        "CAG-HNM (seed=456)"
    )
    experiment_results.append({"name": "CAG-HNM-seed456", "success": success, "runtime": runtime})
    
    # Ablation study
    print("\n[10/10] Ablation: Fixed vs Curriculum")
    success, runtime = run_experiment(
        'exp/ablation_fixed_vs_curriculum/run.py',
        ['--seed', '42', '--epochs', '100', '--save_dir', './results'],
        "Ablation: Fixed vs Curriculum"
    )
    experiment_results.append({"name": "Ablation", "success": success, "runtime": runtime})
    
    # Summary
    total_runtime = (time.time() - total_start) / 60
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Experiment':<20} {'Status':<10} {'Runtime (min)':<15}")
    print("-"*70)
    
    for exp in experiment_results:
        status = "SUCCESS" if exp["success"] else "FAILED"
        print(f"{exp['name']:<20} {status:<10} {exp['runtime']:>10.1f}")
    
    print("-"*70)
    print(f"{'TOTAL':<20} {'':<10} {total_runtime:>10.1f}")
    print(f"\nEnd time: {datetime.now()}")
    
    # Save summary
    summary = {
        "experiments": experiment_results,
        "total_runtime_minutes": total_runtime,
        "start_time": datetime.now().isoformat()
    }
    
    with open('results/experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nSummary saved to results/experiment_summary.json")


if __name__ == '__main__':
    main()

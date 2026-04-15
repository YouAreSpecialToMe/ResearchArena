#!/usr/bin/env python
"""Fast execution of experiments using seed 42 baseline."""
import subprocess
import sys
import os
import time
import json

def run_command(cmd, description, timeout=None):
    """Run a command and log output."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ {description} FAILED with exit code {result.returncode}")
        return False
    else:
        print(f"\n✓ {description} COMPLETED in {elapsed/60:.1f} minutes")
        return True


def main():
    """Run experiments using available baseline."""
    exp_dir = '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01'
    os.chdir(exp_dir)
    
    env_cmd = "source .venv/bin/activate"
    
    total_start = time.time()
    
    # Wait for baseline seed 42 to be available
    baseline_model = os.path.join(exp_dir, 'exp/baseline_unpruned/model_seed42.pt')
    print("Waiting for baseline model (seed 42) to be available...")
    while not os.path.exists(baseline_model):
        time.sleep(10)
    print(f"✓ Baseline model found: {baseline_model}")
    
    # Run pruning experiments in sequence
    experiments = [
        (f"{env_cmd} && python exp/magnitude_pruning/run.py", "Magnitude Pruning"),
        (f"{env_cmd} && python exp/hybrid_pruning/run.py", "Hybrid Pruning"),
        (f"{env_cmd} && python exp/g3p/run.py", "G3P"),
        (f"{env_cmd} && python exp/taylor_pruning/run.py", "Taylor Pruning"),
    ]
    
    for cmd, desc in experiments:
        run_command(cmd, desc, timeout=7200)
    
    # Run MIA evaluation
    run_command(f"{env_cmd} && python exp/mia_threshold/run.py", "MIA Evaluation", timeout=1800)
    
    # Generate results
    run_command(f"{env_cmd} && python generate_results.py", "Results Generation", timeout=600)
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

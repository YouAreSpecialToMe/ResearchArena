#!/usr/bin/env python3
"""
Run a minimal set of experiments efficiently to get actual results.
Given time constraints, we run key experiments with slightly reduced epochs if needed.
"""
import subprocess
import sys
import time
from pathlib import Path

def run_experiment(script_path, args, log_file):
    """Run a single experiment and wait for completion."""
    cmd = [sys.executable, str(script_path)] + args
    
    print(f"\n{'='*60}")
    print(f"Running: {script_path.name}")
    print(f"Args: {' '.join(args)}")
    print(f"{'='*60}")
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).parent.parent
        )
        
        try:
            process.wait()
            return process.returncode == 0
        except KeyboardInterrupt:
            process.terminate()
            return False


def main():
    """Run essential experiments."""
    experiments = [
        # (script, args, log_file)
        ("exp/baseline_uniform/run.py", ["--seed", "42", "--epochs", "70", "--batch_size", "32"], 
         "logs/experiments/baseline_uniform_s42.log"),
        ("exp/baseline_uniform/run.py", ["--seed", "123", "--epochs", "70", "--batch_size", "32"], 
         "logs/experiments/baseline_uniform_s123.log"),
        ("exp/baseline_density/run.py", ["--seed", "42", "--epochs", "70", "--batch_size", "32"], 
         "logs/experiments/baseline_density_s42.log"),
        ("exp/distflow_idw/run.py", ["--seed", "42", "--epochs", "70", "--batch_size", "32"], 
         "logs/experiments/distflow_idw_s42.log"),
        ("exp/distflow_idw/run.py", ["--seed", "123", "--epochs", "70", "--batch_size", "32"], 
         "logs/experiments/distflow_idw_s123.log"),
        ("exp/distflow_idw/run.py", ["--seed", "456", "--epochs", "70", "--batch_size", "32"], 
         "logs/experiments/distflow_idw_s456.log"),
    ]
    
    print("="*60)
    print("Running Essential Experiments")
    print("="*60)
    
    completed = 0
    failed = 0
    
    for script, args, log_file in experiments:
        script_path = Path(script)
        if not script_path.exists():
            print(f"Script not found: {script}")
            failed += 1
            continue
        
        success = run_experiment(script_path, args, log_file)
        if success:
            completed += 1
            print(f"✓ Completed: {script} {' '.join(args)}")
        else:
            failed += 1
            print(f"✗ Failed: {script} {' '.join(args)}")
    
    print("\n" + "="*60)
    print(f"Results: {completed} completed, {failed} failed")
    print("="*60)


if __name__ == "__main__":
    main()

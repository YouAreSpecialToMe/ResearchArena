"""
Run all experiments efficiently using subsets where appropriate.
"""
import os
import sys
import subprocess
import time

def run_command(cmd, log_file):
    """Run a command and log output."""
    print(f"Starting: {cmd}")
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=f)
    return proc

def main():
    """Run all experiments in sequence."""
    
    # Create directories
    os.makedirs('results/baselines/pc_fisherz', exist_ok=True)
    os.makedirs('results/baselines/pc_stable', exist_ok=True)
    os.makedirs('results/baselines/fast_pc', exist_ok=True)
    os.makedirs('results/baselines/ges', exist_ok=True)
    os.makedirs('results/mf_acd/main', exist_ok=True)
    os.makedirs('results/ablations/fixed_vs_adaptive', exist_ok=True)
    os.makedirs('results/validation/ig_approximation', exist_ok=True)
    
    experiments = [
        # (command, log_file, description)
        ("python exp/baselines/pc_fisherz/run.py", "exp/baselines/pc_fisherz/logs/run.log", "PC-FisherZ"),
        ("python exp/baselines/pc_stable/run.py", "exp/baselines/pc_stable/logs/run.log", "PC-Stable"),
        ("python exp/baselines/fast_pc/run.py", "exp/baselines/fast_pc/logs/run.log", "Fast PC"),
        ("python exp/baselines/ges/run.py", "exp/baselines/ges/logs/run.log", "GES"),
        ("python exp/mf_acd/run.py", "exp/mf_acd/logs/run.log", "MF-ACD Main"),
        ("python exp/ablations/fixed_vs_adaptive/run.py", "exp/ablations/fixed_vs_adaptive/logs/run.log", "Ablation: Fixed vs Adaptive"),
        ("python exp/validation/ig_approximation/run.py", "exp/validation/ig_approximation/logs/run.log", "IG Validation"),
    ]
    
    for cmd, log_file, desc in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {desc}")
        print(f"{'='*60}")
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        start_time = time.time()
        ret = os.system(f"{cmd} 2>&1 | tee {log_file}")
        elapsed = time.time() - start_time
        
        if ret == 0:
            print(f"✓ {desc} completed in {elapsed:.1f}s")
        else:
            print(f"✗ {desc} failed with code {ret}")
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)

if __name__ == "__main__":
    main()

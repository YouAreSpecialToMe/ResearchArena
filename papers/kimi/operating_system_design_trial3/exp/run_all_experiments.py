"""
Master script to run all KAPHE experiments.
"""

import sys
sys.path.insert(0, '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp')

import os
import time
import json
import subprocess

def run_script(script_path):
    """Run a Python script and capture output."""
    print(f"\n{'#'*70}")
    print(f"Running: {script_path}")
    print('#'*70)
    
    start_time = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start_time
    
    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print(f"\nCompleted in {elapsed:.1f}s")
    return result.returncode == 0


def main():
    exp_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp'
    
    print("="*70)
    print("KAPHE EXPERIMENT SUITE")
    print("="*70)
    
    start_time = time.time()
    
    # Step 1: Generate workloads
    if not run_script(f'{exp_dir}/01_generate_workloads.py'):
        print("ERROR: Workload generation failed!")
        return
    
    # Step 2: Collect profiling data
    if not run_script(f'{exp_dir}/02_collect_profiling.py'):
        print("ERROR: Profiling failed!")
        return
    
    # Baselines
    baselines = [
        ('baseline_default', f'{exp_dir}/baseline_default/run.py'),
        ('baseline_expert', f'{exp_dir}/baseline_expert/run.py'),
        ('baseline_mlkaps', f'{exp_dir}/baseline_mlkaps/run.py'),
    ]
    
    for name, script in baselines:
        print(f"\n{'='*70}")
        print(f"Running baseline: {name}")
        print('='*70)
        if not run_script(script):
            print(f"WARNING: {name} failed!")
    
    # Main KAPHE experiment
    print(f"\n{'='*70}")
    print("Running KAPHE main experiment")
    print('='*70)
    if not run_script(f'{exp_dir}/kaphe/run.py'):
        print("ERROR: KAPHE failed!")
        return
    
    # Ablations
    ablations = [
        ('ablation_no_char', f'{exp_dir}/ablation_no_char/run.py'),
        ('ablation_knn', f'{exp_dir}/ablation_knn/run.py'),
    ]
    
    for name, script in ablations:
        print(f"\n{'='*70}")
        print(f"Running ablation: {name}")
        print('='*70)
        if not run_script(script):
            print(f"WARNING: {name} failed!")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print("="*70)


if __name__ == '__main__':
    main()

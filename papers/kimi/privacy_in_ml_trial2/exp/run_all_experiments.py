#!/usr/bin/env python
"""Master script to run all G3P experiments."""
import subprocess
import sys
import os
import time
import json

def run_command(cmd, description, timeout=None):
    """Run a command and log output."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    elapsed = time.time() - start_time
    
    if process.returncode != 0:
        print(f"\n❌ {description} FAILED with exit code {process.returncode}")
        print(f"Elapsed: {elapsed/60:.1f} minutes")
        return False
    else:
        print(f"\n✓ {description} COMPLETED in {elapsed/60:.1f} minutes")
        return True


def check_file_exists(path):
    """Check if a file exists."""
    return os.path.exists(path)


def main():
    """Run all experiments in sequence."""
    exp_dir = '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01'
    os.chdir(exp_dir)
    
    # Load environment
    env_cmd = "source .venv/bin/activate"
    
    # Track overall timing
    total_start = time.time()
    
    # =========================================================================
    # Step 1: Train baseline models (if not already done)
    # =========================================================================
    baseline_results = os.path.join(exp_dir, 'exp/baseline_unpruned/results.json')
    if not check_file_exists(baseline_results):
        print("\n" + "="*70)
        print("STEP 1: Training baseline models")
        print("="*70)
        success = run_command(
            f"{env_cmd} && python exp/baseline_unpruned/run.py",
            "Baseline Training",
            timeout=7200  # 2 hours
        )
        if not success:
            print("Baseline training failed. Stopping.")
            return
    else:
        print("\n✓ Baseline models already trained")
    
    # =========================================================================
    # Step 2: Run pruning experiments
    # =========================================================================
    
    # Magnitude Pruning
    mag_results = os.path.join(exp_dir, 'exp/magnitude_pruning/results.json')
    if not check_file_exists(mag_results):
        print("\n" + "="*70)
        print("STEP 2a: Magnitude Pruning")
        print("="*70)
        run_command(
            f"{env_cmd} && python exp/magnitude_pruning/run.py",
            "Magnitude Pruning",
            timeout=5400  # 1.5 hours
        )
    else:
        print("\n✓ Magnitude pruning already done")
    
    # Taylor Pruning
    taylor_results = os.path.join(exp_dir, 'exp/taylor_pruning/results.json')
    if not check_file_exists(taylor_results):
        print("\n" + "="*70)
        print("STEP 2b: Taylor Pruning")
        print("="*70)
        run_command(
            f"{env_cmd} && python exp/taylor_pruning/run.py",
            "Taylor Pruning",
            timeout=3600  # 1 hour
        )
    else:
        print("\n✓ Taylor pruning already done")
    
    # Hybrid (Magnitude + KL)
    hybrid_results = os.path.join(exp_dir, 'exp/hybrid_pruning/results.json')
    if not check_file_exists(hybrid_results):
        print("\n" + "="*70)
        print("STEP 2c: Hybrid (Magnitude + KL)")
        print("="*70)
        run_command(
            f"{env_cmd} && python exp/hybrid_pruning/run.py",
            "Hybrid Pruning",
            timeout=5400  # 1.5 hours
        )
    else:
        print("\n✓ Hybrid pruning already done")
    
    # G3P
    g3p_results = os.path.join(exp_dir, 'exp/g3p/results.json')
    if not check_file_exists(g3p_results):
        print("\n" + "="*70)
        print("STEP 2d: G3P")
        print("="*70)
        run_command(
            f"{env_cmd} && python exp/g3p/run.py",
            "G3P",
            timeout=7200  # 2 hours
        )
    else:
        print("\n✓ G3P already done")
    
    # =========================================================================
    # Step 3: Run MIA evaluation
    # =========================================================================
    mia_results = os.path.join(exp_dir, 'exp/mia_threshold/results.json')
    if not check_file_exists(mia_results):
        print("\n" + "="*70)
        print("STEP 3: MIA Evaluation")
        print("="*70)
        run_command(
            f"{env_cmd} && python exp/mia_threshold/run.py",
            "MIA Evaluation",
            timeout=1800  # 30 minutes
        )
    else:
        print("\n✓ MIA evaluation already done")
    
    # =========================================================================
    # Step 4: Generate visualizations and final results
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: Generating visualizations and final results")
    print("="*70)
    run_command(
        f"{env_cmd} && python generate_results.py",
        "Results Generation",
        timeout=600  # 10 minutes
    )
    
    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

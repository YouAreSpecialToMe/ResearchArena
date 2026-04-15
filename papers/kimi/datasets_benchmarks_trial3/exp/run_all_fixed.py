#!/usr/bin/env python3
"""
Master script to run all fixed experiments.
Executes the entire pipeline with corrected implementations.
"""
import sys
import os
import json
import time
import subprocess

sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')


def run_command(cmd, log_file):
    """Run a command and log output."""
    print(f"\n{'='*70}")
    print(f"Running: {cmd}")
    print(f"{'='*70}")
    
    with open(log_file, 'w') as f:
        f.write(f"Command: {cmd}\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    output = []
    for line in process.stdout:
        print(line, end='')
        output.append(line)
        with open(log_file, 'a') as f:
            f.write(line)
    
    process.wait()
    elapsed = time.time() - start_time
    
    with open(log_file, 'a') as f:
        f.write(f"\nFinished at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Elapsed time: {elapsed/60:.2f} minutes\n")
        f.write(f"Return code: {process.returncode}\n")
    
    return process.returncode == 0, elapsed


def main():
    print("=" * 70)
    print("PopBench: Running All Fixed Experiments")
    print("=" * 70)
    
    start_time = time.time()
    results = {}
    
    # 1. Train population model
    success, elapsed = run_command(
        "cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01 && python exp/popbench_train/run_v2.py",
        "logs/popbench_train_v2.log"
    )
    results['popbench_train'] = {'success': success, 'time': elapsed}
    if not success:
        print("ERROR: Training failed!")
        return 1
    
    # 2. Zero-shot prediction
    success, elapsed = run_command(
        "cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01 && python exp/popbench_zeroshot/run_v2.py",
        "logs/popbench_zeroshot_v2.log"
    )
    results['popbench_zeroshot'] = {'success': success, 'time': elapsed}
    
    # 3. Adaptive evaluation
    success, elapsed = run_command(
        "cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01 && python exp/popbench_adaptive/run_v2.py",
        "logs/popbench_adaptive_v2.log"
    )
    results['popbench_adaptive'] = {'success': success, 'time': elapsed}
    
    # 4. Joint evaluation
    success, elapsed = run_command(
        "cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01 && python exp/popbench_joint/run_v2.py",
        "logs/popbench_joint_v2.log"
    )
    results['popbench_joint'] = {'success': success, 'time': elapsed}
    
    # 5. Ablation: No population prior
    success, elapsed = run_command(
        "cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01 && python exp/ablation_no_population_prior/run_v2.py",
        "logs/ablation_no_population_prior_v2.log"
    )
    results['ablation_no_population_prior'] = {'success': success, 'time': elapsed}
    
    # 6. Ablation: Standard EIG
    success, elapsed = run_command(
        "cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01 && python exp/ablation_standard_eig/run_v2.py",
        "logs/ablation_standard_eig_v2.log"
    )
    results['ablation_standard_eig'] = {'success': success, 'time': elapsed}
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("Experiment Summary")
    print("=" * 70)
    
    for name, data in results.items():
        status = "✓" if data['success'] else "✗"
        print(f"  {status} {name}: {data['time']/60:.2f} min")
    
    print(f"\n  Total time: {total_time/60:.2f} minutes")
    print(f"  All passed: {all(d['success'] for d in results.values())}")
    print("=" * 70)
    
    # Save summary
    with open('logs/experiment_summary_v2.json', 'w') as f:
        json.dump({
            'results': results,
            'total_time_minutes': total_time / 60,
            'all_success': all(d['success'] for d in results.values())
        }, f, indent=2)
    
    return 0 if all(d['success'] for d in results.values()) else 1


if __name__ == "__main__":
    exit(main())

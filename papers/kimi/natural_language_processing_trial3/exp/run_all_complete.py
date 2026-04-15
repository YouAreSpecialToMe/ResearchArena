"""
Master script to run all experiments efficiently.
Uses parallel execution where possible.
"""

import subprocess
import json
import time
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


def run_command(cmd: str, description: str) -> tuple:
    """Run a command and return result."""
    print(f"\n{'='*70}")
    print(f"Starting: {description}")
    print(f"Command: {cmd}")
    print('='*70)
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=False,
            text=True,
            timeout=7200  # 2 hour timeout per task
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"\n✓ Completed: {description} in {elapsed/60:.1f} minutes")
            return (True, description, elapsed, None)
        else:
            print(f"\n✗ Failed: {description} (exit code {result.returncode})")
            return (False, description, elapsed, f"Exit code {result.returncode}")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"\n✗ Timeout: {description}")
        return (False, description, elapsed, "Timeout")
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n✗ Error: {description}: {e}")
        return (False, description, elapsed, str(e))


def run_threshold_tuning():
    """Run threshold tuning."""
    cmd = "cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/natural_language_processing/idea_01 && source .venv/bin/activate && python exp/threshold_tuning/tune_thresholds_proper.py"
    return run_command(cmd, "Threshold Tuning")


def run_method(method: str, dataset: str, model: str, seed: int, max_problems: int = None):
    """Run a single method."""
    max_flag = f"--max_problems {max_problems}" if max_problems else ""
    cmd = f"cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/natural_language_processing/idea_01 && source .venv/bin/activate && python exp/run_complete_experiments.py --method {method} --model {model} --dataset {dataset} --seed {seed} {max_flag}"
    desc = f"{method} on {dataset} (seed={seed}, model={model.split('/')[-1]})"
    return run_command(cmd, desc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick run with reduced problems")
    parser.add_argument("--skip_tuning", action="store_true", help="Skip threshold tuning")
    args = parser.parse_args()
    
    print("="*70)
    print("Complete ESR Experiment Suite")
    print("="*70)
    print(f"Quick mode: {args.quick}")
    print(f"Skip tuning: {args.skip_tuning}")
    
    start_time = time.time()
    results = []
    
    # Step 1: Threshold tuning
    if not args.skip_tuning:
        success, desc, elapsed, error = run_threshold_tuning()
        results.append((success, desc, elapsed, error))
        if not success:
            print("Warning: Threshold tuning failed, using defaults")
    
    # Determine experiment scale
    if args.quick:
        # Quick test mode
        methods = ["vanilla", "esr", "entropy_only"]
        datasets = [("gsm8k", 100)]  # (dataset, max_problems)
        models = ["Qwen/Qwen3-1.7B"]
        seeds = [42]
    else:
        # Full experiment mode
        methods = ["vanilla", "esr", "entropy_only", "egl", "bestofn"]
        datasets = [("gsm8k", None), ("math500", None)]  # None = all problems
        models = ["Qwen/Qwen3-1.7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]
        seeds = [42, 123]
    
    # Generate all experiment tasks
    tasks = []
    for model in models:
        for dataset, max_problems in datasets:
            for seed in seeds:
                for method in methods:
                    # Skip DeepSeek for some methods to save time
                    if "DeepSeek" in model and method in ["egl", "bestofn"] and seed == 123:
                        continue
                    tasks.append((method, dataset, model, seed, max_problems))
    
    print(f"\nTotal experiments to run: {len(tasks)}")
    print(f"Methods: {methods}")
    print(f"Datasets: {[d[0] for d in datasets]}")
    print(f"Models: {[m.split('/')[-1] for m in models]}")
    print(f"Seeds: {seeds}")
    
    # Run experiments sequentially (safer for GPU memory)
    for i, (method, dataset, model, seed, max_problems) in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Running experiment...")
        success, desc, elapsed, error = run_method(method, dataset, model, seed, max_problems)
        results.append((success, desc, elapsed, error))
        
        # Progress summary
        completed = sum(1 for r in results if r[0])
        failed = sum(1 for r in results if not r[0])
        elapsed_total = time.time() - start_time
        print(f"\nProgress: {completed} completed, {failed} failed, {elapsed_total/60:.1f} min elapsed")
    
    # Final summary
    print("\n" + "="*70)
    print("Experiment Suite Completed!")
    print("="*70)
    
    completed = sum(1 for r in results if r[0])
    failed = sum(1 for r in results if not r[0])
    total_time = (time.time() - start_time) / 60
    
    print(f"\nSummary:")
    print(f"  Total experiments: {len(results)}")
    print(f"  Successful: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Total time: {total_time:.1f} minutes ({total_time/60:.2f} hours)")
    
    # Generate aggregate results
    print("\nGenerating aggregate results...")
    try:
        subprocess.run(
            "cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/natural_language_processing/idea_01 && source .venv/bin/activate && python exp/create_final_results.py",
            shell=True,
            check=True
        )
        print("✓ Aggregate results generated")
    except Exception as e:
        print(f"✗ Failed to generate aggregate results: {e}")
    
    print("\n" + "="*70)
    print("All done!")
    print("="*70)


if __name__ == "__main__":
    main()

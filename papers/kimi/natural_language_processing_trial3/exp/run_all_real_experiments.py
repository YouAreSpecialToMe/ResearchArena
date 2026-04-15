"""
Master experiment runner for ESR experiments with REAL model inference.
This script runs all baselines and ESR with actual model inference.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import subprocess
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def run_experiment(method: str, dataset: str, model: str, seed: int, max_problems: int):
    """Run a single experiment and return results."""
    
    method_runners = {
        "vanilla": "exp/vanilla_cot/run.py",
        "esr": "exp/esr/run.py",
        "entropy_only": "exp/entropy_only/run.py",
        "egb": "exp/egb_beam/run.py",
        "egl": "exp/egl_posthoc/run.py",
        "bestofn": "exp/bestofn/run.py",
        "entropix": "exp/entropix/run.py",
        "halt_cot": "exp/halt_cot/run.py",
    }
    
    if method not in method_runners:
        print(f"Unknown method: {method}")
        return None
    
    script = method_runners[method]
    cmd = [
        "python", script,
        "--dataset", dataset,
        "--model", model,
        "--seed", str(seed),
        "--max_problems", str(max_problems)
    ]
    
    print(f"\n{'='*70}")
    print(f"Running: {method} on {dataset} with seed {seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per experiment
        )
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed/60:.1f} minutes")
        
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return None
        
        print(result.stdout)
        return {"success": True, "elapsed_minutes": elapsed/60}
        
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after 2 hours")
        return None
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return None


def load_and_aggregate_results(methods, datasets, seeds):
    """Load all experiment results and aggregate statistics."""
    
    all_results = {}
    
    for method in methods:
        all_results[method] = {}
        for dataset in datasets:
            all_results[method][dataset] = {}
            
            for seed in seeds:
                result_file = Path(f"exp/results/{method}_{dataset}_seed{seed}.json")
                
                if result_file.exists():
                    with open(result_file) as f:
                        data = json.load(f)
                        all_results[method][dataset][seed] = {
                            "accuracy": data.get("accuracy", 0),
                            "avg_tokens": data.get("avg_tokens", 0),
                            "correct_count": data.get("correct_count", 0),
                            "total_problems": data.get("total_problems", 0)
                        }
                        
                        # Add method-specific metrics
                        if "revision_rate" in data:
                            all_results[method][dataset][seed]["revision_rate"] = data["revision_rate"]
                        if "avg_revisions" in data:
                            all_results[method][dataset][seed]["avg_revisions"] = data["avg_revisions"]
                        if "refinement_rate" in data:
                            all_results[method][dataset][seed]["refinement_rate"] = data["refinement_rate"]
    
    # Compute aggregate statistics
    aggregate = {}
    for method in methods:
        aggregate[method] = {}
        for dataset in datasets:
            if all_results[method][dataset]:
                accuracies = [all_results[method][dataset][s]["accuracy"] 
                             for s in seeds if s in all_results[method][dataset]]
                tokens = [all_results[method][dataset][s]["avg_tokens"] 
                         for s in seeds if s in all_results[method][dataset]]
                
                if accuracies:
                    import numpy as np
                    aggregate[method][dataset] = {
                        "accuracy_mean": np.mean(accuracies),
                        "accuracy_std": np.std(accuracies),
                        "tokens_mean": np.mean(tokens),
                        "tokens_std": np.std(tokens),
                        "n_seeds": len(accuracies)
                    }
    
    return all_results, aggregate


def print_summary_table(aggregate):
    """Print a summary table of all results."""
    
    print("\n" + "="*90)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*90)
    print(f"{'Method':<20} {'Dataset':<12} {'Accuracy':<20} {'Avg Tokens':<20}")
    print("-"*90)
    
    for method in aggregate:
        for dataset in aggregate[method]:
            stats = aggregate[method][dataset]
            acc_str = f"{stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f}"
            tok_str = f"{stats['tokens_mean']:.1f} ± {stats['tokens_std']:.1f}"
            print(f"{method:<20} {dataset:<12} {acc_str:<20} {tok_str:<20}")
    
    print("="*90)


def main():
    parser = argparse.ArgumentParser(description="Run all ESR experiments with real inference")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", 
                       help="Model to use (default: Qwen/Qwen3-1.7B)")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k"],
                       help="Datasets to evaluate on")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                       help="Random seeds to use")
    parser.add_argument("--methods", nargs="+", 
                       default=["vanilla", "esr", "entropy_only", "egb", "egl", "bestofn"],
                       help="Methods to evaluate")
    parser.add_argument("--max_problems", type=int, default=150,
                       help="Maximum problems to evaluate per dataset (for time constraints)")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip experiments that already have results")
    args = parser.parse_args()
    
    # Verify GPU
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Experiments will be very slow.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nExperiment Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Max problems per dataset: {args.max_problems}")
    print(f"  Estimated time: ~{len(args.methods) * len(args.datasets) * len(args.seeds) * 20} minutes")
    
    # Create results directory
    Path("exp/results").mkdir(parents=True, exist_ok=True)
    
    # Track overall progress
    total_experiments = len(args.methods) * len(args.datasets) * len(args.seeds)
    completed = 0
    failed = 0
    
    start_time = time.time()
    
    # Run all experiments
    for method in args.methods:
        for dataset in args.datasets:
            for seed in args.seeds:
                # Check if result already exists
                if args.skip_existing:
                    result_file = Path(f"exp/results/{method}_{dataset}_seed{seed}.json")
                    if result_file.exists():
                        print(f"\nSkipping existing: {method}/{dataset}/seed{seed}")
                        completed += 1
                        continue
                
                result = run_experiment(method, dataset, args.model, seed, args.max_problems)
                
                if result:
                    completed += 1
                else:
                    failed += 1
                
                elapsed = time.time() - start_time
                remaining = (elapsed / max(completed, 1)) * (total_experiments - completed)
                
                print(f"\nProgress: {completed}/{total_experiments} completed, {failed} failed")
                print(f"Elapsed: {elapsed/3600:.2f}h, Estimated remaining: {remaining/3600:.2f}h")
    
    # Aggregate and display results
    all_results, aggregate = load_and_aggregate_results(args.methods, args.datasets, args.seeds)
    
    # Print summary
    print_summary_table(aggregate)
    
    # Save aggregate results
    output = {
        "all_results": all_results,
        "aggregate": aggregate,
        "config": {
            "model": args.model,
            "datasets": args.datasets,
            "methods": args.methods,
            "seeds": args.seeds,
            "max_problems": args.max_problems
        },
        "total_experiments": total_experiments,
        "completed": completed,
        "failed": failed
    }
    
    with open("exp/results/aggregate_results.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nAggregate results saved to: exp/results/aggregate_results.json")
    
    # Also save to root
    with open("results.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results also saved to: results.json")
    
    total_time = (time.time() - start_time) / 3600
    print(f"\nTotal execution time: {total_time:.2f} hours")


if __name__ == "__main__":
    main()

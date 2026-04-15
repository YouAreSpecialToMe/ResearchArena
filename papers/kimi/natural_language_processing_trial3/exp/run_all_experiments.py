"""Master script to run all ESR experiments."""

import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict
import time

sys.path.insert(0, str(Path(__file__).parent))


def run_command(cmd: List[str], desc: str):
    """Run a command and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.1f}s with return code {result.returncode}")
    return result.returncode == 0


def main():
    # Configuration
    models = ["Qwen/Qwen3-1.7B"]  # Start with smaller model
    datasets = ["gsm8k", "math500"]
    seeds = [42, 123]  # Reduced to 2 seeds for speed
    
    # Use smaller subset for faster iteration
    # In full run, set to None for full dataset
    limit = None
    
    results_dir = Path("exp/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Threshold Tuning
    print("\n" + "="*60)
    print("STEP 1: Threshold Tuning")
    print("="*60)
    
    success = run_command(
        [sys.executable, "exp/threshold_tuning/tune_thresholds.py",
         "--model", models[0],
         "--output", "exp/results/threshold_tuning.json",
         "--limit", "100"],
        "Threshold Tuning"
    )
    
    if not success:
        print("Threshold tuning failed! Using default thresholds.")
        best_tau_h, best_tau_v = 2.5, 1.5
    else:
        # Load best thresholds
        with open("exp/results/threshold_tuning.json") as f:
            tuning_results = json.load(f)
        best_tau_h = tuning_results["best_tau_h"]
        best_tau_v = tuning_results["best_tau_v"]
    
    print(f"\nUsing thresholds: tau_H={best_tau_h}, tau_V={best_tau_v}")
    
    # Step 2: Run all experiments
    experiments = []
    
    # Vanilla CoT baseline
    for dataset in datasets:
        for seed in seeds:
            experiments.append({
                "name": f"vanilla_{dataset}_seed{seed}",
                "cmd": [sys.executable, "exp/run_experiment.py",
                       "--method", "vanilla",
                       "--model", models[0],
                       "--dataset", dataset,
                       "--seed", str(seed),
                       "--output", f"exp/results/vanilla_{dataset}_seed{seed}.json"]
            })
    
    # Entropy-only baseline
    for dataset in datasets:
        for seed in seeds:
            experiments.append({
                "name": f"entropy_only_{dataset}_seed{seed}",
                "cmd": [sys.executable, "exp/run_experiment.py",
                       "--method", "entropy_only",
                       "--model", models[0],
                       "--dataset", dataset,
                       "--tau_h", str(best_tau_h),
                       "--seed", str(seed),
                       "--output", f"exp/results/entropy_only_{dataset}_seed{seed}.json"]
            })
    
    # ESR full method
    for dataset in datasets:
        for seed in seeds:
            experiments.append({
                "name": f"esr_{dataset}_seed{seed}",
                "cmd": [sys.executable, "exp/run_experiment.py",
                       "--method", "esr",
                       "--model", models[0],
                       "--dataset", dataset,
                       "--tau_h", str(best_tau_h),
                       "--tau_v", str(best_tau_v),
                       "--seed", str(seed),
                       "--output", f"exp/results/esr_{dataset}_seed{seed}.json"]
            })
    
    # EGL post-hoc
    for dataset in datasets:
        for seed in seeds:
            experiments.append({
                "name": f"egl_{dataset}_seed{seed}",
                "cmd": [sys.executable, "exp/run_experiment.py",
                       "--method", "egl",
                       "--model", models[0],
                       "--dataset", dataset,
                       "--tau_h", str(best_tau_h),
                       "--seed", str(seed),
                       "--output", f"exp/results/egl_{dataset}_seed{seed}.json"]
            })
    
    # Best-of-N
    for dataset in datasets:
        for seed in seeds:
            experiments.append({
                "name": f"bestofn_{dataset}_seed{seed}",
                "cmd": [sys.executable, "exp/run_experiment.py",
                       "--method", "bestofn",
                       "--model", models[0],
                       "--dataset", dataset,
                       "--n_samples", "4",
                       "--seed", str(seed),
                       "--output", f"exp/results/bestofn_{dataset}_seed{seed}.json"]
            })
    
    # Run all experiments
    print(f"\n{'='*60}")
    print(f"STEP 2: Running {len(experiments)} experiments")
    print(f"{'='*60}")
    
    completed = 0
    failed = 0
    
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] ", end="")
        success = run_command(exp["cmd"], exp["name"])
        if success:
            completed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Experiments completed: {completed}/{len(experiments)}")
    print(f"Failed: {failed}")
    print(f"{'='*60}")
    
    # Step 3: Aggregate results
    print("\n" + "="*60)
    print("STEP 3: Aggregating Results")
    print("="*60)
    
    aggregate_results()


def aggregate_results():
    """Aggregate results from all experiments."""
    results_dir = Path("exp/results")
    
    methods = ["vanilla", "entropy_only", "esr", "egl", "bestofn"]
    datasets = ["gsm8k", "math500"]
    seeds = [42, 123]
    
    summary = {}
    
    for method in methods:
        summary[method] = {}
        for dataset in datasets:
            accuracies = []
            tokens = []
            
            for seed in seeds:
                result_file = results_dir / f"{method}_{dataset}_seed{seed}.json"
                if result_file.exists():
                    with open(result_file) as f:
                        data = json.load(f)
                    accuracies.append(data.get("accuracy", 0))
                    tokens.append(data.get("avg_tokens", 0))
            
            if accuracies:
                import numpy as np
                summary[method][dataset] = {
                    "accuracy_mean": float(np.mean(accuracies)),
                    "accuracy_std": float(np.std(accuracies)),
                    "tokens_mean": float(np.mean(tokens)),
                    "tokens_std": float(np.std(tokens))
                }
    
    # Save summary
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'Dataset':<10} {'Accuracy':<20} {'Tokens':<15}")
    print("-"*60)
    
    for method in methods:
        for dataset in datasets:
            if dataset in summary.get(method, {}):
                s = summary[method][dataset]
                acc_str = f"{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f}"
                tok_str = f"{s['tokens_mean']:.1f}"
                print(f"{method:<20} {dataset:<10} {acc_str:<20} {tok_str:<15}")
    
    print("="*60)


if __name__ == "__main__":
    main()

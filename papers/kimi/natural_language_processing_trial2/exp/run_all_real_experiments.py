#!/usr/bin/env python3
"""
Master script to run ALL real CDHR experiments.
Executes experiments sequentially with proper resource management.
"""
import os
import sys
import json
import time
import subprocess
from datetime import datetime
from typing import List, Dict

# Experiment configuration
MODEL = "llama-3.1-8b"
SEEDS = [42, 123, 456]

# Dataset sizes for different experiment types
FULL_DATASETS = {
    "gsm8k": 1319,
    "gpqa": 198,
    "aime": 30,  # Small dataset
}

SUBSET_SIZES = {
    "gsm8k": 300,  # For ablations and faster experiments
    "gpqa": 100,
}

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and track progress."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"\n✓ Completed: {description} in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed: {description} after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False

def experiment_exists(method: str, dataset: str, seed: int) -> bool:
    """Check if experiment result already exists."""
    result_file = f"results/{method}_{dataset}_s{seed}.json"
    return os.path.exists(result_file)

def run_baseline_cot():
    """Run Baseline 1: Standard Chain-of-Thought."""
    print("\n" + "="*70)
    print("PHASE 1: BASELINE - STANDARD CHAIN-OF-THOUGHT")
    print("="*70)
    
    results = []
    for dataset in ["gsm8k", "gpqa", "aime"]:
        max_prob = FULL_DATASETS.get(dataset)
        for seed in SEEDS:
            if experiment_exists("cot", dataset, seed):
                print(f"Skipping cot_{dataset}_s{seed} (already exists)")
                continue
            
            cmd = [
                "python", "run_real_experiments.py",
                "--model", MODEL,
                "--dataset", dataset,
                "--method", "cot",
                "--seed", str(seed),
                "--output_dir", "results"
            ]
            if max_prob:
                cmd.extend(["--max_problems", str(max_prob)])
            
            success = run_command(cmd, f"CoT on {dataset} (seed={seed})")
            results.append(("cot", dataset, seed, success))
    
    return results

def run_baseline_sc16():
    """Run Baseline 2: Self-Consistency with 16 samples."""
    print("\n" + "="*70)
    print("PHASE 2: BASELINE - SELF-CONSISTENCY (16 samples)")
    print("="*70)
    
    results = []
    # Run on smaller subset due to compute cost
    for dataset in ["gsm8k", "gpqa"]:
        max_prob = SUBSET_SIZES.get(dataset, 100)
        seed = 42  # Single seed for SC16 due to cost
        
        if experiment_exists("sc16", dataset, seed):
            print(f"Skipping sc16_{dataset}_s{seed} (already exists)")
            continue
        
        cmd = [
            "python", "run_real_experiments.py",
            "--model", MODEL,
            "--dataset", dataset,
            "--method", "sc16",
            "--seed", str(seed),
            "--max_problems", str(max_prob),
            "--output_dir", "results"
        ]
        
        success = run_command(cmd, f"SC16 on {dataset} subset (seed={seed})")
        results.append(("sc16", dataset, seed, success))
    
    return results

def run_cdhr_main():
    """Run Main CDHR experiments."""
    print("\n" + "="*70)
    print("PHASE 3: MAIN CDHR METHOD")
    print("="*70)
    
    results = []
    for dataset in ["gsm8k", "gpqa", "aime"]:
        max_prob = FULL_DATASETS.get(dataset)
        for seed in SEEDS:
            if experiment_exists("cdhr", dataset, seed):
                print(f"Skipping cdhr_{dataset}_s{seed} (already exists)")
                continue
            
            cmd = [
                "python", "run_real_experiments.py",
                "--model", MODEL,
                "--dataset", dataset,
                "--method", "cdhr",
                "--seed", str(seed),
                "--beta", "0.5",
                "--theta_v", "0.05",
                "--theta_sigma", "0.1",
                "--output_dir", "results"
            ]
            if max_prob:
                cmd.extend(["--max_problems", str(max_prob)])
            
            success = run_command(cmd, f"CDHR on {dataset} (seed={seed})")
            results.append(("cdhr", dataset, seed, success))
    
    return results

def run_ablations():
    """Run ablation studies."""
    print("\n" + "="*70)
    print("PHASE 4: ABLATION STUDIES")
    print("="*70)
    
    results = []
    dataset = "gsm8k"
    max_prob = SUBSET_SIZES.get(dataset, 200)
    seed = 42
    
    # Ablation 1: Beta sensitivity
    ablations = [
        ("cdhr_beta0.0", "0.0", "0.05", "0.1"),
        ("cdhr_beta0.25", "0.25", "0.05", "0.1"),
        ("cdhr_beta0.75", "0.75", "0.05", "0.1"),
        ("cdhr_beta1.0", "1.0", "0.05", "0.1"),
    ]
    
    for method_name, beta, theta_v, theta_sigma in ablations:
        if experiment_exists(method_name, dataset, seed):
            print(f"Skipping {method_name}_{dataset}_s{seed} (already exists)")
            continue
        
        cmd = [
            "python", "run_real_experiments.py",
            "--model", MODEL,
            "--dataset", dataset,
            "--method", "cdhr",
            "--seed", str(seed),
            "--beta", beta,
            "--theta_v", theta_v,
            "--theta_sigma", theta_sigma,
            "--max_problems", str(max_prob),
            "--output_dir", "results/ablation_beta"
        ]
        
        success = run_command(cmd, f"{method_name} on {dataset}")
        results.append((method_name, dataset, seed, success))
    
    # Ablation 2: Threshold sensitivity
    threshold_ablations = [
        ("cdhr_tv0.03", "0.5", "0.03", "0.1"),
        ("cdhr_tv0.07", "0.5", "0.07", "0.1"),
        ("cdhr_ts0.075", "0.5", "0.05", "0.075"),
        ("cdhr_ts0.125", "0.5", "0.05", "0.125"),
    ]
    
    for method_name, beta, theta_v, theta_sigma in threshold_ablations:
        if experiment_exists(method_name, dataset, seed):
            print(f"Skipping {method_name}_{dataset}_s{seed} (already exists)")
            continue
        
        cmd = [
            "python", "run_real_experiments.py",
            "--model", MODEL,
            "--dataset", dataset,
            "--method", "cdhr",
            "--seed", str(seed),
            "--beta", beta,
            "--theta_v", theta_v,
            "--theta_sigma", theta_sigma,
            "--max_problems", str(max_prob),
            "--output_dir", "results/ablation_thresholds"
        ]
        
        success = run_command(cmd, f"{method_name} on {dataset}")
        results.append((method_name, dataset, seed, success))
    
    return results

def aggregate_all_results():
    """Aggregate all results into final results.json."""
    print("\n" + "="*70)
    print("PHASE 5: AGGREGATING RESULTS")
    print("="*70)
    
    results_by_experiment = {}
    
    # Collect all result files
    for root, dirs, files in os.walk("results"):
        for file in files:
            if file.endswith(".json") and not file.startswith("aggregated"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    method = data.get("method", "unknown")
                    dataset = data.get("dataset", "unknown")
                    seed = data.get("seed", 0)
                    
                    key = f"{method}_{dataset}"
                    if key not in results_by_experiment:
                        results_by_experiment[key] = {
                            "method": method,
                            "dataset": dataset,
                            "seeds": [],
                            "accuracies": [],
                            "tokens": [],
                            "latencies": [],
                        }
                    
                    results_by_experiment[key]["seeds"].append(seed)
                    results_by_experiment[key]["accuracies"].append(data.get("accuracy", 0))
                    results_by_experiment[key]["tokens"].append(data.get("avg_tokens", 0))
                    results_by_experiment[key]["latencies"].append(data.get("avg_latency", 0))
                    
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
    
    # Compute statistics
    final_results = {}
    for key, data in results_by_experiment.items():
        accs = data["accuracies"]
        toks = data["tokens"]
        lats = data["latencies"]
        
        final_results[key] = {
            "method": data["method"],
            "dataset": data["dataset"],
            "num_seeds": len(accs),
            "accuracy_mean": sum(accs) / len(accs) if accs else 0,
            "accuracy_std": (sum((x - sum(accs)/len(accs))**2 for x in accs) / len(accs))**0.5 if len(accs) > 1 else 0,
            "tokens_mean": sum(toks) / len(toks) if toks else 0,
            "tokens_std": (sum((x - sum(toks)/len(toks))**2 for x in toks) / len(toks))**0.5 if len(toks) > 1 else 0,
            "latency_mean": sum(lats) / len(lats) if lats else 0,
            "latency_std": (sum((x - sum(lats)/len(lats))**2 for x in lats) / len(lats))**0.5 if len(lats) > 1 else 0,
            "raw_results": data,
        }
    
    # Save aggregated results
    with open("results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nAggregated results saved to results.json")
    print(f"Total experiments: {len(final_results)}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    for key, data in sorted(final_results.items()):
        print(f"{key:30s}: Acc={data['accuracy_mean']:.4f} ± {data['accuracy_std']:.4f} "
              f"({data['num_seeds']} seeds)")
    
    return final_results

def main():
    """Run all experiments."""
    start_time = time.time()
    
    print("="*70)
    print("CDHR REAL EXPERIMENTS - FULL EXECUTION")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL}")
    print("="*70)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/ablation_beta", exist_ok=True)
    os.makedirs("results/ablation_thresholds", exist_ok=True)
    
    all_results = []
    
    # Phase 1: Baseline CoT
    all_results.extend(run_baseline_cot())
    
    # Phase 2: Baseline SC16
    all_results.extend(run_baseline_sc16())
    
    # Phase 3: Main CDHR
    all_results.extend(run_cdhr_main())
    
    # Phase 4: Ablations
    all_results.extend(run_ablations())
    
    # Phase 5: Aggregate results
    aggregate_all_results()
    
    # Final summary
    total_time = time.time() - start_time
    success_count = sum(1 for r in all_results if r[3])
    fail_count = len(all_results) - success_count
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total experiments: {len(all_results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()

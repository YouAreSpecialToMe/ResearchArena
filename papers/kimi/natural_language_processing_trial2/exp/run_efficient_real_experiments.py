#!/usr/bin/env python3
"""
Efficient real CDHR experiments within 8-hour time budget.
Uses smaller subsets but ensures all experiments are ACTUALLY executed.
"""
import os
import sys
import json
import time
import subprocess
from datetime import datetime
from typing import List, Dict, Tuple

MODEL = "llama-3.1-8b"
SEEDS = [42, 123, 456]

# Optimized subset sizes for 8-hour budget
# At ~10s per problem, we can process ~2880 problems total
SUBSET_SIZES = {
    "gsm8k": 200,     # 200 problems for main experiments
    "gpqa": 100,      # 100 problems (full dataset is 198)
    "aime": 30,       # Small dataset
}

def run_experiment(method: str, dataset: str, seed: int, 
                   max_problems: int, extra_args: List[str] = None,
                   output_subdir: str = "") -> Tuple[bool, str]:
    """Run a single experiment."""
    output_dir = f"results/{output_subdir}" if output_subdir else "results"
    os.makedirs(output_dir, exist_ok=True)
    
    result_file = f"{output_dir}/{method}_{dataset}_s{seed}.json"
    if os.path.exists(result_file):
        print(f"  [SKIP] {result_file} already exists")
        return True, result_file
    
    cmd = [
        "python", "run_real_experiments.py",
        "--model", MODEL,
        "--dataset", dataset,
        "--method", method,
        "--seed", str(seed),
        "--max_problems", str(max_problems),
        "--output_dir", output_dir
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n  [RUN] {method} on {dataset} (seed={seed}, n={max_problems})")
    start = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=7200)
        elapsed = time.time() - start
        print(f"  [DONE] in {elapsed/60:.1f} min -> {result_file}")
        return True, result_file
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] after 2 hours")
        return False, ""
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"  [FAILED] after {elapsed/60:.1f} min: {e.stderr[:200]}")
        return False, ""

def run_all_experiments():
    """Execute all experiments in optimal order."""
    start_time = time.time()
    results_log = []
    
    print("="*70)
    print("CDHR REAL EXPERIMENTS - EFFICIENT EXECUTION PLAN")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Budget: ~8 hours | Model: Llama-3.1-8B")
    print("="*70)
    
    # PHASE 1: Standard CoT Baseline (3 seeds, main datasets)
    print("\n" + "-"*70)
    print("PHASE 1: Baseline - Standard CoT")
    print("-"*70)
    for dataset in ["gsm8k", "gpqa", "aime"]:
        for seed in SEEDS:
            success, path = run_experiment("cot", dataset, seed, SUBSET_SIZES[dataset])
            results_log.append(("cot", dataset, seed, success))
    
    # PHASE 2: Self-Consistency SC16 (1 seed, subset)
    print("\n" + "-"*70)
    print("PHASE 2: Baseline - Self-Consistency (16 samples)")
    print("-"*70)
    for dataset in ["gsm8k", "gpqa"]:
        success, path = run_experiment("sc16", dataset, 42, SUBSET_SIZES[dataset] // 2)  # Smaller for SC16
        results_log.append(("sc16", dataset, 42, success))
    
    # PHASE 3: Main CDHR (3 seeds)
    print("\n" + "-"*70)
    print("PHASE 3: Main CDHR Method")
    print("-"*70)
    for dataset in ["gsm8k", "gpqa", "aime"]:
        for seed in SEEDS:
            success, path = run_experiment("cdhr", dataset, seed, SUBSET_SIZES[dataset],
                extra_args=["--beta", "0.5", "--theta_v", "0.05", "--theta_sigma", "0.1"])
            results_log.append(("cdhr", dataset, seed, success))
    
    # PHASE 4: Ablation - Beta values (seed 42, GSM8K only)
    print("\n" + "-"*70)
    print("PHASE 4: Ablation - Beta Sensitivity")
    print("-"*70)
    for beta in ["0.0", "0.25", "0.75", "1.0"]:
        method_name = f"cdhr_beta{beta}"
        success, path = run_experiment("cdhr", "gsm8k", 42, 100,
            extra_args=["--beta", beta, "--theta_v", "0.05", "--theta_sigma", "0.1"],
            output_subdir="ablation_beta")
        results_log.append((method_name, "gsm8k", 42, success))
    
    # PHASE 5: Ablation - Threshold sensitivity (seed 42, GSM8K only)
    print("\n" + "-"*70)
    print("PHASE 5: Ablation - Threshold Sensitivity")
    print("-"*70)
    threshold_configs = [
        ("cdhr_tv0.03", "0.5", "0.03", "0.1"),
        ("cdhr_tv0.07", "0.5", "0.07", "0.1"),
        ("cdhr_ts0.075", "0.5", "0.05", "0.075"),
        ("cdhr_ts0.125", "0.5", "0.05", "0.125"),
    ]
    for method_name, beta, theta_v, theta_sigma in threshold_configs:
        success, path = run_experiment("cdhr", "gsm8k", 42, 100,
            extra_args=["--beta", beta, "--theta_v", theta_v, "--theta_sigma", theta_sigma],
            output_subdir="ablation_thresholds")
        results_log.append((method_name, "gsm8k", 42, success))
    
    # PHASE 6: Aggregate results
    print("\n" + "-"*70)
    print("PHASE 6: Aggregating Results")
    print("-"*70)
    aggregate_results()
    
    # Final summary
    total_time = time.time() - start_time
    success_count = sum(1 for _, _, _, s in results_log if s)
    
    print("\n" + "="*70)
    print("EXPERIMENT EXECUTION COMPLETE")
    print("="*70)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Experiments: {len(results_log)} total, {success_count} successful")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return results_log

def aggregate_results():
    """Aggregate all results into results.json."""
    all_results = {}
    
    # Walk through results directory
    for root, dirs, files in os.walk("results"):
        for file in files:
            if not file.endswith(".json"):
                continue
            
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                method = data.get("method", "unknown")
                dataset = data.get("dataset", "unknown")
                seed = data.get("seed", 0)
                
                # Handle ablation methods
                if "ablation_beta" in filepath:
                    beta = data.get("config", {}).get("beta", "unknown")
                    method = f"cdhr_beta{beta}"
                elif "ablation_thresholds" in filepath:
                    tv = data.get("config", {}).get("theta_v", "unknown")
                    ts = data.get("config", {}).get("theta_sigma", "unknown")
                    if tv != 0.05:
                        method = f"cdhr_tv{tv}"
                    else:
                        method = f"cdhr_ts{ts}"
                
                key = f"{method}_{dataset}"
                if key not in all_results:
                    all_results[key] = {
                        "method": method,
                        "dataset": dataset,
                        "seeds": [],
                        "accuracies": [],
                        "tokens": [],
                        "latencies": [],
                    }
                
                all_results[key]["seeds"].append(seed)
                all_results[key]["accuracies"].append(data.get("accuracy", 0))
                all_results[key]["tokens"].append(data.get("avg_tokens", 0))
                all_results[key]["latencies"].append(data.get("avg_latency", 0))
                
            except Exception as e:
                print(f"  Error reading {filepath}: {e}")
    
    # Compute statistics
    final_results = {}
    for key, data in all_results.items():
        accs = data["accuracies"]
        toks = data["tokens"]
        lats = data["latencies"]
        
        n = len(accs)
        final_results[key] = {
            "method": data["method"],
            "dataset": data["dataset"],
            "num_seeds": n,
            "accuracy": {
                "mean": sum(accs) / n,
                "std": (sum((x - sum(accs)/n)**2 for x in accs) / n)**0.5 if n > 1 else 0,
                "values": accs,
            },
            "tokens": {
                "mean": sum(toks) / n,
                "std": (sum((x - sum(toks)/n)**2 for x in toks) / n)**0.5 if n > 1 else 0,
            },
            "latency": {
                "mean": sum(lats) / n,
                "std": (sum((x - sum(lats)/n)**2 for x in lats) / n)**0.5 if n > 1 else 0,
            },
        }
    
    # Save
    with open("results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"  Saved aggregated results to results.json ({len(final_results)} experiment groups)")
    
    # Print summary table
    print("\n  Summary:")
    print(f"  {'Experiment':<30} {'Accuracy':<20} {'Tokens':<15} {'N':<5}")
    print("  " + "-"*75)
    for key in sorted(final_results.keys()):
        r = final_results[key]
        acc_str = f"{r['accuracy']['mean']:.4f} ± {r['accuracy']['std']:.4f}"
        tok_str = f"{r['tokens']['mean']:.1f}"
        print(f"  {key:<30} {acc_str:<20} {tok_str:<15} {r['num_seeds']:<5}")

if __name__ == "__main__":
    run_all_experiments()

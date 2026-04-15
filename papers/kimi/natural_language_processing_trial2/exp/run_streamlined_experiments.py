#!/usr/bin/env python3
"""
Streamlined real CDHR experiments - focused on getting actual results.
Uses smaller problem sets to ensure completion within time budget.
"""
import os
import sys
import json
import time
import subprocess
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

# Even smaller subsets for faster execution
# At ~10s per problem, 100 problems = ~17 minutes
# With model loading overhead (~2 min per run)
SUBSETS = {
    "gsm8k": 100,
    "gpqa": 50,
    "aime": 30,
}

def run_single_experiment(method: str, dataset: str, seed: int, 
                          max_probs: int, extra_args: List[str] = None,
                          output_subdir: str = "") -> Dict:
    """Run a single experiment and return results directly."""
    output_dir = f"results/{output_subdir}" if output_subdir else "results"
    os.makedirs(output_dir, exist_ok=True)
    
    result_file = f"{output_dir}/{method}_{dataset}_s{seed}.json"
    
    # Check if already exists
    if os.path.exists(result_file):
        print(f"  [EXIST] {result_file}")
        with open(result_file, 'r') as f:
            return json.load(f)
    
    cmd = [
        "python", "run_real_experiments.py",
        "--model", "llama-3.1-8b",
        "--dataset", dataset,
        "--method", method,
        "--seed", str(seed),
        "--max_problems", str(max_probs),
        "--output_dir", output_dir
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n[RUN] {' '.join(cmd[-6:])}")
    start = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - start
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
            print(f"  [DONE] {elapsed/60:.1f}min - Acc: {data.get('accuracy', 0):.3f}")
            return data
        else:
            print(f"  [ERROR] Result file not created")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] after 1 hour")
        return None
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"  [FAILED] after {elapsed/60:.1f}min")
        print(f"  STDERR: {e.stderr[:300]}")
        return None

def run_experiment_set(experiments: List[Tuple], description: str) -> List[Dict]:
    """Run a set of experiments."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    
    results = []
    for exp in experiments:
        if len(exp) == 4:
            method, dataset, seed, max_probs = exp
            extra = None
            subdir = ""
        else:
            method, dataset, seed, max_probs, extra, subdir = exp
        
        result = run_single_experiment(method, dataset, seed, max_probs, extra, subdir)
        if result:
            results.append(result)
    
    return results

def aggregate_and_save():
    """Aggregate all results and save to results.json."""
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)
    
    all_results = {}
    
    # Walk through results directory
    for root, dirs, files in os.walk("results"):
        for file in files:
            if not file.endswith(".json") or file == "results.json":
                continue
            
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                method = data.get("method", "unknown")
                dataset = data.get("dataset", "unknown")
                seed = data.get("seed", 0)
                
                # Handle ablations
                config = data.get("config", {})
                if "ablation_beta" in filepath:
                    beta = config.get("beta", "?")
                    method = f"cdhr_beta{beta}"
                elif "ablation_thresholds" in filepath:
                    tv = config.get("theta_v", 0.05)
                    ts = config.get("theta_sigma", 0.1)
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
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)) if n > 1 else 0.0,
                "values": accs,
            },
            "tokens": {
                "mean": float(np.mean(toks)),
                "std": float(np.std(toks)) if n > 1 else 0.0,
            },
            "latency": {
                "mean": float(np.mean(lats)),
                "std": float(np.std(lats)) if n > 1 else 0.0,
            },
        }
    
    # Save
    with open("results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nSaved results.json with {len(final_results)} experiment groups")
    
    # Print summary
    print("\n" + "-"*70)
    print(f"{'Experiment':<30} {'Accuracy':<20} {'Tokens':<15} {'N':<5}")
    print("-"*70)
    for key in sorted(final_results.keys()):
        r = final_results[key]
        acc_str = f"{r['accuracy']['mean']*100:.2f} ± {r['accuracy']['std']*100:.2f}"
        tok_str = f"{r['tokens']['mean']:.1f}"
        print(f"{key:<30} {acc_str:<20} {tok_str:<15} {r['num_seeds']:<5}")
    print("-"*70)
    
    return final_results

def main():
    """Main experiment execution."""
    start_time = time.time()
    
    print("="*70)
    print("CDHR STREAMLINED EXPERIMENTS")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    all_results = []
    
    # Phase 1: Standard CoT (3 seeds)
    cot_exps = [
        ("cot", "gsm8k", 42, SUBSETS["gsm8k"]),
        ("cot", "gsm8k", 123, SUBSETS["gsm8k"]),
        ("cot", "gsm8k", 456, SUBSETS["gsm8k"]),
        ("cot", "gpqa", 42, SUBSETS["gpqa"]),
        ("cot", "gpqa", 123, SUBSETS["gpqa"]),
        ("cot", "gpqa", 456, SUBSETS["gpqa"]),
        ("cot", "aime", 42, SUBSETS["aime"]),
    ]
    all_results.extend(run_experiment_set(cot_exps, "PHASE 1: Standard CoT Baseline"))
    
    # Phase 2: Self-Consistency (1 seed, smaller subset)
    sc_exps = [
        ("sc16", "gsm8k", 42, 50),
        ("sc16", "gpqa", 42, 30),
    ]
    all_results.extend(run_experiment_set(sc_exps, "PHASE 2: Self-Consistency (16 samples)"))
    
    # Phase 3: CDHR Main (3 seeds)
    cdhr_exps = [
        ("cdhr", "gsm8k", 42, SUBSETS["gsm8k"], ["--beta", "0.5", "--theta_v", "0.05", "--theta_sigma", "0.1"], ""),
        ("cdhr", "gsm8k", 123, SUBSETS["gsm8k"], ["--beta", "0.5", "--theta_v", "0.05", "--theta_sigma", "0.1"], ""),
        ("cdhr", "gsm8k", 456, SUBSETS["gsm8k"], ["--beta", "0.5", "--theta_v", "0.05", "--theta_sigma", "0.1"], ""),
        ("cdhr", "gpqa", 42, SUBSETS["gpqa"], ["--beta", "0.5", "--theta_v", "0.05", "--theta_sigma", "0.1"], ""),
        ("cdhr", "aime", 42, SUBSETS["aime"], ["--beta", "0.5", "--theta_v", "0.05", "--theta_sigma", "0.1"], ""),
    ]
    all_results.extend(run_experiment_set(cdhr_exps, "PHASE 3: CDHR Main Method"))
    
    # Phase 4: Beta ablation
    beta_exps = [
        ("cdhr", "gsm8k", 42, 50, ["--beta", "0.0", "--theta_v", "0.05", "--theta_sigma", "0.1"], "ablation_beta"),
        ("cdhr", "gsm8k", 42, 50, ["--beta", "0.25", "--theta_v", "0.05", "--theta_sigma", "0.1"], "ablation_beta"),
        ("cdhr", "gsm8k", 42, 50, ["--beta", "0.75", "--theta_v", "0.05", "--theta_sigma", "0.1"], "ablation_beta"),
        ("cdhr", "gsm8k", 42, 50, ["--beta", "1.0", "--theta_v", "0.05", "--theta_sigma", "0.1"], "ablation_beta"),
    ]
    all_results.extend(run_experiment_set(beta_exps, "PHASE 4: Beta Ablation"))
    
    # Phase 5: Threshold ablation
    thresh_exps = [
        ("cdhr", "gsm8k", 42, 50, ["--beta", "0.5", "--theta_v", "0.03", "--theta_sigma", "0.1"], "ablation_thresholds"),
        ("cdhr", "gsm8k", 42, 50, ["--beta", "0.5", "--theta_v", "0.07", "--theta_sigma", "0.1"], "ablation_thresholds"),
        ("cdhr", "gsm8k", 42, 50, ["--beta", "0.5", "--theta_v", "0.05", "--theta_sigma", "0.075"], "ablation_thresholds"),
        ("cdhr", "gsm8k", 42, 50, ["--beta", "0.5", "--theta_v", "0.05", "--theta_sigma", "0.125"], "ablation_thresholds"),
    ]
    all_results.extend(run_experiment_set(thresh_exps, "PHASE 5: Threshold Ablation"))
    
    # Aggregate results
    final_results = aggregate_and_save()
    
    # Summary
    total_time = time.time() - start_time
    success_count = len([r for r in all_results if r is not None])
    
    print("\n" + "="*70)
    print("EXPERIMENT EXECUTION COMPLETE")
    print("="*70)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Successful experiments: {success_count}/{len(all_results)}")
    print(f"Results saved to: results.json")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()

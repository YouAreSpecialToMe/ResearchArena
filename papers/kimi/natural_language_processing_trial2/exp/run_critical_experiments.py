#!/usr/bin/env python3
"""
Critical CDHR Experiments - Execute the minimum required experiments to address feedback.
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

# Configuration
DATASETS = {
    "gsm8k": ("data/gsm8k.json", 100),  # (path, limit)
    "math": ("data/math.json", 50),
    "gpqa": ("data/gpqa.json", 50),
}

SEEDS = [42, 123, 456]

def run_experiment_script(script_path: str, args: dict, timeout: int = 7200) -> Dict:
    """Run an experiment script with given arguments."""
    import subprocess
    
    cmd = ["python", script_path]
    for key, val in args.items():
        cmd.extend([f"--{key}", str(val)])
    
    print(f"\n[RUN] {' '.join(cmd)}")
    start = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start
        print(f"  [DONE] {elapsed/60:.1f}min")
        return {"success": True, "time": elapsed, "stdout": result.stdout[-500:], "stderr": result.stderr[-500:]}
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] after {timeout}s")
        return {"success": False, "error": "timeout"}
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"  [FAILED] after {elapsed/60:.1f}min")
        print(f"  Error: {e.stderr[-300:]}")
        return {"success": False, "error": str(e)}

def run_cot_baseline(model: str, dataset: str, seed: int, limit: int) -> Dict:
    """Run CoT baseline."""
    dataset_path, _ = DATASETS[dataset]
    output = f"results/baseline_cot/{model}_{dataset}_seed{seed}.json"
    
    if os.path.exists(output):
        print(f"  [SKIP] Already exists: {output}")
        with open(output, 'r') as f:
            return json.load(f)
    
    args = {
        "model": model,
        "dataset": dataset_path,
        "output": output,
        "seed": seed,
        "limit": limit,
    }
    
    result = run_experiment_script("exp/baseline_cot/run.py", args)
    if result["success"] and os.path.exists(output):
        with open(output, 'r') as f:
            return json.load(f)
    return None

def run_cdhr_main(model: str, dataset: str, seed: int, limit: int, beta: float = 0.5) -> Dict:
    """Run CDHR main experiment."""
    dataset_path, _ = DATASETS[dataset]
    output = f"results/cdhr_main/{model}_{dataset}_beta{beta}_seed{seed}.json"
    
    if os.path.exists(output):
        print(f"  [SKIP] Already exists: {output}")
        with open(output, 'r') as f:
            return json.load(f)
    
    args = {
        "model": model,
        "dataset": dataset_path,
        "output": output,
        "seed": seed,
        "limit": limit,
        "beta": beta,
    }
    
    result = run_experiment_script("exp/cdhr_main/run.py", args)
    if result["success"] and os.path.exists(output):
        with open(output, 'r') as f:
            return json.load(f)
    return None

def run_sc16_baseline(model: str, dataset: str, seed: int, limit: int) -> Dict:
    """Run Self-Consistency with 16 samples."""
    dataset_path, _ = DATASETS[dataset]
    output = f"results/baseline_sc16/{model}_{dataset}_seed{seed}.json"
    
    if os.path.exists(output):
        print(f"  [SKIP] Already exists: {output}")
        with open(output, 'r') as f:
            return json.load(f)
    
    args = {
        "model": model,
        "dataset": dataset_path,
        "output": output,
        "seed": seed,
        "limit": limit,
        "samples": 16,
    }
    
    result = run_experiment_script("exp/baseline_sc16/run.py", args)
    if result["success"] and os.path.exists(output):
        with open(output, 'r') as f:
            return json.load(f)
    return None

def run_com_baseline(model: str, dataset: str, seed: int, limit: int) -> Dict:
    """Run Chain of Mindset baseline."""
    dataset_path, _ = DATASETS[dataset]
    output = f"results/baseline_com/{model}_{dataset}_seed{seed}.json"
    
    if os.path.exists(output):
        print(f"  [SKIP] Already exists: {output}")
        with open(output, 'r') as f:
            return json.load(f)
    
    args = {
        "model": model,
        "dataset": dataset_path,
        "output": output,
        "seed": seed,
        "limit": limit,
    }
    
    result = run_experiment_script("exp/baseline_com/run.py", args)
    if result["success"] and os.path.exists(output):
        with open(output, 'r') as f:
            return json.load(f)
    return None

def run_beta_ablation(model: str, dataset: str, seed: int, limit: int, beta: float) -> Dict:
    """Run CDHR with specific beta value."""
    dataset_path, _ = DATASETS[dataset]
    output = f"results/ablation_beta/{model}_{dataset}_beta{beta}_seed{seed}.json"
    
    os.makedirs("results/ablation_beta", exist_ok=True)
    
    if os.path.exists(output):
        print(f"  [SKIP] Already exists: {output}")
        with open(output, 'r') as f:
            return json.load(f)
    
    args = {
        "model": model,
        "dataset": dataset_path,
        "output": output,
        "seed": seed,
        "limit": limit,
        "beta": beta,
    }
    
    result = run_experiment_script("exp/cdhr_main/run.py", args)
    if result["success"] and os.path.exists(output):
        with open(output, 'r') as f:
            return json.load(f)
    return None

def aggregate_results():
    """Aggregate all results into results.json."""
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)
    
    results = {
        "experiment_info": {
            "title": "CDHR Experiments",
            "date": datetime.now().isoformat(),
            "models": ["llama-3.1-8b"],
            "datasets": list(DATASETS.keys()),
        },
        "main_results": {},
        "ablation_studies": {},
    }
    
    # Aggregate baselines
    baseline_methods = {
        "cot": "results/baseline_cot",
        "sc16": "results/baseline_sc16",
        "com": "results/baseline_com",
    }
    
    for method_name, result_dir in baseline_methods.items():
        if not os.path.exists(result_dir):
            continue
        
        for file in os.listdir(result_dir):
            if not file.endswith('.json'):
                continue
            
            filepath = os.path.join(result_dir, file)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                dataset = os.path.basename(data.get("dataset", "unknown")).replace('.json', '')
                key = f"{method_name}_{dataset}"
                
                if key not in results["main_results"]:
                    results["main_results"][key] = {
                        "method": method_name,
                        "dataset": dataset,
                        "accuracies": [],
                        "tokens": [],
                        "latencies": [],
                    }
                
                metrics = data.get("metrics", {})
                results["main_results"][key]["accuracies"].append(metrics.get("accuracy", 0))
                results["main_results"][key]["tokens"].append(metrics.get("avg_tokens", 0))
                results["main_results"][key]["latencies"].append(metrics.get("avg_latency", 0))
                
            except Exception as e:
                print(f"  Error reading {filepath}: {e}")
    
    # Aggregate CDHR results
    if os.path.exists("results/cdhr_main"):
        for file in os.listdir("results/cdhr_main"):
            if not file.endswith('.json'):
                continue
            
            filepath = os.path.join("results/cdhr_main", file)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                dataset = os.path.basename(data.get("dataset", "unknown")).replace('.json', '')
                params = data.get("parameters", {})
                beta = params.get("beta", 0.5)
                
                key = f"cdhr_beta{beta}_{dataset}"
                
                if key not in results["main_results"]:
                    results["main_results"][key] = {
                        "method": f"cdhr_beta{beta}",
                        "dataset": dataset,
                        "accuracies": [],
                        "tokens": [],
                        "latencies": [],
                        "strategy_entropies": [],
                    }
                
                metrics = data.get("metrics", {})
                results["main_results"][key]["accuracies"].append(metrics.get("accuracy", 0))
                results["main_results"][key]["tokens"].append(metrics.get("avg_tokens", 0))
                results["main_results"][key]["latencies"].append(metrics.get("avg_latency", 0))
                results["main_results"][key]["strategy_entropies"].append(metrics.get("strategy_entropy", 0))
                
            except Exception as e:
                print(f"  Error reading {filepath}: {e}")
    
    # Aggregate beta ablation
    if os.path.exists("results/ablation_beta"):
        for file in os.listdir("results/ablation_beta"):
            if not file.endswith('.json'):
                continue
            
            filepath = os.path.join("results/ablation_beta", file)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                dataset = os.path.basename(data.get("dataset", "unknown")).replace('.json', '')
                params = data.get("parameters", {})
                beta = params.get("beta", 0.5)
                
                key = f"beta_{beta}"
                
                if key not in results["ablation_studies"]:
                    results["ablation_studies"][key] = {
                        "beta": beta,
                        "dataset": dataset,
                        "accuracies": [],
                        "tokens": [],
                    }
                
                metrics = data.get("metrics", {})
                results["ablation_studies"][key]["accuracies"].append(metrics.get("accuracy", 0))
                results["ablation_studies"][key]["tokens"].append(metrics.get("avg_tokens", 0))
                
            except Exception as e:
                print(f"  Error reading {filepath}: {e}")
    
    # Compute statistics
    for section in ["main_results", "ablation_studies"]:
        for key, data in results[section].items():
            accs = data.get("accuracies", [])
            toks = data.get("tokens", [])
            lats = data.get("latencies", [])
            
            data["accuracy_mean"] = float(np.mean(accs)) if accs else 0
            data["accuracy_std"] = float(np.std(accs)) if len(accs) > 1 else 0
            data["tokens_mean"] = float(np.mean(toks)) if toks else 0
            data["latency_mean"] = float(np.mean(lats)) if lats else 0
            data["num_seeds"] = len(accs)
    
    # Save results
    with open("results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results.json with {len(results['main_results'])} main results and {len(results['ablation_studies'])} ablations")
    
    # Print summary table
    print("\n" + "-"*80)
    print(f"{'Experiment':<35} {'Accuracy':<20} {'Tokens':<15} {'N':<5}")
    print("-"*80)
    for key in sorted(results["main_results"].keys()):
        r = results["main_results"][key]
        acc_str = f"{r['accuracy_mean']*100:.2f} ± {r['accuracy_std']*100:.2f}" if r['accuracy_std'] > 0 else f"{r['accuracy_mean']*100:.2f}"
        tok_str = f"{r['tokens_mean']:.1f}"
        print(f"{key:<35} {acc_str:<20} {tok_str:<15} {r['num_seeds']:<5}")
    print("-"*80)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama-3.1-8b")
    parser.add_argument("--quick", action="store_true", help="Run with smaller limits for testing")
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*70)
    print("CDHR CRITICAL EXPERIMENTS")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Adjust limits if quick mode
    limits = {"gsm8k": 50, "math": 30, "gpqa": 30} if args.quick else {"gsm8k": 100, "math": 50, "gpqa": 50}
    
    all_results = []
    
    # PHASE 1: CoT Baseline (seed 456 - seeds 42, 123 already done)
    print("\n" + "="*70)
    print("PHASE 1: CoT Baseline (seed 456)")
    print("="*70)
    result = run_cot_baseline(args.model, "gsm8k", 456, limits["gsm8k"])
    if result:
        all_results.append(("cot_gsm8k_s456", result))
    
    # Also run CoT on other datasets
    for dataset in ["math", "gpqa"]:
        for seed in SEEDS:
            result = run_cot_baseline(args.model, dataset, seed, limits[dataset])
            if result:
                all_results.append((f"cot_{dataset}_s{seed}", result))
    
    # PHASE 2: CDHR Main (3 seeds on GSM8K)
    print("\n" + "="*70)
    print("PHASE 2: CDHR Main (GSM8K, 3 seeds)")
    print("="*70)
    for seed in SEEDS:
        result = run_cdhr_main(args.model, "gsm8k", seed, limits["gsm8k"], beta=0.5)
        if result:
            all_results.append((f"cdhr_gsm8k_s{seed}", result))
    
    # PHASE 3: CDHR on other datasets
    print("\n" + "="*70)
    print("PHASE 3: CDHR (Math, GPQA)")
    print("="*70)
    for dataset in ["math", "gpqa"]:
        for seed in SEEDS[:2]:  # 2 seeds for other datasets
            result = run_cdhr_main(args.model, dataset, seed, limits[dataset], beta=0.5)
            if result:
                all_results.append((f"cdhr_{dataset}_s{seed}", result))
    
    # PHASE 4: Self-Consistency Baseline
    print("\n" + "="*70)
    print("PHASE 4: Self-Consistency (16 samples)")
    print("="*70)
    for dataset in ["gsm8k", "math"]:
        result = run_sc16_baseline(args.model, dataset, 42, min(limits[dataset], 50))
        if result:
            all_results.append((f"sc16_{dataset}", result))
    
    # PHASE 5: Chain of Mindset Baseline
    print("\n" + "="*70)
    print("PHASE 5: Chain of Mindset")
    print("="*70)
    for dataset in ["gsm8k", "math"]:
        result = run_com_baseline(args.model, dataset, 42, min(limits[dataset], 50))
        if result:
            all_results.append((f"com_{dataset}", result))
    
    # PHASE 6: Beta Ablation
    print("\n" + "="*70)
    print("PHASE 6: Beta Sensitivity Ablation")
    print("="*70)
    beta_values = [0.0, 0.25, 0.75, 1.0]
    for beta in beta_values:
        result = run_beta_ablation(args.model, "gsm8k", 42, min(limits["gsm8k"], 50), beta)
        if result:
            all_results.append((f"beta_{beta}", result))
    
    # Aggregate results
    final_results = aggregate_results()
    
    # Summary
    total_time = time.time() - start_time
    success_count = len([r for r in all_results if r is not None])
    
    print("\n" + "="*70)
    print("EXPERIMENT EXECUTION COMPLETE")
    print("="*70)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Successful experiments: {success_count}")
    print(f"Results saved to: results.json")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()

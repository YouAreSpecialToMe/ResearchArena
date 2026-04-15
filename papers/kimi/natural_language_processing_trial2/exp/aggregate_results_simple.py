#!/usr/bin/env python3
"""Simple results aggregation."""
import json
import os
import numpy as np
from pathlib import Path

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None

results = {
    "experiment_info": {
        "title": "CDHR Experiments",
        "date": "2026-03-22",
        "models": ["llama-3.1-8b"],
        "datasets": ["gsm8k"],
    },
    "main_results": {},
    "ablation_studies": {}
}

# CoT Baseline
cot_files = [
    "results/cot_gsm8k_s42.json",
    "results/cot_gsm8k_s123.json", 
    "results/baseline_cot/llama-3.1-8b_gsm8k_seed456.json"
]
cot_accs = []
for f in cot_files:
    data = load_json(f)
    if data:
        cot_accs.append(data.get("accuracy", 0))
if cot_accs:
    results["main_results"]["cot_gsm8k"] = {
        "method": "CoT",
        "dataset": "gsm8k",
        "accuracy_mean": float(np.mean(cot_accs)),
        "accuracy_std": float(np.std(cot_accs)),
        "accuracies": cot_accs,
        "num_seeds": len(cot_accs)
    }

# CDHR Main
cdhr_files = [
    "results/cdhr_main/llama-3.1-8b_gsm8k_seed42.json",
    "results/cdhr_main/llama-3.1-8b_gsm8k_seed123.json"
]
cdhr_accs = []
cdhr_entropies = []
for f in cdhr_files:
    data = load_json(f)
    if data:
        metrics = data.get("metrics", {})
        cdhr_accs.append(metrics.get("accuracy", 0))
        cdhr_entropies.append(metrics.get("strategy_entropy", 0))
if cdhr_accs:
    results["main_results"]["cdhr_gsm8k"] = {
        "method": "CDHR",
        "dataset": "gsm8k",
        "accuracy_mean": float(np.mean(cdhr_accs)),
        "accuracy_std": float(np.std(cdhr_accs)),
        "accuracies": cdhr_accs,
        "strategy_entropy_mean": float(np.mean(cdhr_entropies)),
        "num_seeds": len(cdhr_accs)
    }

# SC16 Baseline
data = load_json("results/baseline_sc16/llama-3.1-8b_gsm8k_seed42.json")
if data:
    metrics = data.get("metrics", {})
    results["main_results"]["sc16_gsm8k"] = {
        "method": "Self-Consistency (16 samples)",
        "dataset": "gsm8k",
        "accuracy_mean": metrics.get("accuracy", 0),
        "accuracy_std": 0.0,
        "num_seeds": 1
    }

# CoM Baseline
data = load_json("results/baseline_com/llama-3.1-8b_gsm8k_seed42.json")
if data:
    metrics = data.get("metrics", {})
    results["main_results"]["com_gsm8k"] = {
        "method": "Chain of Mindset",
        "dataset": "gsm8k",
        "accuracy_mean": metrics.get("accuracy", 0),
        "accuracy_std": 0.0,
        "mindset_entropy": metrics.get("mindset_entropy", 0),
        "num_seeds": 1
    }

# Beta Ablation
for beta in [0.0]:
    f = f"results/ablation_beta/llama-3.1-8b_gsm8k_beta{beta}.json"
    data = load_json(f)
    if data:
        metrics = data.get("metrics", {})
        results["ablation_studies"][f"beta_{beta}"] = {
            "beta": beta,
            "accuracy": metrics.get("accuracy", 0),
            "strategy_entropy": metrics.get("strategy_entropy", 0)
        }

# Save results
with open("results.json", 'w') as f:
    json.dump(results, f, indent=2)

print("="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"\n{'Method':<30} {'Accuracy':<20} {'N':<5}")
print("-"*70)
for key, val in results["main_results"].items():
    acc = val['accuracy_mean'] * 100
    std = val['accuracy_std'] * 100
    n = val['num_seeds']
    print(f"{key:<30} {acc:.1f}% ± {std:.1f}%{'':<5} {n:<5}")
print("-"*70)

print("\n" + "="*70)
print("ABLATION STUDIES")
print("="*70)
print(f"\n{'Parameter':<20} {'Value':<15} {'Accuracy':<15}")
print("-"*70)
for key, val in results["ablation_studies"].items():
    print(f"{key:<20} {val.get('beta', 'N/A'):<15} {val.get('accuracy', 0)*100:.1f}%")
print("-"*70)

print("\nResults saved to results.json")

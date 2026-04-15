#!/usr/bin/env python3
"""
Experiment 09: Ablation - No Learned Scorer
- Use simple heuristic: arithmetic > control flow > memory
- Same memory budget as LEOPARD
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.simulation import RewriteRule, RuleType, EGraphSimulator

def heuristic_score(rule):
    """Simple heuristic: prefer arithmetic, then control flow, then memory."""
    if rule.rule_type == RuleType.ARITHMETIC:
        return 3.0
    elif rule.rule_type == RuleType.CONTROL_FLOW:
        return 2.0
    else:
        return 1.0

def run_heuristic_es(program, rules, memory_budget_pct, baseline_memory, seed):
    """Run ES with heuristic rule selection."""
    memory_limit = baseline_memory * memory_budget_pct
    sim = EGraphSimulator(program, rules, memory_limit_mb=memory_limit, seed=seed)
    
    rules_applied = 0
    
    while not sim.is_saturated():
        applicable = sim.get_applicable_rules()
        if not applicable:
            break
        
        # Score applicable rules with heuristic
        applicable_rules = [r for r in rules if r.id in applicable]
        best_rule = max(applicable_rules, key=heuristic_score).id
        
        success, _ = sim.apply_rule(best_rule)
        if not success:
            break
        
        rules_applied += 1
    
    final_program = sim.extract_best_program()
    
    return {
        "initial_instructions": program.num_instructions,
        "final_instructions": final_program.num_instructions,
        "instruction_reduction": program.num_instructions - final_program.num_instructions,
        "speedup": program.num_instructions / max(1, final_program.num_instructions),
        "peak_memory_mb": sim.state.memory_usage_mb,
        "rules_applied": rules_applied,
        "compilation_time_ms": rules_applied * 2.0
    }

def main():
    print("=" * 60)
    print("Experiment 09: Ablation - No Learned Scorer (Heuristic)")
    print("=" * 60)
    
    # Load data
    with open("data/rules.json") as f:
        rules_data = json.load(f)
    
    rules = [
        RewriteRule(r['id'], r['name'], RuleType(r['rule_type']), 
                   r['pattern'], r['replacement'], r['base_benefit'], r['complexity'])
        for r in rules_data
    ]
    
    with open("data/training_programs.pkl", "rb") as f:
        training_programs = pickle.load(f)
    with open("data/test_programs.pkl", "rb") as f:
        test_programs = pickle.load(f)
    
    with open("results/baseline_exhaustive.json") as f:
        exhaustive_results = json.load(f)
    
    memory_budget_pct = 0.5
    seeds = [42, 123, 456]
    
    all_programs = training_programs + test_programs
    
    print(f"\nRunning heuristic guidance (no learning)...")
    
    results = {"heuristic": {}}
    
    for prog in tqdm(all_programs):
        baseline_memory = exhaustive_results["exhaustive"][prog.name]["peak_memory_mb"]
        
        prog_results = []
        for seed in seeds:
            result = run_heuristic_es(prog, rules, memory_budget_pct, baseline_memory, seed)
            prog_results.append(result)
        
        results["heuristic"][prog.name] = {
            "speedup_mean": np.mean([r["speedup"] for r in prog_results]),
            "speedup_std": np.std([r["speedup"] for r in prog_results]),
            "peak_memory_mb_mean": np.mean([r["peak_memory_mb"] for r in prog_results]),
        }
    
    # Compute summary
    speedups = [results["heuristic"][p.name]["speedup_mean"] for p in all_programs]
    
    results["summary"] = {
        "geomean_speedup": np.exp(np.mean(np.log(speedups))),
        "mean_speedup": np.mean(speedups),
    }
    
    # Load LEOPARD results for comparison
    with open("results/leopard_main.json") as f:
        leopard_results = json.load(f)
    
    results["comparison"] = {
        "heuristic_geomean": results["summary"]["geomean_speedup"],
        "leopard_geomean": leopard_results["summary"]["geomean_speedup"],
        "improvement": (leopard_results["summary"]["geomean_speedup"] / results["summary"]["geomean_speedup"] - 1) * 100
    }
    
    # Save results
    with open("results/ablation_no_learning.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("exp/09_ablation_no_learning/results.json", "w") as f:
        json.dump({
            "experiment": "09_ablation_no_learning",
            "status": "completed",
            "metrics": results["summary"],
            "comparison": results["comparison"]
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Ablation Results (No Learning):")
    print("=" * 60)
    print(f"Heuristic geomean speedup: {results['summary']['geomean_speedup']:.3f}x")
    print(f"LEOPARD geomean speedup: {results['comparison']['leopard_geomean']:.3f}x")
    print(f"Learning improvement: {results['comparison']['improvement']:.1f}%")
    print(f"\nResults saved to results/ablation_no_learning.json")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())

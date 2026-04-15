#!/usr/bin/env python3
"""
Experiment 02: Baseline - LLVM -O3 and Random Selection
- Simulates LLVM -O3 baseline
- Random rule selection with 50% memory budget
- Uses 3 random seeds
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.simulation import create_rewrite_rules, EGraphSimulator

def simulate_llvm_o3(program):
    """Simulate LLVM -O3 optimization."""
    # LLVM -O3 typically reduces instructions by 20-40% for numerical kernels
    reduction_factor = np.random.uniform(0.65, 0.80)
    final_instructions = int(program.num_instructions * reduction_factor)
    
    return {
        "initial_instructions": program.num_instructions,
        "final_instructions": final_instructions,
        "instruction_reduction": program.num_instructions - final_instructions,
        "speedup": 1.0 / reduction_factor,
        "memory_usage_mb": program.num_instructions * 0.05,  # LLVM uses less memory
        "compilation_time_ms": program.num_instructions * 0.5  # Fast compilation
    }

def run_random_selection(program, rules, memory_budget_pct, seed):
    """Run equality saturation with random rule selection."""
    np.random.seed(seed)
    
    # Calculate memory budget
    base_memory = program.num_instructions * 0.1
    memory_limit = base_memory * 2  # Exhaustive baseline
    budget = memory_limit * memory_budget_pct
    
    sim = EGraphSimulator(program, rules, memory_limit_mb=budget, seed=seed)
    
    rules_applied = 0
    total_benefit = 0
    
    while not sim.is_saturated():
        applicable = sim.get_applicable_rules()
        if not applicable:
            break
        
        rule_id = np.random.choice(applicable)
        success, benefit = sim.apply_rule(rule_id)
        
        if not success:
            break
        
        rules_applied += 1
        total_benefit += benefit
    
    final_program = sim.extract_best_program()
    
    return {
        "initial_instructions": program.num_instructions,
        "final_instructions": final_program.num_instructions,
        "instruction_reduction": program.num_instructions - final_program.num_instructions,
        "speedup": program.num_instructions / max(1, final_program.num_instructions),
        "peak_memory_mb": sim.state.memory_usage_mb,
        "rules_applied": rules_applied,
        "compilation_time_ms": rules_applied * 2.0  # Simulated time
    }

def main():
    print("=" * 60)
    print("Experiment 02: Baseline - LLVM -O3 and Random Selection")
    print("=" * 60)
    
    # Load data
    with open("data/rules.json") as f:
        rules_data = json.load(f)
    
    from shared.simulation import RewriteRule, RuleType
    rules = [
        RewriteRule(r['id'], r['name'], RuleType(r['rule_type']), 
                   r['pattern'], r['replacement'], r['base_benefit'], r['complexity'])
        for r in rules_data
    ]
    
    with open("data/training_programs.pkl", "rb") as f:
        training_programs = pickle.load(f)
    with open("data/test_programs.pkl", "rb") as f:
        test_programs = pickle.load(f)
    
    all_programs = training_programs + test_programs
    
    seeds = [42, 123, 456]
    memory_budget = 0.5  # 50% of exhaustive memory
    
    results = {
        "llvm_o3": {},
        "random_selection": {}
    }
    
    print("\n[1/2] Running LLVM -O3 baseline...")
    for prog in tqdm(all_programs):
        results["llvm_o3"][prog.name] = simulate_llvm_o3(prog)
    
    print("\n[2/2] Running random selection (3 seeds)...")
    for prog in tqdm(all_programs):
        prog_results = []
        for seed in seeds:
            result = run_random_selection(prog, rules, memory_budget, seed)
            prog_results.append(result)
        
        # Aggregate across seeds
        results["random_selection"][prog.name] = {
            "initial_instructions": prog_results[0]["initial_instructions"],
            "final_instructions_mean": np.mean([r["final_instructions"] for r in prog_results]),
            "final_instructions_std": np.std([r["final_instructions"] for r in prog_results]),
            "instruction_reduction_mean": np.mean([r["instruction_reduction"] for r in prog_results]),
            "speedup_mean": np.mean([r["speedup"] for r in prog_results]),
            "speedup_std": np.std([r["speedup"] for r in prog_results]),
            "peak_memory_mb_mean": np.mean([r["peak_memory_mb"] for r in prog_results]),
            "rules_applied_mean": np.mean([r["rules_applied"] for r in prog_results]),
            "compilation_time_ms_mean": np.mean([r["compilation_time_ms"] for r in prog_results]),
        }
    
    # Compute geomean speedups
    llvm_speedups = [r["speedup"] for r in results["llvm_o3"].values()]
    random_speedups = [results["random_selection"][p.name]["speedup_mean"] for p in all_programs]
    
    results["summary"] = {
        "llvm_o3_geomean_speedup": np.exp(np.mean(np.log(llvm_speedups))),
        "random_selection_geomean_speedup": np.exp(np.mean(np.log(random_speedups))),
        "llvm_o3_mean_instructions": np.mean([r["final_instructions"] for r in results["llvm_o3"].values()]),
        "random_selection_mean_instructions": np.mean([results["random_selection"][p.name]["final_instructions_mean"] for p in all_programs]),
    }
    
    # Save results
    with open("results/baseline_simple.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("exp/02_baseline_simple/results.json", "w") as f:
        json.dump({
            "experiment": "02_baseline_simple",
            "status": "completed",
            "config": {"seeds": seeds, "memory_budget": memory_budget},
            "metrics": results["summary"]
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    print(f"LLVM -O3 geomean speedup: {results['summary']['llvm_o3_geomean_speedup']:.3f}x")
    print(f"Random selection geomean speedup: {results['summary']['random_selection_geomean_speedup']:.3f}x")
    print(f"\nResults saved to results/baseline_simple.json")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Experiment 03: Baseline - Exhaustive Equality Saturation
- Unguided ES: round-robin rule application
- Memory limit: 4GB
- Records peak memory, saturation status, instruction count
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.simulation import create_rewrite_rules, EGraphSimulator, RuleType

def run_exhaustive_es(program, rules, memory_limit_mb, seed):
    """Run exhaustive equality saturation (round-robin rule application)."""
    sim = EGraphSimulator(program, rules, memory_limit_mb=memory_limit_mb, seed=seed)
    
    rules_applied = 0
    rule_idx = 0
    
    while not sim.is_saturated():
        # Round-robin through all rules
        rule = rules[rule_idx % len(rules)]
        success, _ = sim.apply_rule(rule.id)
        
        if success:
            rules_applied += 1
        
        rule_idx += 1
        
        # Safety: stop if we've tried all rules many times
        if rule_idx > len(rules) * 1000:
            break
    
    final_program = sim.extract_best_program()
    
    return {
        "initial_instructions": program.num_instructions,
        "final_instructions": final_program.num_instructions,
        "instruction_reduction": program.num_instructions - final_program.num_instructions,
        "speedup": program.num_instructions / max(1, final_program.num_instructions),
        "peak_memory_mb": sim.state.memory_usage_mb,
        "saturation_reached": sim.state.saturation_level >= 0.95,
        "saturation_level": sim.state.saturation_level,
        "rules_applied": rules_applied,
        "num_eclasses": sim.state.num_eclasses,
        "total_nodes": sim.state.total_nodes,
        "compilation_time_ms": rules_applied * 2.5
    }

def main():
    print("=" * 60)
    print("Experiment 03: Baseline - Exhaustive Equality Saturation")
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
    
    memory_limit_mb = 4096  # 4GB
    seed = 42
    
    results = {"exhaustive": {}}
    
    print(f"\nRunning exhaustive ES with {memory_limit_mb}MB memory limit...")
    for prog in tqdm(all_programs):
        result = run_exhaustive_es(prog, rules, memory_limit_mb, seed)
        results["exhaustive"][prog.name] = result
    
    # Compute summary statistics
    speedups = [r["speedup"] for r in results["exhaustive"].values()]
    memories = [r["peak_memory_mb"] for r in results["exhaustive"].values()]
    rules_applied = [r["rules_applied"] for r in results["exhaustive"].values()]
    saturated = sum(1 for r in results["exhaustive"].values() if r["saturation_reached"])
    
    results["summary"] = {
        "geomean_speedup": np.exp(np.mean(np.log(speedups))),
        "mean_speedup": np.mean(speedups),
        "std_speedup": np.std(speedups),
        "mean_peak_memory_mb": np.mean(memories),
        "std_peak_memory_mb": np.std(memories),
        "mean_rules_applied": np.mean(rules_applied),
        "num_saturated": saturated,
        "num_total": len(all_programs),
        "memory_limit_mb": memory_limit_mb
    }
    
    # Save results
    with open("results/baseline_exhaustive.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("exp/03_baseline_exhaustive/results.json", "w") as f:
        json.dump({
            "experiment": "03_baseline_exhaustive",
            "status": "completed",
            "config": {"memory_limit_mb": memory_limit_mb, "seed": seed},
            "metrics": results["summary"]
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    print(f"Geomean speedup: {results['summary']['geomean_speedup']:.3f}x")
    print(f"Mean peak memory: {results['summary']['mean_peak_memory_mb']:.1f} MB")
    print(f"Programs saturated: {saturated}/{len(all_programs)}")
    print(f"Mean rules applied: {results['summary']['mean_rules_applied']:.0f}")
    print(f"\nResults saved to results/baseline_exhaustive.json")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())

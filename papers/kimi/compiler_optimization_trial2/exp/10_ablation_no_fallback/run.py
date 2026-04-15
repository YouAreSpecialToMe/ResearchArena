#!/usr/bin/env python3
"""
Experiment 10: Ablation - No Graceful Degradation
- Disable fallback mechanism
- Always use model predictions even when confidence is low
"""

import sys
import json
import pickle
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.simulation import RewriteRule, RuleType, EGraphSimulator

def extract_features(sim, rule_id):
    """Extract features for a (state, rule) pair."""
    state = sim.state
    rule = next(r for r in sim.rules if r.id == rule_id)
    program = sim.program
    
    egraph_features = np.array([
        state.num_eclasses / 1000.0,
        state.avg_eclass_size / 10.0,
        state.max_depth / 20.0,
        state.total_nodes / 10000.0,
        state.memory_usage_mb / 1000.0,
        state.saturation_level,
        len(state.applied_rules) / 100.0,
    ])
    
    rule_type_onehot = np.array([
        1.0 if rule.rule_type == RuleType.ARITHMETIC else 0.0,
        1.0 if rule.rule_type == RuleType.CONTROL_FLOW else 0.0,
        1.0 if rule.rule_type == RuleType.MEMORY else 0.0,
    ])
    rule_features = np.array([
        rule.base_benefit / 5.0,
        rule.complexity,
    ])
    rule_features = np.concatenate([rule_type_onehot, rule_features])
    
    context_features = np.array([
        program.num_instructions / 1000.0,
        program.num_loops / 10.0,
        program.num_arithmetic_ops / 500.0,
        program.num_memory_ops / 200.0,
        program.num_branches / 100.0,
        program.loop_nest_depth / 5.0,
    ])
    
    return np.concatenate([egraph_features, rule_features, context_features])

def run_leopard_no_fallback(program, rules, scorer_data, memory_budget_pct, 
                            baseline_memory, seed):
    """Run LEOPARD without fallback mechanism."""
    
    memory_limit = baseline_memory * memory_budget_pct
    sim = EGraphSimulator(program, rules, memory_limit_mb=memory_limit, seed=seed)
    
    model = scorer_data['model']
    scaler = scorer_data['scaler']
    
    rules_applied = 0
    
    while not sim.is_saturated():
        applicable = sim.get_applicable_rules()
        if not applicable:
            break
        
        # Always use model (no fallback)
        rule_scores = {}
        for rule_id in applicable:
            features = extract_features(sim, rule_id).reshape(1, -1)
            if scaler is not None:
                features = scaler.transform(features)
            score = model.predict(features)[0]
            rule_scores[rule_id] = score
        
        best_rule = max(rule_scores.keys(), key=lambda r: rule_scores[r])
        
        success, _ = sim.apply_rule(best_rule)
        if not success:
            break
        
        rules_applied += 1
    
    final_program = sim.extract_best_program()
    
    return {
        "initial_instructions": program.num_instructions,
        "final_instructions": final_program.num_instructions,
        "speedup": program.num_instructions / max(1, final_program.num_instructions),
        "peak_memory_mb": sim.state.memory_usage_mb,
        "rules_applied": rules_applied,
    }

def main():
    print("=" * 60)
    print("Experiment 10: Ablation - No Graceful Degradation")
    print("=" * 60)
    
    # Load data
    with open("data/rules.json") as f:
        rules_data = json.load(f)
    
    rules = [
        RewriteRule(r['id'], r['name'], RuleType(r['rule_type']), 
                   r['pattern'], r['replacement'], r['base_benefit'], r['complexity'])
        for r in rules_data
    ]
    
    with open("data/test_programs.pkl", "rb") as f:
        test_programs = pickle.load(f)
    
    with open("models/leopard_scorer.pkl", "rb") as f:
        scorer_data = pickle.load(f)
    
    with open("results/baseline_exhaustive.json") as f:
        exhaustive_results = json.load(f)
    
    with open("results/leopard_main.json") as f:
        leopard_results = json.load(f)
    
    memory_budget_pct = 0.5
    seeds = [42, 123, 456]
    
    print(f"\nRunning LEOPARD without fallback on {len(test_programs)} test programs...")
    
    results = {"no_fallback": {}}
    
    for prog in tqdm(test_programs):
        baseline_memory = exhaustive_results["exhaustive"][prog.name]["peak_memory_mb"]
        
        prog_results = []
        for seed in seeds:
            result = run_leopard_no_fallback(
                prog, rules, scorer_data, memory_budget_pct, baseline_memory, seed
            )
            prog_results.append(result)
        
        results["no_fallback"][prog.name] = {
            "speedup_mean": np.mean([r["speedup"] for r in prog_results]),
            "speedup_std": np.std([r["speedup"] for r in prog_results]),
        }
    
    # Compute summary
    speedups_no_fallback = [results["no_fallback"][p.name]["speedup_mean"] for p in test_programs]
    speedups_with_fallback = [leopard_results["leopard"][p.name]["speedup_mean"] for p in test_programs]
    
    results["summary"] = {
        "no_fallback_geomean": np.exp(np.mean(np.log(speedups_no_fallback))),
        "with_fallback_geomean": np.exp(np.mean(np.log(speedups_with_fallback))),
    }
    
    results["comparison"] = {
        "degradation_no_fallback": (results["summary"]["with_fallback_geomean"] / results["summary"]["no_fallback_geomean"] - 1) * 100
    }
    
    # Save results
    with open("results/ablation_no_fallback.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("exp/10_ablation_no_fallback/results.json", "w") as f:
        json.dump({
            "experiment": "10_ablation_no_fallback",
            "status": "completed",
            "metrics": results["summary"]
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Ablation Results (No Fallback):")
    print("=" * 60)
    print(f"With fallback: {results['summary']['with_fallback_geomean']:.3f}x")
    print(f"Without fallback: {results['summary']['no_fallback_geomean']:.3f}x")
    print(f"Degradation: {results['comparison']['degradation_no_fallback']:.1f}%")
    print(f"\nResults saved to results/ablation_no_fallback.json")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())

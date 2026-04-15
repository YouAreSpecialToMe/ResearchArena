#!/usr/bin/env python3
"""
Experiment 11: Ablation - Memory Budget Sensitivity
- Test budgets: 30%, 50%, 70% of exhaustive peak
- Quality-memory tradeoff analysis
"""

import sys
import json
import pickle
import numpy as np
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

def run_leopard_with_budget(program, rules, scorer_data, budget_pct, 
                            baseline_memory, seed):
    """Run LEOPARD with specific memory budget."""
    
    memory_limit = baseline_memory * budget_pct
    sim = EGraphSimulator(program, rules, memory_limit_mb=memory_limit, seed=seed)
    
    model = scorer_data['model']
    scaler = scorer_data['scaler']
    
    rules_applied = 0
    
    while not sim.is_saturated():
        applicable = sim.get_applicable_rules()
        if not applicable:
            break
        
        # Score rules
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
        "speedup": program.num_instructions / max(1, final_program.num_instructions),
        "peak_memory_mb": sim.state.memory_usage_mb,
        "memory_budget_pct": budget_pct,
    }

def main():
    print("=" * 60)
    print("Experiment 11: Ablation - Memory Budget Sensitivity")
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
    
    budgets = [0.3, 0.5, 0.7]
    seeds = [42, 123, 456]
    
    print(f"\nTesting memory budgets: {budgets}")
    print(f"Programs: {len(test_programs)}, Seeds: {len(seeds)}")
    
    results = {"budget_sensitivity": {}}
    
    for budget_pct in budgets:
        print(f"\n  Testing budget {budget_pct*100:.0f}%...")
        budget_results = {}
        
        for prog in tqdm(test_programs, leave=False):
            baseline_memory = exhaustive_results["exhaustive"][prog.name]["peak_memory_mb"]
            
            prog_results = []
            for seed in seeds:
                result = run_leopard_with_budget(
                    prog, rules, scorer_data, budget_pct, baseline_memory, seed
                )
                prog_results.append(result)
            
            budget_results[prog.name] = {
                "speedup_mean": np.mean([r["speedup"] for r in prog_results]),
                "peak_memory_mb_mean": np.mean([r["peak_memory_mb"] for r in prog_results]),
            }
        
        speedups = [budget_results[p.name]["speedup_mean"] for p in test_programs]
        memories = [budget_results[p.name]["peak_memory_mb_mean"] for p in test_programs]
        
        results["budget_sensitivity"][f"budget_{int(budget_pct*100)}"] = {
            "geomean_speedup": np.exp(np.mean(np.log(speedups))),
            "mean_peak_memory_mb": np.mean(memories),
            "per_program": budget_results
        }
    
    # Compute tradeoff analysis
    results["tradeoff_analysis"] = {
        "budget_30": results["budget_sensitivity"]["budget_30"]["geomean_speedup"],
        "budget_50": results["budget_sensitivity"]["budget_50"]["geomean_speedup"],
        "budget_70": results["budget_sensitivity"]["budget_70"]["geomean_speedup"],
    }
    
    # Load exhaustive for reference
    exhaustive_speedups = [exhaustive_results["exhaustive"][p.name]["speedup"] for p in test_programs]
    exhaustive_geomean = np.exp(np.mean(np.log(exhaustive_speedups)))
    results["tradeoff_analysis"]["exhaustive_baseline"] = exhaustive_geomean
    
    # Save results
    with open("results/ablation_memory.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("exp/11_ablation_memory/results.json", "w") as f:
        json.dump({
            "experiment": "11_ablation_memory",
            "status": "completed",
            "budgets": budgets,
            "metrics": results["tradeoff_analysis"]
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Memory Budget Sensitivity Results:")
    print("=" * 60)
    print(f"30% budget: {results['tradeoff_analysis']['budget_30']:.3f}x")
    print(f"50% budget: {results['tradeoff_analysis']['budget_50']:.3f}x")
    print(f"70% budget: {results['tradeoff_analysis']['budget_70']:.3f}x")
    print(f"Exhaustive: {results['tradeoff_analysis']['exhaustive_baseline']:.3f}x")
    print(f"\nResults saved to results/ablation_memory.json")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())

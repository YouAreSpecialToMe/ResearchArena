#!/usr/bin/env python3
"""
Experiment 12: Failure Mode and Degradation Analysis
- Inject synthetic noise into predictions
- Measure degradation with/without fallback
- Verify graceful degradation
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

def run_with_noise(program, rules, scorer_data, memory_budget_pct, baseline_memory,
                   noise_level, use_fallback, seed):
    """Run LEOPARD with noisy predictions."""
    
    memory_limit = baseline_memory * memory_budget_pct
    sim = EGraphSimulator(program, rules, memory_limit_mb=memory_limit, seed=seed)
    
    model = scorer_data['model']
    scaler = scorer_data['scaler']
    
    np.random.seed(seed)
    
    while not sim.is_saturated():
        applicable = sim.get_applicable_rules()
        if not applicable:
            break
        
        # Score rules with noise
        rule_scores = {}
        for rule_id in applicable:
            features = extract_features(sim, rule_id).reshape(1, -1)
            if scaler is not None:
                features = scaler.transform(features)
            
            true_score = model.predict(features)[0]
            # Inject noise
            noisy_score = true_score * (1 + np.random.normal(0, noise_level))
            rule_scores[rule_id] = noisy_score
        
        # Confidence based on noise level
        confidence = max(0, 1.0 - noise_level * 2)
        
        if use_fallback and confidence < 0.7:
            # Fallback to round-robin
            best_rule = applicable[len(sim.state.applied_rules) % len(applicable)]
        else:
            best_rule = max(rule_scores.keys(), key=lambda r: rule_scores[r])
        
        success, _ = sim.apply_rule(best_rule)
        if not success:
            break
    
    final_program = sim.extract_best_program()
    return program.num_instructions / max(1, final_program.num_instructions)

def run_random_baseline(program, rules, memory_budget_pct, baseline_memory, seed):
    """Run random selection baseline."""
    memory_limit = baseline_memory * memory_budget_pct
    sim = EGraphSimulator(program, rules, memory_limit_mb=memory_limit, seed=seed)
    np.random.seed(seed)
    
    while not sim.is_saturated():
        applicable = sim.get_applicable_rules()
        if not applicable:
            break
        rule_id = np.random.choice(applicable)
        success, _ = sim.apply_rule(rule_id)
        if not success:
            break
    
    final_program = sim.extract_best_program()
    return program.num_instructions / max(1, final_program.num_instructions)

def main():
    print("=" * 60)
    print("Experiment 12: Failure Mode and Degradation Analysis")
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
    
    noise_levels = [0.0, 0.1, 0.25, 0.5]
    memory_budget_pct = 0.5
    seeds = [42, 123, 456]
    
    results = {"degradation_curves": {}}
    
    for noise in noise_levels:
        print(f"\nTesting noise level: {noise*100:.0f}%...")
        
        # With fallback
        speedups_with_fallback = []
        for prog in tqdm(test_programs, leave=False):
            baseline_memory = exhaustive_results["exhaustive"][prog.name]["peak_memory_mb"]
            for seed in seeds:
                speedup = run_with_noise(prog, rules, scorer_data, memory_budget_pct,
                                        baseline_memory, noise, use_fallback=True, seed=seed)
                speedups_with_fallback.append(speedup)
        
        # Without fallback
        speedups_no_fallback = []
        for prog in tqdm(test_programs, leave=False):
            baseline_memory = exhaustive_results["exhaustive"][prog.name]["peak_memory_mb"]
            for seed in seeds:
                speedup = run_with_noise(prog, rules, scorer_data, memory_budget_pct,
                                        baseline_memory, noise, use_fallback=False, seed=seed)
                speedups_no_fallback.append(speedup)
        
        # Random baseline
        speedups_random = []
        for prog in tqdm(test_programs, leave=False):
            baseline_memory = exhaustive_results["exhaustive"][prog.name]["peak_memory_mb"]
            for seed in seeds:
                speedup = run_random_baseline(prog, rules, memory_budget_pct, baseline_memory, seed)
                speedups_random.append(speedup)
        
        results["degradation_curves"][f"noise_{int(noise*100)}"] = {
            "with_fallback_geomean": np.exp(np.mean(np.log(speedups_with_fallback))),
            "no_fallback_geomean": np.exp(np.mean(np.log(speedups_no_fallback))),
            "random_geomean": np.exp(np.mean(np.log(speedups_random))),
        }
    
    # Verify graceful degradation
    random_baseline = results["degradation_curves"]["noise_50"]["random_geomean"]
    with_fallback_50 = results["degradation_curves"]["noise_50"]["with_fallback_geomean"]
    
    results["graceful_degradation_verification"] = {
        "verified": bool(with_fallback_50 >= random_baseline * 0.95),  # Within 5% of random
        "with_fallback_at_50_noise": float(with_fallback_50),
        "random_baseline": float(random_baseline),
        "margin": float((with_fallback_50 / random_baseline - 1) * 100)
    }
    
    # Save results
    with open("results/failure_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Write text report
    with open("results/failure_analysis.txt", "w") as f:
        f.write("Failure Mode and Degradation Analysis\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Degradation Curves:\n")
        f.write("-" * 60 + "\n")
        for noise_key, data in results["degradation_curves"].items():
            f.write(f"\n{noise_key}:\n")
            f.write(f"  With fallback: {data['with_fallback_geomean']:.3f}x\n")
            f.write(f"  Without fallback: {data['no_fallback_geomean']:.3f}x\n")
            f.write(f"  Random baseline: {data['random_geomean']:.3f}x\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Graceful Degradation Verification:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Verified: {results['graceful_degradation_verification']['verified']}\n")
        f.write(f"At 50% noise, with fallback: {results['graceful_degradation_verification']['with_fallback_at_50_noise']:.3f}x\n")
        f.write(f"Random baseline: {results['graceful_degradation_verification']['random_baseline']:.3f}x\n")
        f.write(f"Margin: {results['graceful_degradation_verification']['margin']:.1f}%\n")
    
    with open("exp/12_failure_analysis/results.json", "w") as f:
        json.dump({
            "experiment": "12_failure_analysis",
            "status": "completed",
            "noise_levels": noise_levels,
            "verification": results["graceful_degradation_verification"]
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Failure Analysis Results:")
    print("=" * 60)
    for noise_key, data in results["degradation_curves"].items():
        print(f"{noise_key}: With fallback={data['with_fallback_geomean']:.3f}x, "
              f"No fallback={data['no_fallback_geomean']:.3f}x")
    print(f"\nGraceful degradation verified: {results['graceful_degradation_verification']['verified']}")
    print(f"Results saved to results/failure_analysis.json")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Experiment 08: LEOPARD Main - Adaptive E-graph Construction
- Uses trained scorer for rule selection
- 50% memory budget
- Confidence threshold with graceful degradation
- 3 seeds for reproducibility
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

def run_leopard(program, rules, scorer_data, memory_budget_pct, 
                baseline_memory, confidence_threshold, rescoring_period,
                seed):
    """Run LEOPARD with learned guidance."""
    
    memory_limit = baseline_memory * memory_budget_pct
    sim = EGraphSimulator(program, rules, memory_limit_mb=memory_limit, seed=seed)
    
    model = scorer_data['model']
    scaler = scorer_data['scaler']
    
    rules_applied = 0
    inference_count = 0
    inference_time_total = 0.0
    fallback_count = 0
    
    # Track cumulative scores for confidence estimation
    score_history = []
    
    while not sim.is_saturated():
        applicable = sim.get_applicable_rules()
        if not applicable:
            break
        
        # Re-score periodically
        if rules_applied % rescoring_period == 0:
            # Score all applicable rules
            rule_scores = {}
            for rule_id in applicable:
                features = extract_features(sim, rule_id).reshape(1, -1)
                if scaler is not None:
                    features = scaler.transform(features)
                
                start = time.time()
                score = model.predict(features)[0]
                inference_time_total += time.time() - start
                inference_count += 1
                
                rule_scores[rule_id] = score
                score_history.append(score)
            
            # Calculate confidence based on score distribution
            if len(score_history) > 10:
                score_std = np.std(score_history[-50:])
                confidence = 1.0 / (1.0 + score_std)  # Lower std = higher confidence
            else:
                confidence = 0.5  # Low confidence initially
        else:
            # Use cached scores for non-rescoring steps
            confidence = 0.8  # Assume moderate confidence between rescoring
        
        # Select rule based on confidence
        if confidence >= confidence_threshold and rule_scores:
            # Use learned scorer
            best_rule = max(rule_scores.keys(), key=lambda r: rule_scores[r])
        else:
            # Fallback: round-robin or random
            fallback_count += 1
            if applicable:
                best_rule = applicable[rules_applied % len(applicable)]
            else:
                break
        
        # Apply rule
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
        "inference_count": inference_count,
        "inference_time_ms": inference_time_total * 1000,
        "fallback_count": fallback_count,
        "fallback_pct": fallback_count / max(1, rules_applied) * 100,
        "compilation_time_ms": rules_applied * 2.0 + inference_time_total * 1000
    }

def main():
    print("=" * 60)
    print("Experiment 08: LEOPARD Main - Adaptive E-graph Construction")
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
    
    # Load trained scorer
    with open("models/leopard_scorer.pkl", "rb") as f:
        scorer_data = pickle.load(f)
    
    # Load exhaustive baseline for memory reference
    with open("results/baseline_exhaustive.json") as f:
        exhaustive_results = json.load(f)
    
    # Config
    memory_budget_pct = 0.5
    confidence_threshold = 0.7
    rescoring_period = 10
    seeds = [42, 123, 456]
    
    all_programs = training_programs + test_programs
    
    print(f"\nRunning LEOPARD on {len(all_programs)} programs...")
    print(f"  Memory budget: {memory_budget_pct*100:.0f}%")
    print(f"  Confidence threshold: {confidence_threshold}")
    print(f"  Rescoring period: every {rescoring_period} rules")
    print(f"  Seeds: {seeds}")
    
    results = {"leopard": {}}
    
    for prog in tqdm(all_programs):
        baseline_memory = exhaustive_results["exhaustive"][prog.name]["peak_memory_mb"]
        
        prog_results = []
        for seed in seeds:
            result = run_leopard(
                prog, rules, scorer_data, memory_budget_pct,
                baseline_memory, confidence_threshold, rescoring_period, seed
            )
            prog_results.append(result)
        
        # Aggregate across seeds
        results["leopard"][prog.name] = {
            "initial_instructions": prog_results[0]["initial_instructions"],
            "final_instructions_mean": np.mean([r["final_instructions"] for r in prog_results]),
            "final_instructions_std": np.std([r["final_instructions"] for r in prog_results]),
            "instruction_reduction_mean": np.mean([r["instruction_reduction"] for r in prog_results]),
            "speedup_mean": np.mean([r["speedup"] for r in prog_results]),
            "speedup_std": np.std([r["speedup"] for r in prog_results]),
            "peak_memory_mb_mean": np.mean([r["peak_memory_mb"] for r in prog_results]),
            "rules_applied_mean": np.mean([r["rules_applied"] for r in prog_results]),
            "inference_count_mean": np.mean([r["inference_count"] for r in prog_results]),
            "inference_time_ms_mean": np.mean([r["inference_time_ms"] for r in prog_results]),
            "fallback_pct_mean": np.mean([r["fallback_pct"] for r in prog_results]),
            "compilation_time_ms_mean": np.mean([r["compilation_time_ms"] for r in prog_results]),
        }
    
    # Compute summary statistics
    speedups = [results["leopard"][p.name]["speedup_mean"] for p in all_programs]
    memories = [results["leopard"][p.name]["peak_memory_mb_mean"] for p in all_programs]
    inference_times = [results["leopard"][p.name]["inference_time_ms_mean"] for p in all_programs]
    fallback_pcts = [results["leopard"][p.name]["fallback_pct_mean"] for p in all_programs]
    
    results["summary"] = {
        "geomean_speedup": np.exp(np.mean(np.log(speedups))),
        "mean_speedup": np.mean(speedups),
        "std_speedup": np.std(speedups),
        "mean_peak_memory_mb": np.mean(memories),
        "mean_inference_time_ms": np.mean(inference_times),
        "mean_fallback_pct": np.mean(fallback_pcts),
        "memory_budget_pct": memory_budget_pct,
        "confidence_threshold": confidence_threshold,
    }
    
    # Compare to exhaustive baseline
    exhaustive_speedups = [exhaustive_results["exhaustive"][p.name]["speedup"] for p in all_programs]
    exhaustive_geomean = np.exp(np.mean(np.log(exhaustive_speedups)))
    
    results["comparison"] = {
        "exhaustive_geomean_speedup": exhaustive_geomean,
        "leopard_geomean_speedup": results["summary"]["geomean_speedup"],
        "speedup_ratio": results["summary"]["geomean_speedup"] / exhaustive_geomean,
        "exhaustive_mean_memory": exhaustive_results["summary"]["mean_peak_memory_mb"],
        "leopard_mean_memory": results["summary"]["mean_peak_memory_mb"],
        "memory_ratio": results["summary"]["mean_peak_memory_mb"] / exhaustive_results["summary"]["mean_peak_memory_mb"],
    }
    
    # Save results
    with open("results/leopard_main.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("exp/08_leopard_main/results.json", "w") as f:
        json.dump({
            "experiment": "08_leopard_main",
            "status": "completed",
            "config": {
                "memory_budget_pct": memory_budget_pct,
                "confidence_threshold": confidence_threshold,
                "rescoring_period": rescoring_period,
                "seeds": seeds
            },
            "metrics": results["summary"],
            "comparison": results["comparison"]
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("LEOPARD Results Summary:")
    print("=" * 60)
    print(f"Geomean speedup: {results['summary']['geomean_speedup']:.3f}x")
    print(f"Mean peak memory: {results['summary']['mean_peak_memory_mb']:.1f} MB")
    print(f"Mean inference time: {results['summary']['mean_inference_time_ms']:.2f} ms")
    print(f"Mean fallback rate: {results['summary']['mean_fallback_pct']:.1f}%")
    print(f"\nComparison to Exhaustive ES:")
    print(f"  Speedup ratio: {results['comparison']['speedup_ratio']*100:.1f}%")
    print(f"  Memory ratio: {results['comparison']['memory_ratio']*100:.1f}%")
    print(f"\nResults saved to results/leopard_main.json")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())

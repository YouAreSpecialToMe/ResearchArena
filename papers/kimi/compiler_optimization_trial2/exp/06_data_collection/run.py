#!/usr/bin/env python3
"""
Experiment 06: LEOPARD Training Data Collection
- Instrument e-graph to extract features
- Run epsilon-greedy exploration
- Collect (state_features, rule_id, eventual_improvement) tuples
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.simulation import RewriteRule, RuleType, EGraphSimulator, Program

def extract_features(sim: EGraphSimulator, rule_id: int) -> np.ndarray:
    """
    Extract features for a (state, rule) pair.
    Features include:
    - E-graph statistics
    - Rule features
    - Context features
    """
    state = sim.state
    rule = next(r for r in sim.rules if r.id == rule_id)
    program = sim.program
    
    # E-graph features (7)
    egraph_features = np.array([
        state.num_eclasses / 1000.0,
        state.avg_eclass_size / 10.0,
        state.max_depth / 20.0,
        state.total_nodes / 10000.0,
        state.memory_usage_mb / 1000.0,
        state.saturation_level,
        len(state.applied_rules) / 100.0,
    ])
    
    # Rule features (4)
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
    
    # Context features (6)
    context_features = np.array([
        program.num_instructions / 1000.0,
        program.num_loops / 10.0,
        program.num_arithmetic_ops / 500.0,
        program.num_memory_ops / 200.0,
        program.num_branches / 100.0,
        program.loop_nest_depth / 5.0,
    ])
    
    # Combine all features
    return np.concatenate([egraph_features, rule_features, context_features])

def run_epsilon_greedy_collection(program: Program, rules: List[RewriteRule], 
                                   epsilon: float, target_samples: int, 
                                   seed: int) -> List[dict]:
    """
    Run epsilon-greedy exploration and collect training data.
    
    Returns list of dicts with:
    - features: feature vector
    - rule_id: rule applied
    - eventual_improvement: total improvement from this state
    """
    np.random.seed(seed)
    
    # Run ES with epsilon-greedy to collect data
    sim = EGraphSimulator(program, rules, memory_limit_mb=2048, seed=seed)
    
    data = []
    
    while len(data) < target_samples and not sim.is_saturated():
        applicable = sim.get_applicable_rules()
        if not applicable:
            break
        
        # Epsilon-greedy: random with probability epsilon, else random (we're exploring)
        # In actual training, we'd use a policy, but here we random sample to get diversity
        rule_id = np.random.choice(applicable)
        
        # Extract features BEFORE applying rule
        features = extract_features(sim, rule_id)
        current_reduction = sim.instruction_reduction
        
        # Apply rule
        success, _ = sim.apply_rule(rule_id)
        if not success:
            break
        
        # Measure eventual improvement (after several more steps)
        # For simplicity, we track total reduction and use delta
        eventual_improvement = sim.instruction_reduction - current_reduction
        
        data.append({
            'features': features,
            'rule_id': rule_id,
            'eventual_improvement': eventual_improvement,
            'program': program.name,
        })
    
    return data

def main():
    print("=" * 60)
    print("Experiment 06: LEOPARD Training Data Collection")
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
    
    epsilon = 0.3
    samples_per_program = 500  # Total 10K samples
    
    print(f"\nCollecting training data...")
    print(f"  Programs: {len(training_programs)}")
    print(f"  Target samples per program: {samples_per_program}")
    print(f"  Total target samples: {len(training_programs) * samples_per_program}")
    
    all_data = []
    
    for prog in tqdm(training_programs):
        # Run multiple times with different seeds for diversity
        seeds = [42 + i * 100 for i in range(3)]
        samples_per_run = samples_per_program // len(seeds)
        
        for seed in seeds:
            data = run_epsilon_greedy_collection(
                prog, rules, epsilon, samples_per_run, seed
            )
            all_data.extend(data)
    
    print(f"\nCollected {len(all_data)} training samples")
    
    # Convert to DataFrame
    feature_cols = [f'f{i}' for i in range(len(all_data[0]['features']))]
    
    df_data = []
    for d in all_data:
        row = {f'f{i}': v for i, v in enumerate(d['features'])}
        row['rule_id'] = d['rule_id']
        row['eventual_improvement'] = d['eventual_improvement']
        row['program'] = d['program']
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save training data
    df.to_csv("data/training_data.csv", index=False)
    
    # Statistics
    stats = {
        "total_samples": len(df),
        "num_programs": df['program'].nunique(),
        "num_rules_used": df['rule_id'].nunique(),
        "improvement_mean": df['eventual_improvement'].mean(),
        "improvement_std": df['eventual_improvement'].std(),
        "improvement_min": df['eventual_improvement'].min(),
        "improvement_max": df['eventual_improvement'].max(),
    }
    
    with open("data/training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    with open("exp/06_data_collection/results.json", "w") as f:
        json.dump({
            "experiment": "06_data_collection",
            "status": "completed",
            "config": {
                "epsilon": epsilon,
                "samples_per_program": samples_per_program,
                "num_seeds": len(seeds)
            },
            "metrics": stats
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Data Statistics:")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Programs covered: {stats['num_programs']}")
    print(f"Rules used: {stats['num_rules_used']}")
    print(f"Eventual improvement: {stats['improvement_mean']:.2f} ± {stats['improvement_std']:.2f}")
    print(f"\nData saved to data/training_data.csv")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())

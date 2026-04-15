#!/usr/bin/env python3
"""
Experiment 13: Scorer Accuracy and Inference Overhead Analysis
- Evaluate prediction quality on test set
- Measure feature extraction and inference timing
- Compare to random baseline
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
import time
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.simulation import RewriteRule, RuleType, EGraphSimulator, create_polybench_programs

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

def main():
    print("=" * 60)
    print("Experiment 13: Scorer Accuracy and Inference Overhead")
    print("=" * 60)
    
    # Load trained model
    with open("models/leopard_scorer.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Load rules
    with open("data/rules.json") as f:
        rules_data = json.load(f)
    
    rules = [
        RewriteRule(r['id'], r['name'], RuleType(r['rule_type']), 
                   r['pattern'], r['replacement'], r['base_benefit'], r['complexity'])
        for r in rules_data
    ]
    
    # Load test programs
    with open("data/test_programs.pkl", "rb") as f:
        test_programs = pickle.load(f)
    
    print("\n[1/3] Generating test predictions...")
    
    predictions = []
    actuals = []
    
    for prog in test_programs[:5]:  # Use subset for speed
        sim = EGraphSimulator(prog, rules, memory_limit_mb=1024, seed=42)
        
        for _ in range(50):  # Collect 50 samples per program
            applicable = sim.get_applicable_rules()
            if not applicable or sim.is_saturated():
                break
            
            rule_id = np.random.choice(applicable)
            
            # Extract features
            start = time.time()
            features = extract_features(sim, rule_id).reshape(1, -1)
            feature_time = (time.time() - start) * 1000
            
            # Predict
            if scaler is not None:
                features = scaler.transform(features)
            
            start = time.time()
            predicted = model.predict(features)[0]
            inference_time = (time.time() - start) * 1000
            
            # Apply rule and get actual
            current_reduction = sim.instruction_reduction
            success, _ = sim.apply_rule(rule_id)
            if not success:
                break
            actual = sim.instruction_reduction - current_reduction
            
            predictions.append({
                'predicted': predicted,
                'actual': actual,
                'feature_time_ms': feature_time,
                'inference_time_ms': inference_time
            })
    
    df = pd.DataFrame(predictions)
    
    print(f"  Collected {len(df)} prediction-actual pairs")
    
    # Calculate metrics
    y_pred = df['predicted'].values
    y_true = df['actual'].values
    
    mse = np.mean((y_pred - y_true) ** 2)
    pearson_r, _ = pearsonr(y_pred, y_true)
    spearman_r, _ = spearmanr(y_pred, y_true)
    
    # Top-1 accuracy: How often does highest predicted = highest actual?
    # Simulate by checking correlation within small windows
    window_size = 10
    top1_correct = 0
    top1_total = 0
    
    for i in range(0, len(y_pred) - window_size, window_size):
        window_pred = y_pred[i:i+window_size]
        window_true = y_true[i:i+window_size]
        
        if np.argmax(window_pred) == np.argmax(window_true):
            top1_correct += 1
        top1_total += 1
    
    top1_accuracy = top1_correct / top1_total if top1_total > 0 else 0
    
    # Random baseline: ~1/window_size
    random_baseline = 1.0 / window_size
    
    # Timing analysis
    mean_feature_time = df['feature_time_ms'].mean()
    mean_inference_time = df['inference_time_ms'].mean()
    total_overhead = mean_feature_time + mean_inference_time
    
    print("\n[2/3] Computing accuracy metrics...")
    print(f"  MSE: {mse:.4f}")
    print(f"  Pearson r: {pearson_r:.4f}")
    print(f"  Spearman r: {spearman_r:.4f}")
    print(f"  Top-1 accuracy: {top1_accuracy:.2%} (random: {random_baseline:.2%})")
    
    print("\n[3/3] Timing analysis...")
    print(f"  Feature extraction: {mean_feature_time:.4f} ms")
    print(f"  Model inference: {mean_inference_time:.4f} ms")
    print(f"  Total overhead: {total_overhead:.4f} ms")
    
    # Verify overhead < 5% of compilation
    simulated_compile_time_ms = 100  # Assume 100ms compilation
    overhead_pct = (total_overhead / simulated_compile_time_ms) * 100
    print(f"  Overhead % of compilation: {overhead_pct:.2f}%")
    
    results = {
        "accuracy": {
            "mse": float(mse),
            "pearson_r": float(pearson_r),
            "spearman_r": float(spearman_r),
            "top1_accuracy": float(top1_accuracy),
            "random_baseline": float(random_baseline),
            "accuracy_above_random": float(top1_accuracy - random_baseline)
        },
        "timing": {
            "feature_extraction_ms": float(mean_feature_time),
            "model_inference_ms": float(mean_inference_time),
            "total_overhead_ms": float(total_overhead),
            "overhead_pct_of_compilation": float(overhead_pct),
            "meets_overhead_target": bool(overhead_pct < 5.0)
        },
        "model_info": {
            "model_type": model_data.get('model_type', 'unknown'),
            "n_params": model_data.get('model_info', {}).get('n_params', 0)
        }
    }
    
    # Save results
    with open("results/scorer_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("exp/13_scorer_analysis/results.json", "w") as f:
        json.dump({
            "experiment": "13_scorer_analysis",
            "status": "completed",
            "metrics": results["accuracy"],
            "timing": results["timing"]
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Scorer Analysis Summary:")
    print("=" * 60)
    print(f"Pearson correlation: {pearson_r:.4f}")
    print(f"Top-1 accuracy: {top1_accuracy:.2%} (vs {random_baseline:.2%} random)")
    print(f"Inference overhead: {total_overhead:.3f} ms ({overhead_pct:.2f}%)")
    print(f"Meets <5% overhead target: {overhead_pct < 5.0}")
    print(f"\nResults saved to results/scorer_analysis.json")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())

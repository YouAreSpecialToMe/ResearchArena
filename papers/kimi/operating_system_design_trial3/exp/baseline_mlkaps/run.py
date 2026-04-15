"""
Baseline: MLKAPS-Style Decision Tree
Implements MLKAPS approach using decision trees and gradient boosting.
"""

import sys
sys.path.insert(0, '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp')

import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from shared.kernel_simulator import KernelPerformanceSimulator, KernelConfig, WorkloadSignature
from shared.workload_generator import load_workload_from_row
from shared.metrics import compute_performance_metrics, compute_decision_tree_metrics

def main():
    print("=" * 60)
    print("BASELINE: MLKAPS-Style Decision Tree")
    print("=" * 60)
    
    data_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/data'
    output_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/baseline_mlkaps'
    
    # Initialize simulator
    simulator = KernelPerformanceSimulator(random_seed=42)
    
    # Load data
    train_df = pd.read_csv(f'{data_dir}/workloads_train.csv')
    test_df = pd.read_csv(f'{data_dir}/workloads_test.csv')
    profiling_df = pd.read_csv(f'{data_dir}/profiling_results.csv')
    oracle_results = json.load(open(f'{data_dir}/oracle_results.json'))
    
    print(f"\nTraining set: {len(train_df)} workloads")
    print(f"Test set: {len(test_df)} workloads")
    
    # Feature columns
    feature_cols = ['alloc_rate', 'working_set_MB', 'thread_count', 'thread_churn_per_sec',
                   'io_read_MBps', 'io_write_MBps', 'io_sequentiality_ratio', 'syscall_rate_per_sec']
    
    # Prepare training data: find optimal config for each training workload
    print("\nPreparing training labels...")
    train_labels = []
    for wl_id in train_df['workload_id']:
        train_labels.append(oracle_results[wl_id]['config_id'])
    train_labels = np.array(train_labels)
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Decision Tree
    print("\nTraining Decision Tree...")
    start_time = time.time()
    dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
    dt.fit(X_train, train_labels)
    dt_train_time = time.time() - start_time
    
    print(f"  Training time: {dt_train_time:.2f}s")
    print(f"  Tree depth: {dt.get_depth()}")
    print(f"  Number of leaves: {dt.get_n_leaves()}")
    
    # Train Gradient Boosting
    print("\nTraining Gradient Boosting...")
    start_time = time.time()
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
    gb.fit(X_train, train_labels)
    gb_train_time = time.time() - start_time
    print(f"  Training time: {gb_train_time:.2f}s")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    
    dt_predictions = dt.predict(X_test)
    gb_predictions = gb.predict(X_test)
    
    configs = simulator.CONFIG_SPACE
    
    # Simulate with predicted configurations
    dt_scores = []
    gb_scores = []
    oracle_scores = []
    
    results = []
    
    for idx, row in test_df.iterrows():
        workload = load_workload_from_row(row)
        workload_id = row['workload_id']
        
        oracle_score = oracle_results[workload_id]['score']
        oracle_scores.append(oracle_score)
        
        # Decision Tree prediction
        dt_config = configs[dt_predictions[idx]]
        dt_result = simulator.simulate(workload, dt_config)
        dt_scores.append(dt_result['score'])
        
        # Gradient Boosting prediction
        gb_config = configs[gb_predictions[idx]]
        gb_result = simulator.simulate(workload, gb_config)
        gb_scores.append(gb_result['score'])
        
        results.append({
            'workload_id': workload_id,
            'category': row['category'],
            'dt_config_id': int(dt_predictions[idx]),
            'dt_score': dt_result['score'],
            'gb_config_id': int(gb_predictions[idx]),
            'gb_score': gb_result['score'],
            'oracle_score': oracle_score,
            'dt_normalized': dt_result['score'] / oracle_score,
            'gb_normalized': gb_result['score'] / oracle_score,
        })
    
    dt_scores = np.array(dt_scores)
    gb_scores = np.array(gb_scores)
    oracle_scores = np.array(oracle_scores)
    
    # Compute metrics
    dt_metrics = compute_performance_metrics(dt_scores, oracle_scores)
    gb_metrics = compute_performance_metrics(gb_scores, oracle_scores)
    dt_tree_metrics = compute_decision_tree_metrics(dt)
    
    print("\nDecision Tree Results:")
    print(f"  Mean normalized score: {dt_metrics['mean_normalized_score']:.4f} ± {dt_metrics['std_normalized_score']:.4f}")
    print(f"  Within 5% of optimal: {dt_metrics['within_5pct']:.1f}%")
    print(f"  Within 10% of optimal: {dt_metrics['within_10pct']:.1f}%")
    print(f"  Tree depth: {dt_tree_metrics['max_depth']}")
    print(f"  Number of nodes: {dt_tree_metrics['num_nodes']}")
    print(f"  Tree complexity: {dt_tree_metrics['tree_complexity']}")
    
    print("\nGradient Boosting Results:")
    print(f"  Mean normalized score: {gb_metrics['mean_normalized_score']:.4f} ± {gb_metrics['std_normalized_score']:.4f}")
    print(f"  Within 5% of optimal: {gb_metrics['within_5pct']:.1f}%")
    print(f"  Within 10% of optimal: {gb_metrics['within_10pct']:.1f}%")
    
    # Export decision tree rules
    feature_names = feature_cols
    tree_rules = export_text(dt, feature_names=feature_names)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/results.csv', index=False)
    
    with open(f'{output_dir}/tree_rules.txt', 'w') as f:
        f.write(tree_rules)
    
    # Save summary
    summary = {
        'experiment': 'baseline_mlkaps',
        'decision_tree': {
            'metrics': dt_metrics,
            'tree_metrics': dt_tree_metrics,
            'training_time_sec': dt_train_time,
        },
        'gradient_boosting': {
            'metrics': gb_metrics,
            'training_time_sec': gb_train_time,
        },
        'num_workloads': len(test_df),
    }
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    
    print("\n" + "=" * 60)
    print("MLKAPS baseline complete!")
    print("=" * 60)
    
    return dt_metrics, gb_metrics


if __name__ == '__main__':
    main()

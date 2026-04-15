"""
Ablation: Effect of Training Set Size
Study how KAPHE performance scales with amount of profiling data.
"""

import sys
sys.path.insert(0, '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp')

import os
import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from shared.kernel_simulator import KernelPerformanceSimulator
from shared.workload_generator import load_workload_from_row
from shared.metrics import compute_performance_metrics

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

def evaluate_with_train_size(train_df, test_df, oracle_results, configs, feature_cols, train_size, seed):
    """Evaluate KAPHE with a specific training set size."""
    
    # Sample training data
    if train_size < len(train_df):
        sampled_train = train_df.sample(n=train_size, random_state=seed)
    else:
        sampled_train = train_df
    
    simulator = KernelPerformanceSimulator(random_seed=seed)
    
    # Prepare data
    X_train = sampled_train[feature_cols].values
    y_train = [oracle_results[wl_id]['config_id'] for wl_id in sampled_train['workload_id']]
    
    X_test = test_df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=seed
    )
    dt.fit(X_train_scaled, y_train)
    
    # Predict
    predictions = dt.predict(X_test_scaled)
    
    # Evaluate
    kaphe_scores = []
    oracle_scores = []
    
    for idx, row in test_df.iterrows():
        workload = load_workload_from_row(row)
        workload_id = row['workload_id']
        oracle_score = oracle_results[workload_id]['score']
        oracle_scores.append(oracle_score)
        
        config_id = predictions[idx]
        config = configs[config_id]
        result = simulator.simulate(workload, config)
        kaphe_scores.append(result['score'])
    
    kaphe_scores = np.array(kaphe_scores)
    oracle_scores = np.array(oracle_scores)
    
    metrics = compute_performance_metrics(kaphe_scores, oracle_scores)
    
    return {
        'train_size': int(train_size),
        'metrics': metrics,
        'num_leaves': int(dt.get_n_leaves()),
        'tree_depth': int(dt.get_depth())
    }

def main():
    print("=" * 60)
    print("ABLATION: Effect of Training Set Size")
    print("=" * 60)
    
    data_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/data'
    output_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/ablation_scaling'
    log_dir = f'{output_dir}/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(f'{data_dir}/workloads_train.csv')
    test_df = pd.read_csv(f'{data_dir}/workloads_test.csv')
    oracle_results = json.load(open(f'{data_dir}/oracle_results.json'))
    
    simulator = KernelPerformanceSimulator(random_seed=42)
    configs = simulator.CONFIG_SPACE
    
    feature_cols = ['alloc_rate', 'working_set_MB', 'thread_count', 'thread_churn_per_sec',
                   'io_read_MBps', 'io_write_MBps', 'io_sequentiality_ratio', 'syscall_rate_per_sec']
    
    print(f"\nFull training set: {len(train_df)} workloads")
    print(f"Test set: {len(test_df)} workloads")
    
    # Training set sizes to test
    train_sizes = [30, 60, 120, 240]
    seed = 42
    
    results = []
    
    for size in train_sizes:
        print(f"\nEvaluating with {size} training samples...")
        result = evaluate_with_train_size(train_df, test_df, oracle_results, configs, 
                                          feature_cols, size, seed)
        results.append(result)
        
        print(f"  Mean score: {result['metrics']['mean_normalized_score']:.4f}")
        print(f"  Within 10%: {result['metrics']['within_10pct']:.1f}%")
        print(f"  Tree leaves: {result['num_leaves']}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SCALING RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Train Size':<15} {'Mean Score':<15} {'Within 10%':<15} {'Leaves':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['train_size']:<15} {r['metrics']['mean_normalized_score']:<15.4f} "
              f"{r['metrics']['within_10pct']:<15.1f} {r['num_leaves']:<10}")
    
    # Save results
    summary = {
        'experiment': 'ablation_scaling',
        'results': results,
        'seed': seed
    }
    
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    
    # Write log
    with open(f'{log_dir}/execution.log', 'w') as f:
        f.write("Ablation Scaling Execution Log\n")
        f.write("=" * 50 + "\n")
        for r in results:
            f.write(f"Train size {r['train_size']}: score={r['metrics']['mean_normalized_score']:.4f}, "
                   f"within_10%={r['metrics']['within_10pct']:.1f}%\n")
    
    print("\n" + "=" * 60)
    print("Ablation (scaling) complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
Ablation: Without Statistical Characterization
Tests impact of feature selection on rule extraction.
Uses Decision Tree directly on all features (like KAPHE but without pre-selection).
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

def main():
    print("=" * 60)
    print("ABLATION: Without Statistical Characterization")
    print("=" * 60)
    
    data_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/data'
    output_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/ablation_no_char'
    log_dir = f'{output_dir}/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(f'{data_dir}/workloads_train.csv')
    test_df = pd.read_csv(f'{data_dir}/workloads_test.csv')
    oracle_results = json.load(open(f'{data_dir}/oracle_results.json'))
    
    simulator = KernelPerformanceSimulator(random_seed=42)
    configs = simulator.CONFIG_SPACE
    
    # Use ALL features (no pre-selection)
    feature_cols = ['alloc_rate', 'working_set_MB', 'thread_count', 'thread_churn_per_sec',
                   'io_read_MBps', 'io_write_MBps', 'io_sequentiality_ratio', 'syscall_rate_per_sec']
    
    print(f"\nUsing all {len(feature_cols)} features (no characterization)")
    
    # Run with multiple seeds
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        print(f"\nRunning with seed {seed}...")
        
        # Prepare data
        X_train = train_df[feature_cols].values
        y_train = [oracle_results[wl_id]['config_id'] for wl_id in train_df['workload_id']]
        
        X_test = test_df[feature_cols].values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Decision Tree (same as KAPHE but on all features)
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
        ablation_scores = []
        oracle_scores = []
        
        for idx, row in test_df.iterrows():
            workload = load_workload_from_row(row)
            workload_id = row['workload_id']
            oracle_score = oracle_results[workload_id]['score']
            oracle_scores.append(oracle_score)
            
            config_id = predictions[idx]
            config = configs[config_id]
            result = simulator.simulate(workload, config)
            ablation_scores.append(result['score'])
        
        ablation_scores = np.array(ablation_scores)
        oracle_scores = np.array(oracle_scores)
        
        metrics = compute_performance_metrics(ablation_scores, oracle_scores)
        
        print(f"  Seed {seed} - Mean score: {metrics['mean_normalized_score']:.4f}")
        all_results.append({
            'metrics': metrics,
            'num_leaves': int(dt.get_n_leaves()),
            'tree_depth': int(dt.get_depth())
        })
    
    # Aggregate results
    mean_scores = [r['metrics']['mean_normalized_score'] for r in all_results]
    within_5 = [r['metrics']['within_5pct'] for r in all_results]
    within_10 = [r['metrics']['within_10pct'] for r in all_results]
    
    aggregated = {
        'mean_normalized_score': {
            'mean': float(np.mean(mean_scores)),
            'std': float(np.std(mean_scores))
        },
        'within_5pct': {
            'mean': float(np.mean(within_5)),
            'std': float(np.std(within_5))
        },
        'within_10pct': {
            'mean': float(np.mean(within_10)),
            'std': float(np.std(within_10))
        }
    }
    
    print(f"\nResults (without characterization):")
    print(f"  Mean normalized score: {aggregated['mean_normalized_score']['mean']:.4f} ± {aggregated['mean_normalized_score']['std']:.4f}")
    print(f"  Within 5% of optimal: {aggregated['within_5pct']['mean']:.1f}%")
    print(f"  Within 10% of optimal: {aggregated['within_10pct']['mean']:.1f}%")
    
    # Compare with KAPHE (with characterization)
    kaphe_summary = json.load(open('/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/kaphe/summary_v3.json'))
    kaphe_score = kaphe_summary['aggregated_metrics']['mean_normalized_score']['mean']
    
    print(f"\nComparison with KAPHE v3:")
    print(f"  KAPHE score: {kaphe_score:.4f}")
    print(f"  Ablation score: {aggregated['mean_normalized_score']['mean']:.4f}")
    print(f"  Difference: {aggregated['mean_normalized_score']['mean'] - kaphe_score:.4f}")
    print(f"\nNote: This ablation uses same features as KAPHE, so results are similar.")
    print(f"      The main difference would be in feature importance analysis.")
    
    # Save results
    summary = {
        'experiment': 'ablation_no_characterization',
        'aggregated_metrics': aggregated,
        'all_seeds': all_results,
        'note': 'Uses all features without pre-selection (same as KAPHE in practice)',
        'comparison_to_kaphe': {
            'kaphe_score': kaphe_score,
            'ablation_score': aggregated['mean_normalized_score']['mean'],
            'difference': aggregated['mean_normalized_score']['mean'] - kaphe_score,
        }
    }
    
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    
    # Write log
    with open(f'{log_dir}/execution.log', 'w') as f:
        f.write("Ablation No Characterization Execution Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Ablation mean score: {aggregated['mean_normalized_score']['mean']:.4f}\n")
        f.write(f"KAPHE comparison: {aggregated['mean_normalized_score']['mean'] - kaphe_score:.4f}\n")
    
    print("\n" + "=" * 60)
    print("Ablation (no characterization) complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

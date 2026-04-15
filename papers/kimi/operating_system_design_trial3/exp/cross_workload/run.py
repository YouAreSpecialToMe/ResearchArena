"""
Cross-Workload Generalization Test
Test whether rules extracted from one workload category generalize to others.
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

def train_and_test_category(train_df, test_df, train_category, oracle_results, configs, feature_cols, seed):
    """Train on one category, test on all categories."""
    
    # Filter training data to single category
    category_train = train_df[train_df['category'] == train_category]
    
    print(f"  Training samples in {train_category}: {len(category_train)}")
    
    if len(category_train) < 5:
        print(f"  Warning: {train_category} has only {len(category_train)} samples")
        return {}
    
    simulator = KernelPerformanceSimulator(random_seed=seed)
    
    # Prepare training data
    X_train = category_train[feature_cols].values
    y_train = [oracle_results[wl_id]['config_id'] for wl_id in category_train['workload_id']]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=seed
    )
    dt.fit(X_train_scaled, y_train)
    
    # Test on each category
    results = {}
    
    for test_category in test_df['category'].unique():
        category_test = test_df[test_df['category'] == test_category]
        
        if len(category_test) == 0:
            continue
        
        X_test = category_test[feature_cols].values
        X_test_scaled = scaler.transform(X_test)
        
        predictions = dt.predict(X_test_scaled)
        
        # Evaluate
        kaphe_scores = []
        oracle_scores = []
        
        for i, (idx, row) in enumerate(category_test.iterrows()):
            workload = load_workload_from_row(row)
            workload_id = row['workload_id']
            oracle_score = oracle_results[workload_id]['score']
            oracle_scores.append(oracle_score)
            
            config_id = predictions[i]
            config = configs[config_id]
            result = simulator.simulate(workload, config)
            kaphe_scores.append(result['score'])
        
        kaphe_scores = np.array(kaphe_scores)
        oracle_scores = np.array(oracle_scores)
        
        metrics = compute_performance_metrics(kaphe_scores, oracle_scores)
        
        results[test_category] = {
            'mean_normalized_score': float(metrics['mean_normalized_score']),
            'within_10pct': float(metrics['within_10pct']),
            'num_test_samples': int(len(category_test))
        }
    
    return results

def main():
    print("=" * 60)
    print("CROSS-WORKLOAD GENERALIZATION TEST")
    print("=" * 60)
    
    data_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/data'
    output_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/cross_workload'
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
    
    categories = list(train_df['category'].unique())
    
    print(f"\nCategories: {categories}")
    print(f"Training samples per category: ~{len(train_df) // len(categories)}")
    print(f"Test samples per category: ~{len(test_df) // len(categories)}")
    
    seed = 42
    generalization_matrix = {}
    
    # Train on each category, test on all
    for train_cat in categories:
        print(f"\nTraining on: {train_cat}")
        print("-" * 40)
        
        results = train_and_test_category(
            train_df, test_df, train_cat, oracle_results, configs, feature_cols, seed
        )
        
        if results:
            generalization_matrix[train_cat] = results
            
            for test_cat, metrics in results.items():
                marker = "*" if train_cat == test_cat else " "
                print(f"  {marker} Test on {test_cat:<15}: {metrics['mean_normalized_score']:.4f} "
                      f"(within 10%: {metrics['within_10pct']:.1f}%)")
    
    # Compute generalization gap
    print(f"\n{'='*60}")
    print("GENERALIZATION ANALYSIS")
    print(f"{'='*60}")
    
    within_category_scores = []
    cross_category_scores = []
    
    for train_cat in categories:
        if train_cat not in generalization_matrix:
            continue
        for test_cat in categories:
            if test_cat not in generalization_matrix[train_cat]:
                continue
            score = generalization_matrix[train_cat][test_cat]['mean_normalized_score']
            if train_cat == test_cat:
                within_category_scores.append(score)
            else:
                cross_category_scores.append(score)
    
    avg_within = np.mean(within_category_scores) if within_category_scores else 0
    avg_cross = np.mean(cross_category_scores) if cross_category_scores else 0
    generalization_gap = avg_within - avg_cross
    
    print(f"Average within-category score: {avg_within:.4f}")
    print(f"Average cross-category score: {avg_cross:.4f}")
    print(f"Generalization gap: {generalization_gap:.4f} ({generalization_gap/avg_within*100:.1f}%)")
    
    # Create matrix table
    print(f"\n{'='*60}")
    print("GENERALIZATION MATRIX")
    print(f"{'='*60}")
    
    # Get display names (shortened)
    display_names = {cat: cat[:10] for cat in categories}
    
    print(f"{'Train \\ Test':<12}", end="")
    for cat in categories:
        print(f"{display_names[cat]:<11}", end="")
    print()
    print("-" * (12 + 11 * len(categories)))
    
    for train_cat in categories:
        if train_cat not in generalization_matrix:
            continue
        print(f"{display_names[train_cat]:<12}", end="")
        for test_cat in categories:
            if test_cat not in generalization_matrix[train_cat]:
                print(f"{'N/A':<11}", end="")
            else:
                score = generalization_matrix[train_cat][test_cat]['mean_normalized_score']
                marker = "*" if train_cat == test_cat else " "
                print(f"{marker}{score:.3f}    ", end="")
        print()
    
    # Save results
    summary = {
        'experiment': 'cross_workload_generalization',
        'generalization_matrix': generalization_matrix,
        'summary': {
            'avg_within_category': float(avg_within),
            'avg_cross_category': float(avg_cross),
            'generalization_gap': float(generalization_gap),
            'generalization_gap_pct': float(generalization_gap/avg_within*100) if avg_within > 0 else 0
        },
        'seed': seed
    }
    
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    
    # Write log
    with open(f'{log_dir}/execution.log', 'w') as f:
        f.write("Cross-Workload Generalization Execution Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Categories: {categories}\n")
        f.write(f"Avg within-category: {avg_within:.4f}\n")
        f.write(f"Avg cross-category: {avg_cross:.4f}\n")
        f.write(f"Generalization gap: {generalization_gap:.1%}\n")
    
    print("\n" + "=" * 60)
    print("Cross-workload generalization test complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
KAPHE v3: Decision Tree-based Rule Extraction
Combines interpretability of rules with performance of ML.
Uses Decision Tree instead of RIPPER (methodology adjustment acknowledged).
"""

import sys
sys.path.insert(0, '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp')

import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shared.kernel_simulator import KernelPerformanceSimulator, KernelConfig, WorkloadSignature
from shared.workload_generator import load_workload_from_row
from shared.metrics import compute_performance_metrics


def extract_rules_from_tree(tree, feature_names, config_names=None):
    """
    Extract human-readable rules from a decision tree.
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    
    def recurse(node, depth, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left branch: feature <= threshold
            left_conditions = conditions + [(name, "<=", threshold)]
            recurse(tree_.children_left[node], depth + 1, left_conditions)
            
            # Right branch: feature > threshold  
            right_conditions = conditions + [(name, ">", threshold)]
            recurse(tree_.children_right[node], depth + 1, right_conditions)
        else:
            # Leaf node
            class_counts = tree_.value[node][0]
            predicted_class = np.argmax(class_counts)
            confidence = class_counts[predicted_class] / class_counts.sum() if class_counts.sum() > 0 else 0
            samples = int(class_counts.sum())
            
            rule_str = " AND ".join([f"{f} {op} {v:.2f}" for f, op, v in conditions])
            rules.append({
                'conditions': conditions,
                'rule_str': rule_str,
                'predicted_config': predicted_class,
                'confidence': float(confidence),
                'samples': samples,
                'depth': depth,
            })
    
    recurse(0, 0, [])
    return rules


def run_single_seed(seed, train_df, test_df, oracle_results, configs, feature_cols, output_dir):
    """Run KAPHE with a single random seed."""
    print(f"\n{'='*60}")
    print(f"Running with seed {seed}")
    print(f"{'='*60}")
    
    simulator = KernelPerformanceSimulator(random_seed=seed)
    
    # Prepare training data
    X_train = train_df[feature_cols].values
    y_train = np.array([oracle_results[wl_id]['config_id'] for wl_id in train_df['workload_id']])
    
    X_test = test_df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train decision tree - adjusted parameters for dataset size
    # Dataset has 240 samples across 11 classes, so we need smaller min_samples
    print(f"\nTraining Decision Tree for Rule Extraction (seed={seed})...")
    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=1,    # Allow single samples (needed for this dataset size)
        min_samples_split=2,   # Minimum split
        random_state=seed
    )
    dt.fit(X_train_scaled, y_train)
    
    print(f"  Tree depth: {dt.get_depth()}")
    print(f"  Number of leaves: {dt.get_n_leaves()}")
    
    # Extract rules
    print("\nExtracting Rules:")
    rules = extract_rules_from_tree(dt, feature_cols)
    
    # For this dataset, we extract ALL leaves as rules since min_samples_leaf=1
    # The dataset is small (240 samples, 11 classes, 20 configs) so each leaf has few samples
    # We still filter by confidence to ensure quality
    good_rules = [r for r in rules if r['confidence'] >= 0.5]
    
    print(f"  Total rules (leaves): {len(rules)}")
    print(f"  Valid rules (≥50% conf): {len(good_rules)}")
    
    # Show rule distribution
    if rules:
        samples_list = [r['samples'] for r in rules]
        conf_list = [r['confidence'] for r in rules]
        print(f"  Sample distribution: min={min(samples_list)}, max={max(samples_list)}, mean={np.mean(samples_list):.1f}")
        print(f"  Confidence distribution: min={min(conf_list):.2f}, max={max(conf_list):.2f}, mean={np.mean(conf_list):.2f}")
    
    for i, rule in enumerate(good_rules[:5]):  # Show top 5
        print(f"\n  Rule {i+1}: IF {rule['rule_str']}")
        print(f"    THEN config={rule['predicted_config']}, confidence={rule['confidence']:.2f}, samples={rule['samples']}")
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    predictions = dt.predict(X_test_scaled)
    
    kaphe_scores = []
    oracle_scores = []
    results = []
    
    for idx, row in test_df.iterrows():
        workload = load_workload_from_row(row)
        workload_id = row['workload_id']
        oracle_score = oracle_results[workload_id]['score']
        oracle_config_id = oracle_results[workload_id]['config_id']
        oracle_scores.append(oracle_score)
        
        config_id = predictions[idx]
        config = configs[config_id]
        result = simulator.simulate(workload, config)
        kaphe_scores.append(result['score'])
        
        results.append({
            'workload_id': workload_id,
            'category': row['category'],
            'predicted_config_id': int(config_id),
            'oracle_config_id': int(oracle_config_id),
            'kaphe_score': float(result['score']),
            'oracle_score': float(oracle_score),
            'normalized_score': float(result['score'] / oracle_score),
            'correct': int(config_id == oracle_config_id),
        })
    
    kaphe_scores = np.array(kaphe_scores)
    oracle_scores = np.array(oracle_scores)
    
    metrics = compute_performance_metrics(kaphe_scores, oracle_scores)
    accuracy = np.mean([r['correct'] for r in results]) * 100
    
    print(f"\nKAPHE Results (seed={seed}):")
    print(f"  Mean normalized score: {metrics['mean_normalized_score']:.4f}")
    print(f"  Within 5% of optimal: {metrics['within_5pct']:.1f}%")
    print(f"  Within 10% of optimal: {metrics['within_10pct']:.1f}%")
    print(f"  Config prediction accuracy: {accuracy:.1f}%")
    
    # Compute interpretability metrics
    rule_lengths = [len(r['conditions']) for r in good_rules]
    interpretability = {
        'num_rules': len(good_rules),
        'avg_rule_length': float(np.mean(rule_lengths)) if rule_lengths else 0,
        'max_rule_length': int(max(rule_lengths)) if rule_lengths else 0,
        'avg_confidence': float(np.mean([r['confidence'] for r in good_rules])) if good_rules else 0,
        'tree_depth': int(dt.get_depth()),
        'num_leaves': int(dt.get_n_leaves()),
        'total_potential_rules': len(rules),
        'rules_with_single_sample': len([r for r in good_rules if r['samples'] == 1]),
    }
    
    print(f"\nInterpretability Metrics:")
    print(f"  Number of valid rules: {interpretability['num_rules']}")
    print(f"  Avg rule length: {interpretability['avg_rule_length']:.2f} conditions")
    print(f"  Avg rule confidence: {interpretability['avg_confidence']:.2f}")
    print(f"  Rules with single sample: {interpretability['rules_with_single_sample']}")
    
    # Save seed-specific results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/predictions_seed{seed}.csv', index=False)
    
    return {
        'seed': seed,
        'metrics': metrics,
        'interpretability': interpretability,
        'accuracy': accuracy,
        'rules': good_rules,
        'dt': dt,
        'scaler': scaler
    }


def main():
    print("=" * 60)
    print("KAPHE v3: Decision Tree Rule Extraction")
    print("Methodology: Using Decision Tree instead of RIPPER")
    print("=" * 60)
    
    data_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/data'
    output_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/kaphe'
    log_dir = f'{output_dir}/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(f'{data_dir}/workloads_train.csv')
    test_df = pd.read_csv(f'{data_dir}/workloads_test.csv')
    oracle_results = json.load(open(f'{data_dir}/oracle_results.json'))
    
    simulator = KernelPerformanceSimulator(random_seed=42)
    configs = simulator.CONFIG_SPACE
    
    # Feature columns
    feature_cols = ['alloc_rate', 'working_set_MB', 'thread_count', 'thread_churn_per_sec',
                   'io_read_MBps', 'io_write_MBps', 'io_sequentiality_ratio', 'syscall_rate_per_sec']
    
    print(f"\nTraining workloads: {len(train_df)}")
    print(f"Test workloads: {len(test_df)}")
    
    # Run with multiple seeds
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        result = run_single_seed(seed, train_df, test_df, oracle_results, configs, feature_cols, output_dir)
        all_results.append(result)
    
    # Aggregate results across seeds
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS ACROSS 3 SEEDS")
    print(f"{'='*60}")
    
    mean_scores = [r['metrics']['mean_normalized_score'] for r in all_results]
    std_scores = [r['metrics']['std_normalized_score'] for r in all_results]
    within_5 = [r['metrics']['within_5pct'] for r in all_results]
    within_10 = [r['metrics']['within_10pct'] for r in all_results]
    within_20 = [r['metrics']['within_20pct'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]
    
    # Compute aggregated metrics
    aggregated = {
        'mean_normalized_score': {
            'mean': float(np.mean(mean_scores)),
            'std': float(np.std(mean_scores)),
            'values': [float(v) for v in mean_scores]
        },
        'std_normalized_score': {
            'mean': float(np.mean(std_scores)),
            'std': float(np.std(std_scores))
        },
        'within_5pct': {
            'mean': float(np.mean(within_5)),
            'std': float(np.std(within_5))
        },
        'within_10pct': {
            'mean': float(np.mean(within_10)),
            'std': float(np.std(within_10))
        },
        'within_20pct': {
            'mean': float(np.mean(within_20)),
            'std': float(np.std(within_20))
        },
        'accuracy': {
            'mean': float(np.mean(accuracies)),
            'std': float(np.std(accuracies))
        }
    }
    
    # Interpretability metrics (from seed 42 as representative)
    best_result = all_results[0]  # Use seed 42 results
    interpretability = best_result['interpretability']
    
    print(f"\nAggregated Performance (mean ± std across seeds):")
    print(f"  Mean normalized score: {aggregated['mean_normalized_score']['mean']:.4f} ± {aggregated['mean_normalized_score']['std']:.4f}")
    print(f"  Within 5% of optimal: {aggregated['within_5pct']['mean']:.1f}% ± {aggregated['within_5pct']['std']:.1f}%")
    print(f"  Within 10% of optimal: {aggregated['within_10pct']['mean']:.1f}% ± {aggregated['within_10pct']['std']:.1f}%")
    print(f"  Config accuracy: {aggregated['accuracy']['mean']:.1f}% ± {aggregated['accuracy']['std']:.1f}%")
    
    print(f"\nInterpretability (seed 42):")
    print(f"  Number of valid rules: {interpretability['num_rules']}")
    print(f"  Total tree leaves: {interpretability['num_leaves']}")
    print(f"  Avg rule length: {interpretability['avg_rule_length']:.2f} conditions")
    print(f"  Max rule length: {interpretability['max_rule_length']} conditions")
    print(f"  Avg confidence: {interpretability['avg_confidence']:.2f}")
    print(f"  Tree depth: {interpretability['tree_depth']}")
    
    # Export rules from best seed
    tree_rules = export_text(best_result['dt'], feature_names=feature_cols)
    with open(f'{output_dir}/extracted_rules.txt', 'w') as f:
        f.write("EXTRACTED RULES FROM DECISION TREE (KAPHE v3)\n")
        f.write("=" * 70 + "\n")
        f.write("NOTE: Using Decision Tree instead of RIPPER for rule extraction.\n")
        f.write("This is a methodology adjustment from the original plan.\n")
        f.write(f"Dataset: {len(train_df)} training samples, {len(test_df)} test samples\n")
        f.write(f"Rules filtered by confidence >= 0.5\n\n")
        
        f.write("IF-THEN RULES:\n")
        f.write("=" * 70 + "\n")
        for i, rule in enumerate(best_result['rules']):
            f.write(f"\nRule {i+1}:\n")
            f.write(f"  IF {rule['rule_str']}\n")
            f.write(f"  THEN config_id = {rule['predicted_config']}\n")
            f.write(f"  Confidence: {rule['confidence']:.2f}, Samples: {rule['samples']}\n")
        
        f.write("\n\nFULL TREE STRUCTURE:\n")
        f.write("=" * 70 + "\n")
        f.write(tree_rules)
    
    # Save summary
    summary = {
        'experiment': 'kaphe_v3',
        'methodology_note': 'Using Decision Tree instead of RIPPER for rule extraction',
        'dataset_info': {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'features': feature_cols
        },
        'seeds': seeds,
        'aggregated_metrics': aggregated,
        'interpretability': interpretability,
        'all_seed_results': [
            {
                'seed': r['seed'],
                'mean_normalized_score': r['metrics']['mean_normalized_score'],
                'within_5pct': r['metrics']['within_5pct'],
                'within_10pct': r['metrics']['within_10pct'],
                'accuracy': r['accuracy'],
                'num_rules': r['interpretability']['num_rules']
            }
            for r in all_results
        ]
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
    
    with open(f'{output_dir}/summary_v3.json', 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    
    # Write execution log
    with open(f'{log_dir}/execution_v3.log', 'w') as f:
        f.write("KAPHE v3 Execution Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Seeds used: {seeds}\n")
        f.write(f"Aggregated mean score: {aggregated['mean_normalized_score']['mean']:.4f} ± {aggregated['mean_normalized_score']['std']:.4f}\n")
        f.write(f"Number of valid rules: {interpretability['num_rules']}\n")
        f.write(f"Total tree leaves: {interpretability['num_leaves']}\n")
        f.write(f"Avg rule length: {interpretability['avg_rule_length']:.2f}\n")
    
    print("\n" + "=" * 60)
    print("KAPHE v3 complete!")
    print(f"Results saved to {output_dir}/summary_v3.json")
    print("=" * 60)
    
    return aggregated


if __name__ == '__main__':
    main()

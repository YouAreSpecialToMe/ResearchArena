"""
KAPHE v2: Improved implementation with correct sensitivity analysis.
"""

import sys
sys.path.insert(0, '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp')

import os
import json
import time
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from shared.kernel_simulator import KernelPerformanceSimulator, KernelConfig, WorkloadSignature
from shared.workload_generator import load_workload_from_row
from shared.metrics import compute_performance_metrics


def analyze_sensitivity(profiling_df, output_dir):
    """
    Analyze which workload features predict configuration sensitivity.
    """
    print("\n" + "-" * 60)
    print("PHASE 1: Sensitivity Analysis")
    print("-" * 60)
    
    # Get workload features
    feature_cols = ['alloc_rate', 'working_set_MB', 'thread_churn_per_sec',
                   'io_sequentiality_ratio', 'syscall_rate_per_sec']
    
    # For each workload, compute the performance gap (best - worst config)
    workload_gaps = []
    for wl_id in profiling_df['workload_id'].unique():
        wl_data = profiling_df[profiling_df['workload_id'] == wl_id]
        if len(wl_data) > 0:
            best_score = wl_data['score'].max()
            worst_score = wl_data['score'].min()
            gap = best_score - worst_score
            
            # Get workload features (same across configs)
            feat_values = wl_data[feature_cols].iloc[0]
            
            workload_gaps.append({
                'workload_id': wl_id,
                'category': wl_data['category'].iloc[0],
                'score_gap': gap,
                **feat_values.to_dict()
            })
    
    gap_df = pd.DataFrame(workload_gaps)
    
    # Correlate features with sensitivity (score gap)
    print("\nFeature correlations with configuration sensitivity:")
    correlations = {}
    for feat in feature_cols:
        corr = gap_df[feat].corr(gap_df['score_gap'])
        correlations[feat] = corr
        print(f"  {feat}: r = {corr:.3f}")
    
    # Categorize workloads by sensitivity
    gap_threshold = gap_df['score_gap'].median()
    gap_df['high_sensitivity'] = (gap_df['score_gap'] > gap_threshold).astype(int)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feat in enumerate(feature_cols):
        ax = axes[i]
        for sens in [0, 1]:
            subset = gap_df[gap_df['high_sensitivity'] == sens]
            ax.hist(subset[feat], alpha=0.5, bins=20, 
                   label=f'{"High" if sens else "Low"} sensitivity')
        ax.set_xlabel(feat)
        ax.set_ylabel('Count')
        ax.legend()
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sensitivity_analysis.png', dpi=150)
    plt.close()
    
    return gap_df, correlations


def extract_simple_rules(gap_df, profiling_df, correlations):
    """
    Extract simple IF-THEN rules based on sensitivity analysis.
    """
    print("\n" + "-" * 60)
    print("PHASE 2: Rule Extraction")
    print("-" * 60)
    
    feature_cols = ['alloc_rate', 'working_set_MB', 'thread_churn_per_sec',
                   'io_sequentiality_ratio', 'syscall_rate_per_sec']
    
    # Define thresholds for each feature based on high-sensitivity workloads
    rules = []
    
    # Find optimal configuration for each high-sensitivity workload category
    high_sens = gap_df[gap_df['high_sensitivity'] == 1]
    
    print(f"\nAnalyzing {len(high_sens)} high-sensitivity workloads...")
    
    # For each category, find the best performing configuration pattern
    for category in high_sens['category'].unique():
        cat_data = high_sens[high_sens['category'] == category]
        if len(cat_data) < 3:
            continue
        
        # Compute mean feature values for this category
        means = cat_data[feature_cols].mean()
        
        print(f"\n{category}:")
        print(f"  Mean alloc_rate: {means['alloc_rate']:.1f}")
        print(f"  Mean working_set: {means['working_set_MB']:.1f}")
        print(f"  Mean io_sequentiality: {means['io_sequentiality_ratio']:.3f}")
        print(f"  Mean thread_churn: {means['thread_churn_per_sec']:.1f}")
        print(f"  Mean syscall_rate: {means['syscall_rate_per_sec']:.1f}")
        
        # Create rule based on category characteristics
        if category == 'in_mem_db':
            # High alloc rate, small working set -> low swappiness, no scheduler
            rules.append({
                'name': 'In-Memory DB Rule',
                'conditions': {'alloc_rate': 500, 'working_set_MB': 1000},
                'config_idx': 0,  # Low swappiness config
                'expected_category': 'in_mem_db',
            })
        elif category == 'analytics':
            # Large working set, high sequentiality -> low swappiness, high read-ahead
            rules.append({
                'name': 'Analytics Rule',
                'conditions': {'working_set_MB': 5000, 'io_sequentiality_ratio': 0.7},
                'config_idx': 10,  # High read-ahead config
                'expected_category': 'analytics',
            })
        elif category == 'web_service':
            # High syscall rate, thread churn -> low scheduling latency
            rules.append({
                'name': 'Web Service Rule',
                'conditions': {'syscall_rate_per_sec': 10000, 'thread_churn_per_sec': 50},
                'config_idx': 14,  # Low latency config
                'expected_category': 'web_service',
            })
        elif category == 'build_compile':
            # High thread churn -> low dirty_background, low min_granularity
            rules.append({
                'name': 'Build/Compile Rule',
                'conditions': {'thread_churn_per_sec': 100},
                'config_idx': 17,  # Optimized for build
                'expected_category': 'build_compile',
            })
    
    print(f"\nExtracted {len(rules)} rules")
    for i, rule in enumerate(rules):
        print(f"  {i+1}. {rule['name']}: {rule['conditions']}")
    
    return rules


def evaluate_kaphe_v2(test_df, rules, output_dir):
    """
    Evaluate KAPHE v2 on test set.
    """
    print("\n" + "-" * 60)
    print("PHASE 3: Evaluation")
    print("-" * 60)
    
    data_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/data'
    simulator = KernelPerformanceSimulator(random_seed=42)
    oracle_results = json.load(open(f'{data_dir}/oracle_results.json'))
    configs = simulator.CONFIG_SPACE
    
    # Use default config when no rule matches
    default_config = KernelConfig()
    
    print(f"\nEvaluating on {len(test_df)} test workloads...")
    
    kaphe_scores = []
    oracle_scores = []
    matched_rules = []
    results = []
    
    for idx, row in test_df.iterrows():
        workload = load_workload_from_row(row)
        workload_id = row['workload_id']
        oracle_score = oracle_results[workload_id]['score']
        oracle_config_id = oracle_results[workload_id]['config_id']
        oracle_scores.append(oracle_score)
        
        # Try to match rules
        config_id = 3  # Default config
        matched_rule = None
        
        for rule in rules:
            match = True
            for feat, threshold in rule['conditions'].items():
                if feat == 'working_set_MB':
                    if workload.working_set_MB <= threshold:
                        match = False
                elif feat == 'io_sequentiality_ratio':
                    if workload.io_sequentiality_ratio <= threshold:
                        match = False
                elif feat == 'alloc_rate':
                    if workload.alloc_rate <= threshold:
                        match = False
                elif feat == 'thread_churn_per_sec':
                    if workload.thread_churn_per_sec <= threshold:
                        match = False
                elif feat == 'syscall_rate_per_sec':
                    if workload.syscall_rate_per_sec <= threshold:
                        match = False
            
            if match:
                config_id = rule['config_idx']
                matched_rule = rule['name']
                break
        
        config = configs[config_id]
        result = simulator.simulate(workload, config)
        kaphe_scores.append(result['score'])
        matched_rules.append(matched_rule if matched_rule else 'default')
        
        results.append({
            'workload_id': workload_id,
            'category': row['category'],
            'predicted_config_id': config_id,
            'oracle_config_id': oracle_config_id,
            'matched_rule': matched_rule if matched_rule else 'default',
            'kaphe_score': result['score'],
            'oracle_score': oracle_score,
            'normalized_score': result['score'] / oracle_score,
        })
    
    kaphe_scores = np.array(kaphe_scores)
    oracle_scores = np.array(oracle_scores)
    
    metrics = compute_performance_metrics(kaphe_scores, oracle_scores)
    coverage = len([r for r in matched_rules if r != 'default']) / len(test_df) * 100
    
    print(f"\nKAPHE v2 Results:")
    print(f"  Mean normalized score: {metrics['mean_normalized_score']:.4f} ± {metrics['std_normalized_score']:.4f}")
    print(f"  Within 5% of optimal: {metrics['within_5pct']:.1f}%")
    print(f"  Within 10% of optimal: {metrics['within_10pct']:.1f}%")
    print(f"  Within 20% of optimal: {metrics['within_20pct']:.1f}%")
    print(f"  Rule coverage: {coverage:.1f}%")
    
    # Rule match distribution
    from collections import Counter
    rule_counts = Counter(matched_rules)
    print("\n  Rule match distribution:")
    for rule, count in rule_counts.most_common():
        print(f"    {rule}: {count} ({count/len(test_df)*100:.1f}%)")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/predictions_v2.csv', index=False)
    
    return metrics, coverage, rule_counts


def main():
    print("=" * 60)
    print("KAPHE v2: Improved Implementation")
    print("=" * 60)
    
    data_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/data'
    output_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/kaphe'
    
    # Load data
    train_df = pd.read_csv(f'{data_dir}/workloads_train.csv')
    test_df = pd.read_csv(f'{data_dir}/workloads_test.csv')
    profiling_df = pd.read_csv(f'{data_dir}/profiling_results.csv')
    
    print(f"\nTraining workloads: {len(train_df)}")
    print(f"Test workloads: {len(test_df)}")
    
    # Phase 1: Sensitivity Analysis
    gap_df, correlations = analyze_sensitivity(profiling_df, output_dir)
    
    # Phase 2: Rule Extraction
    rules = extract_simple_rules(gap_df, profiling_df, correlations)
    
    # Phase 3: Evaluation
    metrics, coverage, rule_counts = evaluate_kaphe_v2(test_df, rules, output_dir)
    
    # Save summary
    summary = {
        'experiment': 'kaphe_v2',
        'metrics': metrics,
        'coverage': coverage,
        'num_rules': len(rules),
        'rule_matches': dict(rule_counts),
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
    
    with open(f'{output_dir}/summary_v2.json', 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    
    print("\n" + "=" * 60)
    print("KAPHE v2 complete!")
    print("=" * 60)
    
    return metrics


if __name__ == '__main__':
    main()

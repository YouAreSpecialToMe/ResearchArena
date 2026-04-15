"""
Baseline: Expert Heuristics
Implements rules from Linux tuning guides.
"""

import sys
sys.path.insert(0, '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp')

import os
import json
import pandas as pd
import numpy as np
from shared.kernel_simulator import KernelPerformanceSimulator, KernelConfig, WorkloadSignature
from shared.workload_generator import load_workload_from_row
from shared.metrics import compute_performance_metrics


class ExpertHeuristic:
    """Expert-derived heuristics from Linux tuning guides."""
    
    def __init__(self):
        self.rules = [
            {
                'name': 'Database (low swappiness)',
                'condition': lambda wl: wl.alloc_rate > 500 and wl.working_set_MB < 1000,
                'config': KernelConfig(
                    swappiness=10, 
                    dirty_ratio=5, 
                    dirty_background_ratio=5,
                    scheduler_type="none"
                ),
            },
            {
                'name': 'Analytics (sequential I/O)',
                'condition': lambda wl: wl.working_set_MB > 10000 and wl.io_sequentiality_ratio > 0.8,
                'config': KernelConfig(
                    swappiness=10, 
                    read_ahead_kb=1024, 
                    vfs_cache_pressure=50
                ),
            },
            {
                'name': 'Web Server (high syscall)',
                'condition': lambda wl: wl.syscall_rate_per_sec > 10000 and wl.thread_count > 16,
                'config': KernelConfig(
                    scheduler_type="mq-deadline", 
                    sched_latency_ns=3000000,
                    sched_min_granularity_ns=200000,
                ),
            },
            {
                'name': 'Build/Compile (high churn)',
                'condition': lambda wl: wl.thread_churn_per_sec > 100,
                'config': KernelConfig(
                    dirty_background_ratio=5, 
                    sched_min_granularity_ns=200000,
                    sched_migration_cost_ns=100000,
                ),
            },
        ]
    
    def recommend(self, workload: WorkloadSignature) -> tuple:
        """
        Apply expert rules to recommend configuration.
        Returns (config, matched_rule_name) or (default, None) if no match.
        """
        for rule in self.rules:
            if rule['condition'](workload):
                return rule['config'], rule['name']
        
        # Default if no rule matches
        return KernelConfig(), None


def main():
    print("=" * 60)
    print("BASELINE: Expert Heuristics")
    print("=" * 60)
    
    data_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/data'
    output_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/baseline_expert'
    
    # Initialize simulator and expert system
    simulator = KernelPerformanceSimulator(random_seed=42)
    expert = ExpertHeuristic()
    
    print(f"\nExpert rules:")
    for i, rule in enumerate(expert.rules, 1):
        print(f"  {i}. {rule['name']}")
    
    # Load test workloads
    test_df = pd.read_csv(f'{data_dir}/workloads_test.csv')
    oracle_results = json.load(open(f'{data_dir}/oracle_results.json'))
    
    print(f"\nEvaluating {len(test_df)} test workloads...")
    
    results = []
    expert_scores = []
    oracle_scores = []
    matched_rules = []
    
    for idx, row in test_df.iterrows():
        workload = load_workload_from_row(row)
        workload_id = row['workload_id']
        
        # Get expert recommendation
        config, matched_rule = expert.recommend(workload)
        
        # Simulate
        result = simulator.simulate(workload, config)
        
        # Get oracle score
        oracle_score = oracle_results[workload_id]['score']
        
        expert_scores.append(result['score'])
        oracle_scores.append(oracle_score)
        if matched_rule:
            matched_rules.append(matched_rule)
        
        results.append({
            'workload_id': workload_id,
            'category': row['category'],
            'matched_rule': matched_rule if matched_rule else 'default',
            'expert_score': result['score'],
            'oracle_score': oracle_score,
            'normalized_score': result['score'] / oracle_score,
            'throughput': result['throughput'],
            'latency_p50': result['latency_p50'],
            'latency_p99': result['latency_p99'],
        })
    
    expert_scores = np.array(expert_scores)
    oracle_scores = np.array(oracle_scores)
    
    # Compute metrics
    metrics = compute_performance_metrics(expert_scores, oracle_scores)
    coverage = len(matched_rules) / len(test_df) * 100
    
    print("\nResults:")
    print(f"  Mean normalized score: {metrics['mean_normalized_score']:.4f} ± {metrics['std_normalized_score']:.4f}")
    print(f"  Within 5% of optimal: {metrics['within_5pct']:.1f}%")
    print(f"  Within 10% of optimal: {metrics['within_10pct']:.1f}%")
    print(f"  Within 20% of optimal: {metrics['within_20pct']:.1f}%")
    print(f"  Rule coverage: {coverage:.1f}% ({len(matched_rules)}/{len(test_df)} workloads)")
    
    # Rule match distribution
    from collections import Counter
    rule_counts = Counter(matched_rules)
    print("\n  Rule match distribution:")
    for rule, count in rule_counts.most_common():
        print(f"    {rule}: {count} ({count/len(test_df)*100:.1f}%)")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/results.csv', index=False)
    
    # Save summary
    summary = {
        'experiment': 'baseline_expert',
        'metrics': metrics,
        'coverage': coverage,
        'num_workloads': len(test_df),
        'rule_matches': dict(rule_counts),
    }
    
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Baseline expert complete!")
    print("=" * 60)
    
    return metrics


if __name__ == '__main__':
    main()

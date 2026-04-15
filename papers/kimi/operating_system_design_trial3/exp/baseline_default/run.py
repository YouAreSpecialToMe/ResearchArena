"""
Baseline: Default Kernel Configuration
Evaluates performance using stock Linux kernel settings.
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

def main():
    print("=" * 60)
    print("BASELINE: Default Kernel Configuration")
    print("=" * 60)
    
    data_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/data'
    output_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/baseline_default'
    
    # Initialize simulator
    simulator = KernelPerformanceSimulator(random_seed=42)
    default_config = KernelConfig()  # Default values
    
    print(f"\nDefault configuration:")
    for k, v in default_config.to_dict().items():
        print(f"  {k}: {v}")
    
    # Load test workloads
    test_df = pd.read_csv(f'{data_dir}/workloads_test.csv')
    oracle_results = json.load(open(f'{data_dir}/oracle_results.json'))
    
    print(f"\nEvaluating {len(test_df)} test workloads...")
    
    results = []
    default_scores = []
    oracle_scores = []
    
    for idx, row in test_df.iterrows():
        workload = load_workload_from_row(row)
        workload_id = row['workload_id']
        
        # Simulate with default config
        result = simulator.simulate(workload, default_config)
        
        # Get oracle score
        oracle_score = oracle_results[workload_id]['score']
        
        default_scores.append(result['score'])
        oracle_scores.append(oracle_score)
        
        results.append({
            'workload_id': workload_id,
            'category': row['category'],
            'default_score': result['score'],
            'oracle_score': oracle_score,
            'normalized_score': result['score'] / oracle_score,
            'throughput': result['throughput'],
            'latency_p50': result['latency_p50'],
            'latency_p99': result['latency_p99'],
        })
    
    default_scores = np.array(default_scores)
    oracle_scores = np.array(oracle_scores)
    
    # Compute metrics
    metrics = compute_performance_metrics(default_scores, oracle_scores)
    
    print("\nResults:")
    print(f"  Mean normalized score: {metrics['mean_normalized_score']:.4f} ± {metrics['std_normalized_score']:.4f}")
    print(f"  Within 5% of optimal: {metrics['within_5pct']:.1f}%")
    print(f"  Within 10% of optimal: {metrics['within_10pct']:.1f}%")
    print(f"  Within 20% of optimal: {metrics['within_20pct']:.1f}%")
    print(f"  Min normalized score: {metrics['min_normalized_score']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/results.csv', index=False)
    
    # Save summary
    summary = {
        'experiment': 'baseline_default',
        'metrics': metrics,
        'config': default_config.to_dict(),
        'num_workloads': len(test_df),
    }
    
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Baseline default complete!")
    print("=" * 60)
    
    return metrics

if __name__ == '__main__':
    main()

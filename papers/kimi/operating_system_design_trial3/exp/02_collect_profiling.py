"""
Step 2: Collect profiling data via simulation.
Simulates running workloads under different kernel configurations.
"""

import sys
sys.path.insert(0, '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp')

import os
import json
import pandas as pd
import numpy as np
from shared.kernel_simulator import KernelPerformanceSimulator, WorkloadSignature
from shared.workload_generator import load_workload_from_row

def main():
    print("=" * 60)
    print("STEP 2: Collecting Profiling Data via Simulation")
    print("=" * 60)
    
    data_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/data'
    
    # Initialize simulator
    simulator = KernelPerformanceSimulator(random_seed=42)
    
    # Get configuration space
    configs = simulator.CONFIG_SPACE
    print(f"\nConfiguration space size: {len(configs)} configurations")
    
    # Load all workloads
    train_df = pd.read_csv(f'{data_dir}/workloads_train.csv')
    test_df = pd.read_csv(f'{data_dir}/workloads_test.csv')
    all_workloads = pd.concat([train_df, test_df], ignore_index=True)
    all_workloads['workload_id'] = list(train_df['workload_id']) + list(test_df['workload_id'])
    
    print(f"Total workloads to profile: {len(all_workloads)}")
    print(f"Total experiments: {len(all_workloads)} × {len(configs)} = {len(all_workloads) * len(configs)}")
    
    # Collect profiling data
    results = []
    oracle_results = {}
    
    print("\nRunning simulations...")
    for idx, row in all_workloads.iterrows():
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(all_workloads)} workloads...")
        
        workload = load_workload_from_row(row)
        workload_id = row['workload_id']
        category = row['category']
        
        best_score = -float('inf')
        best_config_idx = -1
        
        for config_idx, config in enumerate(configs):
            result = simulator.simulate(workload, config)
            
            # Track best configuration
            if result['score'] > best_score:
                best_score = result['score']
                best_config_idx = config_idx
            
            results.append({
                'workload_id': workload_id,
                'category': category,
                'alloc_rate': workload.alloc_rate,
                'working_set_MB': workload.working_set_MB,
                'thread_count': workload.thread_count,
                'thread_churn_per_sec': workload.thread_churn_per_sec,
                'io_read_MBps': workload.io_read_MBps,
                'io_write_MBps': workload.io_write_MBps,
                'io_sequentiality_ratio': workload.io_sequentiality_ratio,
                'syscall_rate_per_sec': workload.syscall_rate_per_sec,
                'config_id': config_idx,
                'swappiness': config.swappiness,
                'dirty_ratio': config.dirty_ratio,
                'dirty_background_ratio': config.dirty_background_ratio,
                'vfs_cache_pressure': config.vfs_cache_pressure,
                'scheduler_type': config.scheduler_type,
                'nr_requests': config.nr_requests,
                'read_ahead_kb': config.read_ahead_kb,
                'sched_latency_ns': config.sched_latency_ns,
                'sched_min_granularity_ns': config.sched_min_granularity_ns,
                'sched_migration_cost_ns': config.sched_migration_cost_ns,
                'throughput': result['throughput'],
                'latency_p50': result['latency_p50'],
                'latency_p99': result['latency_p99'],
                'score': result['score'],
            })
        
        oracle_results[workload_id] = {
            'config_id': best_config_idx,
            'score': best_score,
        }
    
    # Mark optimal configurations
    print("\nMarking optimal configurations...")
    for i, row in enumerate(results):
        wl_id = row['workload_id']
        results[i]['is_optimal'] = (row['config_id'] == oracle_results[wl_id]['config_id'])
        results[i]['normalized_score'] = row['score'] / oracle_results[wl_id]['score']
    
    # Save profiling data
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{data_dir}/profiling_results.csv', index=False)
    
    # Save oracle results
    with open(f'{data_dir}/oracle_results.json', 'w') as f:
        json.dump(oracle_results, f, indent=2)
    
    # Compute ground truth sensitivity
    print("\nComputing ground truth sensitivity...")
    train_results = results_df[results_df['workload_id'].isin(train_df['workload_id'])]
    
    feature_cols = ['alloc_rate', 'working_set_MB', 'thread_churn_per_sec', 
                   'io_sequentiality_ratio', 'syscall_rate_per_sec']
    config_cols = ['swappiness', 'dirty_ratio', 'read_ahead_kb', 'sched_latency_ns']
    
    sensitivity = {}
    for feat in feature_cols:
        sensitivity[feat] = {}
        for cfg in config_cols:
            corr = train_results[feat].corr(train_results[cfg])
            sensitivity[feat][cfg] = corr if not np.isnan(corr) else 0
    
    with open(f'{data_dir}/ground_truth_sensitivity.json', 'w') as f:
        json.dump(sensitivity, f, indent=2)
    
    print("\nGround truth correlations (feature vs config parameter):")
    for feat in feature_cols:
        print(f"  {feat}:")
        for cfg in config_cols:
            print(f"    vs {cfg}: {sensitivity[feat][cfg]:.3f}")
    
    print("\n" + "=" * 60)
    print("Profiling data collection complete!")
    print(f"Results saved to: {data_dir}/profiling_results.csv")
    print(f"Total rows: {len(results_df)}")
    print("=" * 60)

if __name__ == '__main__':
    main()

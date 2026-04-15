#!/usr/bin/env python3
"""
Targeted experiments to demonstrate scheduler differences.
Focuses on workloads where NUMA awareness matters.
"""

import json
import os
import sys
import numpy as np
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator.numa_scheduler_sim import (
    EEVDFScheduler, AutoNUMAScheduler, TiresiasScheduler,
    CXLAimPodScheduler, UniSchedScheduler, Task
)


def create_memory_intensive_workload(num_tasks, seed, cxl_ratio=0.5):
    """Create workload with high CXL memory access to stress NUMA scheduling."""
    random.seed(seed)
    np.random.seed(seed)
    
    tasks = []
    arrival_time = 0.0
    
    for i in range(num_tasks):
        # Tight arrivals to create contention
        arrival_time += random.expovariate(0.2)  # 5 tasks per 1000ms
        
        # Memory-intensive tasks
        duration = random.uniform(200, 800)
        footprint = random.uniform(1000, 5000)
        
        # Access pattern with specified CXL ratio
        local_ratio = 1.0 - cxl_ratio
        access_dist = {
            0: local_ratio * 0.6,
            1: local_ratio * 0.4,
            2: cxl_ratio * 0.6,
            3: cxl_ratio * 0.4
        }
        
        tasks.append(Task(
            id=i,
            arrival_time=arrival_time,
            duration_ms=duration,
            memory_footprint_mb=footprint,
            access_pattern='cxl_bandwidth' if cxl_ratio > 0.3 else 'local_dominant',
            access_distribution=access_dist,
            compute_demand=random.uniform(0.5, 0.9)
        ))
    
    return tasks


def run_experiment(scheduler_class, scheduler_name, tasks, topology_config, seed):
    """Run a single experiment."""
    try:
        scheduler = scheduler_class(topology_config, seed=seed)
        scheduler.run_simulation(tasks)
        metrics = scheduler.calculate_metrics()
        
        return {
            'scheduler': scheduler_name,
            'throughput': metrics['throughput_tasks_per_sec'],
            'avg_latency': metrics['avg_latency_ms'],
            'p95_latency': metrics['p95_latency_ms'],
            'fairness': metrics['jain_fairness'],
            'migrations': metrics['migration_count']
        }
    except Exception as e:
        print(f"Error with {scheduler_name}: {e}")
        return None


def run_comparison(tasks, topology_config, seed, label):
    """Run comparison of all schedulers."""
    results = []
    
    schedulers = [
        (EEVDFScheduler, 'EEVDF'),
        (AutoNUMAScheduler, 'AutoNUMA'),
        (TiresiasScheduler, 'Tiresias'),
        (UniSchedScheduler, 'UniSched_Full')
    ]
    
    for sched_class, sched_name in schedulers:
        result = run_experiment(sched_class, sched_name, tasks, topology_config, seed)
        if result:
            results.append(result)
    
    return results


def main():
    """Run targeted experiments."""
    print("="*60)
    print("Targeted UniSched Experiments")
    print("="*60)
    
    with open('exp/simulator/topology_config.json') as f:
        topology_config = json.load(f)
    
    all_results = []
    
    # Experiment 1: High CXL memory ratio
    print("\n1. High CXL Memory Ratio (50% remote access)...")
    for seed in [42, 43, 44]:
        tasks = create_memory_intensive_workload(100, seed, cxl_ratio=0.5)
        results = run_comparison(tasks, topology_config, seed, "high_cxl")
        for r in results:
            r['experiment'] = 'high_cxl'
            r['seed'] = seed
        all_results.extend(results)
    
    # Experiment 2: Mixed workload with varying patterns
    print("2. Mixed Workload Patterns...")
    for seed in [42, 43, 44]:
        tasks = create_memory_intensive_workload(100, seed, cxl_ratio=0.3)
        results = run_comparison(tasks, topology_config, seed, "mixed")
        for r in results:
            r['experiment'] = 'mixed'
            r['seed'] = seed
        all_results.extend(results)
    
    # Experiment 3: Local-dominant baseline
    print("3. Local-Dominant Baseline...")
    for seed in [42, 43, 44]:
        tasks = create_memory_intensive_workload(100, seed, cxl_ratio=0.1)
        results = run_comparison(tasks, topology_config, seed, "local")
        for r in results:
            r['experiment'] = 'local'
            r['seed'] = seed
        all_results.extend(results)
    
    # Save results
    os.makedirs('exp/results', exist_ok=True)
    with open('exp/results/targeted_experiments.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Aggregate and display
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    import pandas as pd
    df = pd.DataFrame(all_results)
    
    for experiment in df['experiment'].unique():
        print(f"\n{experiment.upper()}:")
        exp_data = df[df['experiment'] == experiment]
        
        for scheduler in exp_data['scheduler'].unique():
            sched_data = exp_data[exp_data['scheduler'] == scheduler]
            throughput = sched_data['throughput'].mean()
            latency = sched_data['avg_latency'].mean()
            fairness = sched_data['fairness'].mean()
            print(f"  {scheduler:20s}: throughput={throughput:.2f}, latency={latency:.1f}, fairness={fairness:.3f}")
    
    # Calculate improvements
    print("\n" + "="*60)
    print("IMPROVEMENT OVER AutoNUMA")
    print("="*60)
    
    for experiment in df['experiment'].unique():
        exp_data = df[df['experiment'] == experiment]
        autonuma_data = exp_data[exp_data['scheduler'] == 'AutoNUMA']
        unisched_data = exp_data[exp_data['scheduler'] == 'UniSched_Full']
        
        if len(autonuma_data) > 0 and len(unisched_data) > 0:
            autonuma_throughput = autonuma_data['throughput'].mean()
            unisched_throughput = unisched_data['throughput'].mean()
            improvement = ((unisched_throughput - autonuma_throughput) / autonuma_throughput) * 100
            
            print(f"{experiment:15s}: {improvement:+.2f}% ({autonuma_throughput:.2f} -> {unisched_throughput:.2f})")


if __name__ == '__main__':
    main()

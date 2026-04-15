#!/usr/bin/env python3
"""
WattSched Ablation: No Classification

Tests WattSched with workload classification disabled (all workloads treated as mixed).
This isolates the contribution of classification to energy savings.
"""

import sys
import os
import json
import random
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from workload_simulator import (
    WattSchedScheduler, create_heterogeneous_topology,
    generate_workload_mix, run_simulation, WorkloadType, Process
)


def run_experiment(seed: int, workload_config: dict) -> dict:
    """Run a single experiment without classification."""
    random.seed(seed)
    np.random.seed(seed)
    
    cores = create_heterogeneous_topology(n_big=4, n_little=4)
    processes = generate_workload_mix(
        count_per_type=workload_config['count_per_type'],
        instructions_per_process=workload_config['instructions_per_process']
    )
    
    start_time = datetime.now()
    stats = run_simulation(
        WattSchedScheduler,
        cores,
        processes,
        scheduler_kwargs={
            'time_slice_ms': 4.0,
            'enable_classification': False,  # Disabled
            'enable_topology': True,
            'enable_adaptive_slice': True
        },
        dt=0.001
    )
    end_time = datetime.now()
    
    return {
        'seed': seed,
        'workload_config': workload_config,
        **stats,
        'runtime_seconds': (end_time - start_time).total_seconds()
    }


def main():
    """Run ablation experiments."""
    seeds = [42, 123, 999]
    
    configs = [
        {
            'name': 'mixed_workload_small',
            'workload': {'count_per_type': 4, 'instructions_per_process': 100_000_000}
        },
        {
            'name': 'mixed_workload_large',
            'workload': {'count_per_type': 8, 'instructions_per_process': 100_000_000}
        },
    ]
    
    all_results = []
    
    for config in configs:
        print(f"Running {config['name']} (no classification)...")
        results_for_config = []
        
        for seed in seeds:
            result = run_experiment(seed, config['workload'])
            results_for_config.append(result)
        
        energies = [r['total_energy_joules'] for r in results_for_config]
        times = [r['total_time_seconds'] for r in results_for_config]
        
        summary = {
            'experiment': config['name'],
            'scheduler': 'WattSched_NoClassify',
            'seeds': seeds,
            'raw_results': results_for_config,
            'energy_mean_joules': np.mean(energies),
            'energy_std_joules': np.std(energies),
            'time_mean_seconds': np.mean(times),
            'time_std_seconds': np.std(times)
        }
        
        all_results.append(summary)
        print(f"  Energy: {summary['energy_mean_joules']:.2f} ± {summary['energy_std_joules']:.2f} J")
        print(f"  Time: {summary['time_mean_seconds']:.2f} ± {summary['time_std_seconds']:.2f} s")
    
    output = {
        'scheduler': 'WattSched_NoClassify',
        'description': 'WattSched without workload classification (all treated as mixed)',
        'experiments': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results.json")
    return all_results


if __name__ == '__main__':
    main()

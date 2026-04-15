#!/usr/bin/env python3
"""
WattSched Full Implementation Experiment

This scheduler combines:
1. Runtime workload classification (CPU, memory, mixed, I/O-bound)
2. Topology-aware scheduling (big/little core assignment)
3. Adaptive time slicing based on workload type
"""

import sys
import os
import json
import random
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from workload_simulator import (
    WattSchedScheduler, create_heterogeneous_topology, create_homogeneous_topology,
    generate_workload_mix, run_simulation, WorkloadType, Process
)


def run_experiment(seed: int, workload_config: dict, topology: str = "heterogeneous",
                   enable_classification: bool = True, enable_topology: bool = True,
                   enable_adaptive_slice: bool = True) -> dict:
    """Run a single experiment with given configuration."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Create topology
    if topology == "heterogeneous":
        cores = create_heterogeneous_topology(n_big=4, n_little=4)
    else:
        cores = create_homogeneous_topology(n_cores=8)
    
    # Generate workload
    processes = generate_workload_mix(
        count_per_type=workload_config['count_per_type'],
        instructions_per_process=workload_config['instructions_per_process']
    )
    
    # Run simulation
    start_time = datetime.now()
    stats = run_simulation(
        WattSchedScheduler,
        cores,
        processes,
        scheduler_kwargs={
            'time_slice_ms': 4.0,
            'enable_classification': enable_classification,
            'enable_topology': enable_topology,
            'enable_adaptive_slice': enable_adaptive_slice
        },
        dt=0.001
    )
    end_time = datetime.now()
    
    return {
        'seed': seed,
        'topology': topology,
        'workload_config': workload_config,
        'enable_classification': enable_classification,
        'enable_topology': enable_topology,
        'enable_adaptive_slice': enable_adaptive_slice,
        **stats,
        'runtime_seconds': (end_time - start_time).total_seconds()
    }


def main():
    """Run all WattSched experiments."""
    seeds = [42, 123, 999]
    
    # Experiment configurations
    configs = [
        {
            'name': 'mixed_workload_small',
            'workload': {'count_per_type': 4, 'instructions_per_process': 100_000_000},
            'topology': 'heterogeneous'
        },
        {
            'name': 'mixed_workload_large',
            'workload': {'count_per_type': 8, 'instructions_per_process': 100_000_000},
            'topology': 'heterogeneous'
        },
        {
            'name': 'cpu_only',
            'workload': {'count_per_type': 0, 'instructions_per_process': 100_000_000},
            'topology': 'heterogeneous'
        },
        {
            'name': 'memory_only',
            'workload': {'count_per_type': 0, 'instructions_per_process': 100_000_000},
            'topology': 'heterogeneous'
        },
        {
            'name': 'homogeneous_mixed',
            'workload': {'count_per_type': 4, 'instructions_per_process': 100_000_000},
            'topology': 'homogeneous'
        },
    ]
    
    def generate_single_type(workload_type, count, instructions):
        processes = []
        for i in range(count):
            processes.append(Process(
                pid=i+1,
                workload_type=workload_type,
                arrival_time=0.0,
                total_instructions=instructions
            ))
        return processes
    
    all_results = []
    
    for config in configs:
        print(f"Running {config['name']}...")
        results_for_config = []
        
        for seed in seeds:
            if config['name'] == 'cpu_only':
                random.seed(seed)
                np.random.seed(seed)
                cores = create_heterogeneous_topology(n_big=4, n_little=4)
                processes = generate_single_type(WorkloadType.CPU_BOUND, 8, 100_000_000)
                stats = run_simulation(WattSchedScheduler, cores, processes,
                                       scheduler_kwargs={
                                           'time_slice_ms': 4.0,
                                           'enable_classification': True,
                                           'enable_topology': True,
                                           'enable_adaptive_slice': True
                                       }, dt=0.001)
                result = {'seed': seed, 'topology': config['topology'], 'workload_config': config['workload'],
                          'enable_classification': True, 'enable_topology': True, 'enable_adaptive_slice': True, **stats}
            elif config['name'] == 'memory_only':
                random.seed(seed)
                np.random.seed(seed)
                cores = create_heterogeneous_topology(n_big=4, n_little=4)
                processes = generate_single_type(WorkloadType.MEMORY_BOUND, 8, 100_000_000)
                stats = run_simulation(WattSchedScheduler, cores, processes,
                                       scheduler_kwargs={
                                           'time_slice_ms': 4.0,
                                           'enable_classification': True,
                                           'enable_topology': True,
                                           'enable_adaptive_slice': True
                                       }, dt=0.001)
                result = {'seed': seed, 'topology': config['topology'], 'workload_config': config['workload'],
                          'enable_classification': True, 'enable_topology': True, 'enable_adaptive_slice': True, **stats}
            elif config['name'] == 'homogeneous_mixed':
                result = run_experiment(seed, config['workload'], config['topology'])
            else:
                result = run_experiment(seed, config['workload'], config['topology'])
            
            results_for_config.append(result)
        
        # Calculate statistics across seeds
        energies = [r['total_energy_joules'] for r in results_for_config]
        times = [r['total_time_seconds'] for r in results_for_config]
        
        summary = {
            'experiment': config['name'],
            'scheduler': 'WattSched',
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
    
    # Save results
    output = {
        'scheduler': 'WattSched',
        'description': 'Full WattSched with workload classification, topology optimization, and adaptive slicing',
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

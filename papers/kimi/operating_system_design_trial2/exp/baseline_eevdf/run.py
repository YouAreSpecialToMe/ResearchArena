#!/usr/bin/env python3
"""
Baseline EEVDF (Linux CFS) Experiment

Simulates standard Linux EEVDF scheduler behavior without workload awareness.
This serves as the primary baseline for comparison.
"""

import sys
import os
import json
import random
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from workload_simulator import (
    EEVDFScheduler, create_heterogeneous_topology, create_homogeneous_topology,
    generate_workload_mix, run_simulation, WorkloadType, Process
)


def run_experiment(seed: int, workload_config: dict, topology: str = "heterogeneous") -> dict:
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
        EEVDFScheduler,
        cores,
        processes,
        scheduler_kwargs={'time_slice_ms': 4.0},
        dt=0.001
    )
    end_time = datetime.now()
    
    return {
        'seed': seed,
        'topology': topology,
        'workload_config': workload_config,
        'total_energy_joules': stats['total_energy_joules'],
        'total_time_seconds': stats['total_time_seconds'],
        'energy_per_million_instr': stats['energy_per_million_instr'],
        'processes_completed': stats['processes_completed'],
        'scheduling_decisions': stats['scheduling_decisions'],
        'runtime_seconds': (end_time - start_time).total_seconds()
    }


def main():
    """Run all baseline experiments."""
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
    ]
    
    # Special handling for cpu_only and memory_only
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
                # Generate only CPU-bound processes
                random.seed(seed)
                np.random.seed(seed)
                cores = create_heterogeneous_topology(n_big=4, n_little=4)
                processes = generate_single_type(WorkloadType.CPU_BOUND, 8, 100_000_000)
                stats = run_simulation(EEVDFScheduler, cores, processes, 
                                       scheduler_kwargs={'time_slice_ms': 4.0}, dt=0.001)
                result = {
                    'seed': seed,
                    'topology': config['topology'],
                    'workload_config': config['workload'],
                    **stats
                }
            elif config['name'] == 'memory_only':
                # Generate only memory-bound processes
                random.seed(seed)
                np.random.seed(seed)
                cores = create_heterogeneous_topology(n_big=4, n_little=4)
                processes = generate_single_type(WorkloadType.MEMORY_BOUND, 8, 100_000_000)
                stats = run_simulation(EEVDFScheduler, cores, processes,
                                       scheduler_kwargs={'time_slice_ms': 4.0}, dt=0.001)
                result = {
                    'seed': seed,
                    'topology': config['topology'],
                    'workload_config': config['workload'],
                    **stats
                }
            else:
                result = run_experiment(seed, config['workload'], config['topology'])
            
            results_for_config.append(result)
        
        # Calculate statistics across seeds
        energies = [r['total_energy_joules'] for r in results_for_config]
        times = [r['total_time_seconds'] for r in results_for_config]
        
        summary = {
            'experiment': config['name'],
            'scheduler': 'EEVDF',
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
        'scheduler': 'EEVDF',
        'description': 'Linux EEVDF baseline without workload awareness',
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

#!/usr/bin/env python3
"""
Comprehensive Experiment Runner for UniSched Evaluation.
Runs all baselines and UniSched variants across all workloads with multiple seeds.
"""

import json
import os
import sys
import glob
import time
import multiprocessing as mp
from pathlib import Path

# Ensure simulator is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator.numa_scheduler_sim import (
    EEVDFScheduler, AutoNUMAScheduler, TiresiasScheduler,
    CXLAimPodScheduler, UniSchedScheduler, Task
)


def load_workload(filepath: str) -> list:
    """Load workload from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    tasks = []
    for t in data['tasks']:
        # Convert string keys to int for access_distribution
        access_dist = {int(k): v for k, v in t['access_distribution'].items()}
        tasks.append(Task(
            id=t['id'],
            arrival_time=t['arrival_time'],
            duration_ms=t['duration_ms'],
            memory_footprint_mb=t['memory_footprint_mb'],
            access_pattern=t['access_pattern'],
            access_distribution=access_dist,
            compute_demand=t['compute_demand']
        ))
    
    return tasks


def run_single_experiment(args):
    """Run a single experiment configuration."""
    scheduler_name, workload_file, topology_config, seed = args
    
    try:
        # Load workload
        tasks = load_workload(workload_file)
        
        # Create scheduler
        if scheduler_name == 'EEVDF':
            scheduler = EEVDFScheduler(topology_config, seed=seed)
        elif scheduler_name == 'AutoNUMA':
            scheduler = AutoNUMAScheduler(topology_config, seed=seed)
        elif scheduler_name == 'Tiresias':
            scheduler = TiresiasScheduler(topology_config, seed=seed)
        elif scheduler_name == 'CXLAimPod':
            scheduler = CXLAimPodScheduler(topology_config, seed=seed)
        elif scheduler_name == 'UniSched_Full':
            scheduler = UniSchedScheduler(topology_config, seed=seed,
                                         enable_profiling=True,
                                         enable_topology=True,
                                         enable_coordination=True)
        elif scheduler_name == 'UniSched_TopologyOnly':
            scheduler = UniSchedScheduler(topology_config, seed=seed,
                                         enable_profiling=False,
                                         enable_topology=True,
                                         enable_coordination=False)
        elif scheduler_name == 'UniSched_ProfilingOnly':
            scheduler = UniSchedScheduler(topology_config, seed=seed,
                                         enable_profiling=True,
                                         enable_topology=False,
                                         enable_coordination=False)
        elif scheduler_name == 'UniSched_NoCoord':
            scheduler = UniSchedScheduler(topology_config, seed=seed,
                                         enable_profiling=True,
                                         enable_topology=True,
                                         enable_coordination=False)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        # Run simulation
        start_time = time.time()
        scheduler.run_simulation(tasks)
        metrics = scheduler.calculate_metrics()
        runtime = time.time() - start_time
        
        # Add metadata
        result = {
            'scheduler': scheduler_name,
            'workload': os.path.basename(workload_file),
            'seed': seed,
            'metrics': metrics,
            'runtime_seconds': runtime
        }
        
        return result
    
    except Exception as e:
        return {
            'scheduler': scheduler_name,
            'workload': os.path.basename(workload_file),
            'seed': seed,
            'error': str(e)
        }


def run_experiments_parallel(schedulers, workloads, topology_config, seeds, max_workers=None):
    """Run all experiments in parallel."""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)
    
    # Build experiment list
    experiments = []
    for scheduler in schedulers:
        for workload in workloads:
            for seed in seeds:
                experiments.append((scheduler, workload, topology_config, seed))
    
    print(f"Running {len(experiments)} experiments with {max_workers} workers...")
    
    # Run in parallel
    with mp.Pool(max_workers) as pool:
        results = pool.map(run_single_experiment, experiments)
    
    return results


def run_sensitivity_sampling_frequency(topology_config, base_workload_file):
    """Run sensitivity analysis for PMU sampling frequency."""
    print("\n" + "="*60)
    print("Sensitivity Analysis: PMU Sampling Frequency")
    print("="*60)
    
    frequencies = [0.1, 0.5, 1.0, 2.0, 5.0]
    tasks = load_workload(base_workload_file)
    results = []
    
    for freq in frequencies:
        print(f"  Testing {freq}% sampling...", end='', flush=True)
        
        # Modify config
        config = json.loads(json.dumps(topology_config))
        config['pmu_config']['sampling_frequency_pct'] = freq
        config['pmu_config']['overhead_pct'] = freq * 0.5  # Rough correlation
        
        scheduler = UniSchedScheduler(config, seed=42)
        scheduler.run_simulation(tasks)
        metrics = scheduler.calculate_metrics()
        
        results.append({
            'sampling_frequency_pct': freq,
            'throughput': metrics['throughput_tasks_per_sec'],
            'avg_latency': metrics['avg_latency_ms'],
            'fairness': metrics['jain_fairness']
        })
        print(f" throughput={metrics['throughput_tasks_per_sec']:.2f}")
    
    return results


def run_sensitivity_cxl_latency(topology_config, workloads):
    """Run sensitivity analysis for CXL latency variations."""
    print("\n" + "="*60)
    print("Sensitivity Analysis: CXL Latency Variations")
    print("="*60)
    
    latency_scenarios = [
        ('low', 150, 250),
        ('medium', 200, 350),
        ('high', 300, 500)
    ]
    
    schedulers_to_test = ['EEVDF', 'AutoNUMA', 'Tiresias', 'UniSched_Full']
    results = []
    
    # Use medium mixed workload as representative
    workload_file = [w for w in workloads if 'mixed_workload_medium' in w][0]
    tasks = load_workload(workload_file)
    
    for scenario, local_cxl, remote_cxl in latency_scenarios:
        print(f"\n  Scenario: {scenario} (local={local_cxl}ns, remote={remote_cxl}ns)")
        
        # Modify config
        config = json.loads(json.dumps(topology_config))
        config['numa_nodes'][2]['latency_ns'] = local_cxl
        config['numa_nodes'][3]['latency_ns'] = remote_cxl
        config['latency_matrix_ns'][0][2] = local_cxl
        config['latency_matrix_ns'][0][3] = remote_cxl
        config['latency_matrix_ns'][1][2] = local_cxl
        config['latency_matrix_ns'][1][3] = remote_cxl
        config['latency_matrix_ns'][2][0] = local_cxl
        config['latency_matrix_ns'][2][1] = local_cxl
        config['latency_matrix_ns'][3][0] = remote_cxl
        config['latency_matrix_ns'][3][1] = remote_cxl
        
        for scheduler_name in schedulers_to_test:
            print(f"    {scheduler_name}...", end='', flush=True)
            
            if scheduler_name == 'EEVDF':
                scheduler = EEVDFScheduler(config, seed=42)
            elif scheduler_name == 'AutoNUMA':
                scheduler = AutoNUMAScheduler(config, seed=42)
            elif scheduler_name == 'Tiresias':
                scheduler = TiresiasScheduler(config, seed=42)
            else:
                scheduler = UniSchedScheduler(config, seed=42)
            
            scheduler.run_simulation(tasks)
            metrics = scheduler.calculate_metrics()
            
            results.append({
                'scenario': scenario,
                'local_cxl_ns': local_cxl,
                'remote_cxl_ns': remote_cxl,
                'scheduler': scheduler_name,
                'throughput': metrics['throughput_tasks_per_sec'],
                'avg_latency': metrics['avg_latency_ms']
            })
            print(f" throughput={metrics['throughput_tasks_per_sec']:.2f}")
    
    return results


def main():
    """Main experiment runner."""
    print("="*60)
    print("UniSched Comprehensive Experiment Suite")
    print("="*60)
    
    # Load topology config
    with open('exp/simulator/topology_config.json') as f:
        topology_config = json.load(f)
    
    # Find all workloads
    workload_files = sorted(glob.glob('exp/simulator/workloads/*.json'))
    print(f"Found {len(workload_files)} workloads")
    
    # Define schedulers to test
    schedulers = [
        'EEVDF',
        'AutoNUMA', 
        'Tiresias',
        'CXLAimPod',
        'UniSched_Full',
        'UniSched_TopologyOnly',
        'UniSched_ProfilingOnly',
        'UniSched_NoCoord'
    ]
    
    # Seeds for reproducibility
    seeds = [42, 43, 44]
    
    # Run main experiments
    print(f"\nSchedulers: {len(schedulers)}")
    print(f"Workloads: {len(workload_files)}")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {len(schedulers) * len(workload_files) * len(seeds)}")
    
    # Use fewer workers due to SimPy's single-threaded nature per simulation
    # But we can parallelize across experiments
    results = run_experiments_parallel(schedulers, workload_files, topology_config, seeds, max_workers=4)
    
    # Save main results
    os.makedirs('exp/results', exist_ok=True)
    with open('exp/results/main_experiments.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nMain results saved to: exp/results/main_experiments.json")
    
    # Run sensitivity analyses
    sampling_results = run_sensitivity_sampling_frequency(
        topology_config, 
        [w for w in workload_files if 'mixed_workload_medium' in w][0]
    )
    
    with open('exp/results/sensitivity_sampling.json', 'w') as f:
        json.dump(sampling_results, f, indent=2)
    
    latency_results = run_sensitivity_cxl_latency(topology_config, workload_files)
    
    with open('exp/results/sensitivity_latency.json', 'w') as f:
        json.dump(latency_results, f, indent=2)
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)
    print("\nOutput files:")
    print("  - exp/results/main_experiments.json")
    print("  - exp/results/sensitivity_sampling.json")
    print("  - exp/results/sensitivity_latency.json")


if __name__ == '__main__':
    main()

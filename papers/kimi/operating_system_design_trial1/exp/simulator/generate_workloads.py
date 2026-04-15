#!/usr/bin/env python3
"""Generate workload traces for NUMA scheduler simulation."""

import json
import random
import numpy as np
from typing import List, Dict


class Task:
    """Lightweight task representation for serialization."""
    def __init__(self, id: int, arrival_time: float, duration_ms: float,
                 memory_footprint_mb: float, access_pattern: str,
                 access_distribution: Dict[int, float], compute_demand: float):
        self.id = id
        self.arrival_time = arrival_time
        self.duration_ms = duration_ms
        self.memory_footprint_mb = memory_footprint_mb
        self.access_pattern = access_pattern
        self.access_distribution = access_distribution
        self.compute_demand = compute_demand
    
    def to_dict(self):
        return {
            'id': self.id,
            'arrival_time': self.arrival_time,
            'duration_ms': self.duration_ms,
            'memory_footprint_mb': self.memory_footprint_mb,
            'access_pattern': self.access_pattern,
            'access_distribution': self.access_distribution,
            'compute_demand': self.compute_demand
        }


def generate_redis_ycsb(num_tasks: int, seed: int) -> List[Dict]:
    """Redis/YCSB-like workload: Key-value operations with Zipfian distribution."""
    random.seed(seed)
    np.random.seed(seed)
    
    tasks = []
    arrival_time = 0.0
    
    for i in range(num_tasks):
        # Poisson arrival
        arrival_time += random.expovariate(0.1)  # 10 ops per ms avg
        
        # Short duration - KV operations
        duration = random.expovariate(1/20.0)  # mean 20ms
        duration = max(5, min(duration, 100))
        
        # Memory footprint - hot keys fit in cache
        footprint = random.uniform(50, 500)
        
        # Zipfian-like access: 50% to hot data (node 0), rest distributed
        # This mimics YCSB workload A/B with hot records
        r = random.random()
        if r < 0.5:
            access_dist = {0: 0.7, 1: 0.2, 2: 0.08, 3: 0.02}
        elif r < 0.8:
            access_dist = {0: 0.3, 1: 0.5, 2: 0.15, 3: 0.05}
        else:
            access_dist = {0: 0.2, 1: 0.2, 2: 0.4, 3: 0.2}
        
        tasks.append(Task(
            id=i,
            arrival_time=arrival_time,
            duration_ms=duration,
            memory_footprint_mb=footprint,
            access_pattern='mixed',
            access_distribution=access_dist,
            compute_demand=random.uniform(0.2, 0.6)
        ).to_dict())
    
    return tasks


def generate_graph_pagerank(num_tasks: int, seed: int) -> List[Dict]:
    """Graph analytics: PageRank-style iterative computation."""
    random.seed(seed)
    np.random.seed(seed)
    
    tasks = []
    arrival_time = 0.0
    
    # Graph processing has phases - many concurrent tasks
    for phase in range(5):
        phase_tasks = num_tasks // 5
        for i in range(phase_tasks):
            # Batch arrivals within phase
            arrival_time += random.expovariate(0.5)
            
            # Longer duration - iterative computation
            duration = random.uniform(200, 1000)
            
            # Large memory footprint - graph exceeds local DRAM
            footprint = random.uniform(2000, 10000)
            
            # Access pattern: distributed across all nodes (graph traversals)
            # Some locality from vertex clustering
            r = random.random()
            if r < 0.4:
                access_dist = {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}
            elif r < 0.7:
                access_dist = {0: 0.2, 1: 0.2, 2: 0.35, 3: 0.25}
            else:
                access_dist = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
            
            tasks.append(Task(
                id=len(tasks),
                arrival_time=arrival_time,
                duration_ms=duration,
                memory_footprint_mb=footprint,
                access_pattern='cxl_bandwidth',
                access_distribution=access_dist,
                compute_demand=random.uniform(0.6, 0.95)
            ).to_dict())
        
        # Gap between phases
        arrival_time += random.uniform(100, 300)
    
    return tasks


def generate_stream_bandwidth(num_tasks: int, seed: int) -> List[Dict]:
    """STREAM-like bandwidth benchmark: Sequential memory access."""
    random.seed(seed)
    np.random.seed(seed)
    
    tasks = []
    arrival_time = 0.0
    
    for i in range(num_tasks):
        arrival_time += random.expovariate(0.05)  # Lower arrival rate
        
        # Long duration - sustained bandwidth test
        duration = random.uniform(300, 1500)
        
        # Large sequential memory footprint
        footprint = random.uniform(1000, 5000)
        
        # Read-heavy access pattern
        read_ratio = random.uniform(0.6, 0.9)
        
        # Sequential accesses often benefit from CXL bandwidth
        r = random.random()
        if r < 0.33:
            # Local DRAM bound
            access_dist = {0: 0.7, 1: 0.2, 2: 0.08, 3: 0.02}
        elif r < 0.66:
            # CXL bandwidth bound
            access_dist = {0: 0.2, 1: 0.2, 2: 0.35, 3: 0.25}
        else:
            # Distributed
            access_dist = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        
        tasks.append(Task(
            id=i,
            arrival_time=arrival_time,
            duration_ms=duration,
            memory_footprint_mb=footprint,
            access_pattern='cxl_bandwidth',
            access_distribution=access_dist,
            compute_demand=random.uniform(0.4, 0.8)
        ).to_dict())
    
    return tasks


def generate_latency_sensitive(num_tasks: int, seed: int) -> List[Dict]:
    """Latency-sensitive workload: Pointer chasing, small working set."""
    random.seed(seed)
    np.random.seed(seed)
    
    tasks = []
    arrival_time = 0.0
    
    for i in range(num_tasks):
        arrival_time += random.expovariate(0.2)
        
        # Short duration but very latency-sensitive
        duration = random.uniform(10, 100)
        
        # Small working set that fits in cache
        footprint = random.uniform(10, 100)
        
        # Strong preference for local memory
        r = random.random()
        if r < 0.7:
            # Local DRAM preferred
            access_dist = {0: 0.8, 1: 0.15, 2: 0.04, 3: 0.01}
        else:
            # Some remote access but low latency critical
            access_dist = {0: 0.6, 1: 0.3, 2: 0.08, 3: 0.02}
        
        tasks.append(Task(
            id=i,
            arrival_time=arrival_time,
            duration_ms=duration,
            memory_footprint_mb=footprint,
            access_pattern='latency_sensitive',
            access_distribution=access_dist,
            compute_demand=random.uniform(0.3, 0.7)
        ).to_dict())
    
    return tasks


def generate_mixed_workload(num_tasks: int, seed: int) -> List[Dict]:
    """Mixed workload: Combination of all patterns."""
    random.seed(seed)
    np.random.seed(seed)
    
    tasks = []
    arrival_time = 0.0
    
    # Mix: 30% local, 30% bandwidth, 20% latency, 20% balanced
    for i in range(num_tasks):
        arrival_time += random.expovariate(0.1)
        
        r = random.random()
        if r < 0.3:
            # Local dominant
            access_dist = {0: 0.75, 1: 0.15, 2: 0.08, 3: 0.02}
            pattern = 'local_dominant'
            duration = random.uniform(100, 500)
            footprint = random.uniform(200, 1000)
        elif r < 0.6:
            # CXL bandwidth
            access_dist = {0: 0.2, 1: 0.15, 2: 0.4, 3: 0.25}
            pattern = 'cxl_bandwidth'
            duration = random.uniform(300, 1000)
            footprint = random.uniform(1000, 5000)
        elif r < 0.8:
            # Latency sensitive
            access_dist = {0: 0.7, 1: 0.2, 2: 0.08, 3: 0.02}
            pattern = 'latency_sensitive'
            duration = random.uniform(20, 150)
            footprint = random.uniform(20, 200)
        else:
            # Balanced
            access_dist = {0: 0.3, 1: 0.2, 2: 0.3, 3: 0.2}
            pattern = 'mixed'
            duration = random.uniform(100, 600)
            footprint = random.uniform(100, 2000)
        
        tasks.append(Task(
            id=i,
            arrival_time=arrival_time,
            duration_ms=duration,
            memory_footprint_mb=footprint,
            access_pattern=pattern,
            access_distribution=access_dist,
            compute_demand=random.uniform(0.3, 0.9)
        ).to_dict())
    
    return tasks


def main():
    """Generate all workload files."""
    import os
    os.makedirs('exp/simulator/workloads', exist_ok=True)
    
    workload_generators = {
        'redis_ycsb': generate_redis_ycsb,
        'graph_pagerank': generate_graph_pagerank,
        'stream_bandwidth': generate_stream_bandwidth,
        'latency_sensitive': generate_latency_sensitive,
        'mixed_workload': generate_mixed_workload
    }
    
    sizes = {
        'small': 50,
        'medium': 100,
        'large': 200
    }
    
    seeds = [42, 43, 44]  # 3 random seeds for statistical confidence
    
    total_workloads = 0
    
    for workload_name, generator in workload_generators.items():
        for size_name, num_tasks in sizes.items():
            for seed in seeds:
                tasks = generator(num_tasks, seed)
                
                filename = f'exp/simulator/workloads/{workload_name}_{size_name}_seed{seed}.json'
                with open(filename, 'w') as f:
                    json.dump({
                        'workload_type': workload_name,
                        'size': size_name,
                        'num_tasks': num_tasks,
                        'seed': seed,
                        'tasks': tasks
                    }, f, indent=2)
                
                total_workloads += 1
                print(f"Generated: {filename} ({num_tasks} tasks)")
    
    print(f"\nTotal workloads generated: {total_workloads}")


if __name__ == '__main__':
    main()

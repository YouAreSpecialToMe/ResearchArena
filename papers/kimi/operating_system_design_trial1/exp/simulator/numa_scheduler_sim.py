#!/usr/bin/env python3
"""
NUMA Scheduler Simulator - Discrete Event Simulation for CXL Memory Systems
Uses SimPy for event-driven simulation of task scheduling on NUMA architectures.
"""

import simpy
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import statistics


@dataclass
class Task:
    """Represents a computational task with memory access patterns."""
    id: int
    arrival_time: float
    duration_ms: float
    memory_footprint_mb: float
    access_pattern: str  # 'local_dominant', 'cxl_bandwidth', 'latency_sensitive', 'mixed'
    access_distribution: Dict[int, float]  # node_id -> percentage
    compute_demand: float  # 0.0 to 1.0
    priority: int = 0
    
    # Runtime state
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    assigned_cpu: Optional[int] = None
    current_node: Optional[int] = None
    migration_count: int = 0
    cpu_time_ms: float = 0.0
    memory_accesses: int = 0
    remote_accesses: int = 0


@dataclass
class NumaNode:
    """Represents a NUMA node (local DRAM or CXL-attached)."""
    id: int
    node_type: str  # 'local_dram', 'cxl_attached', 'cxl_remote'
    cpus: List[int]
    memory_gb: float
    latency_ns: float
    bandwidth_gb_s: float
    is_cxl: bool
    cxl_controller: Optional[int] = None
    
    # Runtime state
    cpu_load: Dict[int, float] = field(default_factory=dict)  # cpu -> load
    memory_used_gb: float = 0.0
    bandwidth_used_gb_s: float = 0.0


class PMUProfiler:
    """Simulates PMU-based memory access profiling."""
    
    def __init__(self, config: Dict, overhead_pct: float):
        self.sampling_frequency = config.get('sampling_frequency_pct', 1.0) / 100.0
        self.overhead_pct = overhead_pct
        self.task_profiles = defaultdict(lambda: defaultdict(int))  # task_id -> node_accesses
        
    def profile_task(self, task: Task, actual_accesses: Dict[int, int]):
        """Profile a task's memory accesses with sampling."""
        # Sample accesses based on frequency
        for node, count in actual_accesses.items():
            sampled = np.random.binomial(count, self.sampling_frequency)
            self.task_profiles[task.id][node] += sampled
    
    def get_task_classification(self, task_id: int) -> str:
        """Classify task based on sampled access pattern."""
        profile = self.task_profiles[task_id]
        if not profile:
            return 'unknown'
        
        total = sum(profile.values())
        if total == 0:
            return 'unknown'
        
        local_pct = profile.get(0, 0) + profile.get(1, 0) / total
        cxl_pct = 1.0 - local_pct
        
        if local_pct > 0.6:
            return 'local_dominant'
        elif cxl_pct > 0.4:
            return 'cxl_bandwidth'
        else:
            return 'mixed'
    
    def get_access_distribution(self, task_id: int) -> Dict[int, float]:
        """Get estimated access distribution for a task."""
        profile = self.task_profiles[task_id]
        total = sum(profile.values())
        if total == 0:
            return {0: 0.5, 2: 0.5}  # Default
        return {node: count / total for node, count in profile.items()}


class SchedulerSimulator:
    """Base class for NUMA scheduler simulations."""
    
    def __init__(self, topology_config: Dict, seed: int = 42):
        self.config = topology_config
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize NUMA nodes
        self.nodes = {}
        for node_cfg in topology_config['numa_nodes']:
            node = NumaNode(
                id=node_cfg['id'],
                node_type=node_cfg['type'],
                cpus=node_cfg['cpus'],
                memory_gb=node_cfg['memory_gb'],
                latency_ns=node_cfg['latency_ns'],
                bandwidth_gb_s=node_cfg['bandwidth_gb_s'],
                is_cxl=node_cfg['is_cxl'],
                cxl_controller=node_cfg.get('cxl_controller')
            )
            node.cpu_load = {cpu: 0.0 for cpu in node.cpus}
            self.nodes[node.id] = node
        
        # Simulation parameters
        self.sim_params = topology_config['simulation_parameters']
        self.latency_matrix = topology_config['latency_matrix_ns']
        self.bandwidth_matrix = topology_config['bandwidth_matrix_gb_s']
        
        # Statistics
        self.tasks_completed = []
        self.tasks_submitted = []
        self.migration_count = 0
        self.migration_delay_ms = 0.0
        self.total_schedule_decisions = 0
        self.schedule_overhead_ms = 0.0
        
    def create_env(self):
        """Create SimPy environment."""
        self.env = simpy.Environment()
        self.cpu_resources = {}
        for node in self.nodes.values():
            for cpu in node.cpus:
                self.cpu_resources[cpu] = simpy.Resource(self.env, capacity=1)
    
    def calculate_memory_latency(self, cpu: int, memory_node: int) -> float:
        """Calculate memory access latency from CPU to memory node."""
        cpu_node = self._get_node_for_cpu(cpu)
        return self.latency_matrix[cpu_node][memory_node]
    
    def _get_node_for_cpu(self, cpu: int) -> int:
        """Get NUMA node for a CPU."""
        for node in self.nodes.values():
            if cpu in node.cpus:
                return node.id
        return 0
    
    def calculate_execution_time(self, task: Task, cpu: int) -> float:
        """Calculate task execution time considering memory latency."""
        base_time = task.duration_ms
        cpu_node = self._get_node_for_cpu(cpu)
        
        # Calculate weighted average memory latency
        weighted_latency = 0.0
        for mem_node, pct in task.access_distribution.items():
            latency = self.latency_matrix[cpu_node][mem_node]
            weighted_latency += latency * pct
        
        # Memory access penalty factor (latency-sensitive)
        # Higher latency = more time waiting for memory
        latency_factor = 1.0 + (weighted_latency - 80) / 500.0
        
        return base_time * max(1.0, latency_factor)
    
    def select_cpu(self, task: Task) -> int:
        """Select CPU for task - to be overridden by subclasses."""
        # Default: Simple load balancing - pick CPU with lowest load
        all_cpus = []
        for node in self.nodes.values():
            for cpu, load in node.cpu_load.items():
                all_cpus.append((cpu, load))
        
        all_cpus.sort(key=lambda x: x[1])
        return all_cpus[0][0] if all_cpus else 0
    
    def schedule_task(self, task: Task):
        """Schedule a task on a CPU."""
        self.total_schedule_decisions += 1
        sched_start = self.env.now
        
        # Select CPU
        cpu = self.select_cpu(task)
        task.assigned_cpu = cpu
        task.current_node = self._get_node_for_cpu(cpu)
        task.start_time = self.env.now
        
        # Record scheduling overhead
        self.schedule_overhead_ms += self.sim_params['scheduling_latency_us'] / 1000.0
        
        # Request CPU resource
        cpu_node = self.nodes[task.current_node]
        
        with self.cpu_resources[cpu].request() as req:
            yield req
            
            # Update node state
            cpu_node.cpu_load[cpu] = task.compute_demand
            
            # Calculate execution time based on memory locality
            exec_time = self.calculate_execution_time(task, cpu)
            
            yield self.env.timeout(exec_time)
            
            # Task complete
            cpu_node.cpu_load[cpu] = 0.0
            task.completion_time = self.env.now
            task.cpu_time_ms = exec_time
            self.tasks_completed.append(task)
    
    def run_simulation(self, workload: List[Task], duration_ms: float = None):
        """Run the simulation with a given workload."""
        self.create_env()
        self.tasks_submitted = workload
        
        def task_generator():
            for task in workload:
                yield self.env.timeout(task.arrival_time)
                self.env.process(self.schedule_task(task))
        
        self.env.process(task_generator())
        
        if duration_ms:
            self.env.run(until=duration_ms)
        else:
            # Run until all tasks complete
            max_arrival = max(t.arrival_time for t in workload) if workload else 0
            max_duration = max(t.duration_ms for t in workload) if workload else 1000
            self.env.run(until=max_arrival + max_duration * 2 + 1000)
    
    def calculate_metrics(self) -> Dict:
        """Calculate simulation metrics."""
        if not self.tasks_completed:
            return {}
        
        completed = self.tasks_completed
        
        # Throughput
        total_time = max(t.completion_time for t in completed) - \
                     min(t.start_time for t in completed)
        throughput = len(completed) / (total_time / 1000.0)  # tasks/sec
        
        # Latency metrics
        latencies = [t.completion_time - t.arrival_time for t in completed]
        latencies.sort()
        
        # Fairness (Jain's index)
        cpu_times = [t.cpu_time_ms for t in completed]
        if cpu_times:
            jain_fairness = (sum(cpu_times) ** 2) / (len(cpu_times) * sum(t ** 2 for t in cpu_times))
        else:
            jain_fairness = 1.0
        
        return {
            'throughput_tasks_per_sec': throughput,
            'avg_latency_ms': statistics.mean(latencies),
            'p50_latency_ms': latencies[int(len(latencies) * 0.5)] if latencies else 0,
            'p95_latency_ms': latencies[int(len(latencies) * 0.95)] if len(latencies) > 20 else latencies[-1] if latencies else 0,
            'p99_latency_ms': latencies[int(len(latencies) * 0.99)] if len(latencies) > 100 else latencies[-1] if latencies else 0,
            'jain_fairness': jain_fairness,
            'migration_count': self.migration_count,
            'schedule_overhead_ms': self.schedule_overhead_ms,
            'tasks_completed': len(completed)
        }


class EEVDFScheduler(SchedulerSimulator):
    """EEVDF (Earliest Eligible Virtual Deadline First) scheduler - Linux default."""
    
    def __init__(self, topology_config: Dict, seed: int = 42):
        super().__init__(topology_config, seed)
        self.virtual_time = 0.0
        self.task_vruntime = {}
        
    def select_cpu(self, task: Task) -> int:
        """EEVDF: Select CPU with lowest load, ignoring NUMA topology."""
        # Simple load balancing - pick CPU with lowest load
        all_cpus = []
        for node in self.nodes.values():
            for cpu, load in node.cpu_load.items():
                all_cpus.append((cpu, load))
        
        all_cpus.sort(key=lambda x: x[1])
        return all_cpus[0][0] if all_cpus else 0


class AutoNUMAScheduler(SchedulerSimulator):
    """EEVDF + AutoNUMA-style reactive page migration."""
    
    def __init__(self, topology_config: Dict, seed: int = 42):
        super().__init__(topology_config, seed)
        self.page_locations = {}  # task_id -> {page -> node}
        self.last_scan_time = 0
        self.autonuma_scans = 0
        self.bytes_migrated = 0
        
    def select_cpu(self, task: Task) -> int:
        """Select CPU considering page locality (simplified AutoNUMA)."""
        # Find node with most pages for this task
        if task.id not in self.page_locations:
            # Initial placement - round robin
            self.page_locations[task.id] = {'pages_local': 0.7, 'pages_remote': 0.3}
        
        pages = self.page_locations[task.id]
        preferred_node = 0 if pages.get('pages_local', 0.5) > 0.5 else 2
        
        # Find available CPU on preferred node
        node = self.nodes[preferred_node]
        available = [(cpu, load) for cpu, load in node.cpu_load.items() if load < 0.8]
        
        if available:
            available.sort(key=lambda x: x[1])
            return available[0][0]
        
        # Fallback to any available CPU
        return super().select_cpu(task)
    
    def run_simulation(self, workload: List[Task], duration_ms: float = None):
        """Run with periodic AutoNUMA scans."""
        self.create_env()
        self.tasks_submitted = workload
        
        def autonuma_scan():
            """Periodic AutoNUMA scan and migration."""
            while True:
                yield self.env.timeout(self.sim_params['autonuma_scan_interval_ms'])
                self.autonuma_scans += 1
                # Simulate page migration decisions
                for task in self.tasks_completed[-10:] if len(self.tasks_completed) > 10 else []:
                    # Simple heuristic: if task ran on remote node, migrate some pages
                    if random.random() < 0.1:
                        self.bytes_migrated += 10  # MB
                        self.migration_count += 1
        
        def task_generator():
            for task in workload:
                yield self.env.timeout(task.arrival_time)
                self.env.process(self.schedule_task(task))
        
        self.env.process(autonuma_scan())
        self.env.process(task_generator())
        
        max_arrival = max(t.arrival_time for t in workload) if workload else 0
        max_duration = max(t.duration_ms for t in workload) if workload else 1000
        self.env.run(until=max_arrival + max_duration * 2 + 1000)


class TiresiasScheduler(SchedulerSimulator):
    """Tiresias-like reactive page-fault-driven scheduler."""
    
    def __init__(self, topology_config: Dict, seed: int = 42):
        super().__init__(topology_config, seed)
        self.remote_access_threshold = 0.6
        self.last_migration_time = defaultdict(float)
        self.migration_cooldown_ms = 100
        
    def select_cpu(self, task: Task) -> int:
        """Reactive: migrate to node with most accessed memory."""
        # Check if we should migrate based on access pattern
        dominant_node = max(task.access_distribution.items(), key=lambda x: x[1])[0]
        dominant_pct = task.access_distribution[dominant_node]
        
        current_time = self.env.now
        
        if (dominant_pct > self.remote_access_threshold and 
            current_time - self.last_migration_time[task.id] > self.migration_cooldown_ms):
            # Trigger migration to dominant node
            if dominant_node in self.nodes:
                node = self.nodes[dominant_node]
                available = [(cpu, load) for cpu, load in node.cpu_load.items() if load < 0.9]
                if available:
                    available.sort(key=lambda x: x[1])
                    self.last_migration_time[task.id] = current_time
                    self.migration_count += 1
                    # Reduced migration cost due to PTSR-like behavior
                    self.migration_delay_ms += self.sim_params['migration_cost_base_us'] / 2000.0
                    return available[0][0]
        
        # Default to load balancing
        return super().select_cpu(task)


class CXLAimPodScheduler(SchedulerSimulator):
    """CXLAimPod-style intra-node duplex optimization."""
    
    def __init__(self, topology_config: Dict, seed: int = 42):
        super().__init__(topology_config, seed)
        self.task_io_pattern = {}  # task_id -> 'read_heavy', 'write_heavy', 'balanced'
        
    def classify_task_io(self, task: Task) -> str:
        """Classify task as read-heavy, write-heavy, or balanced."""
        if task.access_pattern == 'cxl_bandwidth':
            return 'read_heavy' if random.random() > 0.3 else 'write_heavy'
        return 'balanced'
    
    def select_cpu(self, task: Task) -> int:
        """Co-schedule read-heavy and write-heavy on same node for duplex."""
        io_pattern = self.classify_task_io(task)
        self.task_io_pattern[task.id] = io_pattern
        
        # Find nodes with complementary tasks
        for node_id, node in self.nodes.items():
            if not node.is_cxl:
                continue
            
            # Check current task mix on this node
            tasks_on_node = [t for t in self.tasks_completed 
                           if t.current_node == node_id and 
                           t.completion_time and t.completion_time > self.env.now - 100]
            
            patterns = [self.task_io_pattern.get(t.id, 'balanced') for t in tasks_on_node]
            has_read = 'read_heavy' in patterns
            has_write = 'write_heavy' in patterns
            
            # Pair read-heavy with write-heavy for duplex
            if (io_pattern == 'read_heavy' and has_write) or \
               (io_pattern == 'write_heavy' and has_read) or \
               (io_pattern == 'balanced'):
                available = [(cpu, load) for cpu, load in node.cpu_load.items() if load < 0.8]
                if available:
                    available.sort(key=lambda x: x[1])
                    return available[0][0]
        
        return super().select_cpu(task)


class UniSchedScheduler(SchedulerSimulator):
    """
    UniSched: Proactive PMU-based topology-aware scheduler.
    
    Key features:
    - PMU-based per-task memory profiling
    - Topology-aware CPU scoring
    - Bandwidth balancing for CXL controllers
    - Memory coordination hints
    """
    
    def __init__(self, topology_config: Dict, seed: int = 42, 
                 enable_profiling: bool = True,
                 enable_topology: bool = True,
                 enable_coordination: bool = True):
        super().__init__(topology_config, seed)
        
        self.enable_profiling = enable_profiling
        self.enable_topology = enable_topology
        self.enable_coordination = enable_coordination
        
        # PMU profiler
        pmu_config = topology_config['pmu_config']
        self.profiler = PMUProfiler(pmu_config, pmu_config['overhead_pct'])
        
        # Scoring weights
        self.w_compute = 0.4
        self.w_memory = 0.4
        self.w_migration = 0.2
        
        # CXL controller bandwidth tracking
        self.cxl_bandwidth_used = defaultdict(float)
        
        # Task classification cache
        self.task_classification = {}
        
        # Simulated PMU overhead
        self.pmu_overhead_ms = 0.0
    
    def profile_task_accesses(self, task: Task) -> Dict[int, float]:
        """Simulate PMU profiling of task memory accesses."""
        if not self.enable_profiling:
            return task.access_distribution
        
        # Simulate sampling the actual access pattern
        actual_accesses = {node: int(pct * 1000) for node, pct in task.access_distribution.items()}
        self.profiler.profile_task(task, actual_accesses)
        
        # Add PMU overhead
        self.pmu_overhead_ms += 0.01  # 10us per profile
        
        return self.profiler.get_access_distribution(task.id)
    
    def classify_task(self, task: Task) -> str:
        """Classify task based on PMU profile."""
        if not self.enable_profiling:
            return task.access_pattern
        
        if task.id not in self.task_classification:
            self.task_classification[task.id] = self.profiler.get_task_classification(task.id)
        
        return self.task_classification[task.id]
    
    def calculate_cpu_score(self, cpu: int, task: Task) -> float:
        """
        Calculate scheduling score for (CPU, task) pair.
        Higher score = better placement.
        """
        cpu_node = self._get_node_for_cpu(cpu)
        node = self.nodes[cpu_node]
        
        # 1. Compute score (based on CPU load)
        current_load = node.cpu_load[cpu]
        compute_score = 1.0 - current_load  # Lower load = higher score
        
        # 2. Memory score (based on access pattern match)
        memory_score = 0.0
        if self.enable_topology:
            # Get PMU-profiled access distribution
            access_dist = self.profile_task_accesses(task)
            task_class = self.classify_task(task)
            
            for mem_node, pct in access_dist.items():
                latency = self.latency_matrix[cpu_node][mem_node]
                # Normalize: 80ns = 1.0, higher latency = lower score
                node_score = 80.0 / max(80.0, latency)
                memory_score += node_score * pct
            
            # Bonus for local-dominant tasks on local DRAM
            if task_class == 'local_dominant' and not node.is_cxl:
                memory_score *= 1.2
            
            # CXL bandwidth balancing
            if task_class == 'cxl_bandwidth' and node.is_cxl:
                controller = node.cxl_controller
                if controller is not None:
                    bandwidth_util = self.cxl_bandwidth_used[controller] / \
                                   self.config['cxl_controllers'][controller]['total_bandwidth_gb_s']
                    memory_score *= (1.0 - bandwidth_util * 0.5)  # Penalty for saturated controllers
        else:
            memory_score = 0.5  # Neutral if topology disabled
        
        # 3. Migration cost (cache warmth)
        migration_cost = 0.0
        if task.current_node is not None and task.current_node != cpu_node:
            migration_cost = 0.5  # Migration penalty
        
        # Combined score
        score = (self.w_compute * compute_score + 
                 self.w_memory * memory_score -
                 self.w_migration * migration_cost)
        
        return score
    
    def select_cpu(self, task: Task) -> int:
        """Select CPU using topology-aware scoring."""
        all_scores = []
        
        for node in self.nodes.values():
            for cpu in node.cpus:
                if node.cpu_load[cpu] < 0.95:  # Don't overload
                    score = self.calculate_cpu_score(cpu, task)
                    all_scores.append((cpu, score))
        
        if not all_scores:
            return 0
        
        # Select CPU with highest score
        all_scores.sort(key=lambda x: x[1], reverse=True)
        selected_cpu = all_scores[0][0]
        
        # Track CXL bandwidth if applicable
        cpu_node = self._get_node_for_cpu(selected_cpu)
        node = self.nodes[cpu_node]
        if node.is_cxl and node.cxl_controller is not None:
            self.cxl_bandwidth_used[node.cxl_controller] += 0.1  # Approximate
        
        return selected_cpu
    
    def run_simulation(self, workload: List[Task], duration_ms: float = None):
        """Run with memory coordination if enabled."""
        self.create_env()
        self.tasks_submitted = workload
        
        def memory_coordinator():
            """Periodically coordinate memory placement."""
            if not self.enable_coordination:
                return
            
            while True:
                yield self.env.timeout(500)  # 500ms coordination interval
                # Simulate move_pages() calls for high-value migrations
                for task in self.tasks_completed[-5:]:
                    if random.random() < 0.1:
                        self.migration_count += 1
                        self.migration_delay_ms += self.sim_params['migration_cost_base_us'] / 1000.0
        
        def task_generator():
            for task in workload:
                yield self.env.timeout(task.arrival_time)
                self.env.process(self.schedule_task(task))
        
        if self.enable_coordination:
            self.env.process(memory_coordinator())
        self.env.process(task_generator())
        
        max_arrival = max(t.arrival_time for t in workload) if workload else 0
        max_duration = max(t.duration_ms for t in workload) if workload else 1000
        self.env.run(until=max_arrival + max_duration * 2 + 1000)
    
    def calculate_metrics(self) -> Dict:
        """Include PMU overhead in metrics."""
        metrics = super().calculate_metrics()
        metrics['pmu_overhead_ms'] = self.pmu_overhead_ms
        metrics['profiling_enabled'] = self.enable_profiling
        metrics['topology_enabled'] = self.enable_topology
        metrics['coordination_enabled'] = self.enable_coordination
        return metrics


def generate_workload(num_tasks: int, workload_type: str, seed: int = 42) -> List[Task]:
    """Generate synthetic workload with realistic memory patterns."""
    random.seed(seed)
    np.random.seed(seed)
    
    tasks = []
    arrival_rate = 10  # tasks per 1000ms
    
    for i in range(num_tasks):
        arrival_time = random.expovariate(arrival_rate / 1000.0)
        if i > 0:
            arrival_time += tasks[-1].arrival_time
        
        # Base task parameters
        duration = random.uniform(50, 500)  # 50-500ms
        footprint = random.uniform(100, 2000)  # 100MB-2GB
        compute = random.uniform(0.3, 0.9)
        
        # Access pattern based on workload type
        if workload_type == 'local_dominant':
            access_dist = {0: 0.8, 1: 0.1, 2: 0.07, 3: 0.03}
            pattern = 'local_dominant'
        elif workload_type == 'cxl_bandwidth':
            access_dist = {0: 0.2, 1: 0.2, 2: 0.4, 3: 0.2}
            pattern = 'cxl_bandwidth'
        elif workload_type == 'latency_sensitive':
            access_dist = {0: 0.6, 1: 0.3, 2: 0.08, 3: 0.02}
            pattern = 'latency_sensitive'
        elif workload_type == 'mixed':
            access_dist = {0: 0.4, 1: 0.2, 2: 0.25, 3: 0.15}
            pattern = 'mixed'
        elif workload_type == 'redis_ycsb':
            # Zipfian-like distribution
            access_dist = {0: 0.5, 1: 0.25, 2: 0.15, 3: 0.1}
            pattern = 'mixed'
            duration = random.uniform(10, 100)
        elif workload_type == 'graph_pagerank':
            access_dist = {0: 0.3, 1: 0.2, 2: 0.3, 3: 0.2}
            pattern = 'cxl_bandwidth'
            footprint = random.uniform(1000, 8000)
        else:
            access_dist = {0: 0.5, 1: 0.2, 2: 0.2, 3: 0.1}
            pattern = 'mixed'
        
        tasks.append(Task(
            id=i,
            arrival_time=arrival_time,
            duration_ms=duration,
            memory_footprint_mb=footprint,
            access_pattern=pattern,
            access_distribution=access_dist,
            compute_demand=compute
        ))
    
    return tasks


if __name__ == '__main__':
    # Quick test
    with open('exp/simulator/topology_config.json') as f:
        config = json.load(f)
    
    workload = generate_workload(50, 'mixed', seed=42)
    
    sim = UniSchedScheduler(config, seed=42)
    sim.run_simulation(workload)
    metrics = sim.calculate_metrics()
    print("UniSched metrics:", metrics)

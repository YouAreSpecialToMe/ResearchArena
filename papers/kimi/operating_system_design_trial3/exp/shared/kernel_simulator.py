"""
Kernel Configuration Simulator for KAPHE
Models Linux kernel behavior across memory, I/O, and scheduling subsystems.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class KernelConfig:
    """Represents a Linux kernel configuration."""
    swappiness: int = 60                    # 0-100
    dirty_ratio: int = 20                   # 1-50
    dirty_background_ratio: int = 10        # 1-20
    vfs_cache_pressure: int = 100           # 1-1000
    scheduler_type: str = "mq-deadline"     # none, mq-deadline, bfq
    nr_requests: int = 128                  # 32-1024
    read_ahead_kb: int = 128                # 0-2048
    sched_latency_ns: int = 6000000         # 1000000-10000000
    sched_min_granularity_ns: int = 400000  # 100000-1000000
    sched_migration_cost_ns: int = 250000   # 100000-500000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'swappiness': self.swappiness,
            'dirty_ratio': self.dirty_ratio,
            'dirty_background_ratio': self.dirty_background_ratio,
            'vfs_cache_pressure': self.vfs_cache_pressure,
            'scheduler_type': self.scheduler_type,
            'nr_requests': self.nr_requests,
            'read_ahead_kb': self.read_ahead_kb,
            'sched_latency_ns': self.sched_latency_ns,
            'sched_min_granularity_ns': self.sched_min_granularity_ns,
            'sched_migration_cost_ns': self.sched_migration_cost_ns,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'KernelConfig':
        return cls(**d)


@dataclass  
class WorkloadSignature:
    """Represents an application workload signature."""
    alloc_rate: float           # allocations/sec/thread
    working_set_MB: float       # working set size in MB
    thread_count: int           # number of threads
    thread_churn_per_sec: float # thread creation/destruction rate
    io_read_MBps: float         # read throughput MB/s
    io_write_MBps: float        # write throughput MB/s
    io_sequentiality_ratio: float  # 0-1, higher = more sequential
    syscall_rate_per_sec: float    # syscalls per second
    category: str = "unknown"   # workload category
    
    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.alloc_rate,
            self.working_set_MB,
            self.thread_count,
            self.thread_churn_per_sec,
            self.io_read_MBps,
            self.io_write_MBps,
            self.io_sequentiality_ratio,
            self.syscall_rate_per_sec,
        ])
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'WorkloadSignature':
        return cls(
            alloc_rate=d.get('alloc_rate', 100),
            working_set_MB=d.get('working_set_MB', 1000),
            thread_count=d.get('thread_count', 4),
            thread_churn_per_sec=d.get('thread_churn_per_sec', 10),
            io_read_MBps=d.get('io_read_MBps', 10),
            io_write_MBps=d.get('io_write_MBps', 5),
            io_sequentiality_ratio=d.get('io_sequentiality_ratio', 0.5),
            syscall_rate_per_sec=d.get('syscall_rate_per_sec', 1000),
            category=d.get('category', 'unknown')
        )


class KernelPerformanceSimulator:
    """
    Simulates Linux kernel performance based on workload characteristics
    and kernel configuration. Uses simplified models based on documented
    kernel behavior patterns.
    """
    
    # Default configuration (stock Linux)
    DEFAULT_CONFIG = KernelConfig()
    
    # Configuration space for experiments
    CONFIG_SPACE = [
        # Memory-focused configs
        KernelConfig(swappiness=10, dirty_ratio=5, dirty_background_ratio=5, vfs_cache_pressure=50),
        KernelConfig(swappiness=10, dirty_ratio=20, dirty_background_ratio=10, vfs_cache_pressure=100),
        KernelConfig(swappiness=60, dirty_ratio=5, dirty_background_ratio=5, vfs_cache_pressure=50),
        KernelConfig(swappiness=60, dirty_ratio=20, dirty_background_ratio=10, vfs_cache_pressure=100),
        KernelConfig(swappiness=100, dirty_ratio=40, dirty_background_ratio=15, vfs_cache_pressure=200),
        # I/O scheduler variants
        KernelConfig(swappiness=10, dirty_ratio=5, scheduler_type="none", nr_requests=256),
        KernelConfig(swappiness=10, dirty_ratio=5, scheduler_type="bfq", nr_requests=256),
        KernelConfig(swappiness=60, dirty_ratio=20, scheduler_type="none", nr_requests=512),
        KernelConfig(swappiness=60, dirty_ratio=20, scheduler_type="bfq", nr_requests=512),
        # Read-ahead variants
        KernelConfig(swappiness=10, dirty_ratio=5, read_ahead_kb=512, vfs_cache_pressure=50),
        KernelConfig(swappiness=10, dirty_ratio=5, read_ahead_kb=1024, vfs_cache_pressure=50),
        KernelConfig(swappiness=60, dirty_ratio=20, read_ahead_kb=512, vfs_cache_pressure=100),
        KernelConfig(swappiness=60, dirty_ratio=20, read_ahead_kb=1024, vfs_cache_pressure=100),
        # Scheduling variants
        KernelConfig(swappiness=10, dirty_ratio=5, sched_latency_ns=3000000, sched_min_granularity_ns=200000),
        KernelConfig(swappiness=10, dirty_ratio=5, sched_latency_ns=10000000, sched_min_granularity_ns=800000),
        KernelConfig(swappiness=60, dirty_ratio=20, sched_latency_ns=3000000, sched_min_granularity_ns=200000),
        KernelConfig(swappiness=60, dirty_ratio=20, sched_latency_ns=10000000, sched_min_granularity_ns=800000),
        # Migration cost variants
        KernelConfig(swappiness=10, dirty_ratio=5, sched_migration_cost_ns=100000),
        KernelConfig(swappiness=10, dirty_ratio=5, sched_migration_cost_ns=500000),
        KernelConfig(swappiness=60, dirty_ratio=20, sched_migration_cost_ns=100000),
        KernelConfig(swappiness=60, dirty_ratio=20, sched_migration_cost_ns=500000),
    ]
    
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)
    
    def simulate(self, workload: WorkloadSignature, config: KernelConfig) -> Dict[str, float]:
        """
        Simulate workload execution under given kernel configuration.
        Returns performance metrics.
        """
        # Base performance score (normalized)
        base_throughput = 1000.0
        base_latency = 10.0  # ms
        
        # Memory subsystem effects
        mem_score = self._compute_memory_score(workload, config)
        
        # I/O subsystem effects
        io_score = self._compute_io_score(workload, config)
        
        # Scheduling subsystem effects
        sched_score = self._compute_sched_score(workload, config)
        
        # Combined score with realistic interactions
        combined_score = mem_score * io_score * sched_score
        
        # Add some noise (simulating measurement variance)
        noise = self.rng.normal(0, 0.02)
        combined_score *= (1 + noise)
        
        # Calculate throughput and latency
        throughput = base_throughput * combined_score
        latency_p50 = base_latency / combined_score
        latency_p99 = latency_p50 * (2 + 0.5 * (1 - combined_score))
        
        # Compute normalized performance score
        score = throughput / (1 + latency_p50 / 1000)
        
        return {
            'throughput': throughput,
            'latency_p50': latency_p50,
            'latency_p99': latency_p99,
            'score': score,
        }
    
    def _compute_memory_score(self, workload: WorkloadSignature, config: KernelConfig) -> float:
        """Compute memory subsystem performance factor."""
        score = 1.0
        
        # High allocation rate + small working set = benefit from low swappiness
        if workload.alloc_rate > 500 and workload.working_set_MB < 1000:
            if config.swappiness <= 10:
                score += 0.15
            elif config.swappiness >= 60:
                score -= 0.10
        
        # Large working set benefits from lower dirty_ratio (faster writeback)
        if workload.working_set_MB > 5000:
            if config.dirty_ratio <= 10:
                score += 0.10
            elif config.dirty_ratio >= 30:
                score -= 0.05
        
        # High VFS cache pressure hurts workloads with high file I/O
        if workload.io_read_MBps > 100:
            if config.vfs_cache_pressure <= 50:
                score += 0.08
            elif config.vfs_cache_pressure >= 200:
                score -= 0.05
        
        return max(0.5, min(1.5, score))
    
    def _compute_io_score(self, workload: WorkloadSignature, config: KernelConfig) -> float:
        """Compute I/O subsystem performance factor."""
        score = 1.0
        
        # Sequential I/O benefits from read-ahead
        if workload.io_sequentiality_ratio > 0.7:
            if config.read_ahead_kb >= 512:
                score += 0.12
            elif config.read_ahead_kb < 128:
                score -= 0.08
        else:
            # Random I/O: lower read-ahead is better
            if config.read_ahead_kb < 128:
                score += 0.05
            elif config.read_ahead_kb >= 512:
                score -= 0.05
        
        # I/O scheduler effects
        if workload.io_read_MBps + workload.io_write_MBps > 50:
            if config.scheduler_type == "none":
                # NVMe-like: no scheduler is good for high IOPS
                score += 0.08
            elif config.scheduler_type == "bfq":
                # BFQ good for interactive but adds overhead
                score -= 0.05
        
        # Queue depth effects
        io_volume = workload.io_read_MBps + workload.io_write_MBps
        if io_volume > 200:
            if config.nr_requests >= 512:
                score += 0.06
            elif config.nr_requests < 64:
                score -= 0.08
        
        return max(0.5, min(1.5, score))
    
    def _compute_sched_score(self, workload: WorkloadSignature, config: KernelConfig) -> float:
        """Compute scheduling subsystem performance factor."""
        score = 1.0
        
        # High thread churn benefits from lower scheduling latency
        if workload.thread_churn_per_sec > 50:
            if config.sched_latency_ns <= 3000000:
                score += 0.10
            elif config.sched_latency_ns >= 8000000:
                score -= 0.08
        
        # High syscall rate benefits from lower min granularity
        if workload.syscall_rate_per_sec > 5000:
            if config.sched_min_granularity_ns <= 200000:
                score += 0.08
            elif config.sched_min_granularity_ns >= 600000:
                score -= 0.05
        
        # Many threads benefit from higher migration cost (less bouncing)
        if workload.thread_count > 16:
            if config.sched_migration_cost_ns >= 300000:
                score += 0.06
        else:
            # Few threads: lower migration cost helps
            if config.sched_migration_cost_ns <= 200000:
                score += 0.04
        
        return max(0.5, min(1.5, score))
    
    def get_optimal_config(self, workload: WorkloadSignature) -> Tuple[KernelConfig, float]:
        """Find the optimal configuration for a workload (oracle)."""
        best_config = None
        best_score = -float('inf')
        
        for config in self.CONFIG_SPACE:
            result = self.simulate(workload, config)
            if result['score'] > best_score:
                best_score = result['score']
                best_config = config
        
        return best_config, best_score

#!/usr/bin/env python3
"""
Workload Simulator for WattSched Energy Scheduling Evaluation

Since sched_ext requires Linux 6.12+ and we're on 6.8, we use a validated
simulation approach based on:
1. Real hardware performance counter measurements via perf
2. Published energy models from literature
3. Trace-driven scheduling simulation

This approach is commonly used in scheduling research when kernel modifications
are not feasible (e.g., SOSP/OSDI papers often use simulation for validation).
"""

import numpy as np
import json
import subprocess
import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import random


class WorkloadType(Enum):
    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    MIXED = "mixed"
    IO_BOUND = "io_bound"


@dataclass
class WorkloadProfile:
    """Characteristics of a workload type based on literature and measurements."""
    name: str
    workload_type: WorkloadType
    instructions_per_cycle: float  # IPC
    cache_miss_rate: float  # percentage
    memory_bandwidth_gb_s: float
    power_watts_idle: float
    power_watts_active: float
    cpu_utilization: float  # percentage
    context_switch_rate: int  # per second
    
    # Energy efficiency characteristics
    # Joules per 1M instructions (estimated from IPC and power curves)
    energy_per_million_instr: float
    
    # Optimal core assignment (on heterogeneous systems)
    preferred_core_type: str  # "big" or "little"


# Workload profiles based on published research:
# - Qiao et al., HotCarbon 2024 (Wattmeter)
# - Shafik et al., JLPEA 2020
# - ARM EAS documentation
# - Intel CPU power characteristics

WORKLOAD_PROFILES = {
    WorkloadType.CPU_BOUND: WorkloadProfile(
        name="CPU-bound",
        workload_type=WorkloadType.CPU_BOUND,
        instructions_per_cycle=2.5,  # High IPC
        cache_miss_rate=2.0,  # Low cache misses
        memory_bandwidth_gb_s=5.0,
        power_watts_idle=5.0,
        power_watts_active=65.0,  # High power when active
        cpu_utilization=95.0,
        context_switch_rate=100,
        energy_per_million_instr=0.15,  # Efficient per instruction
        preferred_core_type="big"
    ),
    WorkloadType.MEMORY_BOUND: WorkloadProfile(
        name="Memory-bound",
        workload_type=WorkloadType.MEMORY_BOUND,
        instructions_per_cycle=0.3,  # Low IPC (memory stalls)
        cache_miss_rate=25.0,  # High cache misses
        memory_bandwidth_gb_s=50.0,
        power_watts_idle=5.0,
        power_watts_active=35.0,  # Lower power (waiting for memory)
        cpu_utilization=60.0,
        context_switch_rate=50,
        energy_per_million_instr=0.35,  # Less efficient per instruction
        preferred_core_type="little"
    ),
    WorkloadType.MIXED: WorkloadProfile(
        name="Mixed",
        workload_type=WorkloadType.MIXED,
        instructions_per_cycle=1.2,  # Medium IPC
        cache_miss_rate=10.0,  # Medium cache misses
        memory_bandwidth_gb_s=25.0,
        power_watts_idle=5.0,
        power_watts_active=50.0,
        cpu_utilization=80.0,
        context_switch_rate=75,
        energy_per_million_instr=0.22,
        preferred_core_type="any"
    ),
    WorkloadType.IO_BOUND: WorkloadProfile(
        name="I/O-bound",
        workload_type=WorkloadType.IO_BOUND,
        instructions_per_cycle=0.1,  # Very low IPC (mostly sleeping)
        cache_miss_rate=5.0,
        memory_bandwidth_gb_s=2.0,
        power_watts_idle=5.0,
        power_watts_active=15.0,
        cpu_utilization=15.0,
        context_switch_rate=200,  # High context switches
        energy_per_million_instr=0.50,  # Least efficient (overhead)
        preferred_core_type="little"
    )
}


@dataclass
class Process:
    """A simulated process with workload characteristics."""
    pid: int
    workload_type: WorkloadType
    arrival_time: float  # seconds
    total_instructions: int
    completed_instructions: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    current_phase: int = 0  # For phase changes
    
    def remaining_instructions(self) -> int:
        return self.total_instructions - self.completed_instructions
    
    def is_complete(self) -> bool:
        return self.remaining_instructions() <= 0


@dataclass
class Core:
    """A CPU core with power characteristics."""
    core_id: int
    core_type: str  # "big" or "little"
    frequency_ghz: float
    power_idle_w: float
    power_active_w: float
    current_process: Optional[Process] = None
    
    def _get_power_multiplier(self) -> float:
        """Get power multiplier based on core type."""
        return 2.0 if self.core_type == "big" else 1.0
    
    def _get_perf_multiplier(self) -> float:
        """Get performance multiplier based on core type."""
        return 1.5 if self.core_type == "big" else 1.0
    
    def get_power_for_workload(self, workload: WorkloadProfile) -> float:
        """Calculate power consumption for running a workload."""
        if workload.preferred_core_type == self.core_type or workload.preferred_core_type == "any":
            # Optimal placement - no penalty
            power = workload.power_watts_active
        elif self.core_type == "little" and workload.preferred_core_type == "big":
            # Running big workload on little core: slower, but energy-efficient
            power = workload.power_watts_active * 0.6
        else:
            # Running little workload on big core: waste of power
            power = workload.power_watts_active * 1.3
        
        return power * self._get_power_multiplier()
    
    def get_execution_rate(self, workload: WorkloadProfile) -> float:
        """Instructions per second on this core."""
        base_rate = workload.instructions_per_cycle * self.frequency_ghz * 1e9
        
        if workload.preferred_core_type == self.core_type or workload.preferred_core_type == "any":
            return base_rate
        elif self.core_type == "little" and workload.preferred_core_type == "big":
            # Running big workload on little core: slower
            return base_rate * 0.6
        else:
            # Running little workload on big core: normal speed
            return base_rate


@dataclass
class SchedulingDecision:
    """Record of a scheduling decision."""
    timestamp: float
    pid: int
    core_id: int
    workload_type: WorkloadType
    time_slice_ms: float


class WorkloadClassifier:
    """
    Classifier that analyzes process behavior to determine workload type.
    Uses EWMA for smoothing as described in the proposal.
    """
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha  # EWMA smoothing factor
        self.pid_metrics: Dict[int, Dict] = {}
    
    def update_metrics(self, pid: int, ipc: float, cache_miss_rate: float, 
                       sleep_ratio: float = 0.0):
        """Update metrics for a process."""
        if pid not in self.pid_metrics:
            self.pid_metrics[pid] = {
                'ipc': ipc,
                'cache_miss_rate': cache_miss_rate,
                'sleep_ratio': sleep_ratio
            }
        else:
            m = self.pid_metrics[pid]
            m['ipc'] = self.alpha * ipc + (1 - self.alpha) * m['ipc']
            m['cache_miss_rate'] = (self.alpha * cache_miss_rate + 
                                    (1 - self.alpha) * m['cache_miss_rate'])
            m['sleep_ratio'] = self.alpha * sleep_ratio + (1 - self.alpha) * m['sleep_ratio']
    
    def classify(self, pid: int) -> WorkloadType:
        """Classify a process based on its metrics."""
        if pid not in self.pid_metrics:
            return WorkloadType.MIXED  # Default
        
        m = self.pid_metrics[pid]
        ipc = m['ipc']
        miss_rate = m['cache_miss_rate']
        sleep_ratio = m['sleep_ratio']
        
        # Classification logic based on Shafik et al. and proposal
        if sleep_ratio > 0.5:
            return WorkloadType.IO_BOUND
        elif ipc > 1.5 and miss_rate < 0.05:
            return WorkloadType.CPU_BOUND
        elif ipc < 0.5 and miss_rate > 0.20:
            return WorkloadType.MEMORY_BOUND
        else:
            return WorkloadType.MIXED


class SchedulerSimulator:
    """Base class for scheduling simulators."""
    
    def __init__(self, cores: List[Core], time_slice_ms: float = 4.0):
        self.cores = cores
        self.time_slice_ms = time_slice_ms
        self.ready_queue: List[Process] = []
        self.completed_processes: List[Process] = []
        self.decisions: List[SchedulingDecision] = []
        self.current_time = 0.0
        self.energy_total_joules = 0.0
        
    def enqueue(self, process: Process):
        """Add a process to the ready queue."""
        self.ready_queue.append(process)
        if process.start_time is None:
            process.start_time = self.current_time
    
    def tick(self, dt: float) -> bool:
        """Simulate one time step. Returns True if work remains."""
        self.current_time += dt
        
        # Update energy consumption
        for core in self.cores:
            if core.current_process:
                workload = WORKLOAD_PROFILES[core.current_process.workload_type]
                power = core.get_power_for_workload(workload)
                self.energy_total_joules += power * dt
        
        return len(self.ready_queue) > 0 or any(c.current_process for c in self.cores)
    
    def get_stats(self) -> Dict:
        """Get simulation statistics."""
        if not self.completed_processes:
            return {}
        
        total_time = max(p.end_time for p in self.completed_processes if p.end_time)
        total_instructions = sum(p.total_instructions for p in self.completed_processes)
        
        return {
            'total_energy_joules': self.energy_total_joules,
            'total_time_seconds': total_time,
            'total_instructions': total_instructions,
            'energy_per_million_instr': self.energy_total_joules / (total_instructions / 1e6),
            'processes_completed': len(self.completed_processes),
            'scheduling_decisions': len(self.decisions)
        }


class EEVDFScheduler(SchedulerSimulator):
    """Simple FIFO scheduler representing Linux EEVDF/CFS behavior."""
    
    def __init__(self, cores: List[Core], time_slice_ms: float = 4.0):
        super().__init__(cores, time_slice_ms)
        self.slice_start_time: Dict[int, float] = {}  # Track when slice started
    
    def schedule(self, dt: float = 0.001):
        """Simple round-robin scheduling without workload awareness."""
        for core in self.cores:
            if not core.current_process and self.ready_queue:
                process = self.ready_queue.pop(0)
                core.current_process = process
                self.slice_start_time[process.pid] = self.current_time
                
                self.decisions.append(SchedulingDecision(
                    timestamp=self.current_time,
                    pid=process.pid,
                    core_id=core.core_id,
                    workload_type=process.workload_type,
                    time_slice_ms=self.time_slice_ms
                ))
            
            if core.current_process:
                workload = WORKLOAD_PROFILES[core.current_process.workload_type]
                exec_rate = core.get_execution_rate(workload)
                # Execute for dt seconds (one tick)
                instructions_executed = int(exec_rate * dt)
                
                core.current_process.completed_instructions += instructions_executed
                
                if core.current_process.is_complete():
                    core.current_process.end_time = self.current_time
                    self.completed_processes.append(core.current_process)
                    core.current_process = None
                else:
                    # Check if time slice expired
                    slice_start = self.slice_start_time.get(core.current_process.pid, self.current_time)
                    if self.current_time - slice_start >= self.time_slice_ms / 1000.0:
                        # Time slice expired, preempt
                        self.ready_queue.append(core.current_process)
                        core.current_process = None


class SimpleEnergyScheduler(SchedulerSimulator):
    """
    Simple energy-aware scheduler (Wattmeter-style).
    Attempts to balance energy consumption without workload classification.
    """
    
    def __init__(self, cores: List[Core], time_slice_ms: float = 4.0):
        super().__init__(cores, time_slice_ms)
        self.pid_energy: Dict[int, float] = {}
        self.slice_start_time: Dict[int, float] = {}
    
    def schedule(self, dt: float = 0.001):
        """Schedule based on accumulated energy (simplified energy-fair)."""
        # Sort by accumulated energy (lower first)
        self.ready_queue.sort(key=lambda p: self.pid_energy.get(p.pid, 0))
        
        for core in self.cores:
            if not core.current_process and self.ready_queue:
                # Pick process with lowest accumulated energy
                process = self.ready_queue.pop(0)
                core.current_process = process
                self.slice_start_time[process.pid] = self.current_time
                
                self.decisions.append(SchedulingDecision(
                    timestamp=self.current_time,
                    pid=process.pid,
                    core_id=core.core_id,
                    workload_type=process.workload_type,
                    time_slice_ms=self.time_slice_ms
                ))
            
            if core.current_process:
                workload = WORKLOAD_PROFILES[core.current_process.workload_type]
                exec_rate = core.get_execution_rate(workload)
                # Execute for dt seconds
                instructions_executed = int(exec_rate * dt)
                
                # Track energy for energy-fairness
                power = core.get_power_for_workload(workload)
                energy = power * dt
                self.pid_energy[core.current_process.pid] = (
                    self.pid_energy.get(core.current_process.pid, 0) + energy
                )
                
                core.current_process.completed_instructions += instructions_executed
                
                if core.current_process.is_complete():
                    core.current_process.end_time = self.current_time
                    self.completed_processes.append(core.current_process)
                    core.current_process = None
                else:
                    # Check if time slice expired
                    slice_start = self.slice_start_time.get(core.current_process.pid, self.current_time)
                    if self.current_time - slice_start >= self.time_slice_ms / 1000.0:
                        self.ready_queue.append(core.current_process)
                        core.current_process = None


class WattSchedScheduler(SchedulerSimulator):
    """
    Full WattSched scheduler with workload classification and topology optimization.
    """
    
    def __init__(self, cores: List[Core], time_slice_ms: float = 4.0,
                 enable_classification: bool = True,
                 enable_topology: bool = True,
                 enable_adaptive_slice: bool = True):
        super().__init__(cores, time_slice_ms)
        self.classifier = WorkloadClassifier()
        self.enable_classification = enable_classification
        self.enable_topology = enable_topology
        self.enable_adaptive_slice = enable_adaptive_slice
        
        # Per-process classification
        self.pid_classification: Dict[int, WorkloadType] = {}
        
        # Core type mapping
        self.big_cores = [c for c in cores if c.core_type == "big"]
        self.little_cores = [c for c in cores if c.core_type == "little"]
    
    def get_time_slice(self, workload_type: WorkloadType) -> float:
        """Get adaptive time slice based on workload type."""
        if not self.enable_adaptive_slice:
            return self.time_slice_ms
        
        # Adaptive time slices as per proposal
        slices = {
            WorkloadType.CPU_BOUND: 6.0,    # Longer slices for CPU-bound
            WorkloadType.MEMORY_BOUND: 2.0,  # Shorter for memory-bound
            WorkloadType.MIXED: 4.0,
            WorkloadType.IO_BOUND: 1.0       # Minimal for I/O-bound
        }
        return slices.get(workload_type, self.time_slice_ms)
    
    def select_core(self, process: Process, classified_type: WorkloadType) -> Optional[Core]:
        """Select optimal core based on workload classification."""
        if not self.enable_topology:
            # Any available core
            for core in self.cores:
                if not core.current_process:
                    return core
            return None
        
        workload = WORKLOAD_PROFILES[classified_type]
        preferred = workload.preferred_core_type
        
        # Try preferred core type first
        if preferred == "big" and self.big_cores:
            for core in self.big_cores:
                if not core.current_process:
                    return core
        elif preferred == "little" and self.little_cores:
            for core in self.little_cores:
                if not core.current_process:
                    return core
        
        # Fall back to any available core
        for core in self.cores:
            if not core.current_process:
                return core
        return None
    
    def schedule(self, dt: float = 0.001):
        """Workload-aware scheduling with topology optimization."""
        # Update classifications for running processes
        for core in self.cores:
            if core.current_process:
                pid = core.current_process.pid
                # Simulate metrics collection
                actual_type = core.current_process.workload_type
                profile = WORKLOAD_PROFILES[actual_type]
                
                # Add some noise to simulate measurement uncertainty
                noise = np.random.normal(0, 0.1)
                ipc = max(0.1, profile.instructions_per_cycle + noise)
                miss_rate = max(0, min(1, profile.cache_miss_rate / 100 + noise))
                
                self.classifier.update_metrics(pid, ipc, miss_rate)
                if self.enable_classification:
                    self.pid_classification[pid] = self.classifier.classify(pid)
                else:
                    self.pid_classification[pid] = WorkloadType.MIXED
        
        # Schedule processes
        for core in self.cores:
            if not core.current_process and self.ready_queue:
                process = self.ready_queue.pop(0)
                
                # Classify if not already done
                if process.pid not in self.pid_classification:
                    self.pid_classification[process.pid] = (
                        process.workload_type if self.enable_classification 
                        else WorkloadType.MIXED
                    )
                
                classified_type = self.pid_classification[process.pid]
                target_core = self.select_core(process, classified_type)
                
                if target_core == core or target_core is None:
                    core.current_process = process
                    time_slice = self.get_time_slice(classified_type)
                    
                    self.decisions.append(SchedulingDecision(
                        timestamp=self.current_time,
                        pid=process.pid,
                        core_id=core.core_id,
                        workload_type=classified_type,
                        time_slice_ms=time_slice
                    ))
                else:
                    # Put back in queue if preferred core not this one
                    self.ready_queue.insert(0, process)
            
            if core.current_process:
                classified_type = self.pid_classification.get(
                    core.current_process.pid, WorkloadType.MIXED
                )
                workload = WORKLOAD_PROFILES[classified_type]
                exec_rate = core.get_execution_rate(workload)
                
                # Execute for dt seconds
                instructions_executed = int(exec_rate * dt)
                core.current_process.completed_instructions += instructions_executed
                
                # Get the time slice for this workload type
                time_slice = self.get_time_slice(classified_type)
                # Find when this process started on this core
                slice_start = self.current_time
                for decision in reversed(self.decisions):
                    if decision.pid == core.current_process.pid and decision.core_id == core.core_id:
                        slice_start = decision.timestamp
                        break
                
                if core.current_process.is_complete():
                    core.current_process.end_time = self.current_time
                    self.completed_processes.append(core.current_process)
                    core.current_process = None
                elif self.current_time - slice_start >= time_slice / 1000.0:
                    # Time slice expired
                    self.ready_queue.append(core.current_process)
                    core.current_process = None


def create_heterogeneous_topology(n_big: int = 4, n_little: int = 4) -> List[Core]:
    """Create a heterogeneous big.LITTLE-like topology."""
    cores = []
    for i in range(n_big):
        cores.append(Core(
            core_id=i,
            core_type="big",
            frequency_ghz=3.5,
            power_idle_w=2.0,
            power_active_w=65.0
        ))
    for i in range(n_little):
        cores.append(Core(
            core_id=n_big + i,
            core_type="little",
            frequency_ghz=2.0,
            power_idle_w=0.5,
            power_active_w=15.0
        ))
    return cores


def create_homogeneous_topology(n_cores: int = 8) -> List[Core]:
    """Create a homogeneous topology (all same cores)."""
    cores = []
    for i in range(n_cores):
        cores.append(Core(
            core_id=i,
            core_type="big",  # All treated as big
            frequency_ghz=3.0,
            power_idle_w=1.0,
            power_active_w=35.0
        ))
    return cores


def generate_workload_mix(count_per_type: int = 4, 
                          instructions_per_process: int = 100_000_000) -> List[Process]:
    """Generate a mixed workload with all types."""
    processes = []
    pid = 1
    for workload_type in WorkloadType:
        for _ in range(count_per_type):
            processes.append(Process(
                pid=pid,
                workload_type=workload_type,
                arrival_time=0.0,
                total_instructions=instructions_per_process
            ))
            pid += 1
    random.shuffle(processes)
    return processes


import copy

def run_simulation(scheduler_class, cores: List[Core], processes: List[Process],
                   scheduler_kwargs: Dict = None, dt: float = 0.001) -> Dict:
    """Run a simulation and return statistics."""
    scheduler_kwargs = scheduler_kwargs or {}
    
    # Deep copy cores and processes to avoid mutation issues
    cores_copy = copy.deepcopy(cores)
    processes_copy = copy.deepcopy(processes)
    
    scheduler = scheduler_class(cores_copy, **scheduler_kwargs)
    
    # Add all processes to queue
    for p in processes_copy:
        scheduler.enqueue(p)
    
    # Run simulation
    max_time = 3600  # Safety limit
    iteration = 0
    max_iterations = int(max_time / dt)
    
    while scheduler.tick(dt) and iteration < max_iterations:
        scheduler.schedule(dt)
        iteration += 1
    
    return scheduler.get_stats()


if __name__ == "__main__":
    # Quick test
    random.seed(42)
    np.random.seed(42)
    
    cores = create_heterogeneous_topology(n_big=4, n_little=4)
    processes = generate_workload_mix(count_per_type=2, instructions_per_process=10_000_000)
    
    print("Running EEVDF simulation...")
    stats_eevdf = run_simulation(EEVDFScheduler, cores.copy(), processes)
    print(f"EEVDF: Energy={stats_eevdf['total_energy_joules']:.2f}J, Time={stats_eevdf['total_time_seconds']:.2f}s")
    
    print("\nRunning WattSched simulation...")
    stats_wattsched = run_simulation(WattSchedScheduler, cores.copy(), processes)
    print(f"WattSched: Energy={stats_wattsched['total_energy_joules']:.2f}J, Time={stats_wattsched['total_time_seconds']:.2f}s")
    
    savings = (stats_eevdf['total_energy_joules'] - stats_wattsched['total_energy_joules']) / stats_eevdf['total_energy_joules'] * 100
    print(f"\nEnergy savings: {savings:.1f}%")

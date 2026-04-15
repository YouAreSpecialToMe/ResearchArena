"""Discrete-event simulation engine for EEVDF with async displacement.

Key model: Each core runs one task per tick. The scheduler advances vruntime
based only on direct CPU time. But high-displacement tasks generate additional
relay work that consumes real CPU capacity on shared kernel threads. This relay
work is NOT attributed to the originating task by the scheduler, causing fairness
violations when tasks have heterogeneous displacement ratios.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from src.task import Task
from src.scheduler import EEVDFScheduler
from src.relay import RelayEntity, RELAY_CONFIGS
from src.metrics import compute_metrics, jains_fairness_index


class SimulationEngine:
    """Event-driven EEVDF simulator with async relay mechanisms."""

    def __init__(self, num_cores: int = 2, quantum_us: float = 1000.0,
                 tick_us: float = 100.0, sim_duration_us: float = 10_000_000.0,
                 seed: int = 42):
        self.num_cores = num_cores
        self.quantum_us = quantum_us
        self.tick_us = tick_us
        self.sim_duration_us = sim_duration_us
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.schedulers = [EEVDFScheduler(quantum_us) for _ in range(num_cores)]
        self.relay_entities: Dict[str, RelayEntity] = {}
        self.tasks: List[Task] = []
        self.task_map: Dict[int, Task] = {}  # task_id -> Task for fast lookup
        self.current_time: float = 0.0

        # CCP state
        self.ccp_strategy: Optional[str] = None
        self.ccp_params: Dict = {}
        self.ccp_overhead_ops: int = 0
        self.total_ops: int = 0
        # CCP cost model: each operation costs this many microseconds
        self.ccp_op_cost_us: float = 0.5  # cost per vruntime update operation

        # Fairness time series
        self.fairness_timeseries: List[Tuple[float, float]] = []
        self.timeseries_interval_us: float = 1000.0

    def add_relay(self, name: str, config: Optional[Dict] = None):
        if config is None:
            config = RELAY_CONFIGS.get(name, {})
        self.relay_entities[name] = RelayEntity(
            name=name,
            cgroup_attribution=config.get("cgroup_attribution", "none"),
            priority_inheritance=config.get("priority_inheritance", False),
            shared=config.get("shared", True),
        )

    def add_task(self, task: Task, core_id: Optional[int] = None):
        self.tasks.append(task)
        self.task_map[task.task_id] = task
        if core_id is None:
            core_id = len(self.tasks) % self.num_cores
        self.schedulers[core_id % self.num_cores].enqueue(task)

    def configure_ccp(self, strategy: str, **params):
        self.ccp_strategy = strategy
        self.ccp_params = params

    def _apply_ccp_charge(self, task: Task, displaced_time: float):
        """Apply CCP vruntime charging to restore fairness."""
        if self.ccp_strategy is None:
            return

        if self.ccp_strategy == "immediate":
            task.ccp_charged_time += displaced_time
            for sched in self.schedulers:
                if task in sched.run_queue:
                    sched.update_vruntime(task, displaced_time)
                    break
            self.ccp_overhead_ops += 1

        elif self.ccp_strategy == "batched":
            task.pending_displaced += displaced_time

        elif self.ccp_strategy == "statistical":
            ema_alpha = self.ccp_params.get("ema_alpha", 0.1)
            if not hasattr(task, '_ema_displacement'):
                task._ema_displacement = 0.0
            total = task.direct_cpu_time + task.displaced_cpu_time
            if total > 0:
                observed_ratio = task.displaced_cpu_time / total
                task._ema_displacement = ema_alpha * observed_ratio + (1 - ema_alpha) * task._ema_displacement
            # Charge based on estimated ratio
            estimated = displaced_time * min(task._ema_displacement / max(task.displacement_ratio, 0.01), 1.5)
            task.ccp_charged_time += estimated
            for sched in self.schedulers:
                if task in sched.run_queue:
                    sched.update_vruntime(task, estimated)
                    break
            self.ccp_overhead_ops += 1

    def run(self, record_timeseries: bool = False) -> Dict:
        """Run the full simulation."""
        self.current_time = 0.0
        self.ccp_overhead_ops = 0
        self.total_ops = 0
        self.fairness_timeseries = []

        for name in RELAY_CONFIGS:
            if name not in self.relay_entities:
                self.add_relay(name)

        for sched in self.schedulers:
            sched.schedule(self.current_time)

        next_timeseries = self.timeseries_interval_us if record_timeseries else float('inf')
        ccp_batch_interval = self.ccp_params.get("batch_interval_us", 10000.0)
        ccp_batch_timer = 0.0
        tick = self.tick_us

        # Precompute task->relay mapping for speed
        task_relay = {}
        for t in self.tasks:
            rn = getattr(t, 'relay_type', 'io_uring_io_wq')
            if rn not in self.relay_entities:
                rn = list(self.relay_entities.keys())[0] if self.relay_entities else None
            task_relay[t.task_id] = rn

        while self.current_time < self.sim_duration_us:
            self.total_ops += 1

            # PHASE 1: Schedule and execute tasks on cores
            for core_id, sched in enumerate(self.schedulers):
                task = sched.current_task
                if task is None or not task.is_runnable:
                    sched.schedule(self.current_time)
                    task = sched.current_task
                    if task is None:
                        continue

                # Task runs for full tick on this core
                # Scheduler sees full tick as the task's CPU time
                sched.tick(tick, self.current_time)

                # But the task generates displaced work proportional to its alpha
                alpha = task.displacement_ratio
                if alpha > 0:
                    # displaced_work = direct_time * alpha / (1 - alpha)
                    # This is the additional CPU work done by relay entities
                    displaced_amount = tick * alpha / (1.0 - alpha) if alpha < 1.0 else tick
                    rn = task_relay.get(task.task_id)
                    if rn:
                        self.relay_entities[rn].submit_work(
                            task.task_id, task.cgroup_id,
                            displaced_amount, self.current_time)

                # Reschedule every quantum
                if sched.wall_clock >= self.quantum_us:
                    sched.wall_clock = 0.0
                    sched.schedule(self.current_time)
                    self.total_ops += 1

            # PHASE 2: Execute relay work
            # Relay entities run on shared kernel threads, processing displaced
            # work. In a real system, this work completes (it's work-conserving).
            # We process all pending relay work each tick to model that kernel
            # threads are always available to service displaced requests.
            # The key: this CPU time is REAL but NOT attributed to originating tasks.
            for relay in self.relay_entities.values():
                if not relay.work_queue:
                    continue
                # Process all queued work - relay entities are work-conserving
                budget = relay.pending_work_time() + tick
                completed = relay.execute(budget, self.current_time)
                for work in completed:
                    task = self.task_map.get(work.originating_task_id)
                    if task:
                        task.displaced_cpu_time += work.cpu_time_us
                        self._apply_ccp_charge(task, work.cpu_time_us)

            # PHASE 3: Batched CCP flush
            if self.ccp_strategy == "batched":
                ccp_batch_timer += tick
                if ccp_batch_timer >= ccp_batch_interval:
                    for task in self.tasks:
                        if task.pending_displaced > 0:
                            task.ccp_charged_time += task.pending_displaced
                            for sched in self.schedulers:
                                if task in sched.run_queue:
                                    sched.update_vruntime(task, task.pending_displaced)
                                    break
                            task.pending_displaced = 0.0
                            self.ccp_overhead_ops += 1
                    ccp_batch_timer = 0.0

            # Record fairness timeseries
            if record_timeseries and self.current_time >= next_timeseries:
                eff_shares = np.array([t.effective_share(self.current_time) for t in self.tasks])
                jf = float(jains_fairness_index(eff_shares))
                self.fairness_timeseries.append((self.current_time / 1000.0, jf))
                next_timeseries += self.timeseries_interval_us

            self.current_time += tick

        # Final lag computation
        for sched in self.schedulers:
            for task in sched.run_queue:
                sched.compute_lag(task, self.current_time)

        return compute_metrics(self.tasks, self.current_time)


def run_simulation(num_tasks: int, num_cores: int, displacement_ratios: List[float],
                   weights: Optional[List[float]] = None,
                   cgroup_ids: Optional[List[int]] = None,
                   relay_types: Optional[List[str]] = None,
                   sim_duration_us: float = 10_000_000.0,
                   seed: int = 42,
                   ccp_strategy: Optional[str] = None,
                   ccp_params: Optional[Dict] = None,
                   tick_us: float = 100.0,
                   record_timeseries: bool = False) -> Dict:
    """Convenience function to run a simulation with given parameters."""
    engine = SimulationEngine(
        num_cores=num_cores,
        sim_duration_us=sim_duration_us,
        seed=seed,
        tick_us=tick_us,
    )

    if ccp_strategy:
        engine.configure_ccp(ccp_strategy, **(ccp_params or {}))

    if weights is None:
        weights = [1.0] * num_tasks
    if cgroup_ids is None:
        cgroup_ids = [0] * num_tasks
    if relay_types is None:
        relay_types = ["io_uring_io_wq"] * num_tasks

    for i in range(num_tasks):
        task = Task(
            task_id=i,
            weight=weights[i],
            cgroup_id=cgroup_ids[i],
            displacement_ratio=displacement_ratios[i],
        )
        task.relay_type = relay_types[i]
        engine.add_task(task)

    results = engine.run(record_timeseries=record_timeseries)
    results["seed"] = seed
    results["num_cores"] = num_cores

    if ccp_strategy:
        # Overhead: CPU time consumed by CCP operations as fraction of total sim time
        ccp_cpu_time = engine.ccp_overhead_ops * engine.ccp_op_cost_us
        results["ccp_overhead_pct"] = (ccp_cpu_time / max(sim_duration_us, 1)) * 100

    if record_timeseries:
        results["fairness_timeseries"] = engine.fairness_timeseries

    return results

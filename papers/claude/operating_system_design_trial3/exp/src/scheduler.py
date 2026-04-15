"""EEVDF scheduler implementation for the discrete-event simulator."""

import heapq
from typing import List, Optional
from src.task import Task


class EEVDFScheduler:
    """Earliest Eligible Virtual Deadline First scheduler."""

    def __init__(self, quantum_us: float = 1000.0):
        self.quantum_us = quantum_us  # scheduling quantum in microseconds
        self.run_queue: List[Task] = []
        self.current_task: Optional[Task] = None
        self.total_weight: float = 0.0
        self.min_vruntime: float = 0.0
        self.wall_clock: float = 0.0

    def enqueue(self, task: Task):
        """Add a task to the run queue."""
        if task not in self.run_queue:
            # New task: set vruntime to min_vruntime to avoid starvation
            if task.vruntime < self.min_vruntime:
                task.vruntime = self.min_vruntime
            task.eligible_time = task.vruntime
            task.deadline = task.vruntime + self.quantum_us / task.weight
            self.run_queue.append(task)
            self.total_weight += task.weight

    def dequeue(self, task: Task):
        """Remove a task from the run queue."""
        if task in self.run_queue:
            self.run_queue.remove(task)
            self.total_weight -= task.weight

    def update_vruntime(self, task: Task, delta_us: float):
        """Update a task's virtual runtime after it ran for delta_us."""
        task.vruntime += delta_us / task.weight
        # Update min_vruntime
        if self.run_queue:
            self.min_vruntime = min(t.vruntime for t in self.run_queue)

    def compute_lag(self, task: Task, sim_time: float):
        """Compute lag: ideal_time - actual_time for a task."""
        if self.total_weight <= 0:
            return 0.0
        ideal_share = task.weight / self.total_weight
        ideal_time = ideal_share * sim_time
        task.lag = ideal_time - task.direct_cpu_time
        return task.lag

    def select_next(self, current_time: float) -> Optional[Task]:
        """Select the task with the earliest eligible virtual deadline."""
        eligible = [t for t in self.run_queue if t.is_runnable and t.eligible_time <= t.vruntime + 1e-6]
        if not eligible:
            eligible = [t for t in self.run_queue if t.is_runnable]
        if not eligible:
            return None
        # EEVDF: pick the one with the smallest deadline among eligible
        best = min(eligible, key=lambda t: t.deadline)
        return best

    def tick(self, delta_us: float, sim_time: float):
        """Advance the scheduler by delta_us microseconds."""
        self.wall_clock += delta_us
        if self.current_task and self.current_task.is_runnable:
            self.update_vruntime(self.current_task, delta_us)
            self.current_task.direct_cpu_time += delta_us
            self.current_task.total_cpu_time += delta_us
            # Update deadline and eligible time for next selection
            self.current_task.eligible_time = self.current_task.vruntime
            self.current_task.deadline = self.current_task.vruntime + self.quantum_us / self.current_task.weight

    def schedule(self, sim_time: float) -> Optional[Task]:
        """Run the scheduling decision. Returns the newly selected task."""
        next_task = self.select_next(sim_time)
        self.current_task = next_task
        return next_task

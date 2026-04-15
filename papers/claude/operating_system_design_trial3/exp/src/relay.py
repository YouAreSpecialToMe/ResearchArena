"""Relay entity models for async kernel execution mechanisms."""

import dataclasses
from typing import List, Tuple
import numpy as np


@dataclasses.dataclass
class RelayWork:
    """A unit of displaced work submitted by a task to a relay entity."""
    originating_task_id: int
    originating_cgroup_id: int
    cpu_time_us: float  # how much CPU time this work consumes
    submit_time: float
    complete_time: float = 0.0


class RelayEntity:
    """Models an async execution mechanism (io_uring worker, softirq, etc.)."""

    def __init__(self, name: str, cgroup_attribution: str = "none",
                 priority_inheritance: bool = False, shared: bool = True):
        self.name = name
        self.cgroup_attribution = cgroup_attribution  # "none", "partial", "full"
        self.priority_inheritance = priority_inheritance
        self.shared = shared  # shared across tasks or per-task
        self.work_queue: List[RelayWork] = []
        self.completed_work: List[RelayWork] = []
        self.total_cpu_time: float = 0.0

    def submit_work(self, task_id: int, cgroup_id: int, cpu_time_us: float, submit_time: float):
        """Submit displaced work to this relay entity."""
        work = RelayWork(
            originating_task_id=task_id,
            originating_cgroup_id=cgroup_id,
            cpu_time_us=cpu_time_us,
            submit_time=submit_time
        )
        self.work_queue.append(work)
        return work

    def execute(self, available_time_us: float, current_time: float) -> List[RelayWork]:
        """Execute queued work up to available_time_us. Returns completed work items."""
        completed = []
        remaining_time = available_time_us

        while self.work_queue and remaining_time > 0:
            work = self.work_queue[0]
            if work.cpu_time_us <= remaining_time:
                remaining_time -= work.cpu_time_us
                self.total_cpu_time += work.cpu_time_us
                work.complete_time = current_time + (available_time_us - remaining_time)
                self.work_queue.pop(0)
                self.completed_work.append(work)
                completed.append(work)
            else:
                work.cpu_time_us -= remaining_time
                self.total_cpu_time += remaining_time
                remaining_time = 0

        return completed

    def pending_work_time(self) -> float:
        """Total CPU time of pending work."""
        return sum(w.cpu_time_us for w in self.work_queue)


# Standard relay entity configurations
RELAY_CONFIGS = {
    "io_uring_io_wq": {
        "cgroup_attribution": "partial",
        "priority_inheritance": True,
        "shared": False,  # per-task workers
    },
    "io_uring_sqpoll": {
        "cgroup_attribution": "none",
        "priority_inheritance": False,
        "shared": True,
    },
    "softirq_network": {
        "cgroup_attribution": "none",
        "priority_inheritance": False,
        "shared": True,
    },
    "workqueue_cmwq": {
        "cgroup_attribution": "none",
        "priority_inheritance": False,
        "shared": True,
    },
}

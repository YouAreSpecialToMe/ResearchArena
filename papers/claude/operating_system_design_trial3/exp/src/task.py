"""Task (process) model for the EEVDF simulator."""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class Task:
    task_id: int
    weight: float = 1.0
    cgroup_id: int = 0

    # EEVDF state
    vruntime: float = 0.0
    deadline: float = 0.0
    eligible_time: float = 0.0
    lag: float = 0.0

    # CPU time tracking
    total_cpu_time: float = 0.0       # total wall time allocated
    direct_cpu_time: float = 0.0      # time in own context
    displaced_cpu_time: float = 0.0   # time via relay entities

    # Displacement config
    displacement_ratio: float = 0.0   # alpha_i

    # Workload config
    burst_time_mean: float = 1000.0   # mean burst in microseconds
    arrival_rate: float = 100.0       # arrivals per second

    # State
    is_runnable: bool = True
    last_scheduled: float = 0.0

    # CCP state
    ccp_charged_time: float = 0.0     # displaced time charged back via CCP
    pending_displaced: float = 0.0    # accumulated displaced time not yet charged

    def effective_cpu_time(self):
        """Total CPU time including displaced work."""
        return self.direct_cpu_time + self.displaced_cpu_time

    def scheduler_reported_share(self, total_time):
        """Share as seen by the scheduler."""
        if total_time <= 0:
            return 0.0
        return self.direct_cpu_time / total_time

    def effective_share(self, total_time):
        """True share including displaced work."""
        if total_time <= 0:
            return 0.0
        return self.effective_cpu_time() / total_time

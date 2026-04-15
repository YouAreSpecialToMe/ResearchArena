"""Metrics collection for the EEVDF simulator."""

import numpy as np
from typing import List, Dict
from src.task import Task


def jains_fairness_index(shares: np.ndarray) -> float:
    """Compute Jain's fairness index: (sum(x))^2 / (N * sum(x^2))."""
    if len(shares) == 0:
        return 1.0
    s = np.sum(shares)
    ss = np.sum(shares ** 2)
    n = len(shares)
    if ss == 0:
        return 1.0
    return (s ** 2) / (n * ss)


def compute_metrics(tasks: List[Task], sim_time: float) -> Dict:
    """Compute all metrics for a set of tasks after simulation."""
    n = len(tasks)
    if n == 0 or sim_time <= 0:
        return {}

    # Per-task shares
    reported_shares = np.array([t.scheduler_reported_share(sim_time) for t in tasks])
    effective_shares = np.array([t.effective_share(sim_time) for t in tasks])

    # Fairness indices
    jain_reported = jains_fairness_index(reported_shares)
    jain_effective = jains_fairness_index(effective_shares)

    # Share gap
    if np.max(effective_shares) > 0:
        max_share_ratio = np.max(effective_shares) / max(np.min(effective_shares), 1e-12)
    else:
        max_share_ratio = 1.0

    # Lag statistics
    lags = np.array([t.lag for t in tasks])

    # Displacement statistics
    total_direct = sum(t.direct_cpu_time for t in tasks)
    total_displaced = sum(t.displaced_cpu_time for t in tasks)
    total_cpu = total_direct + total_displaced
    displacement_fraction = total_displaced / total_cpu if total_cpu > 0 else 0.0

    # Per-cgroup stats
    cgroup_stats = {}
    cgroup_ids = set(t.cgroup_id for t in tasks)
    for cg in cgroup_ids:
        cg_tasks = [t for t in tasks if t.cgroup_id == cg]
        cg_reported = sum(t.direct_cpu_time for t in cg_tasks)
        cg_actual = sum(t.effective_cpu_time() for t in cg_tasks)
        cgroup_stats[cg] = {
            "reported_cpu": cg_reported,
            "actual_cpu": cg_actual,
            "leakage": cg_actual - cg_reported,
            "leakage_fraction": (cg_actual - cg_reported) / cg_actual if cg_actual > 0 else 0.0
        }

    return {
        "n_tasks": n,
        "sim_time": sim_time,
        "jain_reported": jain_reported,
        "jain_effective": jain_effective,
        "max_share_ratio": max_share_ratio,
        "mean_lag": float(np.mean(lags)),
        "p50_lag": float(np.percentile(lags, 50)) if len(lags) > 0 else 0.0,
        "p99_lag": float(np.percentile(lags, 99)) if len(lags) > 0 else 0.0,
        "max_lag": float(np.max(np.abs(lags))) if len(lags) > 0 else 0.0,
        "total_direct_cpu": total_direct,
        "total_displaced_cpu": total_displaced,
        "displacement_fraction": displacement_fraction,
        "reported_shares": reported_shares.tolist(),
        "effective_shares": effective_shares.tolist(),
        "cgroup_stats": cgroup_stats,
    }


def compute_ccp_metrics(tasks: List[Task], sim_time: float,
                        overhead_ops: int, total_ops: int) -> Dict:
    """Compute metrics specific to CCP evaluation."""
    base = compute_metrics(tasks, sim_time)

    # CCP-corrected shares: scheduler sees direct + charged displaced
    corrected_shares = np.array([
        (t.direct_cpu_time + t.ccp_charged_time) / sim_time if sim_time > 0 else 0.0
        for t in tasks
    ])
    base["jain_ccp"] = float(jains_fairness_index(corrected_shares))
    base["overhead_pct"] = (overhead_ops / total_ops * 100) if total_ops > 0 else 0.0

    return base

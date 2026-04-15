from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policies import POLICY_MAP
from .utils import (
    DEFAULT_HYSTERESIS,
    DEFAULT_MIN_DWELL,
    DEFAULT_SENTINEL_FRACTION,
    DEFAULT_SENTINEL_PLACEMENT,
    EPOCH_LENGTH,
    EXPERTS,
    RECENT_WINDOW_EPOCHS,
    RESIDUAL_HISTORY_EPOCHS,
    SHADOW_RATE,
    hashed_fraction,
    kendall_tau,
    mape,
    spearman_rho,
)


@dataclass
class EpochOutcome:
    epoch: int
    selected_expert: str
    switch: bool
    true_cost: float
    exact_costs: dict[str, float]
    estimated_costs: dict[str, float]
    shadow_sentinel_costs: dict[str, float]
    real_sentinel_costs: dict[str, float]
    corrected_costs: dict[str, float]
    delta: dict[str, float]
    w: dict[str, float]
    ranking_tau_raw: float
    ranking_tau_corrected: float
    ranking_spearman_raw: float
    ranking_spearman_corrected: float
    ranking_mape_raw: float
    ranking_mape_corrected: float


def _is_sampled(page: int, expert: str, seed: int, rate: float) -> bool:
    return hashed_fraction("shadow", expert, seed, int(page)) < rate


def _is_sentinel(page: int, tenant: int, placement: str, fraction: float) -> bool:
    if placement == "uniform-hash":
        return hashed_fraction("sentinel", int(page)) < fraction
    if placement == "contiguous-reserved":
        return hashed_fraction("contiguous", int(page) // 256) < fraction
    if placement == "tenant-local":
        return tenant == 0 and hashed_fraction("tenant-local", int(page)) < min(1.0, 2.0 * fraction)
    raise ValueError(f"unknown sentinel placement: {placement}")


def _estimate_cost(misses: int, observations: int, epoch_accesses: int, fallback: float) -> float:
    if observations <= 0:
        return float(fallback)
    return float((misses / observations) * epoch_accesses)


def _make_policy(name: str, capacity: int, seed: int):
    return POLICY_MAP[name](capacity=capacity, seed=seed)


def simulate_method(
    method: str,
    workload: str,
    trace: np.ndarray,
    tenants: np.ndarray,
    capacity: int,
    exact_epoch_costs: dict[str, list[int]],
    switch_penalty: float,
    seed: int,
    sentinel_fraction: float = DEFAULT_SENTINEL_FRACTION,
    sentinel_placement: str = DEFAULT_SENTINEL_PLACEMENT,
    hysteresis: float = DEFAULT_HYSTERESIS,
    min_dwell: int = DEFAULT_MIN_DWELL,
) -> tuple[dict, list[EpochOutcome]]:
    total_epochs = int(np.ceil(trace.size / EPOCH_LENGTH))
    active = "LRU"
    active_policy = _make_policy(active, capacity, seed)
    shadow_capacity = max(16, int(capacity * SHADOW_RATE))
    sentinel_capacity = max(16, int(capacity * sentinel_fraction))
    shadow_policies = {expert: _make_policy(expert, shadow_capacity, seed) for expert in EXPERTS}
    shadow_sentinel_policies = {expert: _make_policy(expert, sentinel_capacity, seed) for expert in EXPERTS}
    real_sentinel_policies = {expert: _make_policy(expert, sentinel_capacity, seed) for expert in EXPERTS}

    total_misses = 0
    switch_count = 0
    unstable_epochs = 0
    selector_cpu_ms = 0.0
    metadata_bytes = int(
        active_policy.metadata_bytes()
        + sum(policy.metadata_bytes() for policy in shadow_policies.values())
        + sum(policy.metadata_bytes() for policy in shadow_sentinel_policies.values())
        + sum(policy.metadata_bytes() for policy in real_sentinel_policies.values())
    )

    residual_history: dict[str, list[float]] = {expert: [] for expert in EXPERTS}
    recent_hat: dict[str, list[float]] = {expert: [] for expert in EXPERTS}
    recent_real_sentinel: dict[str, list[float]] = {expert: [] for expert in EXPERTS}
    last_hat: dict[str, float] = {expert: float(exact_epoch_costs[expert][0]) for expert in EXPERTS}
    last_shadow_sentinel: dict[str, float] = {expert: float(exact_epoch_costs[expert][0]) for expert in EXPERTS}
    dwell = 0
    ranking_improved = 0
    ranking_degraded = 0
    residual_corr_x: list[float] = []
    residual_corr_y: list[float] = []
    exact_best_match = 0
    records: list[EpochOutcome] = []

    active_epoch_misses = 0
    epoch_accesses = 0
    shadow_obs = {expert: 0 for expert in EXPERTS}
    shadow_misses = {expert: 0 for expert in EXPERTS}
    sentinel_obs = {expert: 0 for expert in EXPERTS}
    sentinel_shadow_misses = {expert: 0 for expert in EXPERTS}
    sentinel_real_obs = {expert: 0 for expert in EXPERTS}
    sentinel_real_misses = {expert: 0 for expert in EXPERTS}

    for t, page_value in enumerate(trace):
        page = int(page_value)
        tenant = int(tenants[t]) if tenants is not None else 0
        active_hit = active_policy.access(page, t)
        if not active_hit:
            active_epoch_misses += 1
            total_misses += 1
        epoch_accesses += 1

        sentinel = _is_sentinel(page, tenant, sentinel_placement, sentinel_fraction)
        for expert in EXPERTS:
            if sentinel:
                real_hit = real_sentinel_policies[expert].access(page, t)
                sentinel_real_obs[expert] += 1
                if not real_hit:
                    sentinel_real_misses[expert] += 1
                if not active_hit:
                    shadow_hit = shadow_sentinel_policies[expert].access(page, t)
                    sentinel_obs[expert] += 1
                    if not shadow_hit:
                        sentinel_shadow_misses[expert] += 1
            if expert != active and not active_hit and _is_sampled(page, expert, seed, SHADOW_RATE):
                shadow_hit = shadow_policies[expert].access(page, t)
                shadow_obs[expert] += 1
                if not shadow_hit:
                    shadow_misses[expert] += 1

        epoch_done = ((t + 1) % EPOCH_LENGTH == 0) or (t + 1 == trace.size)
        if not epoch_done:
            continue

        epoch = len(records)
        exact_costs = {expert: float(exact_epoch_costs[expert][epoch]) for expert in EXPERTS}
        estimated: dict[str, float] = {}
        shadow_sentinel_costs: dict[str, float] = {}
        real_sentinel_costs: dict[str, float] = {}
        delta: dict[str, float] = {}
        weights: dict[str, float] = {}
        corrected: dict[str, float] = {}

        for expert in EXPERTS:
            if expert == active:
                estimated[expert] = float(active_epoch_misses)
            else:
                estimated[expert] = _estimate_cost(shadow_misses[expert], shadow_obs[expert], epoch_accesses, last_hat[expert])
            shadow_sentinel_costs[expert] = _estimate_cost(
                sentinel_shadow_misses[expert], sentinel_obs[expert], epoch_accesses, last_shadow_sentinel[expert]
            )
            real_sentinel_costs[expert] = _estimate_cost(
                sentinel_real_misses[expert], sentinel_real_obs[expert], epoch_accesses, shadow_sentinel_costs[expert]
            )
            delta[expert] = real_sentinel_costs[expert] - shadow_sentinel_costs[expert]
            recent_residuals = residual_history[expert][-RESIDUAL_HISTORY_EPOCHS:]
            baseline_scale = max(1.0, abs(estimated[expert]), abs(real_sentinel_costs[expert]))
            variance = float(np.var(recent_residuals)) if len(recent_residuals) > 1 else 0.0
            normalized_std = np.sqrt(variance) / baseline_scale if variance > 0.0 else 0.0
            coverage = sentinel_real_obs[expert] / max(1, epoch_accesses)
            weights[expert] = float(coverage / max(coverage + normalized_std, 1e-9))
            corrected[expert] = estimated[expert] + weights[expert] * delta[expert]
            residual_history[expert].append(delta[expert])
            recent_hat[expert].append(estimated[expert])
            recent_real_sentinel[expert].append(real_sentinel_costs[expert])
            last_hat[expert] = estimated[expert]
            last_shadow_sentinel[expert] = shadow_sentinel_costs[expert]
            residual_corr_x.append(delta[expert])
            residual_corr_y.append(estimated[expert] - exact_costs[expert])

        if method == "RecentWindow":
            ranking_signal = {
                expert: float(np.mean(recent_hat[expert][-RECENT_WINDOW_EPOCHS:])) for expert in EXPERTS
            }
        elif method == "LeaderSetDuel":
            ranking_signal = dict(real_sentinel_costs)
        elif method == "DirectSentinelOnly":
            ranking_signal = dict(real_sentinel_costs)
        elif method == "NoCalibration":
            ranking_signal = dict(estimated)
        elif method == "NoSentinelScaling":
            ranking_signal = {expert: estimated[expert] + delta[expert] for expert in EXPERTS}
        elif method == "DuelCache":
            ranking_signal = dict(corrected)
        else:
            raise ValueError(f"unknown method {method}")

        selector_cpu_ms += 0.0025 * epoch_accesses + 0.05 * len(EXPERTS)
        best_expert = min(ranking_signal, key=ranking_signal.get)
        exact_best = min(exact_costs, key=exact_costs.get)
        if best_expert == exact_best:
            exact_best_match += 1

        should_switch = False
        if best_expert != active and dwell >= min_dwell:
            gain = ranking_signal[active] - ranking_signal[best_expert]
            should_switch = gain > hysteresis * max(1.0, ranking_signal[active])

        epoch_true_cost = float(active_epoch_misses)
        if should_switch:
            epoch_true_cost += switch_penalty
            resident_pages = active_policy.snapshot_resident_pages()
            active = best_expert
            active_policy = _make_policy(active, capacity, seed)
            active_policy.load_resident_pages(resident_pages, t)
            dwell = 0
            switch_count += 1
        else:
            dwell += 1

        if active != exact_best:
            unstable_epochs += 1

        raw_vector = [estimated[expert] for expert in EXPERTS]
        corr_vector = [corrected[expert] for expert in EXPERTS]
        truth_vector = [exact_costs[expert] for expert in EXPERTS]
        raw_tau = kendall_tau(raw_vector, truth_vector)
        corr_tau = kendall_tau(corr_vector, truth_vector)
        raw_spearman = spearman_rho(raw_vector, truth_vector)
        corr_spearman = spearman_rho(corr_vector, truth_vector)
        raw_mape = mape(raw_vector, truth_vector)
        corr_mape = mape(corr_vector, truth_vector)
        if corr_tau > raw_tau:
            ranking_improved += 1
        elif corr_tau < raw_tau:
            ranking_degraded += 1

        records.append(
            EpochOutcome(
                epoch=epoch,
                selected_expert=active,
                switch=should_switch,
                true_cost=epoch_true_cost,
                exact_costs=exact_costs,
                estimated_costs=estimated,
                shadow_sentinel_costs=shadow_sentinel_costs,
                real_sentinel_costs=real_sentinel_costs,
                corrected_costs=corrected,
                delta=delta,
                w=weights,
                ranking_tau_raw=raw_tau,
                ranking_tau_corrected=corr_tau,
                ranking_spearman_raw=raw_spearman,
                ranking_spearman_corrected=corr_spearman,
                ranking_mape_raw=raw_mape,
                ranking_mape_corrected=corr_mape,
            )
        )

        active_policy.on_epoch_end(epoch)
        for policies in (shadow_policies, shadow_sentinel_policies, real_sentinel_policies):
            for policy in policies.values():
                policy.on_epoch_end(epoch)

        active_epoch_misses = 0
        epoch_accesses = 0
        shadow_obs = {expert: 0 for expert in EXPERTS}
        shadow_misses = {expert: 0 for expert in EXPERTS}
        sentinel_obs = {expert: 0 for expert in EXPERTS}
        sentinel_shadow_misses = {expert: 0 for expert in EXPERTS}
        sentinel_real_obs = {expert: 0 for expert in EXPERTS}
        sentinel_real_misses = {expert: 0 for expert in EXPERTS}

    oracle_fixed = min(sum(costs) for costs in exact_epoch_costs.values())
    weighted_miss_cost = float(total_misses + switch_count * switch_penalty)
    miss_ratio = total_misses / max(1, int(trace.size))
    residual_corr = float(np.corrcoef(residual_corr_x, residual_corr_y)[0, 1]) if len(residual_corr_x) > 1 else 0.0
    summary = {
        "weighted_miss_cost": weighted_miss_cost,
        "miss_ratio": float(miss_ratio),
        "regret_to_oracle_fixed": float(weighted_miss_cost - oracle_fixed),
        "switch_count": int(switch_count),
        "unstable_epoch_fraction": float(unstable_epochs / max(1, total_epochs)),
        "selector_cpu_ms": float(selector_cpu_ms),
        "metadata_bytes": int(metadata_bytes),
        "ranking_improvement_rate": float(ranking_improved / max(1, total_epochs)),
        "ranking_degradation_rate": float(ranking_degraded / max(1, total_epochs)),
        "residual_full_error_corr": residual_corr,
        "exact_best_match_rate": float(exact_best_match / max(1, total_epochs)),
    }
    return summary, records


def evaluate_rankings(
    method: str,
    workload: str,
    trace: np.ndarray,
    tenants: np.ndarray,
    capacity: int,
    exact_epoch_costs: dict[str, list[int]],
    seed: int,
    sentinel_fraction: float,
    sentinel_placement: str,
) -> dict:
    summary, records = simulate_method(
        method=method,
        workload=workload,
        trace=trace,
        tenants=tenants,
        capacity=capacity,
        exact_epoch_costs=exact_epoch_costs,
        switch_penalty=0.0,
        seed=seed,
        sentinel_fraction=sentinel_fraction,
        sentinel_placement=sentinel_placement,
        hysteresis=0.0,
        min_dwell=0,
    )
    return {
        "method": method,
        "kendall_tau": float(np.mean([row.ranking_tau_corrected if method in {"DuelCache", "NoSentinelScaling"} else row.ranking_tau_raw for row in records])),
        "spearman_rho": float(
            np.mean([row.ranking_spearman_corrected if method in {"DuelCache", "NoSentinelScaling"} else row.ranking_spearman_raw for row in records])
        ),
        "mape": float(np.mean([row.ranking_mape_corrected if method in {"DuelCache", "NoSentinelScaling"} else row.ranking_mape_raw for row in records])),
        "ranking_improvement_rate": summary["ranking_improvement_rate"],
        "ranking_degradation_rate": summary["ranking_degradation_rate"],
        "residual_full_error_corr": summary["residual_full_error_corr"],
    }

from __future__ import annotations

import bisect
import collections
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    ACTION_GRID_QUANTA,
    EPOCH_REFS_PER_TENANT,
    ETA_DEFAULT,
    MAX_CPU_WORKERS,
    POLICY_MENU,
    REGRET_WINDOW,
    ROOT,
    RunSpec,
    SHARED_FRACTION_THRESHOLD,
    SRV_ALPHA,
    SRV_CONFIDENCE_ENTROPY_THRESHOLD,
    SRV_CONFIDENCE_MIN_REUSES,
    Timer,
    WINDOW,
    jain_index,
    mean_std,
    peak_rss_mb,
    pin_process_affinity,
    read_json,
    set_reproducible,
    write_json,
    write_jsonl,
)


@dataclass
class PageState:
    file_id: int
    page_id: int
    resident: bool = False
    last_access: int = -1
    freq_total: int = 0
    last_touch_tenant: int = 0
    access_history: list[tuple[int, int]] = field(default_factory=list)
    seq_score: float = 0.0
    freq_by_tenant: dict[int, int] = field(default_factory=lambda: collections.defaultdict(int))


@dataclass
class TenantEpochStats:
    refs: int = 0
    hits: int = 0
    misses: int = 0
    latency: float = 0.0
    shared_refs: int = 0
    controller_changes: int = 0
    charged_debt: float = 0.0
    realized_harm_next: float = 0.0
    realized_shared_regret: int = 0


class ReplaySimulator:
    def __init__(self, spec: RunSpec, calibration: dict[str, Any]):
        self.spec = spec
        self.calibration = calibration
        payload = read_json(Path(spec.trace_path))
        self.trace = payload["events"]
        self.meta = payload["meta"]
        self.tenant_count = spec.tenant_count
        self.total_cache_pages = spec.budget_pages
        self.pages: dict[tuple[int, int], PageState] = {}
        self.resident_pages: set[tuple[int, int]] = set()
        self.debt = [0.0 for _ in range(self.tenant_count)]
        self.target_shares = [self.total_cache_pages / self.tenant_count for _ in range(self.tenant_count)]
        self.policies = ["LRU" for _ in range(self.tenant_count)]
        self.prev_epoch_stats = [TenantEpochStats() for _ in range(self.tenant_count)]
        self.epoch_stats = [TenantEpochStats() for _ in range(self.tenant_count)]
        self.total_stats = [TenantEpochStats() for _ in range(self.tenant_count)]
        self.controller_history = []
        self.eviction_records: dict[tuple[int, int], tuple[int, int]] = {}
        self.future_positions = self._build_future_positions()
        self.latencies = calibration["latency_by_family"][spec.workload_family]
        self.isolated = calibration["isolated_avg_latency"][spec.workload_family]
        self.miss_penalties = self._resolve_miss_penalties()
        self.epoch_rows: list[dict[str, Any]] = []
        self.debt_harm_pairs: list[tuple[float, float]] = []
        self.last_page_by_tenant = [-1 for _ in range(self.tenant_count)]
        self.action_changes = 0
        self.reduction_disabled = not spec.reduction_enabled
        self.run_started = 0.0

    def _uses_private_accounting(self) -> bool:
        return self.spec.method == "PrivateOnly-Utility"

    def _uses_policy_menu(self) -> bool:
        return self.spec.method in {
            "pCache-Account+Policy",
            "ShareArb",
            "ShareArb-NoDebt",
            "ShareArb-UnitCost",
            "ShareArb-HalfLife0.5",
            "ShareArb-HalfLife2.0",
            "UniformSRV",
            "NoReduction",
            "OracleOverlap",
        }

    def _uses_debt_signal(self) -> bool:
        return self.spec.method in {
            "ShareArb",
            "ShareArb-UnitCost",
            "ShareArb-HalfLife0.5",
            "ShareArb-HalfLife2.0",
            "UniformSRV",
            "NoReduction",
        }

    def _uses_local_srv_guard(self) -> bool:
        return self.spec.method in {
            "ShareArb",
            "ShareArb-NoDebt",
            "ShareArb-UnitCost",
            "ShareArb-HalfLife0.5",
            "ShareArb-HalfLife2.0",
            "UniformSRV",
            "NoReduction",
        }

    def _victim_harm_weight(self) -> float:
        if self.spec.oracle:
            return 0.0
        if self.spec.method == "ShareArb-NoDebt":
            return 1.0
        return 1.4 if self._uses_local_srv_guard() else 0.0

    def _resolve_miss_penalties(self) -> list[float]:
        manifest = self.calibration["manifest"][self.spec.workload_family]
        if self.spec.miss_cost_mode == "UnitCost":
            return [1.0 for _ in range(self.tenant_count)]
        if self.spec.miss_cost_mode == "TenantConst":
            penalties = manifest["tenant_penalties"]
            return [float(penalties[str(i)]) for i in range(self.tenant_count)]
        return [float(manifest["family_penalty"]) for _ in range(self.tenant_count)]

    def _build_future_positions(self) -> dict[tuple[int, int], list[tuple[int, int]]]:
        positions: dict[tuple[int, int], list[tuple[int, int]]] = collections.defaultdict(list)
        for idx, event in enumerate(self.trace):
            positions[(event["file_id"], event["page_id"])].append((idx, event["tenant_id"]))
        return positions

    def _page(self, file_id: int, page_id: int) -> PageState:
        key = (file_id, page_id)
        if key not in self.pages:
            self.pages[key] = PageState(file_id=file_id, page_id=page_id)
        return self.pages[key]

    def _current_consumers(self, page: PageState, now: int) -> list[int]:
        recent = [tenant for ts, tenant in page.access_history if now - ts <= WINDOW]
        return sorted(set(recent))

    def _srv_weights(self, page: PageState, now: int) -> dict[int, float]:
        consumers = self._current_consumers(page, now)
        if not consumers:
            return {page.last_touch_tenant: 1.0}
        if self.spec.srv_mode == "uniform":
            weight = 1.0 / len(consumers)
            return {tenant: weight for tenant in consumers}
        weights = collections.defaultdict(float)
        filtered = [(ts, tenant) for ts, tenant in page.access_history if now - ts <= WINDOW]
        for rank, (_, tenant) in enumerate(reversed(filtered[-64:])):
            weights[tenant] += SRV_ALPHA ** rank
        total = sum(weights.values())
        if total <= 0:
            weight = 1.0 / len(consumers)
            return {tenant: weight for tenant in consumers}
        return {tenant: value / total for tenant, value in weights.items()}

    def _occupancy_charges(self, now: int) -> list[float]:
        charges = [0.0 for _ in range(self.tenant_count)]
        for key in self.resident_pages:
            page = self.pages[key]
            consumers = self._current_consumers(page, now)
            if self._uses_private_accounting():
                charges[page.last_touch_tenant] += 1.0
            elif consumers:
                share = 1.0 / len(consumers)
                for tenant in consumers:
                    charges[tenant] += share
            else:
                charges[page.last_touch_tenant] += 1.0
        return charges

    def _occupancy_breakdown(self, now: int) -> tuple[list[float], list[float]]:
        private = [0.0 for _ in range(self.tenant_count)]
        shared = [0.0 for _ in range(self.tenant_count)]
        for key in self.resident_pages:
            page = self.pages[key]
            consumers = self._current_consumers(page, now)
            if self._uses_private_accounting() or len(consumers) <= 1:
                private[page.last_touch_tenant] += 1.0
                continue
            share = 1.0 / len(consumers)
            for tenant in consumers:
                shared[tenant] += share
        return private, shared

    def _page_charge_vector(self, page: PageState, now: int) -> tuple[list[float], list[int]]:
        consumers = self._current_consumers(page, now)
        charges = [0.0 for _ in range(self.tenant_count)]
        if self._uses_private_accounting() or len(consumers) <= 1:
            charges[page.last_touch_tenant] = 1.0
            return charges, consumers
        share = 1.0 / len(consumers)
        for tenant in consumers:
            charges[tenant] = share
        return charges, consumers

    def _policy_cost(self, page: PageState, owner: int, now: int) -> float:
        age = now - page.last_access
        if self.policies[owner] == "LRU":
            return age
        if self.policies[owner] == "SCAN":
            return age + 10.0 * max(0.0, page.seq_score) - 2.0 * page.freq_total
        return age + 2.5 * max(0, 4 - page.freq_total)

    def _oracle_next_use(self, key: tuple[int, int], now: int, evictor: int) -> float:
        positions = self.future_positions.get(key, [])
        idx = bisect.bisect_right(positions, (now, self.tenant_count + 1))
        soonest = REGRET_WINDOW + 10
        for pos, tenant in positions[idx: idx + 10]:
            if tenant == evictor:
                continue
            soonest = min(soonest, pos - now)
        return float(soonest)

    def _pick_victim(self, tenant_id: int, now: int) -> tuple[int, int]:
        charges = self._occupancy_charges(now)
        debt_weight = 0.75 if self._uses_debt_signal() else 0.0
        overloaded = [charges[i] - self.target_shares[i] + debt_weight * self.debt[i] for i in range(self.tenant_count)]
        best_key = None
        best_score = None
        resident_keys = list(self.resident_pages)
        if len(resident_keys) > 32:
            step = max(1, len(resident_keys) // 32)
            resident_keys = resident_keys[::step][:32]
        for key in resident_keys:
            page = self.pages[key]
            page_charges, consumers = self._page_charge_vector(page, now)
            owner = max(range(self.tenant_count), key=lambda t: page_charges[t])
            evictability = self._policy_cost(page, owner, now) / max(1.0, WINDOW / 4.0)
            overload_gain = sum(max(0.0, overloaded[t]) * page_charges[t] for t in range(self.tenant_count))
            underload_penalty = sum(max(0.0, -overloaded[t]) * page_charges[t] for t in range(self.tenant_count))
            external_harm = 0.0
            requester_keep = 0.0
            if len(consumers) > 1:
                weights = self._srv_weights(page, now)
                external_harm = self._victim_harm_weight() * sum(
                    weights.get(other, 0.0) * self.miss_penalties[other]
                    for other in consumers
                    if other != tenant_id
                )
                requester_keep = weights.get(tenant_id, 0.0) * self.miss_penalties[tenant_id]
            oracle_bonus = 0.0
            if self.spec.oracle:
                oracle_bonus = self._oracle_next_use(key, now, tenant_id) / max(1.0, REGRET_WINDOW)
            score = (4.0 * overload_gain) + evictability + oracle_bonus - (2.5 * external_harm) - (0.5 * requester_keep) - underload_penalty
            if best_score is None or score > best_score:
                best_score = score
                best_key = key
        assert best_key is not None
        return best_key

    def _record_regret(self, victim: tuple[int, int], evictor: int, now: int) -> None:
        self.eviction_records[victim] = (evictor, now)
        page = self.pages[victim]
        consumers = self._current_consumers(page, now)
        if len(consumers) > 1:
            weights = self._srv_weights(page, now)
            debt_delta = sum(weights.get(t, 0.0) * self.miss_penalties[t] for t in consumers if t != evictor)
            if self._uses_debt_signal():
                self.debt[evictor] += debt_delta
            self.epoch_stats[evictor].charged_debt += debt_delta

    def _maybe_attribute_harm(self, key: tuple[int, int], tenant: int, now: int) -> None:
        if key not in self.eviction_records:
            return
        evictor, when = self.eviction_records[key]
        if evictor == tenant or now - when > REGRET_WINDOW:
            return
        harm = self.miss_penalties[tenant]
        self.prev_epoch_stats[evictor].realized_harm_next += harm
        self.prev_epoch_stats[evictor].realized_shared_regret += 1

    def _access_page(self, event: dict[str, Any], idx: int) -> None:
        tenant = event["tenant_id"]
        key = (event["file_id"], event["page_id"])
        page = self._page(*key)
        self.epoch_stats[tenant].refs += 1
        current_consumers = self._current_consumers(page, idx)
        if len(current_consumers) > 1:
            self.epoch_stats[tenant].shared_refs += 1
            self.total_stats[tenant].shared_refs += 1
        hit = key in self.resident_pages
        if hit:
            self.epoch_stats[tenant].hits += 1
            self.total_stats[tenant].hits += 1
            self.epoch_stats[tenant].latency += self.latencies["hit_latency_us"]
            self.total_stats[tenant].latency += self.latencies["hit_latency_us"]
        else:
            self.epoch_stats[tenant].misses += 1
            self.total_stats[tenant].misses += 1
            self._maybe_attribute_harm(key, tenant, idx)
            self.epoch_stats[tenant].latency += self.latencies["miss_latency_us"]
            self.total_stats[tenant].latency += self.latencies["miss_latency_us"]
            if len(self.resident_pages) >= self.total_cache_pages:
                victim = self._pick_victim(tenant, idx)
                self.resident_pages.remove(victim)
                self.pages[victim].resident = False
                self._record_regret(victim, tenant, idx)
            self.resident_pages.add(key)
            page.resident = True
        self.total_stats[tenant].refs += 1
        last_page = self.last_page_by_tenant[tenant]
        page.seq_score = 0.8 * page.seq_score + (1.0 if last_page >= 0 and abs(event["page_id"] - last_page) <= 4 else 0.0)
        self.last_page_by_tenant[tenant] = event["page_id"]
        page.last_access = idx
        page.last_touch_tenant = tenant
        page.freq_total += 1
        page.freq_by_tenant[tenant] += 1
        page.access_history.append((idx, tenant))
        if len(page.access_history) > 256:
            page.access_history = page.access_history[-256:]

    def _update_controller(self, idx: int, epoch_id: int) -> None:
        charges = self._occupancy_charges(idx)
        current_stats = self.epoch_stats
        shared_fraction = sum(stat.shared_refs for stat in self.epoch_stats) / max(1, sum(stat.refs for stat in self.epoch_stats))
        confidence = self._srv_confidence(idx)
        active_pages = max(1, len({(e["file_id"], e["page_id"]) for e in self.trace[max(0, idx - WINDOW): idx + 1]}))
        lam = 0.5 ** ((EPOCH_REFS_PER_TENANT * self.tenant_count) / max(1, active_pages * self.spec.debt_half_life_turnovers))
        self.debt = [value * lam for value in self.debt]
        eta = ETA_DEFAULT if self._uses_debt_signal() else 0.0
        if not self.reduction_disabled and (
            shared_fraction < SHARED_FRACTION_THRESHOLD
            or confidence["mean_entropy"] > SRV_CONFIDENCE_ENTROPY_THRESHOLD
            and confidence["avg_reuses"] < SRV_CONFIDENCE_MIN_REUSES
        ):
            eta = 0.0
        pressure = [max(0.0, charges[i] - self.target_shares[i]) + eta * self.debt[i] for i in range(self.tenant_count)]
        utility = []
        slowdown_cost = []
        sequentiality = []
        for tenant in range(self.tenant_count):
            stats = self.epoch_stats[tenant]
            miss_rate = stats.misses / max(1, stats.refs)
            hit_rate = stats.hits / max(1, stats.refs)
            utility.append((hit_rate * self.miss_penalties[tenant]) + (1.25 * miss_rate * self.miss_penalties[tenant]))
            slowdown_cost.append(self._slowdown(stats, tenant))
            sequentiality.append(self._tenant_seq_ratio(tenant))
        allocations = enumerate_allocations(self.tenant_count, ACTION_GRID_QUANTA)
        policy_options = [POLICY_MENU for _ in range(self.tenant_count)] if self._uses_policy_menu() else [["LRU"] for _ in range(self.tenant_count)]
        best = None
        best_score = None
        for alloc in allocations:
            alloc_frac = np.array([a / ACTION_GRID_QUANTA for a in alloc], dtype=float)
            alloc_pages = alloc_frac * self.total_cache_pages
            for pols in _enumerate_policy_assignments(policy_options):
                pol_score = 0.0
                for tenant, policy in enumerate(pols):
                    if policy == "SCAN":
                        pol_score += 0.75 * sequentiality[tenant] - 0.20 * max(0.0, 1.0 - slowdown_cost[tenant])
                    elif policy == "FREQ":
                        pol_score += 0.45 * (1.0 - sequentiality[tenant]) + 0.15 * (self.epoch_stats[tenant].hits / max(1, self.epoch_stats[tenant].refs))
                    else:
                        pol_score += 0.10
                score = (
                    float(np.dot(np.array(utility), alloc_frac))
                    - 0.90 * float(np.dot(np.array(pressure), alloc_frac))
                    - 0.20 * float(np.var(alloc_frac))
                    - 0.10 * float(np.mean(slowdown_cost))
                    + pol_score
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best = (alloc_pages.tolist(), list(pols))
        assert best is not None
        new_shares, new_policies = best
        if [round(v, 5) for v in new_shares] != [round(v, 5) for v in self.target_shares] or new_policies != self.policies:
            self.action_changes += 1
        self.target_shares = new_shares
        self.policies = new_policies
        epoch_slowdowns = [self._slowdown(current_stats[tenant], tenant) for tenant in range(self.tenant_count)]
        fairness = jain_index([1.0 / max(1e-9, value) for value in epoch_slowdowns]) if epoch_slowdowns else 0.0
        occupancy_private, occupancy_shared = self._occupancy_breakdown(idx)
        for tenant in range(self.tenant_count):
            row = {
                "workload_family": self.spec.workload_family,
                "tenant_count": self.tenant_count,
                "cache_budget": self.spec.cache_budget,
                "method": self.spec.method,
                "seed": self.spec.seed,
                "epoch_id": epoch_id,
                "tenant_id": tenant,
                "occupancy_private": occupancy_private[tenant],
                "occupancy_shared": occupancy_shared[tenant],
                "debt": self.prev_epoch_stats[tenant].charged_debt,
                "slowdown": epoch_slowdowns[tenant],
                "throughput_proxy": self._throughput_proxy(current_stats[tenant]),
                "fairness_jain": fairness,
                "shared_regret": current_stats[tenant].realized_shared_regret,
                "controller_changes": self.action_changes,
                "realized_harm_next": self.prev_epoch_stats[tenant].realized_harm_next,
                "runtime_sec": time.perf_counter() - self.run_started,
                "peak_rss_mb": peak_rss_mb(),
            }
            self.epoch_rows.append(row)
            if epoch_id > 0:
                self.debt_harm_pairs.append((self.prev_epoch_stats[tenant].charged_debt, self.prev_epoch_stats[tenant].realized_harm_next))
        self.prev_epoch_stats = self.epoch_stats
        self.epoch_stats = [TenantEpochStats() for _ in range(self.tenant_count)]

    def _slowdown(self, stats: TenantEpochStats, tenant: int) -> float:
        avg_latency = stats.latency / max(1, stats.refs)
        return float(avg_latency / self.isolated[str(tenant)])

    def _throughput_proxy(self, stats: TenantEpochStats) -> float:
        latency = stats.latency
        return float(stats.refs / max(1.0, latency))

    def _tenant_seq_ratio(self, tenant: int) -> float:
        scores = [page.seq_score for page in self.pages.values() if page.last_touch_tenant == tenant]
        return float(np.mean(scores)) if scores else 0.0

    def _srv_confidence(self, now: int) -> dict[str, float]:
        entropies = []
        reuses = []
        for key in self.resident_pages:
            page = self.pages[key]
            consumers = self._current_consumers(page, now)
            if len(consumers) < 2:
                continue
            weights = self._srv_weights(page, now)
            probs = np.array(list(weights.values()), dtype=float)
            entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs))
            entropies.append(float(entropy))
            reuses.append(sum(page.freq_by_tenant.values()) / len(consumers))
        return {
            "mean_entropy": float(np.mean(entropies)) if entropies else 1.0,
            "avg_reuses": float(np.mean(reuses)) if reuses else 0.0,
        }

    def run(self) -> dict[str, Any]:
        with Timer() as timer:
            set_reproducible(self.spec.seed)
            self.cpu_affinity = pin_process_affinity(max_cores=MAX_CPU_WORKERS)
            self.run_started = time.perf_counter()
            epoch_id = 0
            for idx, event in enumerate(self.trace):
                self._access_page(event, idx)
                if all(stat.refs >= EPOCH_REFS_PER_TENANT for stat in self.epoch_stats):
                    self._update_controller(idx, epoch_id)
                    epoch_id += 1
            if any(stat.refs for stat in self.epoch_stats):
                self._update_controller(len(self.trace) - 1, epoch_id)
        metrics = self._summarize(timer.elapsed)
        run_dir = self.spec.run_dir()
        config_payload = self._config_payload()
        write_json(run_dir / "config.json", config_payload)
        metrics["config"] = config_payload
        write_json(run_dir / "results.json", metrics)
        write_jsonl(run_dir / "epoch_metrics.jsonl", self.epoch_rows)
        self._write_log(metrics)
        return metrics

    def _config_payload(self) -> dict[str, Any]:
        payload = self.spec.as_dict()
        input_artifacts = set(payload.get("input_artifacts", []))
        input_artifacts.add("calibration/manifest.json")
        trace_payload = read_json(Path(self.spec.trace_path))
        for dep in trace_payload.get("meta", {}).get("dependencies", []):
            input_artifacts.add(dep)
        payload["input_artifacts"] = sorted(input_artifacts)
        payload["cpu_affinity"] = getattr(self, "cpu_affinity", pin_process_affinity(max_cores=MAX_CPU_WORKERS))
        payload["fixed_hyperparameters"] = {
            "window_refs": WINDOW,
            "epoch_refs_per_tenant": EPOCH_REFS_PER_TENANT,
            "regret_window_refs": REGRET_WINDOW,
            "srv_alpha": SRV_ALPHA,
            "eta_default": ETA_DEFAULT,
            "shared_fraction_threshold": SHARED_FRACTION_THRESHOLD,
            "srv_confidence_entropy_threshold": SRV_CONFIDENCE_ENTROPY_THRESHOLD,
            "srv_confidence_min_reuses": SRV_CONFIDENCE_MIN_REUSES,
            "action_grid_quanta": ACTION_GRID_QUANTA,
            "policy_menu": POLICY_MENU,
            "pressure_weights": {"occupancy": 1.0, "debt": 1.0, "debt_overload_mix": 0.75},
            "victim_scoring": {
                "overload_gain_weight": 4.0,
                "external_harm_weight": 2.5,
                "requester_keep_weight": 0.5,
                "underload_penalty_weight": 1.0,
                "sharearb_harm_weight": 1.4,
                "nodebt_guard_weight": 1.0,
            },
            "controller_objective": {
                "pressure_weight": 0.90,
                "allocation_variance_weight": 0.20,
                "slowdown_weight": 0.10,
                "scan_seq_coeff": 0.75,
                "scan_slowdown_coeff": 0.20,
                "freq_nonseq_coeff": 0.45,
                "freq_hits_coeff": 0.15,
                "lru_bias": 0.10,
                "utility_miss_rate_coeff": 1.25,
            },
            "policy_costs": {
                "scan_seq_weight": 10.0,
                "scan_freq_weight": -2.0,
                "freq_low_count_bonus": 2.5,
            },
        }
        payload["trace_metadata"] = {
            "references_per_tenant": trace_payload["meta"].get("references_per_tenant"),
            "realized_overlap_ratio": trace_payload["meta"].get("realized_overlap_ratio"),
            "shared_pages": trace_payload["meta"].get("shared_pages"),
            "length_regime": trace_payload["meta"].get("length_regime", "planned"),
        }
        return payload

    def _write_log(self, metrics: dict[str, Any]) -> None:
        log_path = ROOT / "exp" / self.spec.experiment / "logs" / (
            f"{self.spec.workload_family}__{self.spec.cache_budget}__{self.spec.method}__seed{self.spec.seed}.log"
        )
        lines = [
            f"run={self.spec.experiment}",
            f"workload={self.spec.workload_family}",
            f"budget={self.spec.cache_budget}",
            f"method={self.spec.method}",
            f"seed={self.spec.seed}",
            f"tenant_count={self.tenant_count}",
            f"trace_path={self.spec.trace_path}",
            f"budget_pages={self.total_cache_pages}",
            f"metrics={json.dumps(metrics['metrics'], sort_keys=True)}",
        ]
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _summarize(self, runtime_sec: float) -> dict[str, Any]:
        shared_regret = 0
        regret_per_tenant = [0.0 for _ in range(self.tenant_count)]
        for page_key, (evictor, when) in self.eviction_records.items():
            positions = self.future_positions.get(page_key, [])
            idx = bisect.bisect_right(positions, (when, self.tenant_count + 1))
            for pos, tenant in positions[idx:]:
                if pos - when > REGRET_WINDOW:
                    break
                if tenant != evictor:
                    shared_regret += 1
                    regret_per_tenant[evictor] += self.miss_penalties[tenant]
                    break
        slowdowns = []
        total_refs = 0
        total_latency = 0.0
        for tenant in range(self.tenant_count):
            total_refs += self.total_stats[tenant].refs
            total_latency += self.total_stats[tenant].latency
            slowdowns.append(self._slowdown(self.total_stats[tenant], tenant))
        pearson, spearman, mae = correlation_metrics(self.debt_harm_pairs)
        throughput_sum = float(total_refs / max(1.0, total_latency))
        fairness = jain_index([1.0 / max(1e-9, s) for s in slowdowns]) if slowdowns else 0.0
        result = {
            "experiment": self.spec.experiment,
            "workload_family": self.spec.workload_family,
            "tenant_count": self.tenant_count,
            "cache_budget": self.spec.cache_budget,
            "method": self.spec.method,
            "seed": self.spec.seed,
            "metrics": {
                "worst_tenant_slowdown": float(max(slowdowns) if slowdowns else 0.0),
                "aggregate_throughput_proxy": throughput_sum,
                "fairness_jain": fairness,
                "shared_regret_raw": int(shared_regret),
                "shared_regret_per_10000_refs": float(shared_regret * 10000 / max(1, len(self.trace))),
                "debt_harm_pearson": pearson,
                "debt_harm_spearman": spearman,
                "debt_harm_mae": mae,
                "controller_changes_per_10000_refs": float(self.action_changes * 10000 / max(1, len(self.trace))),
                "runtime_sec": runtime_sec,
                "peak_rss_mb": peak_rss_mb(),
            },
            "config": self.spec.as_dict(),
        }
        return result


def enumerate_allocations(tenant_count: int, quanta: int) -> list[list[int]]:
    allocs = []
    if tenant_count == 2:
        for first in range(1, quanta):
            allocs.append([first, quanta - first])
        return allocs
    for first in range(1, quanta - 1):
        for second in range(1, quanta - first):
            third = quanta - first - second
            if third >= 1:
                allocs.append([first, second, third])
    return allocs


def _enumerate_policy_assignments(policy_options: list[list[str]]) -> list[tuple[str, ...]]:
    if not policy_options:
        return [tuple()]
    head, *tail = policy_options
    rest = _enumerate_policy_assignments(tail)
    out = []
    for option in head:
        for suffix in rest:
            out.append((option, *suffix))
    return out


def correlation_metrics(pairs: list[tuple[float, float]]) -> tuple[float, float, float]:
    valid = [(a, b) for a, b in pairs if not math.isnan(a) and not math.isnan(b)]
    if len(valid) < 2:
        return 0.0, 0.0, 0.0
    x = np.array([a for a, _ in valid], dtype=float)
    y = np.array([b for _, b in valid], dtype=float)
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    spearman = float(np.corrcoef(xr, yr)[0, 1]) if np.std(xr) > 0 and np.std(yr) > 0 else 0.0
    mae = float(np.mean(np.abs(x - y)))
    return pearson, spearman, mae


def calibration_manifest() -> dict[str, Any]:
    manifest_path = Path("calibration/manifest.json")
    if manifest_path.exists():
        return read_json(manifest_path)
    return {}


def load_calibration() -> dict[str, Any]:
    manifest = read_json(Path("calibration/manifest.json"))
    return {
        "manifest": {item["workload_family"]: item for item in manifest["families"]},
        "latency_by_family": manifest["latency_by_family"],
        "isolated_avg_latency": manifest["isolated_avg_latency"],
    }

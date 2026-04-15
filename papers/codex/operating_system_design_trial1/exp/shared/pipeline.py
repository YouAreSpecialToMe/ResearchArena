from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .methods import evaluate_rankings, simulate_method
from .policies import replay_trace
from .utils import (
    ABLATION_METHODS,
    ACTIVE_METHODS,
    ARTIFACTS,
    CACHE_RATIOS,
    DEFAULT_HYSTERESIS,
    DEFAULT_MIN_DWELL,
    DEFAULT_SENTINEL_FRACTION,
    DEFAULT_SENTINEL_PLACEMENT,
    EPOCH_LENGTH,
    EXPERTS,
    FIGURES_DIR,
    PRIMARY_METHODS,
    RUNS_DIR,
    SEEDS,
    SHADOW_RATE,
    WORKLOADS,
    append_log,
    bootstrap_ci,
    bootstrap_mean_std_ci,
    controller_hyperparameters,
    dump_json,
    dump_jsonl,
    ensure_dirs,
    env_metadata,
    load_json,
    reset_dir,
    run_stem,
    wilcoxon_effect,
)
from .workloads import generate_all, load_tenants, load_trace, manifest_path


def _step_log(step_dir: Path) -> Path:
    return step_dir / "logs" / "run.log"


def _prepare_step_dir(step_dir: Path) -> Path:
    ensure_dirs()
    step_dir.mkdir(parents=True, exist_ok=True)
    reset_dir(step_dir / "logs")
    return _step_log(step_dir)


def _run_paths(experiment: str, family: str, ratio: float, seed: int, method: str, extra: str = "") -> tuple[Path, Path]:
    stem = run_stem(experiment, family, f"ratio{ratio}", f"seed{seed}", method, extra).strip("_")
    return RUNS_DIR / f"{stem}.json", RUNS_DIR / f"{stem}.epochs.jsonl"


def _write_run_artifacts(run_payload: dict, epoch_rows: list[dict], experiment: str, family: str, ratio: float, seed: int, method: str, extra: str = "") -> dict:
    run_path, epoch_path = _run_paths(experiment, family, ratio, seed, method, extra)
    run_payload = dict(run_payload)
    run_payload["run_json"] = str(run_path)
    run_payload["epoch_jsonl"] = str(epoch_path)
    dump_json(run_path, run_payload)
    dump_jsonl(epoch_path, epoch_rows)
    return run_payload


def _load_support_runs_for_primary(methods: set[str]) -> list[dict]:
    rows = []
    for path in RUNS_DIR.glob("03_primary_support__*.json"):
        payload = load_json(path)
        if payload.get("method") in methods:
            rows.append(payload)
    rows.sort(key=lambda row: (row["workload_family"], row["cache_ratio"], row["seed"], row["method"]))
    return rows


def _records_to_epoch_rows(records) -> list[dict]:
    return [
        {
            "epoch": record.epoch,
            "selected_expert": record.selected_expert,
            "switch": record.switch,
            "true_cost": record.true_cost,
            "exact_costs": record.exact_costs,
            "hat_M": record.estimated_costs,
            "ShadowSentinel": record.shadow_sentinel_costs,
            "RealSentinel": record.real_sentinel_costs,
            "delta": record.delta,
            "w": record.w,
            "tilde_M": record.corrected_costs,
            "ranking_tau_raw": record.ranking_tau_raw,
            "ranking_tau_corrected": record.ranking_tau_corrected,
            "ranking_spearman_raw": record.ranking_spearman_raw,
            "ranking_spearman_corrected": record.ranking_spearman_corrected,
            "ranking_mape_raw": record.ranking_mape_raw,
            "ranking_mape_corrected": record.ranking_mape_corrected,
        }
        for record in records
    ]


def write_env(step_dir: Path) -> dict:
    log_path = _prepare_step_dir(step_dir)
    append_log(log_path, "Collecting environment metadata.")
    reset_dir(RUNS_DIR)
    meta = env_metadata()
    lines = [f"{k}: {v}" for k, v in meta.items()]
    (ARTIFACTS / "env.txt").write_text("\n".join(lines) + "\n")
    results = {
        "environment": meta,
        "global_settings": {
            "cpu_workers": 2,
            "seeds": SEEDS,
            "epoch_length": EPOCH_LENGTH,
            "shadow_rate": SHADOW_RATE,
            "sentinel_fraction": DEFAULT_SENTINEL_FRACTION,
            "sentinel_placement": DEFAULT_SENTINEL_PLACEMENT,
            "hysteresis": DEFAULT_HYSTERESIS,
            "minimum_dwell_epochs": DEFAULT_MIN_DWELL,
        },
    }
    dump_json(step_dir / "results.json", results)
    append_log(log_path, "Wrote artifacts/env.txt and exp/01_env/results.json.")
    return meta


def pick_capacity(manifests: list[dict], family: str, seed: int, ratio: float) -> int:
    for item in manifests:
        if item["family"] == family and item["seed"] == seed:
            return int(item["cache_capacities"][str(ratio)])
    raise KeyError((family, seed, ratio))


def generate_data(step_dir: Path) -> dict:
    log_path = _prepare_step_dir(step_dir)
    append_log(log_path, "Generating trace manifests and pilot replays.")
    manifests = generate_all(SEEDS)
    pilots = []
    for family in WORKLOADS:
        trace = load_trace(family, SEEDS[0])
        capacity = pick_capacity(manifests, family, SEEDS[0], 0.5)
        started = time.perf_counter()
        replay_trace(trace[:300_000], "LRU", capacity, seed=SEEDS[0])
        runtime = time.perf_counter() - started
        pilots.append({"family": family, "pilot_runtime_seconds_on_300k_refs": runtime})
        append_log(log_path, f"Pilot {family}: {runtime:.3f}s on 300k refs.")
    results = {"trace_count": len(manifests), "families": WORKLOADS, "manifests": manifests, "pilots": pilots}
    dump_json(step_dir / "results.json", results)
    append_log(log_path, "Trace manifests written.")
    return results


def run_fixed_policy_one(args: tuple[str, int, float, str]) -> dict:
    family, seed, ratio, policy = args
    manifests = load_json(manifest_path())
    trace = load_trace(family, seed)
    capacity = pick_capacity(manifests, family, seed, ratio)
    started = time.perf_counter()
    stats = replay_trace(trace, policy, capacity, seed=seed)
    duration = time.perf_counter() - started
    return {
        "family": family,
        "seed": seed,
        "cache_ratio": ratio,
        "method": policy,
        "capacity": capacity,
        "trace_length": int(trace.size),
        "epoch_misses": stats.epoch_misses,
        "miss_ratio": stats.miss_ratio,
        "weighted_miss_cost": float(stats.total_misses),
        "metadata_bytes": stats.metadata_bytes,
        "runtime_seconds": duration,
        "controller_hyperparameters": controller_hyperparameters(switch_penalty=0.0),
    }


def run_adaptive_method_one(args: tuple[str, int, float, str, dict, float]) -> tuple[dict, list[dict]]:
    family, seed, ratio, method, exact_epoch_costs, switch_penalty = args
    manifests = load_json(manifest_path())
    trace = load_trace(family, seed)
    tenants = load_tenants(family, seed)
    started = time.perf_counter()
    summary, records = simulate_method(
        method=method,
        workload=family,
        trace=trace,
        tenants=tenants,
        capacity=pick_capacity(manifests, family, seed, ratio),
        exact_epoch_costs=exact_epoch_costs,
        switch_penalty=switch_penalty,
        seed=seed,
    )
    runtime = time.perf_counter() - started
    payload = {
        "experiment": "03_primary",
        "workload_family": family,
        "cache_ratio": ratio,
        "seed": seed,
        "method": method,
        "trace_length": int(trace.size),
        "switch_penalty": switch_penalty,
        "runtime_seconds": runtime,
        "controller_hyperparameters": controller_hyperparameters(switch_penalty=switch_penalty),
        **summary,
    }
    return payload, _records_to_epoch_rows(records)


def _write_primary_fixed_support(step_dir: Path, fixed_results: list[dict], switch_penalties: dict[tuple[str, int, float], float], fixed_index: dict[tuple[str, int, float, str], dict]) -> None:
    log_path = _step_log(step_dir)
    for result in fixed_results:
        family = result["family"]
        seed = result["seed"]
        ratio = result["cache_ratio"]
        oracle_fixed = min(
            sum(fixed_index[(family, seed, ratio, expert)]["epoch_misses"]) for expert in EXPERTS
        )
        epoch_rows = [
            {
                "epoch": epoch,
                "true_cost": miss,
                "selected_expert": result["method"],
                "switch": False,
            }
            for epoch, miss in enumerate(result["epoch_misses"])
        ]
        payload = {
            "experiment": "03_primary_support",
            "workload_family": family,
            "cache_ratio": ratio,
            "seed": seed,
            "method": result["method"],
            "trace_length": result["trace_length"],
            "miss_ratio": result["miss_ratio"],
            "weighted_miss_cost": result["weighted_miss_cost"],
            "regret_to_oracle_fixed": float(result["weighted_miss_cost"] - oracle_fixed),
            "switch_count": 0,
            "unstable_epoch_fraction": 0.0,
            "selector_cpu_ms": 0.0,
            "metadata_bytes": result["metadata_bytes"],
            "switch_penalty": switch_penalties[(family, seed, ratio)],
            "runtime_seconds": result["runtime_seconds"],
            "controller_hyperparameters": controller_hyperparameters(switch_penalty=switch_penalties[(family, seed, ratio)]),
        }
        _write_run_artifacts(payload, epoch_rows, "03_primary_support", family, ratio, seed, result["method"])
    append_log(log_path, f"Wrote {len(fixed_results)} fixed-policy support run artifacts.")


def _primary_payload_from_fixed(
    result: dict,
    fixed_index: dict[tuple[str, int, float, str], dict],
    switch_penalties: dict[tuple[str, int, float], float],
) -> tuple[dict, list[dict]]:
    family = result["family"]
    seed = result["seed"]
    ratio = result["cache_ratio"]
    oracle_fixed = min(sum(fixed_index[(family, seed, ratio, expert)]["epoch_misses"]) for expert in EXPERTS)
    epoch_rows = [
        {
            "epoch": epoch,
            "true_cost": miss,
            "selected_expert": result["method"],
            "switch": False,
        }
        for epoch, miss in enumerate(result["epoch_misses"])
    ]
    payload = {
        "experiment": "03_primary",
        "workload_family": family,
        "cache_ratio": ratio,
        "seed": seed,
        "method": result["method"],
        "trace_length": result["trace_length"],
        "miss_ratio": result["miss_ratio"],
        "weighted_miss_cost": result["weighted_miss_cost"],
        "regret_to_oracle_fixed": float(result["weighted_miss_cost"] - oracle_fixed),
        "switch_count": 0,
        "unstable_epoch_fraction": 0.0,
        "selector_cpu_ms": 0.0,
        "metadata_bytes": result["metadata_bytes"],
        "switch_penalty": switch_penalties[(family, seed, ratio)],
        "runtime_seconds": result["runtime_seconds"],
        "controller_hyperparameters": controller_hyperparameters(switch_penalty=switch_penalties[(family, seed, ratio)]),
        "ranking_improvement_rate": 0.0,
        "ranking_degradation_rate": 0.0,
        "residual_full_error_corr": 0.0,
        "exact_best_match_rate": 1.0,
    }
    return payload, epoch_rows


def run_primary(step_dir: Path) -> dict:
    log_path = _prepare_step_dir(step_dir)
    append_log(log_path, "Starting primary matrix.")
    manifests = load_json(manifest_path())
    fixed_jobs = [
        (family, seed, ratio, policy)
        for family in WORKLOADS
        for ratio in CACHE_RATIOS
        for seed in SEEDS
        for policy in ["LRU", "MRU", "LFU-aging", "LHD", "ARC"]
    ]
    fixed_results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_fixed_policy_one, job) for job in fixed_jobs]
        for future in as_completed(futures):
            fixed_results.append(future.result())
    fixed_results.sort(key=lambda row: (row["family"], row["cache_ratio"], row["seed"], row["method"]))
    fixed_index = {(r["family"], r["seed"], r["cache_ratio"], r["method"]): r for r in fixed_results}

    switch_penalties: dict[tuple[str, int, float], float] = {}
    for family in WORKLOADS:
        for ratio in CACHE_RATIOS:
            for seed in SEEDS:
                epoch_costs = [fixed_index[(family, seed, ratio, expert)]["epoch_misses"] for expert in EXPERTS]
                switch_penalties[(family, seed, ratio)] = 0.05 * float(np.mean([np.mean(costs) for costs in epoch_costs]))
    _write_primary_fixed_support(step_dir, fixed_results, switch_penalties, fixed_index)

    primary_rows = []
    for result in fixed_results:
        if result["method"] not in PRIMARY_METHODS:
            continue
        payload, epoch_rows = _primary_payload_from_fixed(result, fixed_index, switch_penalties)
        payload = _write_run_artifacts(payload, epoch_rows, "03_primary", result["family"], result["cache_ratio"], result["seed"], result["method"])
        primary_rows.append(payload)
        append_log(
            log_path,
            f"Primary run {result['family']} ratio={result['cache_ratio']} seed={result['seed']} method={result['method']}: weighted_miss_cost={payload['weighted_miss_cost']:.2f}, switches=0.",
        )

    adaptive_jobs = []
    for family in WORKLOADS:
        for ratio in CACHE_RATIOS:
            for seed in SEEDS:
                exact_epoch_costs = {expert: fixed_index[(family, seed, ratio, expert)]["epoch_misses"] for expert in EXPERTS}
                for method in ACTIVE_METHODS:
                    adaptive_jobs.append((family, seed, ratio, method, exact_epoch_costs, switch_penalties[(family, seed, ratio)]))

    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_adaptive_method_one, job) for job in adaptive_jobs]
        for future in as_completed(futures):
            payload, epoch_rows = future.result()
            payload = _write_run_artifacts(
                payload,
                epoch_rows,
                "03_primary",
                payload["workload_family"],
                payload["cache_ratio"],
                payload["seed"],
                payload["method"],
            )
            primary_rows.append(payload)
            append_log(
                log_path,
                f"Primary run {payload['workload_family']} ratio={payload['cache_ratio']} seed={payload['seed']} method={payload['method']}: weighted_miss_cost={payload['weighted_miss_cost']:.2f}, switches={payload['switch_count']}.",
            )

    primary_rows.sort(key=lambda row: (row["workload_family"], row["cache_ratio"], row["seed"], row["method"]))
    dump_json(step_dir / "results.json", {"run_count": len(primary_rows), "support_run_count": len(fixed_results)})
    dump_json(ARTIFACTS / "primary_runs.json", primary_rows)
    append_log(log_path, f"Primary matrix complete with {len(primary_rows)} planned runs.")
    return {
        "runs": primary_rows,
        "fixed_index": fixed_index,
        "switch_penalties": switch_penalties,
    }


def choose_windows() -> list[dict]:
    windows = []
    for family in ["TwoTenantMix", "SkewShift"]:
        for seed in SEEDS:
            starts = [0, 300_000, 600_000] if family == "TwoTenantMix" else [0, 450_000, 900_000]
            for idx, start in enumerate(starts):
                windows.append({"family": family, "seed": seed, "window_id": idx, "start": start, "length": 120_000})
    return windows


def run_exact_windows(step_dir: Path) -> dict:
    log_path = _prepare_step_dir(step_dir)
    append_log(log_path, "Starting exact-window analysis.")
    manifests = load_json(manifest_path())
    rows = []
    for window in choose_windows():
        family = window["family"]
        seed = window["seed"]
        ratio = 0.5
        trace = np.asarray(load_trace(family, seed)[window["start"] : window["start"] + window["length"]], dtype=np.int32)
        tenants = np.asarray(load_tenants(family, seed)[window["start"] : window["start"] + window["length"]], dtype=np.int8)
        capacity = pick_capacity(manifests, family, seed, ratio)
        exact_epoch_costs = {}
        for expert in EXPERTS:
            stats = replay_trace(trace, expert, capacity, seed=seed)
            exact_epoch_costs[expert] = stats.epoch_misses
            payload = {
                "experiment": "04_exact_windows_support",
                "workload_family": family,
                "cache_ratio": ratio,
                "seed": seed,
                "window_id": window["window_id"],
                "window_start": window["start"],
                "trace_length": int(trace.size),
                "method": expert,
                "miss_ratio": stats.miss_ratio,
                "weighted_miss_cost": float(stats.total_misses),
                "regret_to_oracle_fixed": 0.0,
                "switch_count": 0,
                "unstable_epoch_fraction": 0.0,
                "selector_cpu_ms": 0.0,
                "metadata_bytes": stats.metadata_bytes,
                "controller_hyperparameters": controller_hyperparameters(switch_penalty=0.0),
            }
            epoch_rows = [{"epoch": idx, "true_cost": miss, "selected_expert": expert, "switch": False} for idx, miss in enumerate(stats.epoch_misses)]
            _write_run_artifacts(payload, epoch_rows, "04_exact_windows_support", family, ratio, seed, expert, extra=f"window{window['window_id']}")
        for method in ["RecentWindow", "LeaderSetDuel", "NoCalibration", "DuelCache"]:
            metrics = evaluate_rankings(
                method=method,
                workload=family,
                trace=trace,
                tenants=tenants,
                capacity=capacity,
                exact_epoch_costs=exact_epoch_costs,
                seed=seed,
                sentinel_fraction=DEFAULT_SENTINEL_FRACTION,
                sentinel_placement=DEFAULT_SENTINEL_PLACEMENT,
            )
            row = {**window, "cache_ratio": ratio, **metrics}
            rows.append(row)
            append_log(
                log_path,
                f"Exact window {family} seed={seed} window={window['window_id']} method={method}: tau={metrics['kendall_tau']:.3f}, mape={metrics['mape']:.3f}.",
            )
    dump_json(step_dir / "results.json", {"window_count": len(choose_windows()), "rows": rows})
    dump_json(ARTIFACTS / "exact_window_results.json", rows)
    append_log(log_path, "Exact-window analysis complete.")
    return {"rows": rows}


def run_ablations(step_dir: Path, primary: dict) -> dict:
    log_path = _prepare_step_dir(step_dir)
    append_log(log_path, "Starting ablations and stress tests.")
    manifests = load_json(manifest_path())
    fixed_index = primary["fixed_index"]
    switch_penalties = primary["switch_penalties"]
    ablation_runs = []
    stress_runs = []

    for family in ["TwoTenantMix", "SkewShift"]:
        for ratio in CACHE_RATIOS:
            for seed in SEEDS:
                trace = load_trace(family, seed)
                tenants = load_tenants(family, seed)
                capacity = pick_capacity(manifests, family, seed, ratio)
                exact_epoch_costs = {expert: fixed_index[(family, seed, ratio, expert)]["epoch_misses"] for expert in EXPERTS}
                for method in ABLATION_METHODS:
                    summary, records = simulate_method(
                        method=method,
                        workload=family,
                        trace=trace,
                        tenants=tenants,
                        capacity=capacity,
                        exact_epoch_costs=exact_epoch_costs,
                        switch_penalty=switch_penalties[(family, seed, ratio)],
                        seed=seed,
                    )
                    epoch_rows = _records_to_epoch_rows(records)
                    payload = {
                        "experiment": "05_ablations",
                        "workload_family": family,
                        "cache_ratio": ratio,
                        "seed": seed,
                        "method": method,
                        "trace_length": int(trace.size),
                        "switch_penalty": switch_penalties[(family, seed, ratio)],
                        "controller_hyperparameters": controller_hyperparameters(switch_penalty=switch_penalties[(family, seed, ratio)]),
                        **summary,
                    }
                    payload = _write_run_artifacts(payload, epoch_rows, "05_ablations", family, ratio, seed, method)
                    ablation_runs.append(payload)
                    append_log(log_path, f"Ablation run {family} ratio={ratio} seed={seed} method={method} complete.")

    for family in ["TwoTenantMix", "SkewShift"]:
        ratio = 0.5
        seed = 11
        trace = load_trace(family, seed)
        tenants = load_tenants(family, seed)
        capacity = pick_capacity(manifests, family, seed, ratio)
        exact_epoch_costs = {expert: fixed_index[(family, seed, ratio, expert)]["epoch_misses"] for expert in EXPERTS}
        for fraction in [0.005, 0.01, 0.02, 0.05]:
            for placement in ["uniform-hash", "tenant-local", "contiguous-reserved"]:
                for method in ["NoCalibration", "DuelCache"]:
                    summary, records = simulate_method(
                        method=method,
                        workload=family,
                        trace=trace,
                        tenants=tenants,
                        capacity=capacity,
                        exact_epoch_costs=exact_epoch_costs,
                        switch_penalty=switch_penalties[(family, seed, ratio)],
                        seed=seed,
                        sentinel_fraction=fraction,
                        sentinel_placement=placement,
                    )
                    epoch_rows = _records_to_epoch_rows(records)
                    payload = {
                        "experiment": "05_stress",
                        "workload_family": family,
                        "cache_ratio": ratio,
                        "seed": seed,
                        "method": method,
                        "trace_length": int(trace.size),
                        "sentinel_fraction": fraction,
                        "sentinel_placement": placement,
                        "switch_penalty": switch_penalties[(family, seed, ratio)],
                        "controller_hyperparameters": controller_hyperparameters(
                            switch_penalty=switch_penalties[(family, seed, ratio)],
                            sentinel_fraction=fraction,
                            sentinel_placement=placement,
                        ),
                        **summary,
                    }
                    payload = _write_run_artifacts(
                        payload,
                        epoch_rows,
                        "05_stress",
                        family,
                        ratio,
                        seed,
                        method,
                        extra=f"{placement}_fraction{fraction}",
                    )
                    stress_runs.append(payload)
                    append_log(log_path, f"Stress run {family} placement={placement} fraction={fraction} method={method} complete.")

    dump_json(step_dir / "results.json", {"ablation_runs": ablation_runs, "stress_runs": stress_runs})
    dump_json(ARTIFACTS / "ablation_results.json", ablation_runs)
    dump_json(ARTIFACTS / "stress_results.json", stress_runs)
    append_log(log_path, "Ablation and stress batches complete.")
    return {"ablation_runs": ablation_runs, "stress_runs": stress_runs}


def write_skipped_substrate(step_dir: Path) -> dict:
    log_path = _prepare_step_dir(step_dir)
    reason = (
        "Skipped real programmable-substrate validation: no local cache_ext, PageFlex, or equivalent programmable Linux page-cache substrate exists in this workspace, "
        "and building one from scratch inside this budget would not be a faithful execution of the planned systems step."
    )
    (step_dir / "SKIPPED.md").write_text(reason + "\n")
    results = {
        "status": "skipped",
        "reason": reason,
        "claims_narrowed_to_replay_only": True,
        "deviation_from_plan": "Step 7 was infeasible in this workspace, so claims are narrowed to replay-only evidence.",
    }
    dump_json(step_dir / "results.json", results)
    append_log(log_path, reason)
    return results


def _metric_summary_table(df: pd.DataFrame, group_col: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(group_col):
        row = {group_col: key, "n": int(len(group))}
        for metric in metrics:
            stats = bootstrap_mean_std_ci(group[metric].tolist())
            for stat_name, value in stats.items():
                row[f"{metric}_{stat_name}"] = round(value, 6)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def _paired_metric_summary(duel: pd.DataFrame, baseline: pd.DataFrame, metrics: list[str], by_cols: list[str]) -> dict[str, dict]:
    merged = duel[by_cols + metrics].merge(
        baseline[by_cols + metrics],
        on=by_cols,
        suffixes=("_duel", "_base"),
    ).sort_values(by_cols)
    output = {"n_pairs": int(len(merged))}
    for metric in metrics:
        deltas = (merged[f"{metric}_base"] - merged[f"{metric}_duel"]).tolist()
        baseline_vals = merged[f"{metric}_base"].astype(float)
        if metric in {"kendall_tau", "spearman_rho"} or (baseline_vals <= 0).any():
            rel_mean = None
            rel_ci = {"low": None, "high": None}
        else:
            rel = ((baseline_vals - merged[f"{metric}_duel"].astype(float)) / baseline_vals).tolist()
            rel_mean = float(np.mean(rel))
            rel_ci = bootstrap_ci(rel)
        output[metric] = {
            "absolute_gain_mean": float(np.mean(deltas)),
            "absolute_gain_ci95": bootstrap_ci(deltas),
            "relative_gain_mean": rel_mean,
            "relative_gain_ci95": rel_ci,
        }
        if metric == "weighted_miss_cost":
            output[metric].update(
                wilcoxon_effect(
                    merged[f"{metric}_base"].tolist(),
                    merged[f"{metric}_duel"].tolist(),
                )
            )
    return output


def _method_pair_diagnostics(runs: pd.DataFrame, method_a: str, method_b: str) -> dict:
    a = runs[runs["method"] == method_a].sort_values(["workload_family", "cache_ratio", "seed"])
    b = runs[runs["method"] == method_b].sort_values(["workload_family", "cache_ratio", "seed"])
    if a.empty or b.empty:
        return {}
    merged = a.merge(
        b,
        on=["workload_family", "cache_ratio", "seed"],
        suffixes=("_a", "_b"),
    )
    return {
        "method_a": method_a,
        "method_b": method_b,
        "matched_runs": int(len(merged)),
        "exact_weighted_miss_cost_matches": int((merged["weighted_miss_cost_a"] == merged["weighted_miss_cost_b"]).sum()),
        "exact_switch_count_matches": int((merged["switch_count_a"] == merged["switch_count_b"]).sum()),
        "mean_weighted_miss_cost_gap": float(np.mean(np.abs(merged["weighted_miss_cost_a"] - merged["weighted_miss_cost_b"]))),
        "mean_switch_gap": float(np.mean(np.abs(merged["switch_count_a"] - merged["switch_count_b"]))),
    }


def _diagnose_duelcache_behavior(runs: pd.DataFrame) -> dict:
    duel_rows = runs[runs["method"] == "DuelCache"].sort_values(["workload_family", "cache_ratio", "seed"])
    no_cal_rows = runs[runs["method"] == "NoCalibration"].sort_values(["workload_family", "cache_ratio", "seed"])
    lru_rows = runs[runs["method"] == "LRU"].sort_values(["workload_family", "cache_ratio", "seed"])

    def load_rows(df: pd.DataFrame) -> list[pd.DataFrame]:
        return [_timeline_source(row) for _, row in df.iterrows()]

    duel_epochs = load_rows(duel_rows)
    no_cal_epochs = load_rows(no_cal_rows)
    lru_epochs = load_rows(lru_rows)
    total_epochs = 0
    selected_same_as_nocal = 0
    selected_same_as_lru = 0
    mean_abs_delta = []
    mean_abs_weighted_delta = []
    mean_weight = []
    mean_rank_spread = []
    correction_flip_epochs = 0

    for duel_df, no_cal_df, lru_df in zip(duel_epochs, no_cal_epochs, lru_epochs):
        total_epochs += len(duel_df)
        selected_same_as_nocal += int((duel_df["selected_expert"] == no_cal_df["selected_expert"]).sum())
        selected_same_as_lru += int((duel_df["selected_expert"] == lru_df["selected_expert"]).sum())
        for _, row in duel_df.iterrows():
            delta_vals = np.asarray(list(row.get("delta", {}).values()), dtype=float)
            w_vals = np.asarray(list(row.get("w", {}).values()), dtype=float)
            hat_vals = np.asarray(list(row.get("hat_M", {}).values()), dtype=float)
            corrected_vals = np.asarray(list(row.get("tilde_M", {}).values()), dtype=float)
            if delta_vals.size:
                mean_abs_delta.append(float(np.mean(np.abs(delta_vals))))
            if w_vals.size:
                mean_weight.append(float(np.mean(np.abs(w_vals))))
            if delta_vals.size and w_vals.size:
                mean_abs_weighted_delta.append(float(np.mean(np.abs(w_vals * delta_vals))))
            if hat_vals.size:
                mean_rank_spread.append(float(np.max(hat_vals) - np.min(hat_vals)))
            if hat_vals.size and corrected_vals.size:
                raw_best = min(row["hat_M"], key=row["hat_M"].get)
                corrected_best = min(row["tilde_M"], key=row["tilde_M"].get)
                if raw_best != corrected_best:
                    correction_flip_epochs += 1

    return {
        "duel_vs_nocal": _method_pair_diagnostics(runs, "DuelCache", "NoCalibration"),
        "duel_vs_lru": _method_pair_diagnostics(runs, "DuelCache", "LRU"),
        "epoch_level": {
            "total_epochs": int(total_epochs),
            "selected_expert_match_rate_vs_nocal": float(selected_same_as_nocal / max(1, total_epochs)),
            "selected_expert_match_rate_vs_lru": float(selected_same_as_lru / max(1, total_epochs)),
            "mean_abs_delta": float(np.mean(mean_abs_delta)) if mean_abs_delta else 0.0,
            "mean_abs_weighted_delta": float(np.mean(mean_abs_weighted_delta)) if mean_abs_weighted_delta else 0.0,
            "mean_abs_weight": float(np.mean(mean_weight)) if mean_weight else 0.0,
            "mean_hat_rank_spread": float(np.mean(mean_rank_spread)) if mean_rank_spread else 0.0,
            "correction_flip_rate": float(correction_flip_epochs / max(1, total_epochs)),
        },
    }


def summarize(primary: dict, exact: dict, ablations: dict, substrate: dict, step_dir: Path) -> dict:
    log_path = _prepare_step_dir(step_dir)
    append_log(log_path, "Aggregating results and generating figures.")
    runs = pd.DataFrame(primary["runs"]).sort_values(["workload_family", "cache_ratio", "seed", "method"]).reset_index(drop=True)
    exact_df = pd.DataFrame(exact["rows"])
    ablation_df = pd.DataFrame(ablations["ablation_runs"])
    stress_df = pd.DataFrame(ablations["stress_runs"])

    main_metrics = ["weighted_miss_cost", "miss_ratio", "regret_to_oracle_fixed", "switch_count", "selector_cpu_ms", "metadata_bytes"]
    exact_metrics = ["kendall_tau", "spearman_rho", "mape", "ranking_improvement_rate", "ranking_degradation_rate", "residual_full_error_corr"]
    main_table = _metric_summary_table(runs, "method", main_metrics)
    main_table.to_csv(ARTIFACTS / "main_table.csv", index=False)

    exact_table = _metric_summary_table(exact_df, "method", exact_metrics)
    exact_table.to_csv(ARTIFACTS / "exact_window_table.csv", index=False)

    family_table = _metric_summary_table(runs, "workload_family", ["weighted_miss_cost", "miss_ratio", "regret_to_oracle_fixed"])
    family_table.to_csv(ARTIFACTS / "family_table.csv", index=False)

    paired = {}
    duel = runs[runs["method"] == "DuelCache"].sort_values(["workload_family", "cache_ratio", "seed"])
    for other in ["ARC", "RecentWindow", "LeaderSetDuel"]:
        baseline = runs[runs["method"] == other].sort_values(["workload_family", "cache_ratio", "seed"])
        paired[other] = _paired_metric_summary(
            duel,
            baseline,
            ["weighted_miss_cost", "miss_ratio", "regret_to_oracle_fixed", "switch_count"],
            ["workload_family", "cache_ratio", "seed"],
        )

    exact_paired = {}
    duel_exact = exact_df[exact_df["method"] == "DuelCache"].sort_values(["family", "seed", "window_id"])
    for other in ["RecentWindow", "LeaderSetDuel", "NoCalibration"]:
        other_exact = exact_df[exact_df["method"] == other].sort_values(["family", "seed", "window_id"])
        exact_paired[other] = _paired_metric_summary(
            duel_exact,
            other_exact,
            ["kendall_tau", "spearman_rho", "mape"],
            ["family", "seed", "window_id"],
        )

    claim_families = []
    for family in ["TwoTenantMix", "SkewShift", "PhaseLoop"]:
        fam = runs[runs["workload_family"] == family]
        duel_mean = fam[fam["method"] == "DuelCache"]["weighted_miss_cost"].mean()
        recent_mean = fam[fam["method"] == "RecentWindow"]["weighted_miss_cost"].mean()
        leader_mean = fam[fam["method"] == "LeaderSetDuel"]["weighted_miss_cost"].mean()
        if duel_mean <= 0.95 * recent_mean and duel_mean <= 0.95 * leader_mean:
            claim_families.append(family)

    nocal_exact = exact_df[exact_df["method"] == "NoCalibration"]
    ranking_gain_tau = duel_exact["kendall_tau"].mean() - nocal_exact["kendall_tau"].mean()
    ranking_gain_mape = (nocal_exact["mape"].mean() - duel_exact["mape"].mean()) / max(nocal_exact["mape"].mean(), 1e-9)

    stationary = runs[runs["workload_family"] == "StationaryZipf"]
    duel_stationary = stationary[stationary["method"] == "DuelCache"]["weighted_miss_cost"].mean()
    better_stationary = min(
        stationary[stationary["method"] == "LRU"]["weighted_miss_cost"].mean(),
        stationary[stationary["method"] == "ARC"]["weighted_miss_cost"].mean(),
    )
    transfer_multi = stress_df[(stress_df["workload_family"] == "TwoTenantMix") & (stress_df["method"] == "DuelCache")]
    transfer_single = stress_df[(stress_df["workload_family"] == "SkewShift") & (stress_df["method"] == "DuelCache")]
    failure_regime = stress_df.sort_values("ranking_improvement_rate").iloc[0].to_dict()
    duel_diagnostics = _diagnose_duelcache_behavior(runs)

    deviations = []
    if substrate.get("status") == "skipped":
        deviations.append(
            {
                "step": "07_substrate",
                "status": "skipped",
                "reason": substrate["reason"],
                "impact": "Claims narrowed to replay-only evidence; systems claim treated as unsupported.",
            }
        )

    summary = {
        "claim_scope": "replay_only" if substrate.get("status") == "skipped" else "replay_plus_substrate",
        "confirm_main_replay_claim": len(claim_families) >= 2,
        "confirm_ranking_claim": ranking_gain_tau >= 0.10 and ranking_gain_mape >= 0.15,
        "confirm_negative_control": duel_stationary <= 1.02 * better_stationary,
        "confirm_transfer_claim": bool((transfer_multi["ranking_improvement_rate"] > 0.5).any() and (transfer_single["ranking_improvement_rate"] > 0.5).any()),
        "systems_claim_supported": substrate.get("status") != "skipped",
        "falsifier_little_bias": bool(nocal_exact["kendall_tau"].median() >= 0.85 and nocal_exact["mape"].mean() <= 0.10),
        "falsifier_baselines_match_duelcache": bool(any(v["weighted_miss_cost"]["relative_gain_ci95"]["low"] <= 0.0 for v in paired.values())),
        "falsifier_transfer_narrow": bool(not ((transfer_multi["ranking_improvement_rate"] > 0.5).any() and (transfer_single["ranking_improvement_rate"] > 0.5).any())),
        "falsifier_calibration_hurts": bool((stress_df["ranking_degradation_rate"] > stress_df["ranking_improvement_rate"]).any()),
        "falsifier_systems_no_benefit": substrate.get("status") == "skipped",
    }
    dump_json(ARTIFACTS / "summary.json", summary)
    dump_json(ARTIFACTS / "deviations.json", deviations)

    root_results = {
        "primary_runs": json.loads(runs.to_json(orient="records")),
        "exact_window_results": exact["rows"],
        "ablation_results": ablations["ablation_runs"],
        "stress_results": ablations["stress_runs"],
        "paired_tests": paired,
        "exact_window_paired_tests": exact_paired,
        "main_table": json.loads(main_table.to_json(orient="records")),
        "exact_window_table": json.loads(exact_table.to_json(orient="records")),
        "family_table": json.loads(family_table.to_json(orient="records")),
        "summary": summary,
        "deviations": deviations,
        "duelcache_diagnostics": duel_diagnostics,
        "failure_regime": failure_regime,
        "substate_validation": substrate,
    }
    dump_json(Path(step_dir).parents[1] / "results.json", root_results)
    dump_json(
        step_dir / "results.json",
        {
            "paired_tests": paired,
            "exact_window_paired_tests": exact_paired,
            "summary": summary,
            "failure_regime": failure_regime,
            "duelcache_diagnostics": duel_diagnostics,
            "deviations": deviations,
        },
    )
    _make_figures(runs, exact_df, stress_df)
    append_log(log_path, "Aggregation complete.")
    return root_results


def _timeline_source(run_row: pd.Series) -> pd.DataFrame:
    epoch_path = Path(run_row["epoch_jsonl"])
    return pd.read_json(epoch_path, orient="records", lines=True)


def _make_figures(runs: pd.DataFrame, exact_df: pd.DataFrame, stress_df: pd.DataFrame) -> None:
    reset_dir(FIGURES_DIR)
    for family in ["TwoTenantMix", "SkewShift"]:
        chosen = runs[(runs["workload_family"] == family) & (runs["method"] == "DuelCache") & (runs["cache_ratio"] == 0.5) & (runs["seed"] == 11)].iloc[0]
        timeline = _timeline_source(chosen)
        epochs = timeline["epoch"]
        plt.figure(figsize=(9, 4.5))
        plt.plot(epochs, [row["DuelCache"] if "DuelCache" in row else min(row.values()) for row in timeline["tilde_M"]], label="DuelCache tilde_M(DuelCache)")
        plt.plot(epochs, [min(row.values()) for row in timeline["tilde_M"]], label="Best calibrated estimate")
        plt.plot(epochs, [min(row.values()) for row in timeline["exact_costs"]], label="Exact best expert cost")
        switch_epochs = timeline[timeline["switch"]]["epoch"].tolist()
        for switch_epoch in switch_epochs:
            plt.axvline(switch_epoch, color="black", alpha=0.15, linewidth=1)
        plt.xlabel("Epoch")
        plt.ylabel("Epoch weighted miss cost")
        plt.title(f"{family} per-epoch controller timeline")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{family}_timeline.pdf")
        plt.close()

    plt.figure(figsize=(7.5, 4))
    exact_plot = _metric_summary_table(exact_df, "method", ["kendall_tau", "spearman_rho"]).sort_values("method")
    x = np.arange(len(exact_plot))
    tau = exact_plot["kendall_tau_mean"].to_numpy()
    tau_err = np.vstack(
        [
            tau - exact_plot["kendall_tau_ci95_low"].to_numpy(),
            exact_plot["kendall_tau_ci95_high"].to_numpy() - tau,
        ]
    )
    rho = exact_plot["spearman_rho_mean"].to_numpy()
    rho_err = np.vstack(
        [
            rho - exact_plot["spearman_rho_ci95_low"].to_numpy(),
            exact_plot["spearman_rho_ci95_high"].to_numpy() - rho,
        ]
    )
    plt.errorbar(x - 0.05, tau, yerr=tau_err, fmt="o-", capsize=3, label="Kendall tau")
    plt.errorbar(x + 0.05, rho, yerr=rho_err, fmt="s-", capsize=3, label="Spearman rho")
    plt.xticks(x, exact_plot["method"], rotation=20)
    plt.ylabel("Ranking agreement")
    plt.title("Exact-window ranking quality with 95% bootstrap CIs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exact_window_ranking.pdf")
    plt.close()

    plt.figure(figsize=(7, 4))
    for placement in ["uniform-hash", "tenant-local", "contiguous-reserved"]:
        subset = stress_df[(stress_df["method"] == "DuelCache") & (stress_df["sentinel_placement"] == placement)].sort_values("sentinel_fraction")
        plt.plot(subset["sentinel_fraction"], subset["residual_full_error_corr"], marker="o", label=placement)
    plt.xlabel("Sentinel fraction")
    plt.ylabel("Corr(delta, full error)")
    plt.title("Calibration transfer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "transfer_correlation.pdf")
    plt.close()

    heat = stress_df[stress_df["method"] == "DuelCache"].pivot_table(index="sentinel_placement", columns="sentinel_fraction", values="ranking_improvement_rate").sort_index()
    plt.figure(figsize=(7, 3))
    plt.imshow(heat.values, aspect="auto", cmap="viridis")
    plt.xticks(range(len(heat.columns)), [str(c) for c in heat.columns])
    plt.yticks(range(len(heat.index)), heat.index)
    plt.colorbar(label="Calibration improvement rate")
    plt.xlabel("Sentinel fraction")
    plt.title("Failure-mode heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "failure_heatmap.pdf")
    plt.close()

    plt.figure(figsize=(6, 2.5))
    plt.axis("off")
    plt.text(0.02, 0.62, "Programmable-substrate validation unavailable in this workspace.", fontsize=11)
    plt.text(0.02, 0.36, "Claims are explicitly narrowed to replay-only evidence.", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "systems_validation_placeholder.pdf")
    plt.close()

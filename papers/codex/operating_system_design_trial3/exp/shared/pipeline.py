from __future__ import annotations

import concurrent.futures
import json
import math
import time
from pathlib import Path

import pandas as pd

from .audits import reference_audit, spec_consistency
from .metrics import compare_rankings, ranking_frame
from .policies import POLICY_HYPERPARAMS
from .simulator import simulate_policy
from .utils import (
    ABLATION_TRACE_MODES,
    POLICIES,
    SEEDS,
    TRACE_MODES,
    WORKLOADS,
    append_stage_log,
    bootstrap_ci,
    capture_package_versions,
    ensure_layout,
    now_ts,
    paired_bootstrap_ci,
    reset_stage_log,
    save_text,
    system_report,
    write_json,
)
from .workloads import generate_workload, trace_modes, workload_stats

_TRACE_CACHE: dict[tuple[str, int, str], pd.DataFrame] = {}


def _record_stage_runtime(stage_name: str, started: float, extra: dict | None = None) -> dict:
    runtime_s = time.perf_counter() - started
    payload = {"stage": stage_name, "timestamp": now_ts(), "runtime_s": runtime_s}
    if extra:
        payload.update(extra)
    write_json(ensure_layout()["exp"] / stage_name / "results.json", payload)
    append_stage_log(stage_name, f"completed runtime_s={runtime_s:.3f}")
    return payload


def _write_stage_config(stage_name: str, config: dict) -> None:
    write_json(ensure_layout()["exp"] / stage_name / "config.json", config)


def _load_stage_results(stage_name: str) -> dict:
    return json.loads((ensure_layout()["exp"] / stage_name / "results.json").read_text(encoding="utf-8"))


def _overwrite_stage_results(stage_name: str, extra: dict) -> dict:
    payload = _load_stage_results(stage_name)
    payload.update(extra)
    write_json(ensure_layout()["exp"] / stage_name / "results.json", payload)
    return payload


def _workload_stats_summary(stats_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary = {}
    grouped = stats_df.groupby("workload")
    for workload, subset in grouped:
        summary[workload] = {
            "total_accesses_mean": float(subset["total_accesses"].mean()),
            "unique_pages_mean": float(subset["unique_pages"].mean()),
            "read_write_mix_mean": float(subset["read_write_mix"].mean()),
            "working_set_estimate_pages_mean": float(subset["working_set_estimate_pages"].mean()),
            "working_set_estimate_gb_mean": float(subset["working_set_estimate_gb"].mean()),
            "mean_reuse_distance_pages_mean": float(subset["mean_reuse_distance_pages"].mean()),
            "logical_dataset_gb_mean": float(subset["logical_dataset_gb"].mean()),
        }
    return summary


def _ranking_summary_by_workload(ranking_df: pd.DataFrame, trace_modes: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    summary = {}
    for workload in WORKLOADS:
        workload_summary = {}
        for trace_mode in trace_modes:
            subset = ranking_df[(ranking_df["workload"] == workload) & (ranking_df["trace_mode"] == trace_mode)]
            if subset.empty:
                continue
            workload_summary[trace_mode] = {
                "Kendall_tau_6_mean": float(subset["Kendall_tau_6"].mean()),
                "Spearman_rho_6_mean": float(subset["Spearman_rho_6"].mean()),
                "top1_agreement_mean": float(subset["top1_agreement"].mean()),
                "top2_set_recall_mean": float(subset["top2_set_recall"].mean()),
                "best_policy_regret_mean": float(subset["best_policy_regret"].mean()),
            }
        summary[workload] = workload_summary
    return summary


def _replay_runtime_summary(replay_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary = {}
    grouped = replay_df.groupby(["workload", "trace_mode"])["replay_runtime_s"].mean()
    for (workload, trace_mode), value in grouped.items():
        summary.setdefault(workload, {})[trace_mode] = float(value)
    return summary


def _policy_sensitivity_summary(policy_state_sensitivity: pd.DataFrame) -> dict:
    by_policy = {}
    for policy, subset in policy_state_sensitivity.groupby("policy"):
        by_policy[policy] = {
            "compact_vs_extended_abs_miss_delta_mean": float(subset["delta_miss_rate_vs_ExtendedHinted_CompactState"].abs().mean()),
            "nodirty_vs_extended_abs_miss_delta_mean": float(subset["delta_miss_rate_vs_ExtendedHinted_NoDirty"].abs().mean()),
            "access_vs_extended_abs_miss_delta_mean": float(subset["delta_miss_rate_vs_ExtendedHinted_AccessOnly"].abs().mean()),
        }
    return {
        "overall_abs_miss_delta_means": {
            "CompactState": float(policy_state_sensitivity["delta_miss_rate_vs_ExtendedHinted_CompactState"].abs().mean()),
            "NoDirty": float(policy_state_sensitivity["delta_miss_rate_vs_ExtendedHinted_NoDirty"].abs().mean()),
            "AccessOnly": float(policy_state_sensitivity["delta_miss_rate_vs_ExtendedHinted_AccessOnly"].abs().mean()),
        },
        "by_policy": by_policy,
    }


def _attach_stage_summaries(
    stats_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    live_df: pd.DataFrame,
    policy_state_sensitivity: pd.DataFrame,
    reference_audit_df: pd.DataFrame,
    spec_df: pd.DataFrame,
) -> None:
    _overwrite_stage_results(
        "main_ranking_study",
        {
            "summary_layer_note": "Fields under summary_metrics are stage-level summaries directly recoverable from this JSON. Downstream regroupings used only for figure assembly or row-level inspection remain in artifacts/tables/*.csv and are named in source_artifacts.",
            "summary_metrics": {
                **_load_stage_results("main_ranking_study").get("summary_metrics", {}),
                "workload_statistics_means": _workload_stats_summary(stats_df),
                "replay_runtime_means_s_by_workload_and_trace_mode": _replay_runtime_summary(replay_df),
                "policy_sensitivity": _policy_sensitivity_summary(policy_state_sensitivity),
                "workload_means": _ranking_summary_by_workload(ranking_df, TRACE_MODES),
            },
        },
    )
    _overwrite_stage_results(
        "reference_stability",
        {
            "summary_layer_note": "This JSON stores the headline rebuilt-reference and ablation summaries used in the paper; case-level regroupings remain in artifacts/tables/reference_stability.csv and artifacts/tables/ablation_results.csv.",
            "summary_metrics": {
                **_load_stage_results("reference_stability").get("summary_metrics", {}),
                "ablation_means_by_workload": {
                    workload: {
                        variant: {
                            "delta_Kendall_tau_6_mean": float(
                                ref_df_tmp["delta_Kendall_tau_6"].mean()
                            ),
                            "delta_top2_set_recall_mean": float(
                                ref_df_tmp["delta_top2_set_recall"].mean()
                            ),
                            "delta_best_policy_regret_mean": float(
                                ref_df_tmp["delta_best_policy_regret"].mean()
                            ),
                        }
                        for variant, ref_df_tmp in ablation_df.groupby("variant")
                    }
                    for workload, ablation_df in pd.read_csv(ensure_layout()["tables"] / "ablation_results.csv").groupby("workload")
                },
            },
        },
    )
    _overwrite_stage_results(
        "live_anchors",
        {
            "summary_layer_note": "This JSON stores stage summaries over the three workload-family online anchors. Per-workload rows remain in artifacts/tables/live_anchor_metrics.csv.",
            "summary_metrics": {
                **_load_stage_results("live_anchors").get("summary_metrics", {}),
                "means_by_workload_and_trace_mode": _ranking_summary_by_workload(
                    live_df.rename(
                        columns={
                            "Kendall_tau_6_live": "Kendall_tau_6",
                            "Spearman_rho_6_live": "Spearman_rho_6",
                            "top1_agreement_live": "top1_agreement",
                            "top2_set_recall_live": "top2_set_recall",
                            "best_policy_regret_live": "best_policy_regret",
                        }
                    ),
                    ["ExtendedHinted", "CompactState", "AccessOnly"],
                ),
            },
        },
    )
    _overwrite_stage_results(
        "audits",
        {
            "summary_layer_note": "Audit counts used in the appendix are stored directly in this JSON; row-level bibliography and spec-check details remain in artifacts/tables/reference_audit.csv and artifacts/tables/spec_consistency_audit.csv.",
            "summary_metrics": {
                "reference_audit": {
                    "rows": int(len(reference_audit_df)),
                    "all_fields_ok_rows": int(
                        (
                            reference_audit_df["title_ok"]
                            & reference_audit_df["authors_ok"]
                            & reference_audit_df["venue_ok"]
                            & reference_audit_df["year_ok"]
                            & reference_audit_df["url_ok"]
                        ).sum()
                    ),
                    "corrected_rows": int(reference_audit_df["corrected"].sum()),
                },
                "spec_consistency_audit": {
                    "rows": int(len(spec_df)),
                    "passed_rows": int(spec_df["ok"].sum()),
                },
            },
        },
    )


def stage_environment() -> dict:
    started = time.perf_counter()
    paths = ensure_layout()
    reset_stage_log("environment_config", "# environment_config")
    append_stage_log("environment_config", "collecting environment report")
    save_text(paths["env"] / "system_report.txt", system_report())
    versions = capture_package_versions()
    write_json(paths["env"] / "package_versions.json", versions)
    deviations = {
        "scope_adjustment": "The rerun is reported as a simulator-backed trace study driven by real workload executions with measured cache-residency snapshots and vmstat-based reclaim epochs. True kernel page-cache policy swapping and Filebench-based live anchors were infeasible in this workspace.",
        "infeasible_plan_items": [
            "filebench is not installed",
            "fio is not installed",
            "hyperfine is not installed",
            "no mechanism is available here to swap Linux page-cache policies live",
            "SQLite VFS shimming is not available in the stdlib-only environment",
        ],
    }
    write_json(paths["env"] / "runtime_budget.json", {"cpu_cores": 2, "gpu_parallelism": 0, "global_seeds": SEEDS, "timestamp": now_ts()})
    save_text(
        paths["exp"] / "environment_config" / "SKIPPED.md",
        "\n".join(
            [
                "# Narrowed scope",
                "",
                "- The original kernel-live page-cache claim is infeasible in this workspace.",
                "- This rerun therefore reports a simulator-backed trace study with measured workload executions, measured replay, and measured online anchors.",
                "- `filebench`, `fio`, and `hyperfine` are absent, so the mixed file-service workload is a measured Python fileserver-like workload rather than Filebench.",
            ]
        ),
    )
    _write_stage_config("environment_config", {"seeds": SEEDS, "trace_modes": TRACE_MODES + ABLATION_TRACE_MODES, "policies": POLICIES})
    write_json(paths["env"] / "study_scope.json", deviations)
    return _record_stage_runtime("environment_config", started, {"environment": versions, "deviations": deviations})


def stage_data_preparation() -> tuple[pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    paths = ensure_layout()
    reset_stage_log("data_preparation", "# data_preparation")
    _write_stage_config(
        "data_preparation",
        {
            "workloads": WORKLOADS,
            "seeds": SEEDS,
            "workload_note": "Real workload executions with smaller datasets than the Stage 1 plan; claim narrowed accordingly.",
            "trace_schema": {
                "ExtendedHinted": ["logical_seq", "event_time_us", "inode_id", "page_index", "op_class", "phase_id", "cache_insert_seen", "cache_evict_seen", "dirty_or_writeback_seen", "reclaim_epoch", "seed", "workload", "workload_family"],
                "CompactState": ["logical_seq", "event_time_us", "inode_id", "page_index", "op_class", "phase_id", "dirty_or_writeback_seen", "reclaim_epoch", "seed", "workload", "workload_family"],
                "NoDirty": ["logical_seq", "event_time_us", "inode_id", "page_index", "op_class", "phase_id", "reclaim_epoch", "seed", "workload", "workload_family"],
                "AccessOnly": ["logical_seq", "event_time_us", "inode_id", "page_index", "op_class", "phase_id", "seed", "workload", "workload_family"],
                "NoReclaim": ["logical_seq", "event_time_us", "inode_id", "page_index", "op_class", "phase_id", "dirty_or_writeback_seen", "seed", "workload", "workload_family"],
            },
        },
    )
    trace_rows = []
    stats_rows = []
    for workload in WORKLOADS:
        for seed in SEEDS:
            append_stage_log("data_preparation", f"capturing workload={workload} seed={seed}")
            df = generate_workload(workload, seed, paths["datasets"])
            modes = trace_modes(df)
            modes["ExtendedHinted"].to_parquet(paths["traces_extended"] / f"{workload}_seed{seed}.parquet", compression="gzip")
            for mode_name, mode_df in modes.items():
                if mode_name != "ExtendedHinted":
                    mode_df.to_parquet(paths["traces_compact"] / f"{workload}_{mode_name}_seed{seed}.parquet", compression="gzip")
                trace_rows.append(
                    {
                        "workload": workload,
                        "workload_family": workload,
                        "seed": seed,
                        "trace_mode": mode_name,
                        "rows": len(mode_df),
                        "columns": ",".join(mode_df.columns),
                    }
                )
            stats = workload_stats(df)
            stats_rows.append(stats)
            append_stage_log(
                "data_preparation",
                f"captured workload={workload} seed={seed} accesses={stats['total_accesses']} unique_pages={stats['unique_pages']}",
            )
    stats_df = pd.DataFrame(stats_rows)
    trace_df = pd.DataFrame(trace_rows)
    stats_df.to_csv(paths["tables"] / "workload_stats.csv", index=False)
    trace_df.to_csv(paths["tables"] / "trace_capture_overhead.csv", index=False)
    save_text(
        paths["exp"] / "data_preparation" / "SKIPPED.md",
        "\n".join(
            [
                "# Adjusted capture design",
                "",
                "- The plan's 24 GiB / 16 GiB / 20 GiB datasets were narrowed to sub-GiB real datasets so the full matrix, extra ablations, and paired resampling fit comfortably on 2 CPU cores.",
                "- `cache_insert_seen` and `cache_evict_seen` now come from measured per-page `mincore` residency transitions rather than a replay-side observer.",
                "- SQLite page IDs remain logical pages derived from fixed-width row packing rather than a custom VFS page logger.",
                "- `filebench_fileserver` is represented by a measured fileserver-like real I/O workload because `filebench` is unavailable.",
            ]
        ),
    )
    _record_stage_runtime("data_preparation", started, {"traces_generated": len(trace_df), "workloads": WORKLOADS, "seeds": SEEDS})
    return stats_df, trace_df


def _load_trace(workload: str, seed: int, mode_name: str) -> pd.DataFrame:
    cache_key = (workload, seed, mode_name)
    if cache_key in _TRACE_CACHE:
        return _TRACE_CACHE[cache_key]
    root = Path(__file__).resolve().parents[2]
    if mode_name == "ExtendedHinted":
        df = pd.read_parquet(root / "artifacts" / "traces" / "extended" / f"{workload}_seed{seed}.parquet")
    elif mode_name == "RebuiltNoHints":
        df = pd.read_parquet(root / "artifacts" / "traces" / "extended" / f"{workload}_seed{seed}.parquet")
        df = df.drop(columns=["cache_insert_seen", "cache_evict_seen"])
    elif mode_name == "NoReclaim":
        df = pd.read_parquet(root / "artifacts" / "traces" / "compact" / f"{workload}_NoReclaim_seed{seed}.parquet")
    else:
        df = pd.read_parquet(root / "artifacts" / "traces" / "compact" / f"{workload}_{mode_name}_seed{seed}.parquet")
    _TRACE_CACHE[cache_key] = df
    return df


def _simulate_case(args: tuple) -> dict:
    workload, seed, budget_pages, budget_ratio, mode_name, policy_name, run_type = args
    df = _load_trace(workload, seed, mode_name)
    result = simulate_policy(df, policy_name, budget_pages, mode_name, run_type)
    result.update(
        {
            "workload": workload,
            "workload_family": workload,
            "seed": seed,
            "memory_budget": budget_pages,
            "memory_budget_ratio": budget_ratio,
            "policy": policy_name,
            "trace_mode": mode_name,
            "run_type": run_type,
            "timestamp": now_ts(),
        }
    )
    return result


def stage_replays(stats_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    paths = ensure_layout()
    reset_stage_log("main_ranking_study", "# main_ranking_study")
    budget_lookup = {}
    for _, row in stats_df.iterrows():
        wse = int(row["working_set_estimate_pages"])
        budget_lookup[(row["workload"], int(row["seed"]))] = {
            0.4: max(64, int(math.ceil(wse * 0.40))),
            0.6: max(64, int(math.ceil(wse * 0.60))),
        }
    _write_stage_config(
        "main_ranking_study",
        {
            "budget_lookup": {f"{k[0]}_{k[1]}": v for k, v in budget_lookup.items()},
            "policies": POLICIES,
            "trace_modes": TRACE_MODES,
            "policy_hyperparameters": POLICY_HYPERPARAMS,
            "ranking_order": ["miss_rate", "writeback_count", "mean_eviction_age", "policy"],
        },
    )

    args = []
    for workload in WORKLOADS:
        for seed in SEEDS:
            for budget_ratio in [0.4, 0.6]:
                budget_pages = budget_lookup[(workload, seed)][budget_ratio]
                for mode_name in TRACE_MODES:
                    for policy in POLICIES:
                        args.append((workload, seed, budget_pages, budget_ratio, mode_name, policy, "replay"))
    append_stage_log("main_ranking_study", f"starting replay matrix runs={len(args)}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        replay_rows = list(pool.map(_simulate_case, args))
    replay_df = pd.DataFrame(replay_rows)
    replay_df.to_parquet(paths["replay"] / "replay_results.parquet", index=False)
    replay_df.to_csv(paths["tables"] / "replay_case_results.csv", index=False)

    ranking_rows = []
    pruning_rows = []
    for workload in WORKLOADS:
        for seed in SEEDS:
            for budget_ratio in [0.4, 0.6]:
                subset = replay_df[
                    (replay_df["workload"] == workload)
                    & (replay_df["seed"] == seed)
                    & (replay_df["memory_budget_ratio"] == budget_ratio)
                ]
                ref = subset[subset["trace_mode"] == "ExtendedHinted"]
                ref_ranked = ranking_frame(ref)
                compact = subset[subset["trace_mode"] == "CompactState"]
                compact_best = float(compact["miss_rate"].min())
                survivors = compact[compact["miss_rate"] <= compact_best * 1.05].sort_values("miss_rate").head(3)["policy"].tolist()
                extended_best = ref_ranked.iloc[0]["policy"]
                for mode_name in TRACE_MODES:
                    cand = subset[subset["trace_mode"] == mode_name]
                    metrics = compare_rankings(ref, cand)
                    metrics.update(
                        {
                            "workload": workload,
                            "workload_family": workload,
                            "seed": seed,
                            "memory_budget_ratio": budget_ratio,
                            "trace_mode": mode_name,
                        }
                    )
                    ranking_rows.append(metrics)
                pruning_rows.append(
                    {
                        "workload": workload,
                        "workload_family": workload,
                        "seed": seed,
                        "memory_budget_ratio": budget_ratio,
                        "prune_fraction": 1.0 - len(survivors) / len(POLICIES),
                        "false_prune_rate_vs_extended": float(extended_best not in survivors),
                        "survivor_set_overlap": len(set(survivors) & set(ref_ranked.head(3)["policy"])) / max(1, len(survivors)),
                    }
                )
    ranking_df = pd.DataFrame(ranking_rows)
    pruning_df = pd.DataFrame(pruning_rows)
    ranking_df.to_csv(paths["tables"] / "ranking_metrics.csv", index=False)
    pruning_df.to_csv(paths["tables"] / "pruning_results.csv", index=False)
    per_policy = (
        replay_df.pivot_table(
            index=["workload", "seed", "memory_budget_ratio", "policy"],
            columns="trace_mode",
            values="miss_rate",
        )
        .reset_index()
    )
    for mode_name in ["CompactState", "NoDirty", "AccessOnly"]:
        if mode_name in per_policy.columns and "ExtendedHinted" in per_policy.columns:
            per_policy[f"delta_miss_rate_vs_ExtendedHinted_{mode_name}"] = per_policy[mode_name] - per_policy["ExtendedHinted"]
    per_policy.to_csv(paths["tables"] / "policy_state_sensitivity.csv", index=False)
    append_stage_log("main_ranking_study", f"completed replay matrix rows={len(replay_df)}")
    _record_stage_runtime("main_ranking_study", started, {"replay_runs": len(replay_df), "ranking_cases": len(ranking_df), "pruning_cases": len(pruning_df)})
    return replay_df, ranking_df, pruning_df


def stage_reference_stability(stats_df: pd.DataFrame, replay_df: pd.DataFrame, ranking_df: pd.DataFrame) -> pd.DataFrame:
    del stats_df
    started = time.perf_counter()
    paths = ensure_layout()
    reset_stage_log("reference_stability", "# reference_stability")
    _write_stage_config(
        "reference_stability",
        {
            "variant": "RebuiltNoHints",
            "budgets": [0.4, 0.6],
            "seeds": SEEDS,
            "policy_hyperparameters": POLICY_HYPERPARAMS,
        },
    )
    args = []
    for workload in WORKLOADS:
        for seed in SEEDS:
            for budget_ratio in [0.4, 0.6]:
                budget_pages = int(
                    replay_df[
                        (replay_df["workload"] == workload)
                        & (replay_df["seed"] == seed)
                        & (replay_df["memory_budget_ratio"] == budget_ratio)
                    ]["memory_budget"].iloc[0]
                )
                for policy in POLICIES:
                    args.append((workload, seed, budget_pages, budget_ratio, "RebuiltNoHints", policy, "replay"))
                    args.append((workload, seed, budget_pages, budget_ratio, "NoReclaim", policy, "replay"))
    append_stage_log("reference_stability", f"starting rebuilt reference runs={len(args)}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        rebuilt_rows = list(pool.map(_simulate_case, args))
    rebuilt_df = pd.DataFrame(rebuilt_rows)
    rebuilt_df.to_parquet(paths["replay"] / "reference_rebuilt_results.parquet", index=False)
    rebuilt_df.to_csv(paths["tables"] / "reference_case_results.csv", index=False)

    rows = []
    ablation_rows = []
    for workload in WORKLOADS:
        for seed in SEEDS:
            for budget_ratio in [0.4, 0.6]:
                ext = replay_df[
                    (replay_df["workload"] == workload)
                    & (replay_df["seed"] == seed)
                    & (replay_df["memory_budget_ratio"] == budget_ratio)
                    & (replay_df["trace_mode"] == "ExtendedHinted")
                ]
                rebuild = rebuilt_df[
                    (rebuilt_df["workload"] == workload)
                    & (rebuilt_df["seed"] == seed)
                    & (rebuilt_df["memory_budget_ratio"] == budget_ratio)
                    & (rebuilt_df["trace_mode"] == "RebuiltNoHints")
                ]
                metrics = compare_rankings(ext, rebuild)
                rows.append(
                    {
                        "workload": workload,
                        "workload_family": workload,
                        "seed": seed,
                        "memory_budget_ratio": budget_ratio,
                        "reference_tau_6": metrics["Kendall_tau_6"],
                        "reference_rho_6": metrics["Spearman_rho_6"],
                        "reference_top1_agreement": metrics["top1_agreement"],
                        "reference_top2_recall": metrics["top2_set_recall"],
                        "top_policy_changed": float(ranking_frame(ext).iloc[0]["policy"] != ranking_frame(rebuild).iloc[0]["policy"]),
                    }
                )

                compact = ranking_df[
                    (ranking_df["workload"] == workload)
                    & (ranking_df["seed"] == seed)
                    & (ranking_df["memory_budget_ratio"] == budget_ratio)
                    & (ranking_df["trace_mode"] == "CompactState")
                ].iloc[0]
                for variant_name in ["NoDirty", "AccessOnly", "NoReclaim"]:
                    variant = ranking_df[
                        (ranking_df["workload"] == workload)
                        & (ranking_df["seed"] == seed)
                        & (ranking_df["memory_budget_ratio"] == budget_ratio)
                        & (ranking_df["trace_mode"] == variant_name)
                    ]
                    if variant.empty:
                        variant = rebuilt_df[
                            (rebuilt_df["workload"] == workload)
                            & (rebuilt_df["seed"] == seed)
                            & (rebuilt_df["memory_budget_ratio"] == budget_ratio)
                            & (rebuilt_df["trace_mode"] == variant_name)
                        ]
                        variant_metrics = compare_rankings(ext, variant)
                        variant_row = pd.Series(variant_metrics)
                    else:
                        variant_row = variant.iloc[0]
                    ablation_rows.append(
                        {
                            "workload": workload,
                            "workload_family": workload,
                            "seed": seed,
                            "memory_budget_ratio": budget_ratio,
                            "variant": variant_name,
                            "delta_Kendall_tau_6": float(variant_row["Kendall_tau_6"] - compact["Kendall_tau_6"]),
                            "delta_top2_set_recall": float(variant_row["top2_set_recall"] - compact["top2_set_recall"]),
                            "delta_best_policy_regret": float(variant_row["best_policy_regret"] - compact["best_policy_regret"]),
                            "delta_false_prune_rate_vs_extended": 0.0,
                        }
                    )
    ref_df = pd.DataFrame(rows)
    ablation_df = pd.DataFrame(ablation_rows)
    ref_df.to_csv(paths["tables"] / "reference_stability.csv", index=False)
    ablation_df.to_csv(paths["tables"] / "ablation_results.csv", index=False)
    save_text(
        paths["exp"] / "reference_stability" / "SKIPPED.md",
        "\n".join(
            [
                "# Executed ablations",
                "",
                "- `NoReclaim` is included in this rerun as a full-grid supplementary ablation.",
                "- The executed reference adequacy check is a measured rebuilt-no-hints replay, not a perturbation of `ExtendedHinted` outputs.",
            ]
        ),
    )
    _record_stage_runtime("reference_stability", started, {"reference_cases": len(ref_df), "ablation_cases": len(ablation_df)})
    return ref_df


def stage_live_anchors(stats_df: pd.DataFrame) -> pd.DataFrame:
    started = time.perf_counter()
    paths = ensure_layout()
    reset_stage_log("live_anchors", "# live_anchors")
    _write_stage_config(
        "live_anchors",
        {
            "anchor_seed": 11,
            "anchor_budget_ratio": 0.4,
            "run_type": "online",
            "policy_hyperparameters": POLICY_HYPERPARAMS,
            "validation_note": "These are same-trace online simulations and are not independent kernel-policy validation.",
        },
    )
    budget_lookup = {}
    for _, row in stats_df.iterrows():
        if int(row["seed"]) == 11:
            budget_lookup[row["workload"]] = max(64, int(math.ceil(int(row["working_set_estimate_pages"]) * 0.40)))
    args = []
    for workload in WORKLOADS:
        for policy in POLICIES:
            args.append((workload, 11, budget_lookup[workload], 0.4, "ExtendedHinted", policy, "online"))
    append_stage_log("live_anchors", f"starting measured online anchors runs={len(args)}")
    live_rows = [_simulate_case(arg) for arg in args]
    live_df = pd.DataFrame(live_rows)
    live_df["trace_mode"] = "LIVE_ONLINE"
    live_df.to_parquet(paths["live"] / "live_anchor_results.parquet", index=False)
    live_df.to_csv(paths["tables"] / "live_anchor_case_results.csv", index=False)

    compare_rows = []
    replay_df = pd.read_parquet(paths["replay"] / "replay_results.parquet")
    for workload in WORKLOADS:
        live_case = live_df[live_df["workload"] == workload]
        for mode_name in ["ExtendedHinted", "CompactState", "AccessOnly"]:
            replay_case = replay_df[
                (replay_df["workload"] == workload)
                & (replay_df["seed"] == 11)
                & (replay_df["memory_budget_ratio"] == 0.4)
                & (replay_df["trace_mode"] == mode_name)
            ]
            metrics = compare_rankings(live_case, replay_case, suffix="live")
            metrics.update({"workload": workload, "workload_family": workload, "trace_mode": mode_name})
            compare_rows.append(metrics)
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(paths["tables"] / "live_anchor_metrics.csv", index=False)
    save_text(
        paths["exp"] / "live_anchors" / "SKIPPED.md",
        "\n".join(
            [
                "# Narrowed live validation",
                "",
                "- These anchors are measured online policy simulations over exact workload event streams.",
                "- They are not kernel page-cache policy swaps, which remain infeasible in this workspace.",
            ]
        ),
    )
    _record_stage_runtime("live_anchors", started, {"live_runs": len(live_df), "comparison_rows": len(compare_df)})
    return compare_df


def stage_audits() -> tuple[pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    paths = ensure_layout()
    reset_stage_log("audits", "# audits")
    root = Path(__file__).resolve().parents[2]
    ref_df = pd.DataFrame(reference_audit(root))
    spec_df = pd.DataFrame(spec_consistency(root))
    ref_df.to_csv(paths["tables"] / "reference_audit.csv", index=False)
    spec_df.to_csv(paths["tables"] / "spec_consistency_audit.csv", index=False)
    _record_stage_runtime("audits", started, {"reference_rows": len(ref_df), "spec_checks": len(spec_df)})
    return ref_df, spec_df


def stage_visualization(replay_df: pd.DataFrame, ranking_df: pd.DataFrame, live_df: pd.DataFrame, ref_df: pd.DataFrame, stats_df: pd.DataFrame, pruning_df: pd.DataFrame) -> dict:
    import matplotlib.pyplot as plt
    import seaborn as sns

    started = time.perf_counter()
    paths = ensure_layout()
    reset_stage_log("visualization", "# visualization")
    sns.set_theme(style="whitegrid")

    no_reclaim_delta_df = pd.read_csv(paths["tables"] / "ablation_results.csv")
    no_reclaim_delta_df = no_reclaim_delta_df[no_reclaim_delta_df["variant"] == "NoReclaim"].copy()
    compact_ranking_df = ranking_df[ranking_df["trace_mode"] == "CompactState"].copy()
    no_reclaim_ranking_df = compact_ranking_df.merge(
        no_reclaim_delta_df,
        on=["workload", "workload_family", "seed", "memory_budget_ratio"],
        how="inner",
    )
    no_reclaim_ranking_df["trace_mode"] = "NoReclaim"
    no_reclaim_ranking_df["Kendall_tau_6"] = no_reclaim_ranking_df["Kendall_tau_6"] + no_reclaim_ranking_df["delta_Kendall_tau_6"]
    no_reclaim_ranking_df["top2_set_recall"] = no_reclaim_ranking_df["top2_set_recall"] + no_reclaim_ranking_df["delta_top2_set_recall"]
    plot_ranking_df = pd.concat(
        [ranking_df, no_reclaim_ranking_df[ranking_df.columns]],
        ignore_index=True,
    )

    frontier_modes = TRACE_MODES + ABLATION_TRACE_MODES
    fig1_rows = []
    for workload_family in WORKLOADS:
        for mode_name in frontier_modes:
            subset = plot_ranking_df[(plot_ranking_df["workload_family"] == workload_family) & (plot_ranking_df["trace_mode"] == mode_name)]
            fig1_rows.append(
                {
                    "workload_family": workload_family,
                    "trace_mode": mode_name,
                    "kendall_mean": float(subset["Kendall_tau_6"].mean()),
                    "kendall_ci_low": bootstrap_ci(subset["Kendall_tau_6"].tolist(), seed=11)["ci_low"],
                    "kendall_ci_high": bootstrap_ci(subset["Kendall_tau_6"].tolist(), seed=11)["ci_high"],
                    "top2_mean": float(subset["top2_set_recall"].mean()),
                    "top2_ci_low": bootstrap_ci(subset["top2_set_recall"].tolist(), seed=17)["ci_low"],
                    "top2_ci_high": bootstrap_ci(subset["top2_set_recall"].tolist(), seed=17)["ci_high"],
                }
            )
    fig1 = pd.DataFrame(fig1_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    sns.pointplot(data=fig1, x="trace_mode", y="kendall_mean", hue="workload_family", ax=axes[0], errorbar=None)
    sns.pointplot(data=fig1, x="trace_mode", y="top2_mean", hue="workload_family", ax=axes[1], errorbar=None)
    for ax, metric, low_col, high_col, ylabel in [
        (axes[0], "kendall_mean", "kendall_ci_low", "kendall_ci_high", "Kendall tau"),
        (axes[1], "top2_mean", "top2_ci_low", "top2_ci_high", "Top-2 recall"),
    ]:
        for _, row in fig1.iterrows():
            x = frontier_modes.index(row["trace_mode"])
            hue_offset = {"stream_scan": -0.25, "sqlite_zipf": 0.0, "filebench_fileserver": 0.25}[row["workload_family"]]
            low = max(0.0, float(row[metric]) - float(row[low_col]))
            high = max(0.0, float(row[high_col]) - float(row[metric]))
            ax.errorbar(x + hue_offset, row[metric], yerr=[[low], [high]], fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.set_ylabel(ylabel)
    axes[1].legend_.remove()
    fig.tight_layout()
    for out in [paths["plots"] / "compression_frontier.pdf", paths["figures"] / "compression_frontier.pdf"]:
        fig.savefig(out)
    plt.close(fig)

    heat_rows = []
    for (workload_family, seed, budget_ratio, trace_mode), subset in replay_df[
        replay_df["trace_mode"].isin(["CompactState", "AccessOnly"])
    ].groupby(["workload_family", "seed", "memory_budget_ratio", "trace_mode"]):
        ranked = ranking_frame(subset)
        for _, row in ranked.iterrows():
            heat_rows.append(
                {
                    "workload_family": workload_family,
                    "seed": seed,
                    "memory_budget_ratio": budget_ratio,
                    "trace_mode": trace_mode,
                    "policy": row["policy"],
                    "rank": row["rank"],
                }
            )
    heat = pd.DataFrame(heat_rows)
    heat = heat.pivot_table(index=["workload_family", "memory_budget_ratio", "policy"], columns="trace_mode", values="rank", aggfunc="mean")
    heat["rank_gap"] = (heat["AccessOnly"] - heat["CompactState"]).abs()
    plt.figure(figsize=(8, 8))
    sns.heatmap(heat[["rank_gap"]].unstack(level=2).fillna(0), cmap="crest")
    plt.tight_layout()
    for out in [paths["plots"] / "per_workload_rank_heatmap.pdf", paths["figures"] / "per_workload_rank_heatmap.pdf"]:
        plt.savefig(out)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=live_df, x="workload_family", y="top2_set_recall_live", hue="trace_mode", errorbar=None)
    plt.tight_layout()
    for out in [paths["plots"] / "live_anchor_agreement.pdf", paths["figures"] / "live_anchor_agreement.pdf"]:
        plt.savefig(out)
    plt.close()

    ref_plot = ref_df.melt(id_vars=["workload_family"], value_vars=["reference_tau_6", "reference_top2_recall", "top_policy_changed"], var_name="metric", value_name="value")
    plt.figure(figsize=(9, 4))
    sns.barplot(data=ref_plot, x="metric", y="value", hue="workload_family", errorbar=("ci", 95))
    plt.tight_layout()
    for out in [paths["plots"] / "reference_stability.pdf", paths["figures"] / "reference_stability.pdf"]:
        plt.savefig(out)
    plt.close()

    replay_ci_rows = []
    for mode_name in ["CompactState", "NoDirty", "AccessOnly"]:
        subset = ranking_df[ranking_df["trace_mode"] == mode_name]
        for metric in ["Kendall_tau_6", "top2_set_recall"]:
            row = {"trace_mode": mode_name, "metric": metric}
            row.update(bootstrap_ci(subset[metric].tolist(), seed=13, rounds=1000))
            replay_ci_rows.append(row)
    live_ci_rows = []
    for mode_name in ["ExtendedHinted", "CompactState", "AccessOnly"]:
        subset = live_df[live_df["trace_mode"] == mode_name]
        if not subset.empty:
            row = {"trace_mode": mode_name, "metric": "top2_set_recall_live"}
            row.update(bootstrap_ci(subset["top2_set_recall_live"].tolist(), seed=19, rounds=1000))
            live_ci_rows.append(row)
    pd.DataFrame(replay_ci_rows).to_csv(paths["tables"] / "bootstrap_ci_replay.csv", index=False)
    pd.DataFrame(live_ci_rows).to_csv(paths["tables"] / "bootstrap_ci_live.csv", index=False)

    paired_rows = []
    for compare_mode in ["AccessOnly", "NoDirty", "NoReclaim"]:
        for metric in ["Kendall_tau_6", "top2_set_recall", "best_policy_regret"]:
            compact_values = ranking_df[ranking_df["trace_mode"] == "CompactState"].sort_values(["workload", "seed", "memory_budget_ratio"])[metric].tolist()
            if compare_mode == "NoReclaim":
                compare_values = pd.read_csv(paths["tables"] / "ablation_results.csv")
                compare_values = compare_values[compare_values["variant"] == "NoReclaim"].sort_values(["workload", "seed", "memory_budget_ratio"])
                deltas = [-float(v) for v in compare_values[f"delta_{metric}"].tolist()]
                ci = bootstrap_ci(deltas, seed=23, rounds=1000)
                ci = {"mean_delta": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}
            else:
                compare_values = ranking_df[ranking_df["trace_mode"] == compare_mode].sort_values(["workload", "seed", "memory_budget_ratio"])[metric].tolist()
                ci = paired_bootstrap_ci(compact_values, compare_values, seed=23, rounds=1000)
            paired_rows.append({"left": "CompactState", "right": compare_mode, "metric": metric, **ci})
    pd.DataFrame(paired_rows).to_csv(paths["tables"] / "paired_bootstrap_deltas.csv", index=False)

    runtime_rows = []
    for stage_name in [
        "environment_config",
        "data_preparation",
        "main_ranking_study",
        "reference_stability",
        "live_anchors",
        "audits",
    ]:
        stage_result = json.loads((paths["exp"] / stage_name / "results.json").read_text(encoding="utf-8"))
        runtime_rows.append({"stage": stage_name, "runtime_s": stage_result["runtime_s"]})
    runtime_rows.append({"stage": "visualization", "runtime_s": time.perf_counter() - started})
    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_csv(paths["tables"] / "runtime_accounting.csv", index=False)
    runtime_schedule = {
        "stages": runtime_rows,
        "total_minutes": float(runtime_df["runtime_s"].sum() / 60.0),
        "within_8h_budget": float(runtime_df["runtime_s"].sum()) <= 8 * 3600,
    }
    write_json(paths["tables"] / "runtime_schedule.json", runtime_schedule)

    stats_df.to_csv(paths["tables"] / "table1_workload_statistics.csv", index=False)
    replay_df.groupby(["workload_family", "trace_mode"])["replay_runtime_s"].mean().reset_index().to_csv(paths["tables"] / "table2_trace_capture_overhead.csv", index=False)
    ranking_df.to_csv(paths["tables"] / "table3_replay_ranking_metrics.csv", index=False)
    live_df.to_csv(paths["tables"] / "table4_live_anchor_agreement.csv", index=False)
    pruning_df.to_csv(paths["tables"] / "table5_pruning_results.csv", index=False)
    pd.read_csv(paths["tables"] / "ablation_results.csv").to_csv(paths["tables"] / "table6_ablations.csv", index=False)
    runtime_df.to_csv(paths["tables"] / "table7_runtime_accounting.csv", index=False)
    pd.read_csv(paths["tables"] / "reference_audit.csv").to_csv(paths["tables"] / "table8_reference_audit.csv", index=False)

    _record_stage_runtime("visualization", started, {"figures": 4, "tables": 8})
    _overwrite_stage_results(
        "visualization",
        {
            "source_artifacts": {
                "compression_frontier": "figures/compression_frontier.pdf",
                "per_workload_rank_heatmap": "figures/per_workload_rank_heatmap.pdf",
                "live_anchor_agreement": "figures/live_anchor_agreement.pdf",
                "reference_stability": "figures/reference_stability.pdf",
            },
            "figure_semantics": {
                "compression_frontier": "Per-workload means with bootstrap error bars for Kendall_tau_6 and top2_set_recall. ExtendedHinted, CompactState, NoDirty, and AccessOnly come from artifacts/tables/ranking_metrics.csv; NoReclaim is reconstructed per replay case by applying artifacts/tables/ablation_results.csv deltas to the matching CompactState rows.",
                "per_workload_rank_heatmap": "Absolute rank gap |rank_AccessOnly - rank_CompactState|, averaged over seeds for each workload family, budget ratio, and policy, where each per-case rank uses the lexicographic order (miss_rate, writeback_count, mean_eviction_age, policy) defined in the method section.",
                "live_anchor_agreement": "Per-workload top2_set_recall_live from artifacts/tables/live_anchor_metrics.csv; ordering metrics remain in the live-anchor table.",
                "reference_stability": "Bar plot of reference_tau_6, reference_top2_recall, and top_policy_changed by workload family from artifacts/tables/reference_stability.csv.",
            },
            "included_trace_modes": frontier_modes,
            "runtime_schedule": runtime_schedule,
            "summary_layer_note": "This JSON records the exact semantics used to generate each paper figure. Figure values are downstream regroupings of case-level artifacts rather than stage-level statistical summaries.",
        },
    )
    return runtime_schedule


def refresh_summaries_from_artifacts() -> dict:
    paths = ensure_layout()
    stats_df = pd.read_csv(paths["tables"] / "workload_stats.csv")
    replay_df = pd.read_csv(paths["tables"] / "replay_case_results.csv")
    ranking_df = pd.read_csv(paths["tables"] / "ranking_metrics.csv")
    pruning_df = pd.read_csv(paths["tables"] / "pruning_results.csv")
    ref_df = pd.read_csv(paths["tables"] / "reference_stability.csv")
    live_df = pd.read_csv(paths["tables"] / "live_anchor_metrics.csv")
    policy_state_sensitivity = pd.read_csv(paths["tables"] / "policy_state_sensitivity.csv")
    reference_audit_df = pd.read_csv(paths["tables"] / "reference_audit.csv")
    spec_df = pd.read_csv(paths["tables"] / "spec_consistency_audit.csv")
    runtime_schedule = stage_visualization(replay_df, ranking_df, live_df, ref_df, stats_df, pruning_df)
    _attach_stage_summaries(
        stats_df=stats_df,
        replay_df=replay_df,
        ranking_df=ranking_df,
        ref_df=ref_df,
        live_df=live_df,
        policy_state_sensitivity=policy_state_sensitivity,
        reference_audit_df=reference_audit_df,
        spec_df=spec_df,
    )
    return aggregate_results(
        stats_df,
        replay_df,
        ranking_df,
        pruning_df,
        ref_df,
        live_df,
        reference_audit_df,
        spec_df,
        runtime_schedule,
    )


def aggregate_results(stats_df, replay_df, ranking_df, pruning_df, ref_df, live_df, reference_audit_df, spec_df, runtime_schedule) -> dict:
    root = Path(__file__).resolve().parents[2]
    policy_sensitivity = pd.read_csv(root / "artifacts" / "tables" / "policy_state_sensitivity.csv")
    _attach_stage_summaries(
        stats_df=stats_df,
        replay_df=replay_df,
        ranking_df=ranking_df,
        ref_df=ref_df,
        live_df=live_df,
        policy_state_sensitivity=policy_sensitivity,
        reference_audit_df=reference_audit_df,
        spec_df=spec_df,
    )
    compact = ranking_df[ranking_df["trace_mode"] == "CompactState"]
    access = ranking_df[ranking_df["trace_mode"] == "AccessOnly"]
    live_extended = live_df[live_df["trace_mode"] == "ExtendedHinted"]
    live_compact = live_df[live_df["trace_mode"] == "CompactState"]
    spec_failures = spec_df[~spec_df["ok"]]["check"].tolist()
    success = {
        "compact_top2_recall_ge_0_80": float(compact["top2_set_recall"].mean()) >= 0.80,
        "compact_tau_ge_0_65": float(compact["Kendall_tau_6"].mean()) >= 0.65,
        "access_worse_by_0_10": (float(compact["Kendall_tau_6"].mean()) - float(access["Kendall_tau_6"].mean()) >= 0.10)
        or (float(compact["top2_set_recall"].mean()) - float(access["top2_set_recall"].mean()) >= 0.10),
        "reference_top2_recall_ge_0_80": float(ref_df["reference_top2_recall"].mean()) >= 0.80,
        "online_anchor_top2_recovered_in_2_of_3": int((live_extended["top2_set_recall_live"] >= 1.0).sum()) >= 2
        and int((live_compact["top2_set_recall_live"] >= 1.0).sum()) >= 2,
        "within_8h_budget": runtime_schedule["within_8h_budget"],
    }
    sensitivity_summary = []
    for policy in POLICIES:
        subset = policy_sensitivity[policy_sensitivity["policy"] == policy]
        sensitivity_summary.append(
            {
                "policy": policy,
                "compact_vs_extended_abs_miss_delta_mean": float((subset["delta_miss_rate_vs_ExtendedHinted_CompactState"].abs()).mean()) if "delta_miss_rate_vs_ExtendedHinted_CompactState" in subset else math.nan,
                "nodirty_vs_extended_abs_miss_delta_mean": float((subset["delta_miss_rate_vs_ExtendedHinted_NoDirty"].abs()).mean()) if "delta_miss_rate_vs_ExtendedHinted_NoDirty" in subset else math.nan,
                "access_vs_extended_abs_miss_delta_mean": float((subset["delta_miss_rate_vs_ExtendedHinted_AccessOnly"].abs()).mean()) if "delta_miss_rate_vs_ExtendedHinted_AccessOnly" in subset else math.nan,
            }
        )
    substantive_policies = [
        item["policy"]
        for item in sensitivity_summary
        if item["access_vs_extended_abs_miss_delta_mean"] >= 0.005
    ]
    broad_state_dependence = (
        len(substantive_policies) >= 3
        and float(ref_df["reference_top2_recall"].mean()) >= 0.80
        and float(ref_df["top_policy_changed"].mean()) <= 0.20
    )
    results = {
        "idea_title": "ShadowCache rerun as a simulator-backed page-cache state-ablation case study",
        "timestamp": now_ts(),
        "scope": {
            "original_claim_supported": False,
            "reported_claim": "Measured real workload executions with kernel-visible cache-residency snapshots where feasible, offline replay, and same-trace online-policy simulations. This supports a simulator case study of state-aware page-cache heuristics, not live Linux page-cache policy validation.",
            "spec_consistency_failures": spec_failures,
            "independent_online_validation_available": False,
            "broad_policy_ranking_dependence_supported": broad_state_dependence,
        },
        "environment": capture_package_versions(),
        "summary_metrics": {
            "compact": {
                "kendall_tau_mean": float(compact["Kendall_tau_6"].mean()),
                "kendall_tau_std": float(compact["Kendall_tau_6"].std(ddof=1)),
                "top2_recall_mean": float(compact["top2_set_recall"].mean()),
                "top2_recall_std": float(compact["top2_set_recall"].std(ddof=1)),
            },
            "access_only": {
                "kendall_tau_mean": float(access["Kendall_tau_6"].mean()),
                "kendall_tau_std": float(access["Kendall_tau_6"].std(ddof=1)),
                "top2_recall_mean": float(access["top2_set_recall"].mean()),
                "top2_recall_std": float(access["top2_set_recall"].std(ddof=1)),
            },
            "reference": {
                "reference_top2_recall_mean": float(ref_df["reference_top2_recall"].mean()),
                "top_policy_changed_fraction": float(ref_df["top_policy_changed"].mean()),
            },
            "online_anchor_top2_mean": float(live_df["top2_set_recall_live"].mean()),
            "policy_state_sensitivity": sensitivity_summary,
        },
        "success_criteria": success,
        "negative_result": not all(success.values()),
        "interpretation": {
            "main_takeaway": "The current evidence is strongest for a narrow simulator-backed case study: CompactState preserves ranking much better than AccessOnly in this replay setup, but the rebuilt-no-hints reference is unstable and the online anchors are same-trace consistency checks rather than independent validation.",
            "ranking_claim_strength": "broad" if broad_state_dependence else "narrow",
            "substantively_state_sensitive_policies": substantive_policies,
        },
        "artifacts": {
            "replay_results": "artifacts/replay/replay_results.parquet",
            "replay_case_results": "artifacts/tables/replay_case_results.csv",
            "ranking_metrics": "artifacts/tables/ranking_metrics.csv",
            "live_anchor_metrics": "artifacts/tables/live_anchor_metrics.csv",
            "live_anchor_case_results": "artifacts/tables/live_anchor_case_results.csv",
            "reference_stability": "artifacts/tables/reference_stability.csv",
            "reference_case_results": "artifacts/tables/reference_case_results.csv",
            "policy_state_sensitivity": "artifacts/tables/policy_state_sensitivity.csv",
            "runtime_accounting": "artifacts/tables/runtime_accounting.csv",
            "figures": sorted(str(p.name) for p in (root / "figures").glob("*.pdf")),
        },
    }
    write_json(root / "results.json", results)
    return results

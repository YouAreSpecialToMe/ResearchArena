from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import FIGURE_ROOT, REPLAY_ROOT, ROOT, bootstrap_ci, mean_std, read_json, wilcoxon_signed_rank, write_json
from .verification import verify_or_raise


def gather_run_results() -> pd.DataFrame:
    rows = []
    for path in ROOT.glob("exp/*/runs/*/results.json"):
        payload = read_json(path)
        rows.append(
            {
                "experiment": payload["experiment"],
                "workload_family": payload["workload_family"],
                "cache_budget": payload["cache_budget"],
                "method": payload["method"],
                "seed": payload["seed"],
                **payload["metrics"],
            }
        )
    return pd.DataFrame(rows)


def aggregate_results() -> dict[str, Any]:
    verify_or_raise()
    df = gather_run_results()
    if df.empty:
        return {}
    grouped = []
    for keys, sub in df.groupby(["workload_family", "cache_budget", "method"]):
        row = {
            "workload_family": keys[0],
            "cache_budget": keys[1],
            "method": keys[2],
        }
        for metric in [
            "worst_tenant_slowdown",
            "aggregate_throughput_proxy",
            "fairness_jain",
            "shared_regret_per_10000_refs",
            "debt_harm_pearson",
            "debt_harm_spearman",
            "debt_harm_mae",
            "controller_changes_per_10000_refs",
            "runtime_sec",
            "peak_rss_mb",
        ]:
            row[metric] = mean_std(sub[metric].tolist())
        grouped.append(row)
    stats = paired_stats(df)
    verdicts = hypothesis_checks(df)
    preparation = preparation_summary()
    validation_scope = validation_scope_summary()
    external = external_validity_summary(df)
    method_equivalence = equivalence_summary(df)
    payload = {
        "preparation": preparation,
        "validation_scope": validation_scope,
        "aggregated": grouped,
        "paired_statistics": stats,
        "hypothesis_evaluation": verdicts,
        "ablation_notes": method_equivalence,
        "external_validity": external,
        "claim_assessment": claim_assessment(verdicts, validation_scope, external, preparation, method_equivalence),
    }
    write_json(ROOT / "results.json", payload)
    return payload


def paired_stats(df: pd.DataFrame) -> dict[str, Any]:
    rows = {}
    comparisons = [("ShareArb", "pCache-Account"), ("ShareArb", "ShareArb-NoDebt")]
    for left, right in comparisons:
        merged = pd.merge(
            df[df["method"] == left],
            df[df["method"] == right],
            on=["workload_family", "cache_budget", "seed"],
            suffixes=("_left", "_right"),
        )
        rows[f"{left}_vs_{right}"] = {}
        for metric in ["worst_tenant_slowdown", "aggregate_throughput_proxy", "shared_regret_per_10000_refs", "fairness_jain"]:
            deltas = (merged[f"{metric}_left"] - merged[f"{metric}_right"]).tolist()
            rows[f"{left}_vs_{right}"][metric] = {
                "bootstrap_95ci": bootstrap_ci(deltas),
                "wilcoxon_p": wilcoxon_signed_rank(merged[f"{metric}_left"].tolist(), merged[f"{metric}_right"].tolist()),
            }
    return rows


def hypothesis_checks(df: pd.DataFrame) -> dict[str, Any]:
    overlap_families = ["OverlapShift-2T", "ScanVsLoop-2T", "SQLiteTraceMix-2T", "SQLiteTraceMix-3T"]
    share = df[df["method"] == "ShareArb"]
    pcache = df[df["method"] == "pCache-Account"]
    merged = pd.merge(
        share,
        pcache,
        on=["workload_family", "cache_budget", "seed"],
        suffixes=("_share", "_pcache"),
    )
    overlap = merged[merged["workload_family"].isin(overlap_families)]
    disjoint = merged[merged["workload_family"] == "DisjointPhase-2T"]
    slowdown_improvement = float(
        ((overlap["worst_tenant_slowdown_pcache"] - overlap["worst_tenant_slowdown_share"]) / overlap["worst_tenant_slowdown_pcache"]).mean()
    )
    throughput_delta = float(
        ((overlap["aggregate_throughput_proxy_share"] - overlap["aggregate_throughput_proxy_pcache"]) / overlap["aggregate_throughput_proxy_pcache"]).mean()
    )
    disjoint_slow = float(
        np.abs((disjoint["worst_tenant_slowdown_share"] - disjoint["worst_tenant_slowdown_pcache"]) / disjoint["worst_tenant_slowdown_pcache"]).mean()
    )
    disjoint_thr = float(
        np.abs((disjoint["aggregate_throughput_proxy_share"] - disjoint["aggregate_throughput_proxy_pcache"]) / disjoint["aggregate_throughput_proxy_pcache"]).mean()
    )
    positive_debt = int(
        share.groupby(["workload_family", "cache_budget"])["debt_harm_pearson"].mean().reindex(
            pd.MultiIndex.from_product([overlap_families, ["tight", "medium", "loose"]])
        ).fillna(0.0).gt(0).sum()
    )
    supported = slowdown_improvement >= 0.08 and throughput_delta >= -0.03 and disjoint_slow <= 0.03 and disjoint_thr <= 0.03 and positive_debt >= 9
    return {
        "supported": supported,
        "slowdown_improvement_vs_pcache": slowdown_improvement,
        "throughput_delta_vs_pcache": throughput_delta,
        "disjoint_slowdown_delta": disjoint_slow,
        "disjoint_throughput_delta": disjoint_thr,
        "positive_debt_budget_pairs": positive_debt,
    }


def preparation_summary() -> dict[str, Any]:
    traces = {}
    for path in sorted((ROOT / "traces").glob("*__seed11.json")):
        payload = read_json(path)
        if payload["meta"].get("generator") == "sqlite_external_validation":
            continue
        traces[payload["family"]] = {
            "references_per_tenant": payload["meta"]["references_per_tenant"],
            "event_count": len(payload["events"]),
            "realized_overlap_ratio": payload["meta"]["realized_overlap_ratio"],
            "shared_pages": payload["meta"]["shared_pages"],
            "top_shared_page_ranges": payload["meta"]["top_shared_page_ranges"],
        }
    return {"trace_lengths_seed11": traces}


def validation_scope_summary() -> dict[str, Any]:
    live_path = ROOT / "live_validation" / "results.json"
    if not live_path.exists():
        return {"mode": "unknown"}
    payload = read_json(live_path)
    return {
        "mode": payload.get("status", "unknown"),
        "reason": payload.get("reason"),
        "scope": payload.get("scope"),
    }


def external_validity_summary(df: pd.DataFrame) -> dict[str, Any]:
    external_df = df[df["experiment"] == "external_validation"].copy()
    if external_df.empty:
        return {"status": "missing"}
    merged = pd.merge(
        external_df[external_df["method"] == "ShareArb"],
        external_df[external_df["method"] == "pCache-Account"],
        on=["workload_family", "cache_budget", "seed"],
        suffixes=("_share", "_pcache"),
    )
    rows = []
    sign_agreement = 0
    for _, row in merged.iterrows():
        slowdown_delta = row["worst_tenant_slowdown_pcache"] - row["worst_tenant_slowdown_share"]
        throughput_delta = row["aggregate_throughput_proxy_share"] - row["aggregate_throughput_proxy_pcache"]
        if slowdown_delta != 0:
            sign_agreement += 1 if slowdown_delta > 0 else 0
        rows.append(
            {
                "condition": f"{row['workload_family']}::{row['cache_budget']}::seed{int(row['seed'])}",
                "replay_winner": "ShareArb" if slowdown_delta > 0 else "pCache-Account",
                "sign_agreement": slowdown_delta > 0,
                "slowdown_delta": float(slowdown_delta),
                "throughput_delta": float(throughput_delta),
            }
        )
    return {
        "status": "auxiliary_replay_check",
        "description": "Auxiliary replay-side perturbation check on raw and burst-perturbed captured SQLite streams. This is reported separately and does not replace the skipped live sanity check.",
        "rows": rows,
        "sharearb_wins": int(sum(1 for row in rows if row["replay_winner"] == "ShareArb")),
        "condition_count": len(rows),
    }


def equivalence_summary(df: pd.DataFrame) -> dict[str, Any]:
    left = df[df["method"] == "ShareArb-NoDebt"]
    right = df[df["method"] == "pCache-Account+Policy"]
    merged = pd.merge(left, right, on=["workload_family", "cache_budget", "seed"], suffixes=("_nodebt", "_policy"))
    if merged.empty:
        return {"matched_runs": 0}
    cells = []
    for metric in ["worst_tenant_slowdown", "aggregate_throughput_proxy", "shared_regret_per_10000_refs", "fairness_jain"]:
        same = np.isclose(merged[f"{metric}_nodebt"], merged[f"{metric}_policy"], atol=1e-12, rtol=1e-9)
        cells.append({"metric": metric, "identical_fraction": float(np.mean(same))})
    return {
        "matched_runs": int(len(merged)),
        "metric_identity": cells,
    }


def claim_assessment(
    verdicts: dict[str, Any],
    validation_scope: dict[str, Any],
    external: dict[str, Any],
    preparation: dict[str, Any],
    equivalence: dict[str, Any],
) -> dict[str, Any]:
    trace_lengths = preparation["trace_lengths_seed11"]
    slowdown_gain = verdicts["slowdown_improvement_vs_pcache"]
    if slowdown_gain > 0:
        summary = (
            "In the revised shortened-trace replay regime, ShareArb improves worst-tenant slowdown over pCache-Account on average, "
            "but the overall claim is still unsupported because debt-harm alignment misses the planned threshold and the artifact is "
            "scoped to replay-only evidence after the live sanity check proved infeasible."
        )
    else:
        summary = (
            "Negative main result: ShareArb does not improve worst-tenant slowdown over pCache-Account on this matrix. "
            "Observed wins are regime-specific at best and do not satisfy the planned support threshold."
        )
    return {
        "supported": verdicts["supported"],
        "summary": summary,
        "slowdown_improvement_vs_pcache": slowdown_gain,
        "positive_debt_budget_pairs": verdicts["positive_debt_budget_pairs"],
        "validation_scope": validation_scope,
        "external_sharearb_wins": external.get("sharearb_wins"),
        "trace_length_regime": {
            "synthetic_refs_per_tenant": int(trace_lengths["OverlapShift-2T"]["references_per_tenant"]["0"]),
            "sqlite_refs_per_tenant": int(trace_lengths["SQLiteTraceMix-2T"]["references_per_tenant"]["0"]),
            "matches_plan": int(trace_lengths["OverlapShift-2T"]["references_per_tenant"]["0"]) == 80000
            and int(trace_lengths["SQLiteTraceMix-2T"]["references_per_tenant"]["0"]) == 60000,
        },
        "sharearb_nodebt_distinct_from_policy": equivalence,
        "external_validity_status": external.get("status", "missing"),
    }


def make_figures() -> None:
    df = gather_run_results()
    if df.empty:
        return
    FIGURE_ROOT.mkdir(exist_ok=True, parents=True)
    main_figure(df)
    budget_figure(df)
    ablation_heatmap(df)
    mechanism_scatter()
    live_table()
    runtime_table(df)


def main_figure(df: pd.DataFrame) -> None:
    medium = df[df["cache_budget"] == "medium"]
    families = ["OverlapShift-2T", "ScanVsLoop-2T", "SQLiteTraceMix-2T", "SQLiteTraceMix-3T", "DisjointPhase-2T"]
    methods = ["PrivateOnly-Utility", "pCache-Account", "pCache-Account+Policy", "ShareArb"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.18
    x = np.arange(len(families))
    for idx, method in enumerate(methods):
        sub = medium[medium["method"] == method]
        slow = []
        slow_err = []
        thr = []
        thr_err = []
        for fam in families:
            fam_sub = sub[sub["workload_family"] == fam]
            slow_values = fam_sub["worst_tenant_slowdown"].tolist()
            thr_values = fam_sub["aggregate_throughput_proxy"].tolist()
            slow_ci = bootstrap_ci(slow_values, n_resamples=10_000)
            thr_ci = bootstrap_ci(thr_values, n_resamples=10_000)
            slow.append(float(np.mean(slow_values)) if slow_values else math.nan)
            thr.append(float(np.mean(thr_values)) if thr_values else math.nan)
            slow_err.append(max(0.0, slow_ci["high"] - slow_ci["mid"]))
            thr_err.append(max(0.0, thr_ci["high"] - thr_ci["mid"]))
        axes[0].bar(x + idx * width, slow, width, label=method, yerr=slow_err, capsize=3)
        axes[1].bar(x + idx * width, thr, width, label=method, yerr=thr_err, capsize=3)
    for ax, title, ylabel in [
        (axes[0], "Worst-Tenant Slowdown", "Slowdown"),
        (axes[1], "Aggregate Throughput Proxy", "Refs / latency-us"),
    ]:
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(families, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "main_results.png", dpi=200)
    plt.close(fig)


def budget_figure(df: pd.DataFrame) -> None:
    families = ["OverlapShift-2T", "ScanVsLoop-2T", "SQLiteTraceMix-2T", "SQLiteTraceMix-3T"]
    methods = ["pCache-Account", "pCache-Account+Policy", "ShareArb-NoDebt", "ShareArb"]
    budgets = ["tight", "medium", "loose"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, family in zip(axes.flat, families):
        sub = df[df["workload_family"] == family]
        for method in methods:
            vals = [sub[(sub["method"] == method) & (sub["cache_budget"] == budget)]["worst_tenant_slowdown"].mean() for budget in budgets]
            ax.plot(budgets, vals, marker="o", label=method)
        ax.set_title(family)
        ax.set_ylabel("Worst slowdown")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "budget_sensitivity.png", dpi=200)
    plt.close(fig)


def ablation_heatmap(df: pd.DataFrame) -> None:
    base = df[df["method"] == "ShareArb"]
    mapping = {
        "NoDebt": "ShareArb-NoDebt",
        "UniformSRV": "UniformSRV",
        "UnitCost": "ShareArb-UnitCost",
        "short half-life": "ShareArb-HalfLife0.5",
        "long half-life": "ShareArb-HalfLife2.0",
        "NoReduction": "NoReduction",
    }
    metrics = ["worst_tenant_slowdown", "aggregate_throughput_proxy", "fairness_jain", "shared_regret_per_10000_refs", "controller_changes_per_10000_refs"]
    data = []
    labels = []
    for label, method in mapping.items():
        sub = df[df["method"] == method]
        merged = pd.merge(base, sub, on=["workload_family", "cache_budget", "seed"], suffixes=("_base", "_alt"))
        if merged.empty:
            values = [math.nan] * len(metrics)
        else:
            values = [(merged[f"{metric}_alt"] - merged[f"{metric}_base"]).mean() for metric in metrics]
        data.append(values)
        labels.append(label)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(np.array(data, dtype=float), aspect="auto", cmap="coolwarm")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(["slowdown gain", "throughput", "fairness", "regret", "stability"], rotation=20, ha="right")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "ablation_heatmap.png", dpi=200)
    plt.close(fig)


def mechanism_scatter() -> None:
    rows = []
    for path in ROOT.glob("exp/*/runs/*/epoch_metrics.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["method"] == "ShareArb":
                rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows)
    pearson = float(df["debt"].corr(df["realized_harm_next"], method="pearson"))
    spearman = float(df["debt"].corr(df["realized_harm_next"], method="spearman"))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["debt"], df["realized_harm_next"], s=10, alpha=0.5)
    ax.set_xlabel("Charged debt")
    ax.set_ylabel("Realized downstream harm")
    ax.set_title("Debt vs realized downstream harm")
    ax.text(
        0.03,
        0.97,
        f"Pearson={pearson:.3f}\nSpearman={spearman:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "mechanism_scatter.png", dpi=200)
    plt.close(fig)


def live_table() -> None:
    external_path = ROOT / "exp" / "external_validation" / "results.json"
    if external_path.exists():
        payload = read_json(ROOT / "results.json") if (ROOT / "results.json").exists() else {}
        rows = payload.get("external_validity", {}).get("rows", [])
        if rows:
            fig, ax = plt.subplots(figsize=(10, max(2, 0.45 * len(rows) + 1)))
            ax.axis("off")
            columns = ["condition", "replay_winner", "sign_agreement", "slowdown_delta", "throughput_delta"]
            cell_text = [[row.get(col, "") for col in columns] for row in rows]
            table = ax.table(cellText=cell_text, colLabels=columns, loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            ax.set_title("External-validity replay check", pad=12)
            fig.tight_layout()
            fig.savefig(FIGURE_ROOT / "live_validation_table.png", dpi=200)
            plt.close(fig)
            return
    table_path = ROOT / "live_validation" / "results.json"
    if not table_path.exists():
        return
    payload = read_json(table_path)
    rows = payload.get("rows", [])
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "Live validation skipped: required controls unavailable", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(FIGURE_ROOT / "live_validation_table.png", dpi=200)
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(10, max(2, 0.5 * len(rows) + 1)))
    ax.axis("off")
    columns = ["condition", "replay_winner", "live_winner", "sign_agreement", "effect_range"]
    cell_text = [[row.get(col, "") for col in columns] for row in rows]
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "live_validation_table.png", dpi=200)
    plt.close(fig)


def runtime_table(df: pd.DataFrame) -> None:
    sections = []
    sections.append(["trace_capture", "observed", "see calibration + traces", "CPU-only"])
    sections.append(["primary_replay", float(df[df["experiment"] == "primary"]["runtime_sec"].sum()), "sum of per-run runtime_sec", "no GPU"])
    sections.append(["ablations", float(df[df["experiment"] == "ablations"]["runtime_sec"].sum()), "sum of per-run runtime_sec", "no GPU"])
    sections.append(["oracle", float(df[df["experiment"] == "oracle"]["runtime_sec"].sum()), "sum of per-run runtime_sec", "no GPU"])
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    table = ax.table(cellText=sections, colLabels=["stage", "observed_sec", "note", "gpu"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "runtime_accounting.png", dpi=200)
    plt.close(fig)

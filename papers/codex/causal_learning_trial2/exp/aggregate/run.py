from __future__ import annotations

from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from exp.shared.common import FIGURES_DIR, ensure_dir, load_json, save_json, set_thread_env
from exp.shared.metrics import bootstrap_metric_summary, calibration_metrics, paired_summary


def save_fig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = ["_".join(str(part) for part in col if part).strip("_") if isinstance(col, tuple) else str(col) for col in out.columns]
    return out


def reliability_table(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["bin", "pred_mean", "emp_mean", "count"])
    work = df.copy()
    work["bin"] = pd.cut(work["predicted"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    out = (
        work.groupby("bin", observed=False)
        .agg(pred_mean=("predicted", "mean"), emp_mean=("empirical", "mean"), count=("predicted", "size"))
        .reset_index()
    )
    out["bin"] = out["bin"].astype(str)
    return out


def append_regret(core: pd.DataFrame) -> pd.DataFrame:
    pacer = core[core["method"] == "pacer_cert"][["instance_id", "unused_budget", "directed_f1", "weight_regime", "graph_family", "p"]]
    forced = core[core["method"] == "pacer_full_budget"][["instance_id", "directed_f1"]].rename(columns={"directed_f1": "forced_directed_f1"})
    regret = pacer.merge(forced, on="instance_id", how="left")
    regret["post_stop_regret"] = regret["forced_directed_f1"] - regret["directed_f1"]
    return regret


def grouped_metric_summary(df: pd.DataFrame, group_col: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for group_value, grp in df.groupby(group_col):
        row = {group_col: group_value}
        for metric_idx, metric in enumerate(metrics):
            if metric not in grp:
                continue
            summary = bootstrap_metric_summary(grp[metric].tolist(), seed=metric_idx)
            row[f"{metric}_mean"] = summary["mean"]
            row[f"{metric}_std"] = summary["std"]
            row[f"{metric}_ci95_low"] = summary["ci95"][0]
            row[f"{metric}_ci95_high"] = summary["ci95"][1]
        row["runtime_seconds_median"] = float(grp["runtime_seconds"].median()) if "runtime_seconds" in grp else np.nan
        row["runtime_seconds_p90"] = float(grp["runtime_seconds"].quantile(0.9)) if "runtime_seconds" in grp else np.nan
        if "peak_rss_mb" in grp:
            row["peak_rss_mb_median"] = float(grp["peak_rss_mb"].median())
            row["peak_rss_mb_p90"] = float(grp["peak_rss_mb"].quantile(0.9))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def bootstrap_calibration_summary(df: pd.DataFrame, label_col: str, score_col: str, seed: int = 0, n_boot: int = 1000) -> dict[str, float | list[float] | None]:
    if df.empty:
        return {}
    base = calibration_metrics(df[label_col].tolist(), df[score_col].tolist())
    rng = np.random.default_rng(seed)
    boot_metrics = {key: [] for key in base}
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        sub = df.iloc[idx]
        metrics = calibration_metrics(sub[label_col].tolist(), sub[score_col].tolist())
        for key, value in metrics.items():
            if value is not None and not np.isnan(value):
                boot_metrics[key].append(float(value))
    out: dict[str, float | list[float] | None] = {"count": int(len(df))}
    for key, value in base.items():
        out[key] = None if value is None or np.isnan(value) else float(value)
        vals = sorted(boot_metrics[key])
        if vals:
            out[f"{key}_ci95"] = [float(vals[int(0.025 * len(vals))]), float(vals[int(0.975 * len(vals)) - 1])]
        else:
            out[f"{key}_ci95"] = None
    return out


def agreement_summary(values: list[int], seed: int = 0) -> dict[str, float | list[float]]:
    summary = bootstrap_metric_summary([float(v) for v in values], seed=seed)
    return {
        "rate": summary["mean"],
        "std": summary["std"],
        "ci95": summary["ci95"],
        "count": len(values),
    }


def make_figure1(checkpoints: pd.DataFrame) -> None:
    weak = checkpoints[checkpoints["weight_regime"] == "weak"].copy()
    keep_methods = ["pacer_cert", "pacer_no_d", "aoed_lite", "git", "random_active", "fges_only"]
    weak = weak[weak["method"].isin(keep_methods)]
    summary_rows = []
    for (method, cumulative_samples), grp in weak.groupby(["method", "cumulative_samples"]):
        stats = bootstrap_metric_summary(grp["directed_f1"].tolist(), seed=int(cumulative_samples))
        summary_rows.append(
            {
                "method": method,
                "cumulative_samples": cumulative_samples,
                "mean": stats["mean"],
                "ci_low": stats["ci95"][0],
                "ci_high": stats["ci95"][1],
            }
        )
    summary = pd.DataFrame(summary_rows)
    plt.figure(figsize=(8, 5))
    for method in keep_methods:
        sub = summary[summary["method"] == method].sort_values("cumulative_samples")
        if sub.empty:
            continue
        plt.plot(sub["cumulative_samples"], sub["mean"], marker="o", label=method)
        plt.fill_between(sub["cumulative_samples"], sub["ci_low"], sub["ci_high"], alpha=0.15)
    plt.xlabel("Interventional samples")
    plt.ylabel("Directed F1")
    plt.title("Figure 1: Weak-regime directed F1 vs interventional samples")
    plt.legend()
    save_fig(FIGURES_DIR / "figure1_directed_f1_vs_samples.png")


def make_figure2(calib_tuples: pd.DataFrame) -> pd.DataFrame:
    rel = reliability_table(calib_tuples)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.plot(rel["pred_mean"], rel["emp_mean"], marker="o")
    plt.xlabel("Predicted D(e|I,m)")
    plt.ylabel("Empirical resolvability")
    plt.title("Reliability")
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=calib_tuples, x="predicted", y="empirical", hue="method", s=35)
    plt.xlabel("Predicted D(e|I,m)")
    plt.ylabel("Empirical resolvability")
    plt.title("Scatter")
    save_fig(FIGURES_DIR / "figure2_reliability_scatter.png")
    return rel


def make_figure3(regret: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4.5))
    sns.scatterplot(data=regret, x="unused_budget", y="post_stop_regret", hue="graph_family", style="p", s=70)
    plt.xlabel("Unused budget")
    plt.ylabel("Post-stop regret")
    plt.title("Figure 3: Unused budget vs post-stop regret")
    save_fig(FIGURES_DIR / "figure3_unused_budget_vs_regret.png")


def make_figure4(trust_q: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ess = trust_q[trust_q["stratum_type"] == "ess_bucket"]
    sns.barplot(data=ess, x="stratum_value", y="auroc", ax=ax1)
    ax1.set_xlabel("ESS bucket")
    ax1.set_ylabel("q_e AUROC")
    ax1.set_title("ESS strata")
    ax2 = fig.add_subplot(1, 2, 2)
    nh = trust_q[trust_q["stratum_type"] == "neighborhood_bucket"]
    sns.barplot(data=nh, x="stratum_value", y="auroc", ax=ax2)
    ax2.set_xlabel("Neighborhood size")
    ax2.set_ylabel("q_e AUROC")
    ax2.set_title("Neighborhood strata")
    save_fig(FIGURES_DIR / "figure4_trust_strata.png")


def build_table2(calib_tuples: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({"metric": "D calibration", "subset": "all", **bootstrap_calibration_summary(calib_tuples, "empirical_label", "predicted", seed=11)})
    if not oracle.empty:
        rows.append({"metric": "q calibration", "subset": "all", **bootstrap_calibration_summary(oracle, "oracle_label", "q_e", seed=17)})
        for method, grp in oracle.groupby("method"):
            rows.append({"metric": "q calibration", "subset": method, **bootstrap_calibration_summary(grp, "oracle_label", "q_e", seed=23)})
    return pd.DataFrame(rows)


def build_headline_summary(core: pd.DataFrame, regret: pd.DataFrame, oracle: pd.DataFrame, direct: pd.DataFrame, calib_tuples: pd.DataFrame) -> tuple[dict, str]:
    weak = core[core["weight_regime"] == "weak"].copy()
    weak_regret = regret[regret["weight_regime"] == "weak"].copy()
    pacer_q = oracle[oracle["method"] == "pacer_cert"].copy()
    pacer_q_metrics = calibration_metrics(pacer_q["oracle_label"].tolist(), pacer_q["q_e"].tolist()) if not pacer_q.empty else {}
    d_spearman = float(calib_tuples[["predicted", "empirical"]].corr(method="spearman").iloc[0, 1]) if len(calib_tuples) > 1 else np.nan
    weak_final = weak[weak["method"].isin(["pacer_cert", "aoed_lite", "random_active"])].groupby("method")["directed_f1"].mean().to_dict()
    top1 = agreement_summary(direct["rank_agreement_top1"].astype(int).tolist(), seed=31) if not direct.empty else {}
    stop_agree = agreement_summary(direct["stop_continue_agreement"].astype(int).tolist(), seed=37) if not direct.empty else {}
    weak_regret_summary = bootstrap_metric_summary(weak_regret["post_stop_regret"].tolist(), seed=41) if not weak_regret.empty else {"mean": None, "std": None, "ci95": None}
    success_flags = {
        "preregistered_q_auroc_gt_0p7": bool((pacer_q_metrics.get("auroc") or 0.0) > 0.7),
        "preregistered_d_spearman_positive": bool(not np.isnan(d_spearman) and d_spearman > 0.0),
        "preregistered_direct_majority_top1": bool((top1.get("rate") or 0.0) > 0.5),
        "preregistered_weak_slice_regret_lte_0p02": bool((weak_regret_summary.get("mean") or 1.0) <= 0.02),
    }
    negative_lines = [
        "# Negative-result framing",
        "",
        "PACER-Cert failed the preregistered calibration and weak-slice success criteria in this CPU-only benchmark.",
        f"- Weak-slice final directed F1 means: PACER-Cert {weak_final.get('pacer_cert', float('nan')):.3f}, AOED-lite {weak_final.get('aoed_lite', float('nan')):.3f}, random active {weak_final.get('random_active', float('nan')):.3f}.",
        f"- Weak-slice post-stop regret to forced-full-budget PACER-Cert: mean {weak_regret_summary.get('mean', float('nan')):.3f} with 95% bootstrap CI [{weak_regret_summary.get('ci95', [float('nan'), float('nan')])[0]:.3f}, {weak_regret_summary.get('ci95', [float('nan'), float('nan')])[1]:.3f}].",
        f"- PACER-Cert q_e calibration on oracle labels: AUROC {pacer_q_metrics.get('auroc', float('nan')):.3f}, Brier {pacer_q_metrics.get('brier', float('nan')):.3f}, Spearman {pacer_q_metrics.get('spearman', float('nan')):.3f}.",
        f"- Overall D(e|I,m) versus empirical resolvability Spearman: {d_spearman:.3f}.",
        f"- Direct-lookahead agreement: top-1 action agreement {int((top1.get('rate') or 0.0) * top1.get('count', 0))}/{top1.get('count', 0)}, stop-vs-continue agreement {int((stop_agree.get('rate') or 0.0) * stop_agree.get('count', 0))}/{stop_agree.get('count', 0)}.",
        "- All trust strata remain untrustworthy under the proposal thresholds; the certificate should not be presented as reliable.",
        "- PACER-no-D remains far cheaper than PACER-Cert in the current implementation, so runtime should not be interpreted as a fair compute-matched ablation.",
    ]
    return (
        {
            "weak_slice_final_directed_f1": weak_final,
            "weak_slice_post_stop_regret": weak_regret_summary,
            "pacer_cert_q_metrics": pacer_q_metrics,
            "d_spearman": d_spearman,
            "direct_lookahead_top1": top1,
            "direct_lookahead_stop_continue": stop_agree,
            "success_flags": success_flags,
        },
        "\n".join(negative_lines) + "\n",
    )


def provenance_rows(root: Path, table_paths: list[Path], appendix_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in table_paths:
        rows.append(
            {
                "artifact_path": str(path),
                "artifact_type": path.suffix.lstrip("."),
                "generator": str((root / "exp" / "aggregate" / "run.py").resolve()),
                "source_inputs": "exp/core_benchmark/benchmark_rollouts.csv; exp/core_benchmark/checkpoint_records.csv; exp/calibration/*.csv; exp/calibration/results.json",
            }
        )
    for path in appendix_paths:
        rows.append(
            {
                "artifact_path": str(path),
                "artifact_type": path.suffix.lstrip("."),
                "generator": str((root / "exp" / "calibration" / "run.py").resolve()),
                "source_inputs": "exp/calibration/saved_states/pacer_cert/*/results.json; exp/calibration/saved_states/pacer_cert/*/certificate_*.json",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    set_thread_env()
    root = Path(__file__).resolve().parents[2]
    ensure_dir(Path(__file__).resolve().parent / "logs")
    summaries_dir = ensure_dir(root / "artifacts" / "summaries")
    certs_dir = ensure_dir(root / "artifacts" / "certificates")
    core = pd.read_csv(root / "exp" / "core_benchmark" / "benchmark_rollouts.csv")
    checkpoints = pd.read_csv(root / "exp" / "core_benchmark" / "checkpoint_records.csv")
    calib_tuples = pd.read_csv(root / "exp" / "calibration" / "calibration_tuples.csv")
    oracle = pd.read_csv(root / "exp" / "calibration" / "oracle_labels.csv")
    trust_d = pd.read_csv(root / "exp" / "calibration" / "trust_diagnostics_d.csv")
    trust_q = pd.read_csv(root / "exp" / "calibration" / "trust_diagnostics_q.csv")
    direct = pd.read_csv(root / "exp" / "calibration" / "direct_lookahead.csv")
    calib_summary = load_json(root / "exp" / "calibration" / "results.json")
    memory_probe = pd.read_csv(root / "exp" / "memory_probe" / "memory_probe.csv") if (root / "exp" / "memory_probe" / "memory_probe.csv").exists() else pd.DataFrame()

    make_figure1(checkpoints)
    reliability = make_figure2(calib_tuples)
    regret = append_regret(core)
    make_figure3(regret)
    make_figure4(trust_q)

    metrics = ["directed_f1", "shd", "unused_budget", "runtime_seconds", "auc_directed_f1"]
    if "peak_rss_mb" in core.columns:
        metrics.append("peak_rss_mb")
    core_summary = grouped_metric_summary(core, "method", metrics)
    if not memory_probe.empty:
        probe_cols = memory_probe[["method", "peak_rss_mb", "runtime_seconds"]].rename(
            columns={"peak_rss_mb": "peak_rss_mb_probe", "runtime_seconds": "runtime_seconds_probe"}
        )
        core_summary = core_summary.merge(probe_cols, on="method", how="left")
    core_summary.to_csv(root / "exp" / "core_benchmark" / "summary.csv", index=False)
    core_summary.to_csv(summaries_dir / "table1.csv", index=False)
    (summaries_dir / "table1.md").write_text(core_summary.to_markdown(index=False))

    table2 = build_table2(calib_tuples, oracle)
    table2.to_csv(summaries_dir / "table2.csv", index=False)
    (summaries_dir / "table2.md").write_text(table2.to_markdown(index=False))
    reliability.to_csv(summaries_dir / "reliability.csv", index=False)
    (summaries_dir / "reliability.md").write_text(reliability.to_markdown(index=False))

    comparisons = {}
    for baseline in ["random_active", "git", "aoed_lite", "pacer_no_d"]:
        left = core[core["method"] == "pacer_cert"].sort_values("instance_id")
        right = core[core["method"] == baseline].sort_values("instance_id")
        comparisons[baseline] = {
            "auc_directed_f1": paired_summary(left["auc_directed_f1"].tolist(), right["auc_directed_f1"].tolist()),
            "final_directed_f1": paired_summary(left["directed_f1"].tolist(), right["directed_f1"].tolist()),
            "shd": paired_summary(right["shd"].tolist(), left["shd"].tolist()),
            "unused_budget": paired_summary(left["unused_budget"].tolist(), right["unused_budget"].tolist()),
            "runtime_seconds": paired_summary(right["runtime_seconds"].tolist(), left["runtime_seconds"].tolist()),
        }
        if "peak_rss_mb" in left.columns and "peak_rss_mb" in right.columns:
            comparisons[baseline]["peak_rss_mb"] = paired_summary(right["peak_rss_mb"].tolist(), left["peak_rss_mb"].tolist())

    appendix_src_csv = root / "exp" / "calibration" / "appendix" / "appendix_certificate_table.csv"
    appendix_src_md = root / "exp" / "calibration" / "appendix" / "appendix_certificate_table.md"
    appendix_manifest = load_json(root / "exp" / "calibration" / "appendix" / "appendix_certificate_manifest.json")
    appendix_dst_csv = certs_dir / "appendix_certificate_example.csv"
    appendix_dst_md = certs_dir / "appendix_certificate_example.md"
    shutil.copyfile(appendix_src_csv, appendix_dst_csv)
    shutil.copyfile(appendix_src_md, appendix_dst_md)

    headline_summary, headline_md = build_headline_summary(core, regret, oracle, direct, calib_tuples)
    (summaries_dir / "negative_result_summary.md").write_text(headline_md)

    prov = provenance_rows(
        root,
        [
            summaries_dir / "table1.csv",
            summaries_dir / "table1.md",
            summaries_dir / "table2.csv",
            summaries_dir / "table2.md",
            summaries_dir / "reliability.csv",
            summaries_dir / "reliability.md",
            summaries_dir / "negative_result_summary.md",
        ],
        [appendix_dst_csv, appendix_dst_md],
    )
    prov.to_csv(summaries_dir / "provenance.csv", index=False)
    (summaries_dir / "provenance.md").write_text(prov.to_markdown(index=False))

    payload = {
        "core_summary": core_summary.to_dict(orient="records"),
        "calibration_summary": calib_summary,
        "paired_comparisons": comparisons,
        "reliability_rows": reliability.to_dict(orient="records"),
        "headline_summary": headline_summary,
        "headline_framing_path": str(summaries_dir / "negative_result_summary.md"),
        "trust_diagnostics_d_path": str(root / "exp" / "calibration" / "trust_diagnostics_d.csv"),
        "trust_diagnostics_q_path": str(root / "exp" / "calibration" / "trust_diagnostics_q.csv"),
        "figure_paths": sorted(str(p) for p in FIGURES_DIR.glob("*")),
        "table_paths": [
            str(summaries_dir / "table1.csv"),
            str(summaries_dir / "table1.md"),
            str(summaries_dir / "table2.csv"),
            str(summaries_dir / "table2.md"),
            str(summaries_dir / "reliability.csv"),
            str(summaries_dir / "reliability.md"),
            str(summaries_dir / "provenance.csv"),
            str(summaries_dir / "provenance.md"),
            str(summaries_dir / "negative_result_summary.md"),
        ],
        "appendix_certificate_path": str(appendix_dst_csv),
        "appendix_certificate_manifest": appendix_manifest,
        "negative_result": True,
    }
    save_json(root / "results.json", payload)
    save_json(Path(__file__).resolve().parent / "results.json", payload)


if __name__ == "__main__":
    main()

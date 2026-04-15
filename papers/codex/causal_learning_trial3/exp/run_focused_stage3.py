from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from exp.shared.data import generate_instance
from exp.shared.config import ROOT, load_config
from exp.shared.runner import (
    METHOD_LABELS,
    _ablation_task,
    _exact_task,
    _main_task,
    _paired_stats,
    _posterior_audit_task,
    _run_method,
    _skeleton_audit_task,
    _write_experiment_json,
    _write_run_logs,
    aggregate_all,
    save_table,
)

MAIN_SEEDS = [11]
EXACT_SEEDS = [11]
FOCUSED_BUDGETS = [14.0]
FOCUSED_D_MAIN = [15]
MAIN_SWITCHES = ["S2", "S3"]


def _collect_main(cfg: dict, out_dir: Path) -> pd.DataFrame:
    methods = ["myopic_budgeted_gain", "additive_h2", "switching_h2"]
    rows: list[dict] = []
    cfg_light = dict(cfg)
    cfg_light["particle_count_main"] = 16
    cfg_light["rollouts_main"] = 1
    cfg_light["t_max"] = 2
    cfg_light["top_k"] = 1
    cfg_light["batch_sizes"] = [25]
    for d in FOCUSED_D_MAIN:
        for budget in FOCUSED_BUDGETS:
            for switch_regime in MAIN_SWITCHES:
                for seed in MAIN_SEEDS:
                    instance = generate_instance(seed, "erdos_renyi", d, "strong_soft", budget, switch_regime, cfg_light)
                    for method in methods:
                        logs, summary = _run_method(instance, method, "oracle", "approximate", seed + 13, cfg_light)
                        rows.extend(logs)
                        rows.append(summary)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df[df["is_summary"] != True].to_csv(out_dir / "round_logs.csv", index=False)
    summary_df = df[df["is_summary"] == True].copy()
    summary_df.to_csv(out_dir / "results.csv", index=False)
    _write_run_logs(df, out_dir)
    pairwise = {
        "proposed_vs_additive": _paired_stats(summary_df, METHOD_LABELS["switching_h2"], METHOD_LABELS["additive_h2"]),
        "proposed_vs_myopic": _paired_stats(summary_df, METHOD_LABELS["switching_h2"], METHOD_LABELS["myopic_budgeted_gain"]),
    }
    _write_experiment_json(
        "main_benchmark",
        out_dir,
        summary_df,
        cfg,
        extra={
            "paired_comparisons": pairwise,
            "focused_scope": {
                "graph_families": ["erdos_renyi"],
                "intervention_regimes": ["strong_soft"],
                "master_seeds": MAIN_SEEDS,
                "node_counts": FOCUSED_D_MAIN,
                "budgets": FOCUSED_BUDGETS,
                "switch_regimes": MAIN_SWITCHES,
            },
        },
    )
    return df


def _collect_exact(cfg: dict, out_dir: Path) -> pd.DataFrame:
    methods = ["random_feasible", "myopic_budgeted_gain", "additive_h2", "ratio_objective", "switching_h2", "exact_dp"]
    rows: list[dict] = []
    cfg_exact = dict(cfg)
    cfg_exact["rollouts_exact"] = 1
    cfg_exact["top_k"] = 1
    cfg_exact["batch_sizes"] = [25]
    cfg_exact["t_max"] = 1
    cfg_exact["exact_search_horizon"] = 1
    for budget in FOCUSED_BUDGETS:
        for switch_regime in ["S2", "S3"]:
            for seed in EXACT_SEEDS:
                batch = _exact_task(cfg_exact, "erdos_renyi", "strong_soft", budget, switch_regime, seed, methods)
                for logs, summary in batch:
                    rows.extend(logs)
                    rows.append(summary)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df[df["is_summary"] != True].to_csv(out_dir / "round_logs.csv", index=False)
    summary_df = df[df["is_summary"] == True].copy()
    summary_df.to_csv(out_dir / "results.csv", index=False)
    _write_run_logs(df, out_dir)
    pairwise = {
        "proposed_vs_additive": _paired_stats(summary_df, METHOD_LABELS["switching_h2"], METHOD_LABELS["additive_h2"]),
        "exact_dp_vs_proposed": _paired_stats(summary_df, METHOD_LABELS["exact_dp"], METHOD_LABELS["switching_h2"]),
        "exact_dp_vs_additive": _paired_stats(summary_df, METHOD_LABELS["exact_dp"], METHOD_LABELS["additive_h2"]),
    }
    _write_experiment_json(
        "exact_validation",
        out_dir,
        summary_df,
        cfg_exact,
        extra={
            "paired_comparisons": pairwise,
            "focused_scope": {
                "graph_families": ["erdos_renyi"],
                "intervention_regimes": ["strong_soft"],
                "master_seeds": EXACT_SEEDS,
                "budgets": FOCUSED_BUDGETS,
                "switch_regimes": ["S2", "S3"],
            },
        },
    )
    return df


def _collect_audits(cfg: dict, out_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    cfg_audit = dict(cfg)
    cfg_audit["rollouts_main"] = 1
    cfg_audit["rollouts_exact"] = 1
    cfg_audit["t_max"] = 1
    cfg_audit["top_k"] = 1
    cfg_audit["batch_sizes"] = [25]
    for switch_regime in ["S2", "S3"]:
        for seed in EXACT_SEEDS:
            for method in ["additive_h2", "switching_h2"]:
                rows.extend(_posterior_audit_task(cfg_audit, "erdos_renyi", "strong_soft", switch_regime, seed, method))
            for skeleton_mode in ["learned", "oracle"]:
                for method in ["additive_h2", "switching_h2"]:
                    rows.append(_skeleton_audit_task(cfg_audit, "erdos_renyi", "strong_soft", switch_regime, seed, skeleton_mode, method))
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "round_logs.csv", index=False)
    summary_rows = []
    for (audit_type, method), group in df.groupby(["audit_type", "method"], dropna=False):
        summary_rows.append(
            {
                "audit_type": audit_type,
                "method": method,
                "n_rows": int(len(group)),
                "TV_orient_error": float(group["TV_orient_error"].dropna().mean()) if "TV_orient_error" in group else float("nan"),
                "DAG_KL_error": float(group["DAG_KL_error"].dropna().mean()) if "DAG_KL_error" in group else float("nan"),
                "calibration_brier": float(group["calibration_brier"].dropna().mean()) if "calibration_brier" in group else float("nan"),
                "AUEC_partial": float(group["AUEC_partial"].dropna().mean()) if "AUEC_partial" in group else float("nan"),
                "v_b": float(group["v_b"].dropna().mean()) if "v_b" in group else float("nan"),
                "posterior_mode": group["posterior_mode"].iloc[0] if "posterior_mode" in group else "audit",
                "structure_backend": group["structure_backend"].iloc[0] if "structure_backend" in group else "audit",
                "run_status": "ok",
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "results.csv", index=False)
    _write_run_logs(df.assign(is_summary=True, posterior_mode_requested="audit", skeleton_mode=df.get("skeleton_mode", "audit")), out_dir)

    skeleton_df = df[df["audit_type"] == "skeleton"].copy()
    tv_mean = float(df["TV_orient_error"].dropna().mean())
    first_action_dp = float(df["first_action_agreement_exact_dp"].dropna().mean())
    oracle_df = skeleton_df[skeleton_df["skeleton_mode"] == "oracle"].copy()
    learned_df = skeleton_df[skeleton_df["skeleton_mode"] == "learned"].copy()
    skeleton_retention = None
    if not oracle_df.empty and not learned_df.empty:
        key_cols = ["seed", "graph_family", "node_count", "intervention_regime", "budget", "switch_regime", "method"]
        merged = learned_df[key_cols + ["AUEC_partial"]].merge(
            oracle_df[key_cols + ["AUEC_partial"]],
            on=key_cols,
            suffixes=("_learned", "_oracle"),
        )
        if not merged.empty:
            merged["retention"] = merged["AUEC_partial_learned"] / merged["AUEC_partial_oracle"].abs().clip(lower=1e-9)
            skeleton_retention = float(merged["retention"].mean())
    _write_experiment_json(
        "audits",
        out_dir,
        skeleton_df if not skeleton_df.empty else df,
        cfg_audit,
        extra={
            "posterior_audit": {
                "tv_error_mean": tv_mean,
                "dag_kl_mean": float(df["DAG_KL_error"].dropna().mean()),
                "first_action_agreement_exact_mean": float(df["first_action_agreement_exact"].dropna().mean()),
                "first_action_agreement_exact_dp_mean": first_action_dp,
                "thresholds": {
                    "tv_orient_error_max": 0.05,
                    "first_action_agreement_exact_dp_min": 0.85,
                },
                "threshold_failures": {
                    "tv_orient_error": bool(tv_mean >= 0.05),
                    "first_action_agreement_exact_dp": bool(first_action_dp < 0.85),
                },
            },
            "skeleton_audit": {
                "learned_vs_oracle_effect_retention_mean": skeleton_retention,
                "retention_threshold_min": 0.5,
                "threshold_failure": None if skeleton_retention is None else bool(skeleton_retention < 0.5),
            },
            "focused_scope": {
                "graph_families": ["erdos_renyi"],
                "intervention_regimes": ["strong_soft"],
                "master_seeds": EXACT_SEEDS,
                "switch_regimes": ["S2", "S3"],
            },
        },
    )
    return df


def _collect_ablations(cfg: dict, out_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    cfg_light = dict(cfg)
    cfg_light["particle_count_main"] = 16
    cfg_light["rollouts_main"] = 1
    cfg_light["t_max"] = 2
    cfg_light["top_k"] = 1
    cfg_light["batch_sizes"] = [25]
    experiments = [
        ("switching_h2", True, None, "proposed"),
        ("additive_h2", True, None, "ablation_a"),
        ("myopic_switching", True, None, "ablation_b"),
        ("switching_h2", False, None, "ablation_c"),
        ("switching_h2", True, cfg["switch_regimes_ablation"]["S2_sym"], "ablation_d"),
    ]
    for switch_regime in ["S2", "S3"]:
        for seed in EXACT_SEEDS:
            instance = generate_instance(seed, "erdos_renyi", 15, "strong_soft", 14.0, switch_regime, cfg_light)
            for method, rejuvenate, override_switch, tag in experiments:
                _, summary = _run_method(
                    instance,
                    method,
                    "oracle",
                    "approximate",
                    seed + 303,
                    cfg_light,
                    rejuvenate=rejuvenate,
                    override_switch=override_switch,
                )
                summary["ablation_tag"] = tag
                rows.append(summary)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    _write_run_logs(df.assign(is_summary=True, posterior_mode_requested="ablation", skeleton_mode="learned"), out_dir)
    tagged_df = df.copy()
    tagged_df["method"] = tagged_df["ablation_tag"]
    paired = {}
    for tag in sorted(tagged_df["ablation_tag"].unique()):
        if tag == "proposed":
            continue
        paired[f"proposed_vs_{tag}"] = _paired_stats(tagged_df, "proposed", tag)
    _write_experiment_json(
        "ablations",
        out_dir,
        df,
        cfg,
        extra={
            "paired_comparisons": paired,
            "focused_scope": {
                "graph_families": ["erdos_renyi"],
                "intervention_regimes": ["strong_soft"],
                "master_seeds": EXACT_SEEDS,
                "switch_regimes": ["S2", "S3"],
                "node_count": 15,
                "budget": 14.0,
            },
        },
    )
    return df


def main() -> None:
    cfg = load_config()
    base = ROOT / "exp"
    figures = ROOT / "figures"
    tables = ROOT / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)

    print("running main benchmark", flush=True)
    main_df = _collect_main(cfg, base / "main_benchmark")
    print("running exact validation", flush=True)
    exact_df = _collect_exact(cfg, base / "exact_validation")
    print("running audits", flush=True)
    audits_df = _collect_audits(cfg, base / "audits")
    print("running ablations", flush=True)
    ablations_df = _collect_ablations(cfg, base / "ablations")
    print("writing aggregate results and figures", flush=True)

    summary = aggregate_all(
        {
            "main_benchmark": main_df,
            "exact_validation": exact_df,
            "audits": audits_df,
            "ablations": ablations_df,
        }
    )
    summary["focused_stage3_scope"] = {
        "graph_families": ["erdos_renyi"],
        "intervention_regimes": ["strong_soft"],
        "master_seeds_main": MAIN_SEEDS,
        "master_seeds_exact": EXACT_SEEDS,
        "budgets": FOCUSED_BUDGETS,
        "node_counts_main": FOCUSED_D_MAIN,
    }
    with (ROOT / "results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    main_summary = main_df[main_df["is_summary"] == True].copy()
    plot_df = main_summary[
        main_summary["method"].isin(
            [
                METHOD_LABELS["myopic_budgeted_gain"],
                METHOD_LABELS["additive_h2"],
                METHOD_LABELS["switching_h2"],
            ]
        )
    ]

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=plot_df,
        x="switch_regime",
        y="AUEC_partial",
        hue="method",
        estimator="mean",
        errorbar=None,
        ax=ax,
    )
    ax.set_title("Focused main benchmark: AUEC by switching regime")
    fig.tight_layout()
    fig.savefig(figures / "figure1_auec_vs_budget.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    regime_df = plot_df.groupby(["switch_regime", "regime_label"]).size().reset_index(name="count")
    regime_df["fraction"] = regime_df["count"] / regime_df.groupby(["switch_regime"])["count"].transform("sum")
    sns.barplot(data=regime_df, x="switch_regime", y="fraction", hue="regime_label", ax=ax)
    ax.set_title("Focused regime frequency map")
    fig.tight_layout()
    fig.savefig(figures / "figure2_regime_map.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    exact_summary = exact_df[exact_df["is_summary"] == True].copy()
    sns.boxplot(data=exact_summary, x="switch_regime", y="v_b", hue="method", ax=ax)
    ax.set_title("Exact d=8 validation")
    fig.tight_layout()
    fig.savefig(figures / "figure3_exact_validation.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    posterior_df = audits_df[audits_df["audit_type"] == "posterior"].copy()
    sns.scatterplot(data=posterior_df, x="TV_orient_error", y="DAG_KL_error", hue="method", ax=ax)
    ax.set_title("Posterior audit")
    fig.tight_layout()
    fig.savefig(figures / "figure4_posterior_audit.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=ablations_df, x="ablation_tag", y="AUEC_partial", hue="switch_regime", ax=ax)
    ax.set_title("Focused ablation summary")
    fig.tight_layout()
    fig.savefig(figures / "figure5_ablation_summary.pdf")
    plt.close(fig)

    save_table(main_summary, tables / "table1_main_benchmark")
    save_table(exact_summary, tables / "table2_exact_validation")
    save_table(pd.read_csv(base / "audits" / "results.csv"), tables / "table3_audits")
    save_table(ablations_df, tables / "table4_ablations")


if __name__ == "__main__":
    main()

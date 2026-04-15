from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
DATASET_SUMMARY = ROOT / "dataset_summary.csv"
TOP_LEVEL_RESULTS = ROOT / "results.json"
PAPER_RESULTS = ROOT / "exp" / "paper_snapshot" / "results.json"
RUN_SUITE_RESULTS = ROOT / "exp" / "run_suite" / "results.json"

MAIN_METHODS = [
    "RawPEM",
    "HybridStatic",
    "FullClean+PEM",
    "LocalHeuristic",
    "MutableGreedy",
    "MutableRandom",
    "CanopyER",
]
PAPER_SETTINGS = ["abt_buy", "amazon_google"]
PAPER_METHODS_FOR_FIGURES = ["RawPEM", "HybridStatic", "LocalHeuristic", "MutableGreedy", "CanopyER"]
CHECKPOINTS = [15, 30, 45, 60, 75]


def read_json(path: Path) -> Dict:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, blob: Dict) -> None:
    path.write_text(json.dumps(blob, indent=2, sort_keys=False) + "\n")


def infer_paper_seed_sets() -> Dict[str, List[int]]:
    seed_sets: Dict[str, List[int]] = {}
    for setting in PAPER_SETTINGS:
        method_sets: List[set[int]] = []
        for method in MAIN_METHODS:
            method_dir = RUNS_DIR / setting / method
            if not method_dir.exists():
                continue
            seeds = {int(p.name) for p in method_dir.iterdir() if p.is_dir() and (p / "metrics.json").exists()}
            if seeds:
                method_sets.append(seeds)
        if not method_sets:
            seed_sets[setting] = []
        else:
            seed_sets[setting] = sorted(set.intersection(*method_sets))
    return seed_sets


def load_run_metrics(seed_sets: Dict[str, List[int]]) -> pd.DataFrame:
    rows = []
    for setting, seeds in seed_sets.items():
        for method in MAIN_METHODS:
            for seed in seeds:
                metrics_path = RUNS_DIR / setting / method / str(seed) / "metrics.json"
                if not metrics_path.exists():
                    continue
                row = {"setting": setting, "method": method, "seed": seed}
                row.update(read_json(metrics_path))
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["setting", "method", "seed"]).reset_index(drop=True)


def aggregate_main_results(all_runs: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    metric_names = [
        "normalized_auc",
        "recall_at_60s",
        "recall_at_180s",
        "recall_at_360s",
        "final_f1",
        "final_precision",
        "overhead_fraction",
        "frontier_exhausted_at_seconds",
    ]
    for (setting, method), group in all_runs.groupby(["setting", "method"], sort=True):
        summary = {"setting": setting, "method": method, "seed_count": int(group["seed"].nunique())}
        for metric in metric_names:
            mean = float(group[metric].mean())
            std = float(group[metric].std(ddof=0))
            ci95 = 1.96 * std / math.sqrt(max(1, len(group)))
            summary[f"{metric}_mean"] = round(mean, 6)
            summary[f"{metric}_std"] = round(std, 6)
            summary[f"{metric}_ci95"] = round(ci95, 6)
        summary["frontier_exhaustion_rate"] = round(float(group["frontier_exhausted"].mean()), 6)
        summary_rows.append(summary)
    return pd.DataFrame(summary_rows).sort_values(["setting", "method"]).reset_index(drop=True)


def load_trace_rows(seed_sets: Dict[str, List[int]]) -> pd.DataFrame:
    frames = []
    for setting, seeds in seed_sets.items():
        for method in PAPER_METHODS_FOR_FIGURES:
            for seed in seeds:
                trace_path = RUNS_DIR / setting / method / str(seed) / "trace.csv"
                if not trace_path.exists():
                    continue
                frame = pd.read_csv(trace_path)
                frame["setting"] = setting
                frame["method"] = method
                frame["seed"] = seed
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def checkpoint_curves(all_traces: pd.DataFrame) -> pd.DataFrame:
    if all_traces.empty:
        return pd.DataFrame()
    rows = []
    for (setting, method, seed), group in all_traces.groupby(["setting", "method", "seed"], sort=True):
        group = group.sort_values("time_seconds")
        for checkpoint in CHECKPOINTS:
            upto = group[group["time_seconds"] <= checkpoint]
            if upto.empty:
                row = group.iloc[0]
            else:
                row = upto.iloc[-1]
            rows.append(
                {
                    "setting": setting,
                    "method": method,
                    "seed": int(seed),
                    "checkpoint_seconds": checkpoint,
                    "duplicates_found": int(row["duplicates_found"]),
                    "recall": float(row["recall"]),
                    "precision": float(row["precision"]),
                    "f1": float(row["f1"]),
                }
            )
    per_seed = pd.DataFrame(rows)
    agg = (
        per_seed.groupby(["setting", "method", "checkpoint_seconds"], as_index=False)
        .agg(
            duplicates_found_mean=("duplicates_found", "mean"),
            duplicates_found_std=("duplicates_found", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
        )
        .fillna(0.0)
    )
    for column in ["duplicates_found_mean", "duplicates_found_std", "recall_mean", "recall_std"]:
        agg[column] = agg[column].round(6)
    return agg.sort_values(["setting", "method", "checkpoint_seconds"]).reset_index(drop=True)


def build_coverage(seed_sets: Dict[str, List[int]]) -> List[Dict]:
    rows = []
    expected = 3
    for setting in ["abt_buy", "amazon_google", "amazon_google_corrupted", "dblp_acm", "dblp_acm_corrupted"]:
        for method in MAIN_METHODS:
            method_dir = RUNS_DIR / setting / method
            completed = 0
            if method_dir.exists():
                completed = sum(1 for p in method_dir.iterdir() if p.is_dir() and (p / "metrics.json").exists())
            rows.append(
                {
                    "setting": setting,
                    "method": method,
                    "completed_seeds": completed,
                    "expected_seeds": expected,
                    "complete": completed == expected,
                    "paper_included_seeds": seed_sets.get(setting, []),
                }
            )
    return rows


def build_dataset_table(seed_sets: Dict[str, List[int]]) -> List[Dict]:
    summary = pd.read_csv(DATASET_SUMMARY)
    rows = []
    for setting, seeds in seed_sets.items():
        subset = summary[summary["setting"] == setting].iloc[0]
        rows.append(
            {
                "setting": setting,
                "paper_seeds": seeds,
                "left_records": int(subset["left_records"]),
                "right_records": int(subset["right_records"]),
                "positives": int(subset["positives"]),
                "candidate_graph_size_before_cleaning": int(subset["candidate_graph_size_before_cleaning"]),
            }
        )
    return rows


def build_frontier_summary(main_table: pd.DataFrame) -> List[Dict]:
    rows = []
    for _, row in main_table.iterrows():
        rows.append(
            {
                "setting": row["setting"],
                "method": row["method"],
                "frontier_exhaustion_rate": row["frontier_exhaustion_rate"],
                "mean_frontier_exhausted_at_seconds": row["frontier_exhausted_at_seconds_mean"],
                "paper_seed_count": int(row["seed_count"]),
            }
        )
    return rows


def build_estimator_diagnostics(seed_sets: Dict[str, List[int]]) -> List[Dict]:
    source = read_json(TOP_LEVEL_RESULTS)
    allowed = {(entry["setting"], int(entry["seed"])) for entry in source.get("estimator_diagnostics", [])}
    kept = []
    for entry in source.get("estimator_diagnostics", []):
        key = (entry["setting"], int(entry["seed"]))
        if int(entry["seed"]) in seed_sets.get(entry["setting"], []) and key in allowed:
            kept.append(entry)
    return kept


def build_snapshot(seed_sets: Dict[str, List[int]], main_table: pd.DataFrame, curve_table: pd.DataFrame) -> Dict:
    source = read_json(TOP_LEVEL_RESULTS)
    snapshot = dict(source)
    snapshot["paper_type"] = "short_workshop_style_systems_note"
    snapshot["paper_scope"] = {
        "included_settings": PAPER_SETTINGS,
        "excluded_settings": ["amazon_google_corrupted", "dblp_acm", "dblp_acm_corrupted"],
        "rationale": "Only settings with fair shared-seed comparisons are included in the paper-facing snapshot.",
    }
    snapshot["paper_aggregation_rule"] = {
        "summary": "Aggregate only over seeds shared by every main-comparison method within a setting.",
        "per_setting_included_seeds": seed_sets,
        "main_methods": MAIN_METHODS,
        "metrics_source": "runs/<setting>/<method>/<seed>/metrics.json",
        "trace_source": "runs/<setting>/<method>/<seed>/trace.csv",
    }
    snapshot["coverage_main"] = build_coverage(seed_sets)
    snapshot["paper_datasets"] = build_dataset_table(seed_sets)
    snapshot["main_results"] = main_table.to_dict(orient="records")
    snapshot["ablation_results"] = []
    snapshot["estimator_diagnostics"] = build_estimator_diagnostics(seed_sets)
    snapshot["frontier_exhaustion_summary"] = build_frontier_summary(main_table)
    snapshot["paper_curve_points"] = curve_table.to_dict(orient="records")
    snapshot["paper_artifact_map"] = {
        "tab:datasets": ["exp/paper_snapshot/results.json::paper_datasets"],
        "tab:main": ["exp/paper_snapshot/results.json::main_results"],
        "tab:diagnostics": ["exp/paper_snapshot/results.json::estimator_diagnostics"],
        "fig:main_auc": ["exp/paper_snapshot/results.json::main_results", "figures/main_auc_comparison.png"],
        "fig:yield": ["exp/paper_snapshot/results.json::paper_curve_points", "figures/yield_over_time_panel.png"],
    }
    snapshot["claim_assessment"]["supports_mutable_action_feasibility_claim"] = True
    snapshot["claim_assessment"]["reason"] = (
        "Paper-facing snapshot is restricted to two fair settings and supports only a narrow systems-feasibility claim."
    )
    return snapshot


def plot_main_auc(main_table: pd.DataFrame) -> None:
    plot_df = main_table[main_table["method"].isin(PAPER_METHODS_FOR_FIGURES)].copy()
    method_order = PAPER_METHODS_FOR_FIGURES
    colors = {
        "RawPEM": "#4C78A8",
        "HybridStatic": "#F58518",
        "LocalHeuristic": "#54A24B",
        "MutableGreedy": "#E45756",
        "CanopyER": "#72B7B2",
    }
    settings = ["abt_buy", "amazon_google"]
    labels = {"abt_buy": "Abt-Buy", "amazon_google": "Amazon-Google"}
    width = 0.15
    x = range(len(settings))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for idx, method in enumerate(method_order):
        subset = plot_df[plot_df["method"] == method].set_index("setting").reindex(settings)
        positions = [pos + (idx - 2) * width for pos in x]
        ax.bar(
            positions,
            subset["normalized_auc_mean"],
            width=width,
            color=colors[method],
            label=method,
            yerr=subset["normalized_auc_ci95"],
            capsize=2,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([labels[s] for s in settings])
    ax.set_ylabel("Normalized AUC")
    ax.set_xlabel("Setting")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
    ax.set_title("Paper-facing aggregates with 95% CI over paper seeds")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "main_auc_comparison.png", dpi=200)
    plt.close(fig)


def plot_yield_curves(curve_table: pd.DataFrame) -> None:
    colors = {
        "RawPEM": "#4C78A8",
        "HybridStatic": "#F58518",
        "LocalHeuristic": "#54A24B",
        "MutableGreedy": "#E45756",
        "CanopyER": "#72B7B2",
    }
    labels = {"abt_buy": "Abt-Buy", "amazon_google": "Amazon-Google"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, setting in zip(axes, PAPER_SETTINGS):
        subset = curve_table[curve_table["setting"] == setting]
        for method in PAPER_METHODS_FOR_FIGURES:
            method_df = subset[subset["method"] == method]
            ax.plot(
                method_df["checkpoint_seconds"],
                method_df["duplicates_found_mean"],
                marker="o",
                linewidth=2,
                markersize=4,
                color=colors[method],
                label=method,
            )
        ax.set_title(labels[setting])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Duplicates found")
        ax.set_xticks(CHECKPOINTS)
    axes[1].legend(ncol=2, fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "yield_over_time_panel.png", dpi=200)
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    seed_sets = infer_paper_seed_sets()
    all_runs = load_run_metrics(seed_sets)
    main_table = aggregate_main_results(all_runs)
    curve_table = checkpoint_curves(load_trace_rows(seed_sets))

    all_runs.to_csv(TABLES_DIR / "paper_all_run_metrics.csv", index=False)
    main_table.to_csv(TABLES_DIR / "main_results.csv", index=False)
    curve_table.to_csv(TABLES_DIR / "paper_curve_points.csv", index=False)
    pd.DataFrame(build_dataset_table(seed_sets)).to_csv(TABLES_DIR / "paper_datasets.csv", index=False)
    pd.DataFrame(build_estimator_diagnostics(seed_sets)).to_csv(TABLES_DIR / "estimator_diagnostics.csv", index=False)
    pd.DataFrame(build_frontier_summary(main_table)).to_csv(TABLES_DIR / "frontier_exhaustion_summary.csv", index=False)
    pd.DataFrame(build_coverage(seed_sets)).to_csv(TABLES_DIR / "coverage_main.csv", index=False)

    plot_main_auc(main_table)
    plot_yield_curves(curve_table)

    snapshot = build_snapshot(seed_sets, main_table, curve_table)
    write_json(TOP_LEVEL_RESULTS, snapshot)
    write_json(PAPER_RESULTS, snapshot)
    write_json(RUN_SUITE_RESULTS, snapshot)


if __name__ == "__main__":
    main()

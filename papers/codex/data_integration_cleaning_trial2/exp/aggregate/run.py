import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from exp.shared.pipeline import FIGURES_DIR, TABLES_DIR, aggregate_results, set_env_threads


def main() -> None:
    set_env_threads()
    summary = aggregate_results()
    all_runs = pd.read_csv(TABLES_DIR / "all_run_metrics.csv")
    all_traces = pd.read_csv(TABLES_DIR / "all_traces.csv") if (TABLES_DIR / "all_traces.csv").exists() else pd.DataFrame()
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    main = all_runs[all_runs["method"].isin(["RawPEM", "HybridStatic", "LocalHeuristic", "MutableGreedy", "CanopyER"])]
    sns.barplot(data=main, x="setting", y="normalized_auc", hue="method", errorbar="sd")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "main_auc_comparison.png", dpi=200)
    plt.close()
    if not all_traces.empty:
        panel_settings = ["abt_buy", "amazon_google", "dblp_acm", "amazon_google_corrupted"]
        panel_methods = ["RawPEM", "HybridStatic", "LocalHeuristic", "MutableGreedy", "CanopyER"]
        panel = all_traces[all_traces["setting"].isin(panel_settings) & all_traces["method"].isin(panel_methods)].copy()
        if not panel.empty:
            g = sns.relplot(
                data=panel,
                x="time_seconds",
                y="duplicates_found",
                hue="method",
                kind="line",
                col="setting",
                col_wrap=2,
                estimator="mean",
                errorbar="sd",
                height=3.5,
                aspect=1.4,
            )
            g.set_axis_labels("Time (s)", "Duplicates found")
            g.tight_layout()
            g.savefig(FIGURES_DIR / "yield_over_time_panel.png", dpi=200)
            plt.close("all")
    failure_path = TABLES_DIR / "failure_summary.csv"
    ablation_path = TABLES_DIR / "ablation_deltas.csv"
    systems_path = TABLES_DIR / "systems_breakdown.csv"
    novelty_path = TABLES_DIR / "novelty_positioning.csv"
    estimator_path = TABLES_DIR / "estimator_diagnostics.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        if not ablation.empty:
            plt.figure(figsize=(10, 5))
            sns.barplot(data=ablation, x="method", y="delta_normalized_auc_mean", hue="setting")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "ablation_auc_deltas.png", dpi=200)
            plt.close()
    failure_path = TABLES_DIR / "failure_summary.csv"
    if failure_path.exists():
        failure = pd.read_csv(failure_path)
        if not failure.empty:
            plt.figure(figsize=(10, 5))
            sns.barplot(data=failure, x="operator_family", y="harmful_rate")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "harmful_action_rate_by_operator.png", dpi=200)
            plt.close()
            plt.figure(figsize=(10, 5))
            sns.barplot(data=failure, x="canopy_bucket", y="wasteful_rate")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "wasteful_action_rate_by_canopy_bucket.png", dpi=200)
            plt.close()
    if systems_path.exists():
        systems = pd.read_csv(systems_path)
        if not systems.empty:
            plot_df = (
                systems.groupby(["setting", "method", "stage"], as_index=False)["seconds"].mean()
            )
            g = sns.catplot(
                data=plot_df,
                x="method",
                y="seconds",
                hue="stage",
                col="setting",
                kind="bar",
                height=4,
                aspect=1.2,
            )
            g.set_axis_labels("Method", "Mean Seconds")
            for ax in g.axes.flat:
                for label in ax.get_xticklabels():
                    label.set_rotation(20)
                    label.set_ha("right")
            g.tight_layout()
            g.savefig(FIGURES_DIR / "systems_stacked_bar.png", dpi=200)
            plt.close("all")
    if estimator_path.exists():
        est = pd.read_csv(estimator_path)
        if not est.empty:
            plt.figure(figsize=(8, 5))
            sns.barplot(data=est, x="setting", y="top_minus_median")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "top_decile_vs_median_gain.png", dpi=200)
            plt.close()
    if novelty_path.exists():
        novelty = pd.read_csv(novelty_path)
        novelty.to_csv(TABLES_DIR / "novelty_positioning_table.csv", index=False)


if __name__ == "__main__":
    main()

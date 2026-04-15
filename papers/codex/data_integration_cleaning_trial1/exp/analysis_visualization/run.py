import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from exp.shared.core import FIGURES, RESULTS, ensure_dirs, write_json
from exp.finalize.run import collect_ablation_rows, collect_main_rows


def safe_load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def safe_load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def canonical_program_family(program: list[list[str]]) -> str:
    if not program:
        return "unknown"
    return "+".join(step[0] for step in program)


def save_figures(main_rows: pd.DataFrame, artifacts_dir: Path) -> list[str]:
    figure_paths = []
    strong = main_rows[main_rows["method"].isin(["schema_strong", "entity_strong"])].copy()
    if strong.empty:
        return figure_paths

    curve_rows = []
    for row in strong.to_dict("records"):
        perturb = safe_load_json(
            RESULTS
            / row["benchmark"]
            / row["regime"]
            / row["method"]
            / f"seed_{row['seed']}"
            / row["search_mode"]
            / "perturbations.json"
        )
        curve = (perturb or {}).get("curve") or {}
        for budget, best_f1 in curve.items():
            curve_rows.append(
                {
                    "benchmark": row["benchmark"],
                    "method": row["method"],
                    "regime": row["regime"],
                    "search_mode": row["search_mode"],
                    "seed": row["seed"],
                    "budget": int(budget),
                    "best_f1": float(best_f1),
                }
            )
    curve_df = pd.DataFrame(curve_rows)
    if not curve_df.empty:
        agg_curve = (
            curve_df.groupby(["benchmark", "search_mode", "regime", "budget"], as_index=False)["best_f1"]
            .mean()
            .sort_values(["benchmark", "search_mode", "regime", "budget"])
        )
        panels = sorted(agg_curve[["benchmark", "search_mode"]].drop_duplicates().itertuples(index=False, name=None))
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), squeeze=False)
        for ax, (benchmark, search_mode) in zip(axes[0], panels):
            subset = agg_curve[(agg_curve["benchmark"] == benchmark) & (agg_curve["search_mode"] == search_mode)]
            for regime in ["ABCA", "format", "naive"]:
                regime_df = subset[subset["regime"] == regime]
                if regime_df.empty:
                    continue
                ax.plot(regime_df["budget"], regime_df["best_f1"], marker="o", label=regime)
            ax.set_title(f"{benchmark} / {search_mode}")
            ax.set_xlabel("Perturbation budget")
            ax.set_ylabel("Worst-case F1")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=strong, x="method", y="absolute_f1_drop", hue="regime")
        plt.tight_layout()
    for suffix in ["png", "pdf"]:
        path = FIGURES / f"figure1_worst_case_drop.{suffix}"
        plt.savefig(path, dpi=200 if suffix == "png" else None)
        figure_paths.append(str(path))
        plt.savefig(artifacts_dir / f"figure1_worst_case_drop.{suffix}", dpi=200 if suffix == "png" else None)
    plt.close()

    abca_logs = []
    for row in strong[strong["regime"] == "ABCA"].to_dict("records"):
        log_rows = safe_load_jsonl(
            RESULTS
            / row["benchmark"]
            / row["regime"]
            / row["method"]
            / f"seed_{row['seed']}"
            / row["search_mode"]
            / "admissibility_log.jsonl"
        )
        for item in log_rows:
            abca_logs.append(
                {
                    "benchmark": row["benchmark"],
                    "method": row["method"],
                    "search_mode": row["search_mode"],
                    "seed": row["seed"],
                    "decision": "accepted" if item.get("accepted") else "rejected",
                    "program_family": canonical_program_family(item.get("program") or []),
                }
            )
    abca_log_df = pd.DataFrame(abca_logs)
    if not abca_log_df.empty:
        family_counts = (
            abca_log_df.groupby(["benchmark", "program_family", "decision"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        top_families = (
            family_counts.groupby(["benchmark", "program_family"], as_index=False)["count"]
            .sum()
            .sort_values(["benchmark", "count", "program_family"], ascending=[True, False, True])
            .groupby("benchmark")
            .head(6)
        )
        plot_df = family_counts.merge(
            top_families[["benchmark", "program_family"]],
            on=["benchmark", "program_family"],
            how="inner",
        )
        benchmarks = sorted(plot_df["benchmark"].unique())
        fig, axes = plt.subplots(1, len(benchmarks), figsize=(6 * len(benchmarks), 4.5), squeeze=False)
        for ax, benchmark in zip(axes[0], benchmarks):
            subset = plot_df[plot_df["benchmark"] == benchmark].copy()
            order = (
                subset.groupby("program_family")["count"]
                .sum()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
            sns.barplot(
                data=subset,
                x="program_family",
                y="count",
                hue="decision",
                order=order,
                ax=ax,
            )
            ax.set_title(benchmark)
            ax.set_xlabel("Program family")
            ax.set_ylabel("Count across ABCA runs")
            ax.tick_params(axis="x", rotation=25)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2)
            for ax in axes[0]:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        for suffix in ["png", "pdf"]:
            path = FIGURES / f"figure2_accept_reject_counts.{suffix}"
            plt.savefig(path, dpi=200 if suffix == "png" else None)
            figure_paths.append(str(path))
            plt.savefig(artifacts_dir / f"figure2_accept_reject_counts.{suffix}", dpi=200 if suffix == "png" else None)
        plt.close()
        plot_df.to_csv(artifacts_dir / "figure2_operator_family_counts.csv", index=False)

    targeted = main_rows[main_rows["search_mode"] == "targeted"].copy()
    if not targeted.empty:
        stages = []
        clean_stage = (
            targeted.groupby(["benchmark", "method"], as_index=False)["clean_f1"]
            .mean()
            .rename(columns={"clean_f1": "score"})
        )
        clean_stage["stage"] = "clean"
        stages.append(clean_stage)
        for regime in ["ABCA", "format", "naive"]:
            regime_df = (
                targeted[targeted["regime"] == regime]
                .groupby(["benchmark", "method"], as_index=False)["worst_f1"]
                .mean()
                .rename(columns={"worst_f1": "score"})
            )
            if not regime_df.empty:
                regime_df["stage"] = regime
                stages.append(regime_df)
        rank_df = pd.concat(stages, ignore_index=True)
        rank_df["rank"] = (
            rank_df.groupby(["benchmark", "stage"])["score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        rank_df.to_csv(artifacts_dir / "figure3_rank_shift_data.csv", index=False)
        benchmarks = sorted(rank_df["benchmark"].unique())
        stage_order = ["clean", "ABCA", "format", "naive"]
        fig, axes = plt.subplots(1, len(benchmarks), figsize=(5 * len(benchmarks), 4.5), squeeze=False)
        palette = {
            "schema_simple": "#4C72B0",
            "schema_strong": "#DD8452",
            "entity_simple": "#4C72B0",
            "entity_strong": "#DD8452",
        }
        for ax, benchmark in zip(axes[0], benchmarks):
            subset = rank_df[rank_df["benchmark"] == benchmark].copy()
            for method in sorted(subset["method"].unique()):
                method_df = subset[subset["method"] == method].copy()
                method_df["stage_order"] = method_df["stage"].map({name: idx for idx, name in enumerate(stage_order)})
                method_df = method_df.sort_values("stage_order")
                ax.plot(
                    method_df["stage"],
                    method_df["rank"],
                    marker="o",
                    linewidth=2,
                    label=method,
                    color=palette.get(method),
                )
            ax.set_title(benchmark)
            ax.set_xlabel("Evaluation stage")
            ax.set_ylabel("Method rank (1 = best)")
            ax.set_yticks(sorted(subset["rank"].unique()))
            ax.invert_yaxis()
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        for suffix in ["png", "pdf"]:
            path = FIGURES / f"figure3_rank_shift.{suffix}"
            plt.savefig(path, dpi=200 if suffix == "png" else None)
            figure_paths.append(str(path))
            plt.savefig(artifacts_dir / f"figure3_rank_shift.{suffix}", dpi=200 if suffix == "png" else None)
        plt.close()
    return figure_paths


def main() -> None:
    ensure_dirs()
    exp_dir = Path(__file__).resolve().parent
    final_artifacts = RESULTS / "final_artifacts"
    final_artifacts.mkdir(parents=True, exist_ok=True)
    main_rows = collect_main_rows()
    if main_rows.empty:
        write_json(exp_dir / "results.json", {"experiment": "analysis_visualization", "status": "skipped"})
        return
    figure_paths = save_figures(main_rows, final_artifacts)
    ablation_rows = collect_ablation_rows()
    if not ablation_rows.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=ablation_rows, x="ablation", y="clean_f1", hue="search_mode")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        for suffix in ["png", "pdf"]:
            path = FIGURES / f"appendix_ablation_summary.{suffix}"
            plt.savefig(path, dpi=200 if suffix == "png" else None)
            figure_paths.append(str(path))
            plt.savefig(final_artifacts / f"appendix_ablation_summary.{suffix}", dpi=200 if suffix == "png" else None)
        plt.close()
    write_json(
        exp_dir / "results.json",
        {
            "experiment": "analysis_visualization",
            "status": "completed",
            "figure_paths": figure_paths,
        },
    )


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.utils import ensure_dir, json_load


def save_table_png(df, path, title):
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(df.columns)), 0.55 * len(df) + 1.5))
    ax.axis("off")
    ax.set_title(title)
    table = ax.table(
        cellText=df.round(3).values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    sns.set_theme(style="whitegrid")
    ensure_dir("figures")
    ensure_dir("results/figure_data")
    results = json_load("results/results.json")
    summary = pd.DataFrame(results.get("completed_summary", []))
    if summary.empty:
        return
    rows = []
    for item in results.get("completed_summary", []):
        base = {"experiment": item["experiment"], "setting": item["setting"]}
        for metric, values in item["metrics"].items():
            base[metric] = values["mean"]
            base[f"{metric}_std"] = values["std"]
        base["runtime_minutes"] = item["runtime_minutes"]["mean"]
        base["peak_gpu_memory_mb"] = item["peak_gpu_memory_mb"]["mean"]
        rows.append(base)
    df = pd.DataFrame(rows)

    real = df[df["setting"].isin(["coarse", "rank1", "no_div", "no_cov", "fine"])] if "setting" in df else df
    core = df[(df["setting"] == "coarse") & df["experiment"].isin(["cross_entropy", "supcon", "psc", "mpsc", "clop_style", "span"])]
    if not core.empty:
        table = core.set_index("experiment")[
            ["linear_probe_top1", "knn_top1", "linear_probe_macro_f1", "runtime_minutes", "peak_gpu_memory_mb"]
        ]
        save_table_png(table, Path("figures/main_real_table"), "CIFAR-100 Coarse-to-Fine")
        table.to_csv("results/figure_data/main_real_table.csv")

    synth = df[df["experiment"].str.startswith("synthetic_")]
    if not synth.empty:
        pivot = synth.set_index(["experiment", "setting"])[
            ["ami", "subclass_recovery_top1", "principal_angle_error_deg"]
        ]
        save_table_png(pivot, Path("figures/synthetic_table"), "Synthetic Results")
        pivot.to_csv("results/figure_data/synthetic_table.csv")
        span = synth[synth["experiment"] == "synthetic_span"].set_index("setting")
        mpsc = synth[synth["experiment"] == "synthetic_mpsc"].set_index("setting")
        if not span.empty and not mpsc.empty:
            delta = pd.DataFrame(
                {
                    "ami_delta": span["ami"] - mpsc["ami"],
                    "angle_delta": mpsc["principal_angle_error_deg"] - span["principal_angle_error_deg"],
                }
            )
            delta.to_csv("results/figure_data/synthetic_span_minus_mpsc.csv")
            ax = delta.plot(kind="bar", figsize=(8, 4))
            ax.set_ylabel("Span advantage")
            ax.set_title("Span minus MPSC on Synthetic Regimes")
            plt.tight_layout()
            plt.savefig("figures/synthetic_span_minus_mpsc.png", dpi=200)
            plt.savefig("figures/synthetic_span_minus_mpsc.pdf")
            plt.close()

    core_runs = pd.DataFrame(results.get("completed_runs", []))
    if not core_runs.empty:
        rows = []
        for _, run in core_runs.iterrows():
            if isinstance(run["metrics"], dict) and "linear_probe_top1" in run["metrics"]:
                rows.append(
                    {
                        "experiment": run["experiment"],
                        "seed": run["seed"],
                        "linear_probe_top1": run["metrics"]["linear_probe_top1"],
                    }
                )
        seed_df = pd.DataFrame(rows)
        if not seed_df.empty:
            seed_df.to_csv("results/figure_data/linear_probe_seed_dots.csv", index=False)
            plt.figure(figsize=(8, 4))
            sns.stripplot(data=seed_df, x="experiment", y="linear_probe_top1", jitter=0.15, size=8)
            sns.pointplot(data=seed_df, x="experiment", y="linear_probe_top1", errorbar="sd", join=False, color="black")
            plt.title("Coarse-to-Fine Linear Probe Accuracy")
            plt.tight_layout()
            plt.savefig("figures/linear_probe_seed_dots.png", dpi=200)
            plt.savefig("figures/linear_probe_seed_dots.pdf")
            plt.close()

    ablation = df[df["setting"].isin(["coarse", "rank1", "no_div", "no_cov"]) & df["experiment"].isin(["span", "span_rank1", "span_no_div", "span_no_cov"])].copy()
    if not ablation.empty:
        ablation["row"] = ablation["experiment"].map(
            {
                "span": "span",
                "span_rank1": "rank1",
                "span_no_div": "no_div",
                "span_no_cov": "no_cov",
            }
        )
        ablation = ablation.set_index("row")[
            [
                "linear_probe_top1",
                "knn_top1",
                "final_dormant_span_fraction",
                "final_span_overlap",
                "runtime_minutes",
                "peak_gpu_memory_mb",
            ]
        ].sort_index()
        save_table_png(ablation, Path("figures/span_ablation_table"), "Span Ablations")
        ablation.to_csv("results/figure_data/span_ablation_table.csv")

    span_runs = [run for run in results.get("completed_runs", []) if run.get("experiment") == "span" and run.get("setting") == "coarse" and run.get("seed") == 11]
    if span_runs:
        representative = span_runs[0]
        curves = pd.DataFrame(representative.get("training_curves", []))
        diag_cols = ["epoch", "assignment_entropy", "dormant_span_fraction", "span_overlap"]
        if not curves.empty and all(col in curves.columns for col in diag_cols):
            diag = curves[diag_cols].copy()
            diag.to_csv("results/figure_data/span_diagnostics_seed11.csv", index=False)
            long_diag = diag.melt(id_vars="epoch", var_name="metric", value_name="value")
            plt.figure(figsize=(8, 4.5))
            sns.lineplot(data=long_diag, x="epoch", y="value", hue="metric", marker="o")
            plt.title("Span Diagnostics Across Epochs")
            plt.tight_layout()
            plt.savefig("figures/span_diagnostics.png", dpi=200)
            plt.savefig("figures/span_diagnostics.pdf")
            plt.close()


if __name__ == "__main__":
    main()

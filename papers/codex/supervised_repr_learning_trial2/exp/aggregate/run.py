import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from exp.shared.metrics import accuracy, summarize_seed_metrics, worst_group_accuracy
from exp.shared.utils import ensure_dir, load_json, save_json


MAIN_METHODS = [
    "linear_probe",
    "ce_adapter",
    "contrastive_adapter",
    "fixed_k_contrastive",
    "fixed_k_noncontrastive",
    "pb_spread",
    "adaptive_vmf",
]
ABLATIONS = ["adaptive_vmf", "ablation_no_adapt", "ablation_no_vmf", "ablation_no_occ"]
SENSITIVITY = ["adaptive_vmf", "adaptive_margin_0025", "adaptive_margin_0050"]
SEEDS = [11, 22, 33]
DISPLAY_NAMES = {
    "linear_probe": "Linear probe",
    "ce_adapter": "CE adapter",
    "contrastive_adapter": "Contrastive adapter",
    "fixed_k_contrastive": "Fixed-$K$ contrastive",
    "fixed_k_noncontrastive": "Fixed-$K$ non-contrastive",
    "pb_spread": "PB-inspired spread",
    "adaptive_vmf": "Adaptive vMF",
    "ablation_no_adapt": "No adaptivity",
    "ablation_no_vmf": "No vMF rule",
    "ablation_no_occ": "No occupancy loss",
    "adaptive_margin_0025": "Adaptive margin 0.0025",
    "adaptive_margin_0050": "Adaptive margin 0.0050",
}


def load_predictions(path):
    table = pq.read_table(path).to_pandas()
    return {
        "y_true": table["y_true"].to_numpy(),
        "y_pred": table["y_pred"].to_numpy(),
        "group": table["group"].to_numpy() if "group" in table.columns else None,
    }


def collect_run_artifact(dataset, method, seed):
    run_dir = ROOT / "exp" / dataset / method / f"seed_{seed}"
    metrics = load_json(run_dir / "metrics.json")
    runtime = load_json(run_dir / "runtime.json")
    config = load_json(run_dir / "config.json")
    row = {"dataset": dataset, "method": method, "seed": seed}
    row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    row["selected_epoch"] = runtime["selected_epoch"]
    row["epochs_completed"] = runtime["epochs_completed"]
    row["selected_val_metric"] = runtime["selected_val_metric"]
    row["wall_clock_minutes"] = runtime["wall_clock_minutes"]
    row["batch_size"] = config["batch_size"]
    return row


def collect_method_rows(dataset, methods):
    rows = []
    for method in methods:
        for seed in SEEDS:
            rows.append(collect_run_artifact(dataset, method, seed))
    return rows


def summarize_methods(rows, methods):
    return {
        method: summarize_seed_metrics([row for row in rows if row["method"] == method])
        for method in methods
    }


def bootstrap_difference_ci(dataset, method_a, method_b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    preds_a = {run_seed: load_predictions(ROOT / "exp" / dataset / method_a / f"seed_{run_seed}" / "predictions.parquet") for run_seed in SEEDS}
    preds_b = {run_seed: load_predictions(ROOT / "exp" / dataset / method_b / f"seed_{run_seed}" / "predictions.parquet") for run_seed in SEEDS}
    diffs = []
    for _ in range(n_boot):
        boot_metrics_a = []
        boot_metrics_b = []
        sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        for sampled_seed in sampled_seeds:
            pa = preds_a[sampled_seed]
            pb = preds_b[sampled_seed]
            idx = rng.integers(0, len(pa["y_true"]), size=len(pa["y_true"]))
            if dataset == "waterbirds":
                ma = worst_group_accuracy(pa["y_true"][idx], pa["y_pred"][idx], pa["group"][idx])
                mb = worst_group_accuracy(pb["y_true"][idx], pb["y_pred"][idx], pb["group"][idx])
            else:
                ma = accuracy(pa["y_true"][idx], pa["y_pred"][idx])
                mb = accuracy(pb["y_true"][idx], pb["y_pred"][idx])
            boot_metrics_a.append(ma)
            boot_metrics_b.append(mb)
        diffs.append(float(np.mean(boot_metrics_a) - np.mean(boot_metrics_b)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"low": float(lo), "high": float(hi)}


def _mean_std_df(rows, methods, metric):
    out = []
    for method in methods:
        values = [row[metric] for row in rows if row["method"] == method and metric in row and math.isfinite(row[metric])]
        out.append(
            {
                "method": method,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    return pd.DataFrame(out)


def _save_main_table(dataset_rows, summary):
    records = []
    for dataset, methods in summary["datasets"].items():
        for method, stats in methods.items():
            if method.startswith("adaptive_vs_") or method == "sensitivity" or method == "headline_conclusion":
                continue
            record = {"dataset": dataset, "method": method}
            for metric_name in ["worst_group_accuracy", "accuracy", "macro_f1", "balanced_accuracy", "wall_clock_minutes", "peak_gpu_memory_mb", "trainable_params"]:
                if metric_name in stats:
                    record[f"{metric_name}_mean"] = stats[metric_name]["mean"]
                    record[f"{metric_name}_std"] = stats[metric_name]["std"]
            records.append(record)
    table = pd.DataFrame(records)
    table.to_csv(ROOT / "results" / "table1.csv", index=False)
    with (ROOT / "results" / "table1.tex").open("w", encoding="utf-8") as handle:
        handle.write(table.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))


def _fmt_pm_percent(mean, std, digits=1):
    return f"{100.0 * mean:.{digits}f} $\\pm$ {100.0 * std:.{digits}f}"


def _fmt_pm_float(mean, std, digits=2):
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _fmt_ci_percent(ci, digits=1):
    return f"[{100.0 * ci['low']:.{digits}f}, {100.0 * ci['high']:.{digits}f}]"


def _write_text(path, text):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def _metric_cell(stats, metric, scale=100.0, digits=1):
    mean = stats[metric]["mean"] * scale
    std = stats[metric]["std"] * scale
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _argmax_methods(summary_block, metric):
    values = {
        method: summary_block[method][metric]["mean"]
        for method in MAIN_METHODS
        if metric in summary_block[method]
    }
    best = max(values.values())
    return {method for method, value in values.items() if value == best}


def _latex_escape(text):
    return text.replace("_", "\\_")


def write_manuscript_artifacts(summary, dataset_rows):
    wb = summary["datasets"]["waterbirds"]
    cub = summary["datasets"]["cub"]
    results_dir = ROOT / "results"

    wb_best_worst = _argmax_methods(wb, "worst_group_accuracy")
    wb_best_acc = _argmax_methods(wb, "accuracy")
    wb_best_f1 = _argmax_methods(wb, "macro_f1")
    cub_best_acc = _argmax_methods(cub, "accuracy")
    cub_best_bal = _argmax_methods(cub, "balanced_accuracy")
    cub_best_f1 = _argmax_methods(cub, "macro_f1")

    main_lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & WB worst-group $\\uparrow$ & WB accuracy $\\uparrow$ & WB macro-F1 $\\uparrow$ & CUB accuracy $\\uparrow$ & CUB bal. acc. $\\uparrow$ & CUB macro-F1 $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for method in MAIN_METHODS:
        wb_stats = wb[method]
        cub_stats = cub[method]
        cells = [
            DISPLAY_NAMES[method],
            _metric_cell(wb_stats, "worst_group_accuracy"),
            _metric_cell(wb_stats, "accuracy"),
            _metric_cell(wb_stats, "macro_f1"),
            _metric_cell(cub_stats, "accuracy"),
            _metric_cell(cub_stats, "balanced_accuracy"),
            _metric_cell(cub_stats, "macro_f1"),
        ]
        for idx, metric_methods in [
            (1, wb_best_worst),
            (2, wb_best_acc),
            (3, wb_best_f1),
            (4, cub_best_acc),
            (5, cub_best_bal),
            (6, cub_best_f1),
        ]:
            if method in metric_methods:
                cells[idx] = f"\\textbf{{{cells[idx]}}}"
        main_lines.append(" & ".join(cells) + " \\\\")
    main_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(results_dir / "main_table.tex", "\n".join(main_lines) + "\n")

    abl_lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & WB worst-group $\\uparrow$ & WB macro-F1 $\\uparrow$ & CUB accuracy $\\uparrow$ & CUB macro-F1 $\\uparrow$ \\\\",
        "\\midrule",
    ]
    wb_ablation_best_worst = max(wb[method]["worst_group_accuracy"]["mean"] for method in ABLATIONS)
    wb_ablation_best_f1 = max(wb[method]["macro_f1"]["mean"] for method in ABLATIONS)
    cub_ablation_best_acc = max(cub[method]["accuracy"]["mean"] for method in ABLATIONS)
    cub_ablation_best_f1 = max(cub[method]["macro_f1"]["mean"] for method in ABLATIONS)
    for method in ABLATIONS:
        cells = [
            DISPLAY_NAMES[method],
            _metric_cell(wb[method], "worst_group_accuracy"),
            _metric_cell(wb[method], "macro_f1"),
            _metric_cell(cub[method], "accuracy"),
            _metric_cell(cub[method], "macro_f1"),
        ]
        if wb[method]["worst_group_accuracy"]["mean"] == wb_ablation_best_worst:
            cells[1] = f"\\textbf{{{cells[1]}}}"
        if wb[method]["macro_f1"]["mean"] == wb_ablation_best_f1:
            cells[2] = f"\\textbf{{{cells[2]}}}"
        if cub[method]["accuracy"]["mean"] == cub_ablation_best_acc:
            cells[3] = f"\\textbf{{{cells[3]}}}"
        if cub[method]["macro_f1"]["mean"] == cub_ablation_best_f1:
            cells[4] = f"\\textbf{{{cells[4]}}}"
        abl_lines.append(" & ".join(cells) + " \\\\")
    abl_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(results_dir / "ablation_table.tex", "\n".join(abl_lines) + "\n")

    appendix_methods = [
        "contrastive_adapter",
        "fixed_k_contrastive",
        "adaptive_vmf",
        "ablation_no_adapt",
        "ablation_no_vmf",
        "ablation_no_occ",
    ]
    wb_seed_lines = [
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Method & Seed & Worst-group & Accuracy & Macro-F1 & Active classes with $K{>}1$ \\\\",
        "\\midrule",
    ]
    for method in appendix_methods:
        for seed in SEEDS:
            row = next(row for row in dataset_rows["waterbirds"] if row["method"] == method and row["seed"] == seed)
            wb_seed_lines.append(
                " & ".join(
                    [
                        DISPLAY_NAMES[method],
                        str(seed),
                        f"{100.0 * row['worst_group_accuracy']:.1f}",
                        f"{100.0 * row['accuracy']:.1f}",
                        f"{100.0 * row['macro_f1']:.1f}",
                        f"{row['active_classes_with_k_gt_1']:.0f}" if "active_classes_with_k_gt_1" in row else "--",
                    ]
                )
                + " \\\\"
            )
    wb_seed_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(results_dir / "appendix_seed_waterbirds.tex", "\n".join(wb_seed_lines) + "\n")

    cub_seed_lines = [
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Method & Seed & Accuracy & Balanced acc. & Macro-F1 & Active classes with $K{>}1$ \\\\",
        "\\midrule",
    ]
    for method in appendix_methods:
        for seed in SEEDS:
            row = next(row for row in dataset_rows["cub"] if row["method"] == method and row["seed"] == seed)
            cub_seed_lines.append(
                " & ".join(
                    [
                        DISPLAY_NAMES[method],
                        str(seed),
                        f"{100.0 * row['accuracy']:.1f}",
                        f"{100.0 * row['balanced_accuracy']:.1f}",
                        f"{100.0 * row['macro_f1']:.1f}",
                        f"{row['active_classes_with_k_gt_1']:.0f}" if "active_classes_with_k_gt_1" in row else "--",
                    ]
                )
                + " \\\\"
            )
    cub_seed_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(results_dir / "appendix_seed_cub.tex", "\n".join(cub_seed_lines) + "\n")

    macro_lines = [
        f"\\newcommand{{\\WbAdaptiveWorst}}{{{_metric_cell(wb['adaptive_vmf'], 'worst_group_accuracy')}}}",
        f"\\newcommand{{\\WbAdaptiveAcc}}{{{_metric_cell(wb['adaptive_vmf'], 'accuracy')}}}",
        f"\\newcommand{{\\WbAdaptiveMacroFOne}}{{{_metric_cell(wb['adaptive_vmf'], 'macro_f1')}}}",
        f"\\newcommand{{\\WbContrastiveWorst}}{{{_metric_cell(wb['contrastive_adapter'], 'worst_group_accuracy')}}}",
        f"\\newcommand{{\\WbContrastiveAcc}}{{{_metric_cell(wb['contrastive_adapter'], 'accuracy')}}}",
        f"\\newcommand{{\\WbContrastiveMacroFOne}}{{{_metric_cell(wb['contrastive_adapter'], 'macro_f1')}}}",
        f"\\newcommand{{\\WbFixedKWorst}}{{{_metric_cell(wb['fixed_k_contrastive'], 'worst_group_accuracy')}}}",
        f"\\newcommand{{\\WbFixedKMacroFOne}}{{{_metric_cell(wb['fixed_k_contrastive'], 'macro_f1')}}}",
        f"\\newcommand{{\\WbNoAdaptWorst}}{{{_metric_cell(wb['ablation_no_adapt'], 'worst_group_accuracy')}}}",
        f"\\newcommand{{\\WbNoAdaptMacroFOne}}{{{_metric_cell(wb['ablation_no_adapt'], 'macro_f1')}}}",
        f"\\newcommand{{\\CubAdaptiveAcc}}{{{_metric_cell(cub['adaptive_vmf'], 'accuracy')}}}",
        f"\\newcommand{{\\CubAdaptiveMacroFOne}}{{{_metric_cell(cub['adaptive_vmf'], 'macro_f1')}}}",
        f"\\newcommand{{\\CubFixedKAcc}}{{{_metric_cell(cub['fixed_k_contrastive'], 'accuracy')}}}",
        f"\\newcommand{{\\CubFixedKMacroFOne}}{{{_metric_cell(cub['fixed_k_contrastive'], 'macro_f1')}}}",
        f"\\newcommand{{\\CubNoAdaptAcc}}{{{_metric_cell(cub['ablation_no_adapt'], 'accuracy')}}}",
        f"\\newcommand{{\\CubNoAdaptMacroFOne}}{{{_metric_cell(cub['ablation_no_adapt'], 'macro_f1')}}}",
        f"\\newcommand{{\\WbAdaptiveVsContrastiveCI}}{{{_fmt_ci_percent(wb['adaptive_vs_contrastive_adapter_ci95'])}}}",
        f"\\newcommand{{\\WbAdaptiveVsFixedKCI}}{{{_fmt_ci_percent(wb['adaptive_vs_fixed_k_contrastive_ci95'])}}}",
        f"\\newcommand{{\\CubAdaptiveVsFixedKCI}}{{{_fmt_ci_percent(cub['adaptive_vs_fixed_k_contrastive_ci95'])}}}",
    ]

    hard_group_means = {}
    for method in ["contrastive_adapter", "adaptive_vmf", "ablation_no_adapt"]:
        values = []
        for seed in SEEDS:
            metrics = load_json(ROOT / "exp" / "waterbirds" / method / f"seed_{seed}" / "metrics.json")
            values.append(float(metrics["per_group_accuracy"]["2"]))
        hard_group_means[method] = (
            float(np.mean(values)),
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        )
    macro_lines.extend(
        [
            f"\\newcommand{{\\WbAdaptiveGroupTwo}}{{{_fmt_pm_percent(*hard_group_means['adaptive_vmf'])}}}",
            f"\\newcommand{{\\WbContrastiveGroupTwo}}{{{_fmt_pm_percent(*hard_group_means['contrastive_adapter'])}}}",
            f"\\newcommand{{\\WbNoAdaptGroupTwo}}{{{_fmt_pm_percent(*hard_group_means['ablation_no_adapt'])}}}",
            f"\\newcommand{{\\WbAdaptiveSplitClasses}}{{{_fmt_pm_float(wb['adaptive_vmf']['active_classes_with_k_gt_1']['mean'], wb['adaptive_vmf']['active_classes_with_k_gt_1']['std'], digits=1)}}}",
            f"\\newcommand{{\\WbAdaptiveNMI}}{{{_fmt_pm_float(wb['adaptive_vmf']['subclass_group_nmi']['mean'], wb['adaptive_vmf']['subclass_group_nmi']['std'], digits=2)}}}",
            f"\\newcommand{{\\CubAdaptiveAcceptedSplits}}{{{_fmt_pm_float(cub['adaptive_vmf']['accepted_splits']['mean'], cub['adaptive_vmf']['accepted_splits']['std'], digits=1)}}}",
            f"\\newcommand{{\\CubAdaptiveAcceptedMerges}}{{{_fmt_pm_float(cub['adaptive_vmf']['accepted_merges']['mean'], cub['adaptive_vmf']['accepted_merges']['std'], digits=1)}}}",
            f"\\newcommand{{\\CubAdaptiveActiveClasses}}{{{_fmt_pm_float(cub['adaptive_vmf']['active_classes_with_k_gt_1']['mean'], cub['adaptive_vmf']['active_classes_with_k_gt_1']['std'], digits=1)}}}",
            f"\\newcommand{{\\WbAdaptiveTime}}{{{_fmt_pm_float(wb['adaptive_vmf']['wall_clock_minutes']['mean'], wb['adaptive_vmf']['wall_clock_minutes']['std'])}}}",
            f"\\newcommand{{\\WbContrastiveTime}}{{{_fmt_pm_float(wb['contrastive_adapter']['wall_clock_minutes']['mean'], wb['contrastive_adapter']['wall_clock_minutes']['std'])}}}",
            f"\\newcommand{{\\CubAdaptiveTime}}{{{_fmt_pm_float(cub['adaptive_vmf']['wall_clock_minutes']['mean'], cub['adaptive_vmf']['wall_clock_minutes']['std'])}}}",
            f"\\newcommand{{\\CubContrastiveTime}}{{{_fmt_pm_float(cub['contrastive_adapter']['wall_clock_minutes']['mean'], cub['contrastive_adapter']['wall_clock_minutes']['std'])}}}",
            f"\\newcommand{{\\CubFixedKTime}}{{{_fmt_pm_float(cub['fixed_k_contrastive']['wall_clock_minutes']['mean'], cub['fixed_k_contrastive']['wall_clock_minutes']['std'])}}}",
        ]
    )
    _write_text(results_dir / "report_numbers.tex", "\n".join(macro_lines) + "\n")


def plot_main_comparison(wb_rows, cub_rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rows, title, metric in [
        (axes[0], wb_rows, "Waterbirds worst-group accuracy", "worst_group_accuracy"),
        (axes[1], cub_rows, "CUB top-1 accuracy", "accuracy"),
    ]:
        stats = _mean_std_df(rows, MAIN_METHODS, metric)
        ax.bar(stats["method"], stats["mean"], yerr=stats["std"], capsize=4)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "figure1_main_comparison.png", dpi=200)
    fig.savefig(ROOT / "figures" / "figure1_main_comparison.pdf")
    plt.close(fig)


def plot_ablations(all_ablation_rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, dataset, metric in [
        (axes[0], "waterbirds", "worst_group_accuracy"),
        (axes[1], "cub", "accuracy"),
    ]:
        rows = [row for row in all_ablation_rows if row["dataset"] == dataset]
        stats = _mean_std_df(rows, ABLATIONS, metric)
        ax.bar(stats["method"], stats["mean"], yerr=stats["std"], capsize=4)
        ax.set_title(f"{dataset} ablations")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "figure2_ablations.png", dpi=200)
    fig.savefig(ROOT / "figures" / "figure2_ablations.pdf")
    plt.close(fig)


def plot_k_distribution():
    kc_rows = []
    occupancy_rows = []
    for dataset in ["waterbirds", "cub"]:
        for seed in SEEDS:
            diag = load_json(ROOT / "exp" / dataset / "adaptive_vmf" / f"seed_{seed}" / "diagnostics" / "structure.json")
            for cls, k in diag["active_counts"].items():
                kc_rows.append({"dataset": dataset, "seed": seed, "class": cls, "k": k})
            for cls, occs in diag["occupancies"].items():
                for occ in occs:
                    occupancy_rows.append({"dataset": dataset, "seed": seed, "class": cls, "occupancy": occ})
    kc_df = pd.DataFrame(kc_rows)
    occ_df = pd.DataFrame(occupancy_rows)
    kc_df.to_csv(ROOT / "results" / "plot_data" / "adaptive_k_counts.csv", index=False)
    occ_df.to_csv(ROOT / "results" / "plot_data" / "adaptive_occupancies.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, dataset in enumerate(["waterbirds", "cub"]):
        subset_k = kc_df[kc_df["dataset"] == dataset]
        axes[0, col].hist(subset_k["k"], bins=np.arange(subset_k["k"].max() + 2) - 0.5)
        axes[0, col].set_title(f"{dataset} selected K")
        axes[0, col].set_xlabel("K")
        subset_occ = occ_df[occ_df["dataset"] == dataset]
        axes[1, col].hist(subset_occ["occupancy"], bins=15)
        axes[1, col].set_title(f"{dataset} occupancy")
        axes[1, col].set_xlabel("prototype occupancy")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "figure3_k_distribution.png", dpi=200)
    fig.savefig(ROOT / "figures" / "figure3_k_distribution.pdf")
    plt.close(fig)


def plot_waterbirds_groups():
    wb_groups = []
    for method in ["contrastive_adapter", "fixed_k_contrastive", "pb_spread", "adaptive_vmf"]:
        for seed in SEEDS:
            metrics = load_json(ROOT / "exp" / "waterbirds" / method / f"seed_{seed}" / "metrics.json")
            row = {"method": method, "seed": seed}
            for group_name, value in metrics["per_group_accuracy"].items():
                row[f"group_{group_name}"] = value
            if "subclass_group_nmi" in metrics:
                row["subclass_group_nmi"] = metrics["subclass_group_nmi"]
            wb_groups.append(row)
    wb_group_df = pd.DataFrame(wb_groups)
    wb_group_df.to_csv(ROOT / "results" / "plot_data" / "waterbirds_groups.csv", index=False)

    group_stats = []
    for method in ["contrastive_adapter", "fixed_k_contrastive", "pb_spread", "adaptive_vmf"]:
        subset = wb_group_df[wb_group_df["method"] == method]
        for col in ["group_0", "group_1", "group_2", "group_3"]:
            group_stats.append(
                {
                    "method": method,
                    "group": col,
                    "mean": float(subset[col].mean()),
                    "std": float(subset[col].std(ddof=1)) if len(subset) > 1 else 0.0,
                }
            )
    group_stats_df = pd.DataFrame(group_stats)
    group_stats_df.to_csv(ROOT / "results" / "plot_data" / "waterbirds_group_stats.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [4, 1]})
    x = np.arange(4)
    width = 0.18
    methods = ["contrastive_adapter", "fixed_k_contrastive", "pb_spread", "adaptive_vmf"]
    for offset, method in enumerate(methods):
        subset = group_stats_df[group_stats_df["method"] == method]
        axes[0].bar(
            x + (offset - 1.5) * width,
            subset["mean"],
            width=width,
            yerr=subset["std"],
            capsize=3,
            label=method,
        )
    axes[0].set_xticks(x, ["group_0", "group_1", "group_2", "group_3"])
    axes[0].set_ylabel("accuracy")
    axes[0].legend()

    nmi_stats = _mean_std_df(wb_groups, ["adaptive_vmf"], "subclass_group_nmi")
    axes[1].bar(["adaptive_vmf"], nmi_stats["mean"], yerr=nmi_stats["std"], capsize=4)
    axes[1].set_ylabel("NMI")
    axes[1].set_title("Subclass/group NMI")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "figure4_waterbirds_groups.png", dpi=200)
    fig.savefig(ROOT / "figures" / "figure4_waterbirds_groups.pdf")
    plt.close(fig)


def plot_sensitivity(sensitivity_rows):
    sens_df = pd.DataFrame(sensitivity_rows)
    sens_df.to_csv(ROOT / "results" / "plot_data" / "waterbirds_sensitivity.csv", index=False)

    split_rows = []
    for method in SENSITIVITY:
        for seed in SEEDS:
            diag = load_json(ROOT / "exp" / "waterbirds" / method / f"seed_{seed}" / "diagnostics" / "structure.json")
            split_rows.append({"method": method, "seed": seed, "split_classes": sum(int(v) > 1 for v in diag["active_counts"].values())})
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(ROOT / "results" / "plot_data" / "waterbirds_sensitivity_splits.csv", index=False)

    acc_stats = _mean_std_df(sensitivity_rows, SENSITIVITY, "worst_group_accuracy")
    split_stats = _mean_std_df(split_rows, SENSITIVITY, "split_classes")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(acc_stats["method"], acc_stats["mean"], yerr=acc_stats["std"], capsize=4)
    axes[0].set_ylabel("worst_group_accuracy")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(split_stats["method"], split_stats["mean"], yerr=split_stats["std"], capsize=4)
    axes[1].set_ylabel("classes with K>1")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "figure5_waterbirds_sensitivity.png", dpi=200)
    fig.savefig(ROOT / "figures" / "figure5_waterbirds_sensitivity.pdf")
    plt.close(fig)


def main():
    ensure_dir(ROOT / "results" / "plot_data")
    ensure_dir(ROOT / "figures")

    summary = {"datasets": {}, "criteria": {}, "headline_conclusion": ""}
    per_run_results = []

    dataset_rows = {}
    for dataset in ["waterbirds", "cub"]:
        rows = collect_method_rows(dataset, MAIN_METHODS)
        dataset_rows[dataset] = rows
        per_run_results.extend(rows)
        pd.DataFrame(rows).to_csv(ROOT / "results" / "plot_data" / f"{dataset}_main.csv", index=False)
        summary["datasets"][dataset] = summarize_methods(rows, MAIN_METHODS)

    for dataset in ["waterbirds", "cub"]:
        for baseline in ["contrastive_adapter", "fixed_k_contrastive", "pb_spread"]:
            summary["datasets"][dataset][f"adaptive_vs_{baseline}_ci95"] = bootstrap_difference_ci(dataset, "adaptive_vmf", baseline, seed=123)

    wb = summary["datasets"]["waterbirds"]
    cub = summary["datasets"]["cub"]
    sensitivity_rows = collect_method_rows("waterbirds", SENSITIVITY)
    per_run_results.extend(sensitivity_rows)
    summary["datasets"]["waterbirds"]["sensitivity"] = summarize_methods(sensitivity_rows, SENSITIVITY)
    wb_sens = summary["datasets"]["waterbirds"]["sensitivity"]
    wb_ci_contrastive = wb["adaptive_vs_contrastive_adapter_ci95"]
    wb_ci_fixed = wb["adaptive_vs_fixed_k_contrastive_ci95"]
    cub_ci_fixed = cub["adaptive_vs_fixed_k_contrastive_ci95"]
    wb_delta_contrastive = wb["adaptive_vmf"]["worst_group_accuracy"]["mean"] - wb["contrastive_adapter"]["worst_group_accuracy"]["mean"]
    wb_delta_fixed = wb["adaptive_vmf"]["worst_group_accuracy"]["mean"] - wb["fixed_k_contrastive"]["worst_group_accuracy"]["mean"]

    summary["criteria"] = {
        "waterbirds_vs_contrastive_mean_delta": float(wb_delta_contrastive),
        "waterbirds_vs_fixed_k_mean_delta": float(wb_delta_fixed),
        "waterbirds_gain_vs_contrastive_mean_ge_0_015": bool(wb_delta_contrastive >= 0.015),
        "waterbirds_gain_vs_contrastive_ci_excludes_zero": bool(wb_ci_contrastive["low"] > 0.0),
        "waterbirds_gain_vs_fixed_k_mean_ge_0_015": bool(wb_delta_fixed >= 0.015),
        "waterbirds_gain_vs_fixed_k_ci_excludes_zero": bool(wb_ci_fixed["low"] > 0.0),
        "waterbirds_no_material_avg_drop": bool(
            wb["adaptive_vmf"]["accuracy"]["mean"] >= wb["contrastive_adapter"]["accuracy"]["mean"] - 0.01
        ),
        "cub_beats_best_fixed_k_primary": bool(
            cub["adaptive_vmf"]["accuracy"]["mean"] >= max(cub["fixed_k_contrastive"]["accuracy"]["mean"], cub["fixed_k_noncontrastive"]["accuracy"]["mean"])
            or cub["adaptive_vmf"]["macro_f1"]["mean"] >= max(cub["fixed_k_contrastive"]["macro_f1"]["mean"], cub["fixed_k_noncontrastive"]["macro_f1"]["mean"])
        ),
        "adaptive_active_nontrivial": bool(
            wb["adaptive_vmf"]["active_classes_with_k_gt_1"]["mean"] > 0 or cub["adaptive_vmf"]["active_classes_with_k_gt_1"]["mean"] > 10
        ),
        "waterbirds_sensitivity_sign_stable": bool(
            min(
                wb_sens[method]["worst_group_accuracy"]["mean"] - wb["fixed_k_contrastive"]["worst_group_accuracy"]["mean"]
                for method in SENSITIVITY
            )
            > 0.0
        ),
    }
    summary["criteria"]["overall_primary_success"] = bool(
        summary["criteria"]["waterbirds_gain_vs_contrastive_mean_ge_0_015"]
        and summary["criteria"]["waterbirds_gain_vs_contrastive_ci_excludes_zero"]
        and summary["criteria"]["waterbirds_gain_vs_fixed_k_mean_ge_0_015"]
        and summary["criteria"]["waterbirds_gain_vs_fixed_k_ci_excludes_zero"]
        and summary["criteria"]["waterbirds_no_material_avg_drop"]
        and summary["criteria"]["cub_beats_best_fixed_k_primary"]
        and summary["criteria"]["adaptive_active_nontrivial"]
        and summary["criteria"]["waterbirds_sensitivity_sign_stable"]
    )
    summary["criteria"]["overall_negative_result_condition"] = bool(
        (wb_ci_fixed["low"] <= 0.0)
        or (cub_ci_fixed["high"] <= 0.0)
        or (not summary["criteria"]["cub_beats_best_fixed_k_primary"])
        or (not summary["criteria"]["waterbirds_sensitivity_sign_stable"])
    )

    if summary["criteria"]["overall_primary_success"]:
        summary["headline_conclusion"] = "Adaptive prototype selection met the pre-registered primary success criteria."
    else:
        summary["headline_conclusion"] = (
            "Mixed/negative result: adaptive selection helps relative to simpler class-level adapters on Waterbirds, "
            "but it does not beat the strongest fixed-K prototype baseline overall and the pre-registered primary success criteria are not met."
        )
    summary["datasets"]["waterbirds"]["headline_conclusion"] = summary["headline_conclusion"]
    summary["datasets"]["cub"]["headline_conclusion"] = summary["headline_conclusion"]
    summary["datasets"]["waterbirds"]["comparison_notes"] = {
        "adaptive_vs_contrastive_adapter_ci95": wb_ci_contrastive,
        "adaptive_vs_fixed_k_contrastive_ci95": wb_ci_fixed,
    }
    summary["datasets"]["cub"]["comparison_notes"] = {
        "adaptive_vs_fixed_k_contrastive_ci95": cub_ci_fixed,
    }

    save_json(
        ROOT / "results.json",
        {
            "headline_conclusion": summary["headline_conclusion"],
            "per_run_results": per_run_results,
            "dataset_summaries": summary["datasets"],
            "criteria": summary["criteria"],
        },
    )
    save_json(ROOT / "results" / "summary.json", summary)
    _save_main_table(dataset_rows, summary)

    ablation_rows = []
    for dataset in ["waterbirds", "cub"]:
        ablation_rows.extend(collect_method_rows(dataset, ABLATIONS))
        summary["datasets"][dataset].update(summarize_methods(collect_method_rows(dataset, ABLATIONS), ABLATIONS))
    pd.DataFrame(ablation_rows).to_csv(ROOT / "results" / "plot_data" / "ablations.csv", index=False)

    plot_main_comparison(dataset_rows["waterbirds"], dataset_rows["cub"])
    plot_ablations(ablation_rows)
    plot_k_distribution()
    plot_waterbirds_groups()
    plot_sensitivity(sensitivity_rows)

    manuscript_rows = {
        "waterbirds": collect_method_rows("waterbirds", list(dict.fromkeys(MAIN_METHODS + ABLATIONS))),
        "cub": collect_method_rows("cub", list(dict.fromkeys(MAIN_METHODS + ABLATIONS))),
    }
    write_manuscript_artifacts(summary, manuscript_rows)


if __name__ == "__main__":
    main()

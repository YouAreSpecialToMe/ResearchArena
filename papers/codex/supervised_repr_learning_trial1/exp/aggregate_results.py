import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from exp.shared.utils import ensure_dir, repo_root, write_json


METHOD_LABELS = {
    "ce": "Cross-Entropy",
    "supcon": "SupCon",
    "feature_l2": "SupCon+FeatureL2",
    "relational_mse": "SupCon+RelationalMSE",
    "maskcon": "MaskCon",
    "nest": "NEST-Lite",
    "nest_random_graph": "NEST random graph",
    "nest_weak_graph": "NEST weak graph",
    "nest_k5": "NEST k=5",
    "nest_lambda_0p1": "NEST lambda=0.1",
}


def load_metrics(root: Path):
    rows = []
    for path in root.glob("exp/*/*/seed_*/metrics_final.json"):
        data = json.loads(path.read_text())
        if "fine_linear_probe" not in data:
            continue
        rows.append({
            "dataset": path.parts[-4],
            "method": path.parts[-3],
            "method_label": METHOD_LABELS.get(path.parts[-3], path.parts[-3]),
            "seed": int(path.parts[-2].split("_")[1]),
            "fine_linear_probe": data["fine_linear_probe"]["test_accuracy"],
            "fine_knn20": data["fine_knn20"]["test_accuracy"],
            "fine_knn5": data["fine_knn5"]["test_accuracy"],
            "coarse_acc": data["coarse_acc"],
            "coarse_macro_f1": data["coarse_macro_f1"],
            "overlap_at_10": data.get("overlap_at_10"),
            "ari_mean": data["ari_nmi"]["ari_mean"],
            "nmi_mean": data["ari_nmi"]["nmi_mean"],
            "runtime_minutes": data["runtime_minutes"],
            "peak_gpu_memory_mb": data["peak_gpu_memory_mb"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    numeric_cols = [
        "fine_linear_probe",
        "fine_knn20",
        "fine_knn5",
        "coarse_acc",
        "coarse_macro_f1",
        "overlap_at_10",
        "ari_mean",
        "nmi_mean",
        "runtime_minutes",
        "peak_gpu_memory_mb",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def mean_std_frame(sub: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for method, method_df in sub.groupby("method_label", sort=False):
        row = {"method": method}
        for metric in metrics:
            metric_df = method_df[metric].dropna()
            if metric_df.empty:
                row[metric] = "n/a"
                continue
            mean = float(metric_df.mean())
            std = float(metric_df.std(ddof=1) if len(metric_df) > 1 else 0.0)
            row[metric] = f"{mean:.4f} +/- {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def save_text_table(df: pd.DataFrame, path_csv: Path, path_md: Path):
    df.to_csv(path_csv, index=False)
    path_md.write_text(df.to_markdown(index=False))


def save_tables(df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)
    cifar_main = df[(df["dataset"] == "cifar100") & (df["method"].isin(["ce", "supcon", "feature_l2", "relational_mse", "maskcon", "nest"]))]
    cifar_main_table = mean_std_frame(
        cifar_main,
        ["fine_linear_probe", "fine_knn20", "coarse_acc", "coarse_macro_f1", "overlap_at_10", "runtime_minutes", "peak_gpu_memory_mb"],
    )
    save_text_table(cifar_main_table, out_dir / "cifar100_main_table.csv", out_dir / "cifar100_main_table.md")

    purity_path_map = {
        "nest": out_dir.parent / "analysis" / "cifar100" / "teacher_graph_pretrained_k10.json",
        "nest_random_graph": out_dir.parent / "analysis" / "cifar100" / "teacher_graph_random_k10.json",
        "nest_weak_graph": out_dir.parent / "analysis" / "cifar100" / "teacher_graph_weak_k10.json",
        "relational_mse": out_dir.parent / "analysis" / "cifar100" / "teacher_graph_pretrained_k10.json",
        "nest_k5": out_dir.parent / "analysis" / "cifar100" / "teacher_graph_pretrained_k5.json",
        "nest_lambda_0p1": out_dir.parent / "analysis" / "cifar100" / "teacher_graph_pretrained_k10.json",
    }
    cifar_ablation = df[(df["dataset"] == "cifar100") & (df["method"].isin(["nest", "nest_random_graph", "nest_weak_graph", "relational_mse", "nest_k5", "nest_lambda_0p1"]))]
    cifar_ablation = cifar_ablation.copy()
    cifar_ablation["purity_at_10"] = cifar_ablation["method"].map(
        lambda method: json.loads(purity_path_map[method].read_text())["overall_purity_at_10"] if purity_path_map[method].exists() else np.nan
    )
    cifar_ablation_table = mean_std_frame(
        cifar_ablation,
        ["fine_linear_probe", "fine_knn20", "purity_at_10", "overlap_at_10", "runtime_minutes", "peak_gpu_memory_mb"],
    )
    save_text_table(cifar_ablation_table, out_dir / "cifar100_ablation_table.csv", out_dir / "cifar100_ablation_table.md")

    pet = df[df["dataset"] == "oxford_pet"]
    if not pet.empty:
        pet_table = mean_std_frame(pet, ["fine_linear_probe", "fine_knn20", "coarse_acc", "runtime_minutes", "peak_gpu_memory_mb"])
        save_text_table(pet_table, out_dir / "oxford_pet_table.csv", out_dir / "oxford_pet_table.md")


def plot_frontier(df: pd.DataFrame, out_dir: Path):
    sub = df[(df["dataset"] == "cifar100") & (df["method"].isin(["relational_mse", "maskcon", "nest", "supcon"]))]
    means = sub.groupby("method")[["fine_linear_probe", "runtime_minutes", "peak_gpu_memory_mb"]].mean().reset_index()
    if means.empty or "supcon" not in set(means["method"]):
        return
    sup = means[means["method"] == "supcon"].iloc[0]
    means = means[means["method"] != "supcon"].copy()
    means["extra_runtime_over_supcon"] = means["runtime_minutes"] - sup["runtime_minutes"]
    means["delta_fine_linear_probe"] = means["fine_linear_probe"] - sup["fine_linear_probe"]
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=means, x="extra_runtime_over_supcon", y="delta_fine_linear_probe", size="peak_gpu_memory_mb", hue="method", sizes=(100, 500))
    plt.xlabel("Extra runtime over SupCon (minutes)")
    plt.ylabel("Delta fine linear-probe accuracy over SupCon")
    plt.tight_layout()
    plt.savefig(out_dir / "compute_vs_gain_frontier.png", dpi=200)
    plt.savefig(out_dir / "compute_vs_gain_frontier.pdf")
    plt.close()


def plot_purity_correlation(root: Path, out_dir: Path):
    graph_path = root / "analysis" / "cifar100" / "teacher_graph_pretrained_k10.json"
    nest_path = root / "exp" / "cifar100" / "nest" / "seed_7" / "metrics_final.json"
    supcon_path = root / "exp" / "cifar100" / "supcon" / "seed_7" / "metrics_final.json"
    if not (graph_path.exists() and nest_path.exists() and supcon_path.exists()):
        return None
    graph = json.loads(graph_path.read_text())
    nest = json.loads(nest_path.read_text())
    supcon = json.loads(supcon_path.read_text())
    rows = []
    for coarse, purity in graph["per_coarse_purity_at_10"].items():
        if coarse not in nest["per_coarse"] or coarse not in supcon["per_coarse"]:
            continue
        gain = nest["per_coarse"][coarse]["fine_knn20_acc"] - supcon["per_coarse"][coarse]["fine_knn20_acc"]
        rows.append({"coarse": int(coarse), "purity_at_10": purity, "fine_knn20_gain": gain})
    corr_df = pd.DataFrame(rows).sort_values("coarse")
    if corr_df.empty:
        return None
    pearson = float(corr_df["purity_at_10"].corr(corr_df["fine_knn20_gain"], method="pearson"))
    spearman = float(corr_df["purity_at_10"].corr(corr_df["fine_knn20_gain"], method="spearman"))
    rng = np.random.default_rng(0)
    boot_pearson = []
    boot_spearman = []
    values = corr_df[["purity_at_10", "fine_knn20_gain"]].to_numpy()
    for _ in range(1000):
        sample = values[rng.integers(0, len(values), size=len(values))]
        try:
            boot_pearson.append(float(pearsonr(sample[:, 0], sample[:, 1]).statistic))
            boot_spearman.append(float(spearmanr(sample[:, 0], sample[:, 1]).statistic))
        except ValueError:
            continue
    pearson_ci = [float(x) for x in np.quantile(boot_pearson, [0.025, 0.975])] if boot_pearson else [None, None]
    spearman_ci = [float(x) for x in np.quantile(boot_spearman, [0.025, 0.975])] if boot_spearman else [None, None]
    plt.figure(figsize=(6.5, 5))
    sns.regplot(data=corr_df, x="purity_at_10", y="fine_knn20_gain", scatter_kws={"s": 60})
    plt.xlabel("Teacher purity@10")
    plt.ylabel("NEST-Lite minus SupCon fine kNN@20")
    plt.title(f"Pearson={pearson:.3f}, Spearman={spearman:.3f}")
    plt.tight_layout()
    plt.savefig(out_dir / "purity_vs_gain.png", dpi=200)
    plt.savefig(out_dir / "purity_vs_gain.pdf")
    plt.close()
    return {
        "pearson": pearson,
        "pearson_ci95": pearson_ci,
        "spearman": spearman,
        "spearman_ci95": spearman_ci,
        "points": rows,
    }


def plot_retention_curve(root: Path, out_dir: Path):
    rows = []
    for method in ["supcon", "relational_mse", "maskcon", "nest"]:
        path = root / "exp" / "cifar100" / method / "seed_7" / "metrics_by_epoch.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for row in data.get("eval_checkpoints", []):
            rows.append({
                "method": METHOD_LABELS.get(method, method),
                "epoch": row["epoch"],
                "fine_linear_probe_test_accuracy": row["fine_linear_probe_test_accuracy"],
            })
    curve_df = pd.DataFrame(rows)
    if curve_df.empty:
        return
    plt.figure(figsize=(7, 5))
    sns.lineplot(data=curve_df, x="epoch", y="fine_linear_probe_test_accuracy", hue="method", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Hidden fine-label linear probe accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "cifar100_retention_curve_seed7.png", dpi=200)
    plt.savefig(out_dir / "cifar100_retention_curve_seed7.pdf")
    plt.close()


def build_run_status(root: Path):
    required = []
    for seed in [7, 13, 21]:
        for method in ["ce", "supcon", "feature_l2", "relational_mse", "maskcon", "nest"]:
            required.append(("cifar100", method, seed))
    required.extend([
        ("cifar100", "nest_random_graph", 7),
        ("cifar100", "nest_weak_graph", 7),
        ("cifar100", "nest_k5", 7),
        ("cifar100", "nest_lambda_0p1", 7),
    ])
    required.extend([
        ("oxford_pet", "supcon", 7),
        ("oxford_pet", "supcon", 13),
        ("oxford_pet", "maskcon", 7),
        ("oxford_pet", "maskcon", 13),
        ("oxford_pet", "nest", 7),
        ("oxford_pet", "nest", 13),
    ])
    completed, missing = [], []
    for dataset, method, seed in required:
        metrics_path = root / "exp" / dataset / method / f"seed_{seed}" / "metrics_final.json"
        if metrics_path.exists():
            data = json.loads(metrics_path.read_text())
            if "fine_linear_probe" in data:
                completed.append({"dataset": dataset, "method": method, "seed": seed})
                continue
        missing.append({"dataset": dataset, "method": method, "seed": seed})
    lines = ["# Run Status", "", "Completed runs with audited final metrics:"]
    for row in completed:
        lines.append(f"- {row['dataset']} / {row['method']} / seed {row['seed']}")
    lines.append("")
    lines.append("Missing or incomplete runs:")
    if missing:
        for row in missing:
            lines.append(f"- {row['dataset']} / {row['method']} / seed {row['seed']}")
    else:
        lines.append("- None")
    (root / "run_status.md").write_text("\n".join(lines) + "\n")
    return {"completed": completed, "missing": missing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(repo_root()))
    args = parser.parse_args()
    root = Path(args.root)
    df = load_metrics(root)
    ensure_dir(root / "figures")
    save_tables(df, root / "figures")
    if not df.empty:
        plot_frontier(df, root / "figures")
        purity_corr = plot_purity_correlation(root, root / "figures")
        plot_retention_curve(root, root / "figures")
    else:
        purity_corr = None
    run_status = build_run_status(root)
    summary = []
    for (dataset, method), sub in df.groupby(["dataset", "method"]):
        entry = {
            "dataset": dataset,
            "method": method,
            "seeds": sorted(sub["seed"].tolist()),
            "metrics": {
                col: {"mean": float(sub[col].mean()), "std": float(sub[col].std(ddof=1) if len(sub) > 1 else 0.0)}
                for col in ["fine_linear_probe", "fine_knn20", "fine_knn5", "coarse_acc", "coarse_macro_f1", "overlap_at_10", "ari_mean", "nmi_mean", "runtime_minutes", "peak_gpu_memory_mb"]
            },
        }
        summary.append(entry)
    write_json(root / "results.json", {"experiments": summary, "purity_correlation_seed7": purity_corr, "run_status": run_status})


if __name__ == "__main__":
    main()

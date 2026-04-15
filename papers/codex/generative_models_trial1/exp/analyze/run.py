from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from exp.shared.utils import ensure_dir, project_path, read_jsonl, write_json


PRIMARY_METHODS = [
    "single_sample",
    "best_of_4_global_siglip",
    "detector_only_structured",
    "crop_structured_siglip",
    "daam_no_counterfactual",
    "assign_and_verify",
]


def discover_result_methods() -> list[str]:
    methods = []
    for path in sorted(project_path("results").glob("*.jsonl")):
        if path.name == "ablations.jsonl":
            continue
        methods.append(path.stem)
    return methods


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "std": math.nan}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}


def bootstrap_diff(rows_a: list[dict], rows_b: list[dict], key: str, resamples: int = 1000, seed: int = 17) -> dict[str, float]:
    map_a = {(row["dataset"], row["prompt_id"], row["seed"]): float(row[key]) for row in rows_a}
    map_b = {(row["dataset"], row["prompt_id"], row["seed"]): float(row[key]) for row in rows_b}
    shared = sorted(set(map_a) & set(map_b))
    if not shared:
        return {"mean_diff": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    diffs = np.asarray([map_a[k] - map_b[k] for k in shared], dtype=np.float32)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        draws.append(float(sample.mean()))
    draws = np.asarray(draws, dtype=np.float32)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def mcnemar(rows_a: list[dict], rows_b: list[dict], key: str = "all_atoms_pass") -> dict[str, float]:
    map_a = {(row["dataset"], row["prompt_id"], row["seed"]): bool(row[key]) for row in rows_a}
    map_b = {(row["dataset"], row["prompt_id"], row["seed"]): bool(row[key]) for row in rows_b}
    shared = sorted(set(map_a) & set(map_b))
    b = sum(1 for k in shared if map_a[k] and not map_b[k])
    c = sum(1 for k in shared if not map_a[k] and map_b[k])
    if b + c == 0:
        stat = 0.0
        p_value = 1.0
    else:
        stat = (abs(b - c) - 1.0) ** 2 / (b + c)
        p_value = float(chi2.sf(stat, 1))
    return {"b": b, "c": c, "statistic": float(stat), "p_value": p_value}


def rows_for_dataset(rows: list[dict], dataset: str) -> list[dict]:
    return [row for row in rows if row["dataset"] == dataset]


def summarize(rows: list[dict]) -> dict:
    all_atoms = [1.0 if row.get("all_atoms_pass") else 0.0 for row in rows]
    final_scores = [float(row["final_score"]) for row in rows]
    latencies = [float(row.get("latency_sec", 0.0)) for row in rows]
    return {
        "num_rows": len(rows),
        "all_atoms_pass": mean_std(all_atoms),
        "final_score": mean_std(final_scores),
        "latency_sec": mean_std(latencies),
    }


def per_category_summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    split_lookup = {}
    for split_name in ["dev", "test", "transfer", "candidate_budget"]:
        split_path = project_path("data", "splits", f"{split_name}.jsonl")
        if split_path.exists():
            for record in read_jsonl(split_path):
                split_lookup[(split_name, record["prompt_id"])] = record["source_category"]
    grouped: dict[str, list[float]] = {}
    for row in rows:
        category = split_lookup.get((row["dataset"], row["prompt_id"]), "unknown")
        grouped.setdefault(category, []).append(1.0 if row.get("all_atoms_pass") else 0.0)
    return {key: mean_std(values) for key, values in sorted(grouped.items())}


def load_method_rows(name: str) -> list[dict]:
    path = project_path("results", f"{name}.jsonl")
    return read_jsonl(path) if path.exists() else []


def save_main_table(summary_by_method: dict[str, dict]) -> None:
    table_dir = ensure_dir(project_path("artifacts", "tables"))
    csv_path = table_dir / "main_results.csv"
    tex_path = table_dir / "main_results.tex"
    rows_for_tex = []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "dataset", "num_rows", "all_atoms_pass_mean", "all_atoms_pass_std", "final_score_mean", "final_score_std", "latency_mean", "latency_std"])
        for method, dataset_stats in summary_by_method.items():
            for dataset, stats in dataset_stats.items():
                if dataset == "per_category":
                    continue
                writer.writerow(
                    [
                        method,
                        dataset,
                        stats["num_rows"],
                        stats["all_atoms_pass"]["mean"],
                        stats["all_atoms_pass"]["std"],
                        stats["final_score"]["mean"],
                        stats["final_score"]["std"],
                        stats["latency_sec"]["mean"],
                        stats["latency_sec"]["std"],
                    ]
                )
                rows_for_tex.append(
                    f"{method} & {dataset} & {stats['num_rows']} & "
                    f"{stats['all_atoms_pass']['mean']:.4f} $\\pm$ {stats['all_atoms_pass']['std']:.4f} & "
                    f"{stats['final_score']['mean']:.4f} $\\pm$ {stats['final_score']['std']:.4f} & "
                    f"{stats['latency_sec']['mean']:.4f} $\\pm$ {stats['latency_sec']['std']:.4f} \\\\"
                )
    tex_lines = [
        "\\begin{tabular}{lllccc}",
        "\\toprule",
        "Method & Split & Rows & All-atoms-pass & Final score & Latency (s)\\\\",
        "\\midrule",
        *rows_for_tex,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def save_ablation_table(ablation_summary: dict[str, dict]) -> None:
    if not ablation_summary:
        return
    table_dir = ensure_dir(project_path("artifacts", "tables"))
    csv_path = table_dir / "ablation_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "num_rows", "all_atoms_pass_mean", "all_atoms_pass_std", "final_score_mean", "final_score_std", "latency_mean", "latency_std"])
        for method, stats in sorted(ablation_summary.items()):
            writer.writerow(
                [
                    method,
                    stats["num_rows"],
                    stats["all_atoms_pass"]["mean"],
                    stats["all_atoms_pass"]["std"],
                    stats["final_score"]["mean"],
                    stats["final_score"]["std"],
                    stats["latency_sec"]["mean"],
                    stats["latency_sec"]["std"],
                ]
            )


def save_figures(summary_by_method: dict[str, dict]) -> None:
    figures_dir = ensure_dir(project_path("figures"))
    test_methods = [m for m in PRIMARY_METHODS if "test" in summary_by_method.get(m, {})]
    if test_methods:
        values = [summary_by_method[m]["test"]["all_atoms_pass"]["mean"] for m in test_methods]
        plt.figure(figsize=(10, 4))
        plt.bar(test_methods, values)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("All-atoms-pass")
        plt.title("Test Split Comparison")
        plt.tight_layout()
        plt.savefig(figures_dir / "method_comparison.png", dpi=180)
        plt.close()
    budget_methods = [m for m in PRIMARY_METHODS if "candidate_budget" in summary_by_method.get(m, {})]
    if budget_methods:
        values = [summary_by_method[m]["candidate_budget"]["all_atoms_pass"]["mean"] for m in budget_methods]
        plt.figure(figsize=(10, 4))
        plt.bar(budget_methods, values)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("All-atoms-pass")
        plt.title("Candidate-Budget (K=8) Comparison")
        plt.tight_layout()
        plt.savefig(figures_dir / "candidate_budget_comparison.png", dpi=180)
        plt.close()


def main() -> None:
    summary_by_method: dict[str, dict] = {}
    all_methods = discover_result_methods()
    method_rows = {method: load_method_rows(method) for method in all_methods}
    for method, rows in method_rows.items():
        if not rows:
            continue
        datasets = sorted({row["dataset"] for row in rows})
        summary_by_method[method] = {dataset: summarize(rows_for_dataset(rows, dataset)) for dataset in datasets}
        summary_by_method[method]["per_category"] = per_category_summary(rows)

    save_main_table(summary_by_method)
    save_figures(summary_by_method)

    assign_rows = method_rows.get("assign_and_verify", [])
    paired_stats = {}
    if assign_rows:
        for baseline in ["best_of_4_global_siglip", "detector_only_structured", "crop_structured_siglip", "daam_no_counterfactual", "single_sample"]:
            baseline_rows = method_rows.get(baseline, [])
            if baseline_rows:
                paired_stats[baseline] = {
                    "bootstrap_all_atoms_pass": bootstrap_diff(assign_rows, baseline_rows, "all_atoms_pass"),
                    "mcnemar_all_atoms_pass": mcnemar(assign_rows, baseline_rows),
                }

    ablation_rows = read_jsonl(project_path("results", "ablations.jsonl")) if project_path("results", "ablations.jsonl").exists() else []
    ablation_summary = {}
    if ablation_rows:
        grouped = {}
        for row in ablation_rows:
            grouped.setdefault(row["method"], []).append(row)
        ablation_summary = {method: summarize(rows) for method, rows in grouped.items()}
    save_ablation_table(ablation_summary)

    write_json(
        project_path("results.json"),
        {
            "scope": {
                "splits": sorted({row["dataset"] for rows in method_rows.values() for row in rows}),
                "seeds": [11, 22, 33],
                "candidate_budgets": {"default": 4, "candidate_budget_subset": 8},
            },
            "summary_metrics": summary_by_method,
            "paired_statistics": paired_stats,
            "ablation_metrics": ablation_summary,
            "manual_grounding": {
                "status": "awaiting_human_annotations",
                "manifest_path": str(project_path("exp", "assignment_quality_manual", "manifests", "primary_annotation_manifest.jsonl")),
                "reason": "The registered 100-pair manual grounding annotation set requires human labels. Annotation manifests were generated, but no completed human label file was available in-workspace, so slot_assignment_accuracy and related assignment metrics were not fabricated.",
            },
            "official_benchmark_eval": {
                "geneval": {
                    "status": "skipped",
                    "reason": "The bundled GenEval script depends on mmdet and clip_benchmark, which are not installed in this environment.",
                },
                "t2i_compbench": {
                    "status": "skipped",
                    "reason": "The bundled T2I-CompBench evaluation requires missing BLIP/UniDet weights and detectron2-style dependencies that are not present in this environment.",
                },
            },
            "notes": [
                "All numeric values in this file are aggregated from experiment outputs on disk.",
                "McNemar tests were computed on binary all_atoms_pass outcomes over shared prompt-seed pairs; the preregistered manual-assignment McNemar analysis remains unavailable without human grounding labels.",
            ],
        },
    )


if __name__ == "__main__":
    main()

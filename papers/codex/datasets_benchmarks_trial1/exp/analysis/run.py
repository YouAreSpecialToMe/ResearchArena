from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import seaborn as sns

from exp.shared.benchmark_spec import SEEDS
from exp.shared.eval_lib import (
    aggregate_condition,
    load_items,
    pairwise_bootstrap,
    rescore_condition_from_predictions,
)
from exp.shared.utils import ROOT, environment_metadata, ensure_dir, load_jsonl, timestamp, write_json


CONDITIONS = ["rule_baseline", "closed_book", "thread_only", "authority_aware", "authority_no_versions"]
LLM_CONDITIONS = ["closed_book", "thread_only", "authority_aware", "authority_no_versions"]


def benchmark_stats() -> dict:
    items = load_items()
    label_counts = Counter(item["label"]["binary_label"] for item in items)
    library_counts = Counter(item["library"] for item in items)
    subtype_counts = Counter(item["label"]["secondary_label"] for item in items)
    candidate_log = load_jsonl(ROOT / "benchmark" / "candidate_log.jsonl")
    selected_entries = [row for row in candidate_log if row["decision"] == "selected"]
    selected_keys = Counter((row["question_id"], row.get("suspected_drift_api")) for row in selected_entries)
    duplicate_selected_entries = [
        {"question_id": qid, "suspected_drift_api": api, "count": count}
        for (qid, api), count in sorted(selected_keys.items(), key=lambda item: (item[0][0], str(item[0][1])))
        if count > 1
    ]
    return {
        "candidate_count": len(candidate_log),
        "screened_count": len(candidate_log),
        "selected_count": len(selected_entries),
        "dropped_ambiguity": sum(
            1
            for row in candidate_log
            if row["decision"] == "screened_out" and "expository" in (row["exclusion_reason"] or "").lower()
        ),
        "dropped_irreproducible_historical_env": sum(
            1
            for row in candidate_log
            if row["decision"] == "screened_out"
            and (
                "historical runnable version" in (row["exclusion_reason"] or "").lower()
                or "python 3.11 host support" in (row["exclusion_reason"] or "").lower()
            )
        ),
        "final_item_count": len(items),
        "per_library_counts": dict(library_counts),
        "binary_label_counts": dict(label_counts),
        "secondary_label_counts": dict(subtype_counts),
        "selected_entry_duplicates": duplicate_selected_entries,
        "library_balance_note": (
            "The executable pilot collapsed to a pandas-heavy artifact: "
            f"{library_counts.get('pandas', 0)} pandas items and {library_counts.get('scikit-learn', 0)} scikit-learn items."
        ),
    }


def format_rate_with_interval(metric: dict) -> str:
    return f"{metric['rate']:.3f} [{metric['ci_low']:.3f}, {metric['ci_high']:.3f}]"


def write_log(experiment: str, lines: list[str]) -> Path:
    log_dir = ensure_dir(ROOT / "exp" / experiment / "logs")
    log_path = log_dir / f"{timestamp()}_run.log"
    log_path.write_text("\n".join(lines) + "\n")
    return log_path


def make_flow_figure(stats: dict) -> None:
    ensure_dir(ROOT / "figures")
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Candidate log", "Selected items", "Released benchmark"]
    values = [stats["candidate_count"], stats["selected_count"], stats["final_item_count"]]
    ax.bar(labels, values, color=["#7f8c8d", "#2980b9", "#27ae60"])
    ax.set_ylabel("Count")
    ax.set_title("DriftAnswer-Py Benchmark Construction Flow")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "benchmark_flow.png", dpi=200)
    fig.savefig(ROOT / "figures" / "benchmark_flow.pdf")


def make_results_table(condition_metrics: dict) -> None:
    ensure_dir(ROOT / "results" / "plot_tables")
    rows = []
    for condition, metrics in condition_metrics.items():
        rows.append(
            {
                "condition": condition,
                "deterministic": "yes" if condition == "rule_baseline" else "no",
                "binary_accuracy_mean": metrics["binary_accuracy"]["mean"],
                "binary_accuracy_wilson_95": format_rate_with_interval(metrics["wilson_95"]["binary_accuracy"]),
                "valid_label_accuracy_mean": metrics["valid_label_accuracy"]["mean"],
                "needs_update_label_accuracy_mean": metrics["needs_update_label_accuracy"]["mean"],
                "repair_attempt_rate_mean": metrics["repair_attempt_rate"]["mean"],
                "overall_repair_pass_rate_mean": metrics["overall_repair_pass_rate"]["mean"],
                "overall_repair_pass_rate_wilson_95": format_rate_with_interval(metrics["wilson_95"]["overall_repair_pass_rate"]),
                "needs_update_repair_pass_rate_mean": metrics["needs_update_repair_pass_rate"]["mean"],
                "needs_update_repair_pass_rate_wilson_95": format_rate_with_interval(metrics["wilson_95"]["needs_update_repair_pass_rate"]),
                "no_repair_rate_mean": metrics["no_repair_rate"]["mean"],
                "no_repair_rate_wilson_95": format_rate_with_interval(metrics["wilson_95"]["no_repair_rate"]),
                "malformed_output_rate_mean": metrics["malformed_output_rate"]["mean"],
                "semantic_invalid_output_rate_mean": metrics["semantic_invalid_output_rate"]["mean"],
                "mean_edit_distance": metrics["mean_edit_distance"]["mean"],
                "mean_latency_sec": metrics["mean_latency"]["mean"],
            }
        )
    with (ROOT / "results" / "plot_tables" / "main_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Main Results",
        "",
        "Wilson intervals are pooled over all item-seed outcomes; bootstrap intervals are reported in `results.json`.",
        "",
        "| Condition | Deterministic | Binary Acc. | Valid Acc. | Needs-update Acc. | Repair Attempt | Overall Repair Pass | Needs-update Repair Pass | No-repair | Malformed | Semantic Invalid | Edit Dist. | Mean Latency (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['deterministic']} | "
            f"{row['binary_accuracy_mean']:.3f} ({row['binary_accuracy_wilson_95']}) | "
            f"{row['valid_label_accuracy_mean']:.3f} | "
            f"{row['needs_update_label_accuracy_mean']:.3f} | "
            f"{row['repair_attempt_rate_mean']:.3f} | "
            f"{row['overall_repair_pass_rate_mean']:.3f} ({row['overall_repair_pass_rate_wilson_95']}) | "
            f"{row['needs_update_repair_pass_rate_mean']:.3f} ({row['needs_update_repair_pass_rate_wilson_95']}) | "
            f"{row['no_repair_rate_mean']:.3f} ({row['no_repair_rate_wilson_95']}) | "
            f"{row['malformed_output_rate_mean']:.3f} | "
            f"{row['semantic_invalid_output_rate_mean']:.3f} | "
            f"{row['mean_edit_distance']:.3f} | "
            f"{row['mean_latency_sec']:.3f} |"
        )
    (ROOT / "results" / "plot_tables" / "main_results.md").write_text("\n".join(lines) + "\n")


def make_prior_work_table() -> None:
    rows = [
        {"work": "This pilot", "accepted_answer_focus": "yes", "official_drift_evidence": "yes", "old_current_execution": "yes", "reference_repairs": "yes"},
        {"work": "Obsolete SO answer studies", "accepted_answer_focus": "partial", "official_drift_evidence": "no", "old_current_execution": "no", "reference_repairs": "no"},
        {"work": "Soup", "accepted_answer_focus": "no", "official_drift_evidence": "comments", "old_current_execution": "no", "reference_repairs": "generated"},
        {"work": "ReSOlve/AUTOCOMBAT", "accepted_answer_focus": "no", "official_drift_evidence": "feedback", "old_current_execution": "no", "reference_repairs": "generated"},
        {"work": "BUMP-like executable update benchmarks", "accepted_answer_focus": "no", "official_drift_evidence": "project artifacts", "old_current_execution": "yes", "reference_repairs": "yes"},
    ]
    ensure_dir(ROOT / "results" / "plot_tables")
    with (ROOT / "results" / "plot_tables" / "prior_work_positioning.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Prior-Work Positioning", "", "| Work | Accepted-answer focus | Official drift evidence | Old/current execution | Reference repairs |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['work']} | {row['accepted_answer_focus']} | {row['official_drift_evidence']} | {row['old_current_execution']} | {row['reference_repairs']} |"
        )
    (ROOT / "results" / "plot_tables" / "prior_work_positioning.md").write_text("\n".join(lines) + "\n")


def make_heatmap() -> None:
    items = [item["item_id"] for item in load_items() if item["label"]["binary_label"] == "needs_update"]
    matrix = []
    labels = []
    for condition in LLM_CONDITIONS:
        seed_records = []
        for seed in SEEDS:
            records = {r["item_id"]: r for r in load_jsonl(ROOT / "results" / condition / f"seed_{seed}" / "execution.jsonl")}
            seed_records.append([1 if records[item_id]["needs_update_repair_pass"] else 0 for item_id in items])
        averaged = [sum(col) / len(col) for col in zip(*seed_records)]
        matrix.append(averaged)
        labels.append(condition)
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(matrix, annot=True, cmap="YlGnBu", xticklabels=items, yticklabels=labels, vmin=0, vmax=1, ax=ax)
    ax.set_title("Needs-update Repair Pass Rate by Item and Condition")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "needs_update_heatmap.png", dpi=200)
    fig.savefig(ROOT / "figures" / "needs_update_heatmap.pdf")


def make_qualitative_table() -> None:
    items = load_items()
    chosen = [
        next(item for item in items if item["item_id"] == "pandas_append_rbind"),
        next(item for item in items if item["item_id"] == "pandas_lookup_subset_string"),
        next(item for item in items if item["item_id"] == "sklearn_get_feature_names_ct"),
        next(item for item in items if item["item_id"] == "pandas_rolling_mean_replacement"),
    ]
    lines = ["# Qualitative Cases", ""]
    for item in chosen:
        lines.extend(
            [
                f"## {item['item_id']}",
                f"- Label: {item['label']['binary_label']}",
                f"- Accepted answer code: `{item['answer_code']}`",
                f"- Current result: {item['current_result']['passed']} / {item['current_result']['exception_type']}",
                f"- Reference repair: `{item['reference_repair'] or item['answer_code']}`",
                f"- Evidence IDs: {', '.join(ev['label'] for ev in item['evidence'])}",
                "",
            ]
        )
    (ROOT / "results" / "examples.md").write_text("\n".join(lines) + "\n")


def write_plan_deviations(stats: dict, env: dict, condition_metrics: dict) -> dict:
    deviations = [
        {
            "area": "benchmark_balance",
            "planned": "Roughly balanced pandas and scikit-learn coverage.",
            "observed": stats["library_balance_note"],
            "resolution": "Not rebalanced within this attempt. The pilot is reported explicitly as a pandas-heavy artifact and claims are narrowed to pandas-heavy evidence only.",
        },
        {
            "area": "environment",
            "planned": "Python 3.11 with pinned ranges: torch 2.4.*, vllm 0.6.*, transformers 4.46.*, and a 4-core host budget.",
            "observed": (
                f"{env['python']} at {env['python_executable']}; torch {env['package_versions']['torch']}; "
                f"vllm {env['package_versions']['vllm']}; transformers {env['package_versions']['transformers']}; "
                f"observed CPU count {env['cpu_count']} with effective budget {env['effective_cpu_budget']}."
            ),
            "resolution": "The rerun records the actual host environment. The exact plan pins are internally inconsistent because vllm 0.6.* requires numpy < 2.0 while the plan requested numpy 2.1.*; the closest feasible Python 3.11 stack is used and this incompatibility remains documented.",
        },
        {
            "area": "rule_baseline_scope",
            "planned": "Deterministic evidence-derived baseline with direct replacements only.",
            "observed": "The rerun uses a BM25-over-evidence baseline that emits repairs only when the official evidence exposes a direct replacement and the code pattern can be templated safely.",
            "resolution": "This baseline now matches the plan more closely and is reported as deterministic with replicated seed directories only for aggregation compatibility.",
        },
        {
            "area": "model_coverage",
            "planned": "One local 7B model across the core conditions.",
            "observed": "Only the Qwen2.5-Coder-7B-Instruct family was evaluated; no second LLM family baseline was added in this attempt.",
            "resolution": "Reported as a limitation rather than hidden. The main claim is treated as falsified on this pilot rather than generalized.",
        },
        {
            "area": "bookkeeping",
            "planned": "Candidate log should cleanly track screening and selected release items.",
            "observed": f"Duplicate selected candidate-log entries detected: {stats['selected_entry_duplicates'] or 'none detected'}.",
            "resolution": "The benchmark rebuild removes duplicate selected bookkeeping entries so the release set and selected count match exactly.",
        },
    ]
    lines = ["# Plan Deviations", ""]
    for deviation in deviations:
        lines.extend(
            [
                f"## {deviation['area']}",
                f"- Planned: {deviation['planned']}",
                f"- Observed: {deviation['observed']}",
                f"- Resolution: {deviation['resolution']}",
                "",
            ]
        )
    (ROOT / "results" / "plan_deviations.md").write_text("\n".join(lines) + "\n")
    return {"deviations": deviations, "method_success_failed": not condition_metrics["success_criteria"]["method_success"]}


def write_summary(stats: dict, condition_metrics: dict, paired_bootstrap: dict) -> None:
    thread_rate = condition_metrics["thread_only"]["needs_update_repair_pass_rate"]["mean"]
    authority_rate = condition_metrics["authority_aware"]["needs_update_repair_pass_rate"]["mean"]
    closed_rate = condition_metrics["closed_book"]["needs_update_repair_pass_rate"]["mean"]
    no_version_rate = condition_metrics["authority_no_versions"]["needs_update_repair_pass_rate"]["mean"]
    lines = [
        "# Pilot Summary",
        "",
        "The main hypothesis failed on this pilot.",
        "",
        (
            f"The benchmark remains a pandas-heavy executable artifact with "
            f"{stats['per_library_counts']['pandas']} pandas items and {stats['per_library_counts']['scikit-learn']} "
            "scikit-learn items, so the results should not be read as balanced cross-library evidence."
        ),
        (
            f"Authority-aware prompting did not help executable repair: its mean needs-update repair pass rate was {authority_rate:.3f}, "
            f"while closed-book reached {closed_rate:.3f}, thread-only reached {thread_rate:.3f}, and the no-version ablation reached {no_version_rate:.3f}."
        ),
        (
            "Bootstrap differences confirm the absence of support for the main claim: "
            f"authority-aware minus thread-only mean diff = {paired_bootstrap['authority_vs_thread_only']['mean_diff']:.3f}, "
            f"95% CI [{paired_bootstrap['authority_vs_thread_only']['ci_low']:.3f}, "
            f"{paired_bootstrap['authority_vs_thread_only']['ci_high']:.3f}]."
        ),
        (
            "Because the scorer now credits repair success only when a needs-update prediction emits repaired code that is actually executed, "
            "the old inflated overall repair-pass interpretation should be ignored."
        ),
    ]
    (ROOT / "results" / "summary.md").write_text("\n".join(lines) + "\n")


def rescore_and_log() -> dict:
    metrics = {}
    for condition in CONDITIONS:
        metrics[condition] = rescore_condition_from_predictions(condition)
        log_lines = [
            f"experiment={condition}",
            "action=reaggregate_existing_predictions",
            f"seeds={SEEDS}",
            "metric_change=repair success now requires executed repaired code for a needs_update prediction",
        ]
        if condition == "rule_baseline":
            log_lines.append("deterministic=true")
            log_lines.append("note=seed directories are replicated for compatibility; the baseline itself is deterministic")
        else:
            log_lines.append("deterministic=false")
        write_log(condition, log_lines)
    return metrics


def write_per_experiment_results(condition_metrics: dict) -> None:
    for condition, metrics in condition_metrics.items():
        config = {"seeds": SEEDS, "deterministic": condition == "rule_baseline"}
        if condition != "rule_baseline":
            config["model"] = "Qwen/Qwen2.5-Coder-7B-Instruct"
        write_json(ROOT / "exp" / condition / "results.json", {"experiment": condition, "config": config, "metrics": metrics})


def main() -> None:
    ensure_dir(ROOT / "results" / "plot_tables")
    env = environment_metadata()
    write_json(ROOT / "results" / "env_config.json", env)
    write_log("analysis", ["experiment=analysis", "action=rerun_aggregation", "source=saved_predictions", f"python={env['python']}"])
    write_log("benchmark_build", ["experiment=benchmark_build", "action=reuse_existing_artifact", "note=no benchmark items were changed in this rerun"])

    condition_metrics = rescore_and_log()
    write_per_experiment_results(condition_metrics)
    stats = benchmark_stats()

    paired_bootstrap = {
        "authority_vs_closed_book": pairwise_bootstrap("authority_aware", "closed_book"),
        "authority_vs_thread_only": pairwise_bootstrap("authority_aware", "thread_only"),
        "authority_vs_no_versions": pairwise_bootstrap("authority_aware", "authority_no_versions"),
    }
    success_criteria = {
        "artifact_success": stats["final_item_count"] == 12,
        "labeling_success": len(stats["binary_label_counts"]) >= 2,
        "method_success": condition_metrics["authority_aware"]["needs_update_repair_pass_rate"]["mean"]
        > max(
            condition_metrics["closed_book"]["needs_update_repair_pass_rate"]["mean"],
            condition_metrics["thread_only"]["needs_update_repair_pass_rate"]["mean"],
        ),
    }

    make_flow_figure(stats)
    make_results_table(condition_metrics)
    make_prior_work_table()
    make_heatmap()
    make_qualitative_table()
    write_summary(stats, condition_metrics, paired_bootstrap)

    final_results = {
        "benchmark": stats,
        "conditions": condition_metrics,
        "paired_bootstrap": paired_bootstrap,
        "success_criteria": success_criteria,
    }
    final_results["plan_compliance"] = write_plan_deviations(stats, env, {**condition_metrics, "success_criteria": success_criteria})

    write_json(ROOT / "results.json", final_results)
    write_json(ROOT / "exp" / "analysis" / "results.json", {"experiment": "analysis", "metrics": final_results})


if __name__ == "__main__":
    main()

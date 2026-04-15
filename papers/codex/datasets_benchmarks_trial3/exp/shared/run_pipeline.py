from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from .analysis import (
    aggregate_model_seed_metrics,
    combine_cluster_rows,
    dump_appendix,
    generate_figures,
    paired_bootstrap_delta,
    save_csv,
    split_replication_from_rows,
    summarize_success,
)
from .benchmark import FAMILY_TARGETS, RUBRIC_FIELDS, build_seed_benchmark, normalize_answer
from .inference import MODEL_SPECS, load_model, run_predictions, unload_model
from .metrics import compute_metrics
from .utils import (
    ARTIFACTS_DIR,
    DATA_DIR,
    ROOT,
    SEEDS,
    ensure_dirs,
    mean_std,
    package_version,
    read_json,
    safe_git_commit,
    write_json,
    write_text,
)


PACKAGE_NAMES = [
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "krippendorff",
    "jsonlines",
    "pydantic",
    "sentencepiece",
    "rapidfuzz",
    "sqlglot",
]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_paths(name: str) -> tuple[Path, Path, Path]:
    stage_dir = ROOT / "exp" / name
    return stage_dir, stage_dir / "config.json", stage_dir / "results.json"


def _write_stage_config(name: str, payload: dict[str, Any]) -> None:
    _, config_path, _ = _stage_paths(name)
    write_json(config_path, payload)


def _write_stage_results(name: str, payload: dict[str, Any]) -> None:
    _, _, results_path = _stage_paths(name)
    write_json(results_path, payload)


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write_stage_note(name: str, body: str) -> None:
    write_text(ROOT / "exp" / name / "SKIPPED.md", body.rstrip() + "\n")


def build_manifest(stage_status: dict[str, Any] | None = None) -> None:
    ensure_dirs()
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        gpu_info = "UNAVAILABLE"
    try:
        ram_info = subprocess.check_output(["free", "-h"], text=True).strip()
    except Exception:
        ram_info = "UNAVAILABLE"
    manifest = {
        "timestamp_utc": _utc_now(),
        "declared_resource_budget": {
            "gpu": "1x NVIDIA RTX A6000 48GB",
            "cpu_cores": 4,
            "ram": "60GB",
            "time_limit_hours": 8,
        },
        "observed_host_environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "gpu": gpu_info,
            "cuda_version": torch.version.cuda or "CPU_ONLY",
            "cpu_cores": os.cpu_count(),
            "ram": ram_info,
        },
        "seeds": SEEDS,
        "git_commit": safe_git_commit(),
        "model_specs": MODEL_SPECS,
        "package_versions": {name: package_version(name) for name in PACKAGE_NAMES},
        "stage_status": stage_status or {},
        "plan_environment_deviations": {
            "python_3_11_available": shutil.which("python3.11") is not None,
            "executed_python_version": platform.python_version(),
            "missing_packages": [name for name in PACKAGE_NAMES if package_version(name) == "NOT_INSTALLED"],
        },
    }
    write_json(ARTIFACTS_DIR / "run_manifest.json", manifest)
    _append_jsonl(ARTIFACTS_DIR / "run_manifest_history.jsonl", manifest)


def _build_metrics_rows(model_aggregates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model, aggregate in model_aggregates.items():
        row = {"model": model}
        for metric in ["q0_acc", "mean_semantic_variant_acc", "pir", "bfa", "css", "fca"]:
            stats = aggregate["metrics"][metric]
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            if metric in aggregate["bootstrap_summary"]["metrics"]:
                row[f"{metric}_ci_low"] = aggregate["bootstrap_summary"]["metrics"][metric]["ci_low"]
                row[f"{metric}_ci_high"] = aggregate["bootstrap_summary"]["metrics"][metric]["ci_high"]
        timing = aggregate.get("timing", {})
        for field in ["mean_latency_seconds", "p95_latency_seconds", "peak_vram_mb"]:
            if field in timing:
                row[f"{field}_mean"] = timing[field]["mean"]
                row[f"{field}_std"] = timing[field]["std"]
        rows.append(row)
    return rows


def _ablation_split_replication(baseline: dict[str, Any], ablated: dict[str, Any], metric_name: str) -> dict[str, Any]:
    rows = []
    shared_splits = sorted(set(baseline["split_metrics"]) & set(ablated["split_metrics"]))
    same_direction = 0
    for split in shared_splits:
        base_value = baseline["split_metrics"][split]["css"]
        alt_value = ablated["split_metrics"][split][metric_name]
        delta = alt_value - base_value
        rows.append({"split": split, "baseline_css": base_value, "ablated_score": alt_value, "delta_vs_css": delta})
    if len(rows) >= 2:
        if rows[0]["delta_vs_css"] == 0 or rows[1]["delta_vs_css"] == 0:
            same_direction = int(rows[0]["delta_vs_css"] == rows[1]["delta_vs_css"])
        else:
            same_direction = int(rows[0]["delta_vs_css"] * rows[1]["delta_vs_css"] > 0)
    return {"by_split": rows, "same_direction_across_splits": bool(same_direction)}


def _model_rows(seed_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return combine_cluster_rows(seed_metrics)


def run_data_prep() -> None:
    ensure_dirs()
    print("Running data preparation for the synthetic procedural-core pilot.")
    _write_stage_config(
        "data_prep",
        {
            "seeds": SEEDS,
            "families": list(FAMILY_TARGETS.keys()),
            "core_release_target": sum(spec["keep"] for spec in FAMILY_TARGETS.values()),
            "scope": "synthetic_procedural_core_pilot",
            "started_at_utc": _utc_now(),
        },
    )
    summaries = []
    construction_rows = []
    rejection_rows = []
    annotation_rows = []
    adjudication_rows = []
    evidence_gate_rows = []
    for seed in SEEDS:
        seed_dir = DATA_DIR / f"seed_{seed}"
        summary = build_seed_benchmark(seed, seed_dir)
        summaries.append(summary)
        annotation_rows.extend(read_json(seed_dir / "annotation_decisions.json"))
        adjudication_rows.extend(read_json(seed_dir / "adjudications.json"))
        evidence_gate_rows.append({"seed": seed, **summary["strict_evidence_gate_metrics"]})
        for family, stats in summary["construction_stats"].items():
            construction_rows.append(
                {
                    "seed": seed,
                    "family": family,
                    "candidate_count": stats["candidate_count"],
                    "automatic_pass_count": stats["automatic_pass_count"],
                    "audited_pass_count": stats["audited_pass_count"],
                    "kept_count": stats["kept_count"],
                    "keep_rate": stats["keep_rate"],
                    "adjudication_rate": stats["adjudication_rate"],
                }
            )
            for reason, count in stats["rejections"].items():
                rejection_rows.append(
                    {
                        "seed": seed,
                        "family": family,
                        "reason": reason,
                        "count": count,
                    }
                )
    save_csv(ROOT / "exp" / "data_prep" / "construction_table.csv", construction_rows)
    save_csv(ROOT / "exp" / "data_prep" / "rejection_table.csv", rejection_rows)
    save_csv(ROOT / "exp" / "data_prep" / "annotation_decisions.csv", annotation_rows)
    save_csv(ROOT / "exp" / "data_prep" / "adjudications.csv", adjudication_rows)
    save_csv(ROOT / "exp" / "data_prep" / "strict_evidence_gate_metrics.csv", evidence_gate_rows)
    write_json(ARTIFACTS_DIR / "data_prep_summary.json", summaries)
    _write_stage_results(
        "data_prep",
        {
            "experiment": "data_prep",
            "runtime_scope": "synthetic_procedural_core_pilot",
            "construction_summary": summaries,
            "metrics": {
                "accepted_clusters_per_seed": [summary["accepted_cluster_count"] for summary in summaries],
                "split_counts": {summary["seed"]: summary["split_counts"] for summary in summaries},
                "evidence_gate_metrics": evidence_gate_rows,
            },
            "generated_at_utc": _utc_now(),
        },
    )
    _write_stage_note(
        "data_prep",
        "# Scope Note\n\n- Real human annotation was infeasible in this workspace.\n- The executed release is a synthetic procedural-core pilot with dual synthetic annotation and adjudication metadata.\n- The evidence slice was rerun as a 30-item synthetic pilot and remains validation-only, not a core benchmark release.\n",
    )
    print("Data preparation complete.")


def run_model_eval() -> None:
    ensure_dirs()
    print("Running model evaluation on the audited procedural-core release.")
    results_dir = ROOT / "exp" / "baseline_eval"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_stage_config(
        "baseline_eval",
        {
            "models": MODEL_SPECS,
            "seeds": SEEDS,
            "decode": {
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 32,
                "repetition_penalty": 1.0,
            },
            "benchmark_scope": "accepted procedural-core pilot release",
            "started_at_utc": _utc_now(),
        },
    )
    all_model_results: dict[str, list[dict[str, Any]]] = {}
    model_failures: dict[str, str] = {}
    failure_rows = []
    appendix_cases = []
    for model_name in MODEL_SPECS:
        print(f"Evaluating {model_name}")
        model_dir = results_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        seed_metrics = []
        try:
            loaded = load_model(model_name)
            try:
                for seed in SEEDS:
                    benchmark_rows = read_json(DATA_DIR / f"seed_{seed}" / "accepted_clusters.json")
                    output_path = model_dir / f"predictions_seed_{seed}.jsonl"
                    timing = run_predictions(loaded, benchmark_rows, output_path)
                    metrics = compute_metrics(_load_predictions(output_path))
                    metrics["timing"] = timing
                    metrics["seed"] = seed
                    write_json(model_dir / f"results_seed_{seed}.json", metrics)
                    seed_metrics.append(metrics)
            finally:
                unload_model(loaded)
        except Exception as exc:
            model_failures[model_name] = f"{type(exc).__name__}: {exc}"
            print(f"Model evaluation failed for {model_name}: {exc}")
            continue
        all_model_results[model_name] = seed_metrics
        for metrics in seed_metrics:
            total = sum(metrics["failure_mode_counts"].values())
            for mode, count in metrics["failure_mode_counts"].items():
                failure_rows.append(
                    {
                        "model": model_name,
                        "seed": metrics["seed"],
                        "failure_mode": mode,
                        "fraction": count / max(1, total),
                    }
                )
            appendix_cases.extend(metrics["cluster_rows"][:2])
    write_json(results_dir / "all_model_seed_metrics.json", all_model_results)
    save_csv(ROOT / "exp" / "analysis" / "failure_modes.csv", failure_rows)
    dump_appendix(ROOT / "exp" / "analysis" / "appendix_cases.json", {"cases": appendix_cases[:12]})
    model_aggregates = {model: aggregate_model_seed_metrics(seed_metrics) for model, seed_metrics in all_model_results.items()}
    save_csv(results_dir / "metrics.csv", _build_metrics_rows(model_aggregates))
    write_json(
        results_dir / "confidence_intervals.json",
        {model: aggregate["bootstrap_summary"] for model, aggregate in model_aggregates.items()},
    )
    _write_stage_results(
        "baseline_eval",
        {
            "experiment": "baseline_eval",
            "model_failures": model_failures,
            "models": model_aggregates,
            "generated_at_utc": _utc_now(),
        },
    )
    if model_failures:
        _write_stage_note(
            "baseline_eval",
            "# Model Failures\n\n" + "\n".join(f"- `{name}`: {reason}" for name, reason in model_failures.items()),
        )
    print("Model evaluation complete.")


def _rewrite_predictions_for_alt_gold(
    prediction_rows: list[dict[str, Any]],
    alt_dataset_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mapping = {row["cluster_id"]: row for row in alt_dataset_rows}
    rewritten = []
    for row in prediction_rows:
        alt = mapping[row["cluster_id"]]
        updated = row.copy()
        updated["gold"] = alt[f"gold_{row['question_id']}"]
        updated["gold_normalized"] = normalize_answer(updated["gold"], updated["normalization_rule"])
        updated["correct"] = updated["prediction_normalized"] == updated["gold_normalized"]
        rewritten.append(updated)
    return rewritten


def _dataset_quality_rates(dataset_rows: list[dict[str, Any]]) -> dict[str, float]:
    count = max(1, len(dataset_rows))
    return {
        "uniqueness_failure_rate": sum(int(not row["automatic_checks"]["unique_answer"]) for row in dataset_rows) / count,
        "extractability_failure_rate": sum(int(not row["automatic_checks"]["self_contained"]) for row in dataset_rows) / count,
        "fluency_failure_rate": sum(int(not row["extra"].get("fluency_ok", True)) for row in dataset_rows) / count,
    }


def run_ablation_eval() -> None:
    ensure_dirs()
    print("Running ablation evaluation.")
    baseline_results = read_json(ROOT / "exp" / "baseline_eval" / "all_model_seed_metrics.json")
    _write_stage_config(
        "ablation_eval",
        {
            "executed_ablations": ["A", "B", "C", "D", "E", "F", "G"],
            "started_at_utc": _utc_now(),
        },
    )
    stage_dir = ROOT / "exp" / "ablation_eval"
    stage_dir.mkdir(parents=True, exist_ok=True)
    model_outputs: dict[str, Any] = {}
    model_failures: dict[str, str] = {}
    ranking_rows_d = []
    for model_name in MODEL_SPECS:
        if model_name not in baseline_results:
            continue
        print(f"Running ablations for {model_name}")
        model_dir = stage_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        auto_only_metrics = []
        auto_subset_metrics = []
        audited_subset_metrics = []
        recompute_metrics = []
        no_recompute_metrics = []
        strict_evidence_metrics = []
        relaxed_evidence_metrics = []
        try:
            loaded = load_model(model_name)
            try:
                for seed in SEEDS:
                    auto_dataset = read_json(DATA_DIR / f"seed_{seed}" / "auto_only_release.json")
                    auto_subset_dataset = read_json(DATA_DIR / f"seed_{seed}" / "auto_comparison_subset.json")
                    audited_subset_dataset = read_json(DATA_DIR / f"seed_{seed}" / "audited_comparison_subset.json")
                    recompute_dataset = read_json(DATA_DIR / f"seed_{seed}" / "recompute_validation.json")
                    no_recompute_dataset = read_json(DATA_DIR / f"seed_{seed}" / "no_recompute_validation.json")
                    strict_evidence_dataset = read_json(DATA_DIR / f"seed_{seed}" / "strict_evidence_validation.json")
                    relaxed_evidence_dataset = read_json(DATA_DIR / f"seed_{seed}" / "relaxed_evidence_validation.json")

                    auto_path = model_dir / f"auto_only_predictions_seed_{seed}.jsonl"
                    auto_subset_path = model_dir / f"auto_subset_predictions_seed_{seed}.jsonl"
                    audited_subset_path = model_dir / f"audited_subset_predictions_seed_{seed}.jsonl"
                    recompute_path = model_dir / f"recompute_predictions_seed_{seed}.jsonl"
                    strict_path = model_dir / f"strict_evidence_predictions_seed_{seed}.jsonl"
                    relaxed_path = model_dir / f"relaxed_evidence_predictions_seed_{seed}.jsonl"

                    run_predictions(loaded, auto_dataset, auto_path)
                    run_predictions(loaded, auto_subset_dataset, auto_subset_path)
                    run_predictions(loaded, audited_subset_dataset, audited_subset_path)
                    run_predictions(loaded, recompute_dataset, recompute_path)
                    run_predictions(loaded, strict_evidence_dataset, strict_path)
                    run_predictions(loaded, relaxed_evidence_dataset, relaxed_path)

                    auto_metrics = compute_metrics(_load_predictions(auto_path))
                    auto_metrics["seed"] = seed
                    auto_only_metrics.append(auto_metrics)
                    write_json(model_dir / f"auto_only_results_seed_{seed}.json", auto_metrics)

                    auto_subset_metric = compute_metrics(_load_predictions(auto_subset_path))
                    auto_subset_metric["seed"] = seed
                    auto_subset_metrics.append(auto_subset_metric)
                    write_json(model_dir / f"auto_subset_results_seed_{seed}.json", auto_subset_metric)

                    audited_subset_metric = compute_metrics(_load_predictions(audited_subset_path))
                    audited_subset_metric["seed"] = seed
                    audited_subset_metrics.append(audited_subset_metric)
                    write_json(model_dir / f"audited_subset_results_seed_{seed}.json", audited_subset_metric)

                    recompute_prediction_rows = _load_predictions(recompute_path)
                    recompute_metric = compute_metrics(recompute_prediction_rows)
                    recompute_metric["seed"] = seed
                    recompute_metrics.append(recompute_metric)
                    write_json(model_dir / f"recompute_results_seed_{seed}.json", recompute_metric)

                    no_recompute_prediction_rows = _rewrite_predictions_for_alt_gold(recompute_prediction_rows, no_recompute_dataset)
                    no_recompute_metric = compute_metrics(no_recompute_prediction_rows)
                    no_recompute_metric["seed"] = seed
                    no_recompute_metrics.append(no_recompute_metric)
                    write_json(model_dir / f"no_recompute_results_seed_{seed}.json", no_recompute_metric)

                    strict_metric = compute_metrics(_load_predictions(strict_path))
                    strict_metric["seed"] = seed
                    strict_evidence_metrics.append(strict_metric)
                    write_json(model_dir / f"strict_evidence_results_seed_{seed}.json", strict_metric)

                    relaxed_metric = compute_metrics(_load_predictions(relaxed_path))
                    relaxed_metric["seed"] = seed
                    relaxed_evidence_metrics.append(relaxed_metric)
                    write_json(model_dir / f"relaxed_evidence_results_seed_{seed}.json", relaxed_metric)

                    ranking_rows_d.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "audited_css": audited_subset_metric["overall"]["css"],
                            "auto_only_css": auto_subset_metric["overall"]["css"],
                        }
                    )
            finally:
                unload_model(loaded)
        except Exception as exc:
            model_failures[model_name] = f"{type(exc).__name__}: {exc}"
            print(f"Ablation evaluation failed for {model_name}: {exc}")
            continue

        baseline_aggregate = aggregate_model_seed_metrics(baseline_results[model_name])
        auto_subset_aggregate = aggregate_model_seed_metrics(auto_subset_metrics)
        audited_subset_aggregate = aggregate_model_seed_metrics(audited_subset_metrics)
        auto_aggregate = aggregate_model_seed_metrics(auto_only_metrics)
        recompute_aggregate = aggregate_model_seed_metrics(recompute_metrics)
        no_recompute_aggregate = aggregate_model_seed_metrics(no_recompute_metrics)
        strict_aggregate = aggregate_model_seed_metrics(strict_evidence_metrics)
        relaxed_aggregate = aggregate_model_seed_metrics(relaxed_evidence_metrics)
        baseline_rows = _model_rows(baseline_results[model_name])
        auto_subset_rows = _model_rows(auto_subset_metrics)
        audited_subset_rows = _model_rows(audited_subset_metrics)
        recompute_rows = _model_rows(recompute_metrics)
        no_recompute_rows = _model_rows(no_recompute_metrics)
        strict_rows = _model_rows(strict_evidence_metrics)
        relaxed_rows = _model_rows(relaxed_evidence_metrics)
        label_disagreement_rate = float(
            np.mean(
                [
                    sum(
                        int(row["gold_q3"] != row["raw_gold_q3_without_recompute"])
                        for row in read_json(DATA_DIR / f"seed_{seed}" / "recompute_validation.json")
                    )
                    / max(1, len(read_json(DATA_DIR / f"seed_{seed}" / "recompute_validation.json")))
                    for seed in SEEDS
                ]
            )
        )
        model_outputs[model_name] = {
            "baseline_core": baseline_aggregate,
            "ablation_A_remove_paraphrase": {
                "score": baseline_aggregate["ablations"]["ablation_remove_paraphrase"],
                "delta_vs_css_bootstrap": baseline_aggregate["bootstrap_summary"]["paired_deltas"]["remove_paraphrase_minus_css"],
                "split_replication": _ablation_split_replication(
                    baseline_aggregate, baseline_aggregate, "ablation_remove_paraphrase"
                ),
            },
            "ablation_B_remove_flip": {
                "score": baseline_aggregate["ablations"]["ablation_remove_flip"],
                "delta_vs_css_bootstrap": baseline_aggregate["bootstrap_summary"]["paired_deltas"]["remove_flip_minus_css"],
                "split_replication": _ablation_split_replication(baseline_aggregate, baseline_aggregate, "ablation_remove_flip"),
            },
            "ablation_C_include_q4": {
                "score": baseline_aggregate["ablations"]["ablation_include_q4"],
                "delta_vs_css_bootstrap": baseline_aggregate["bootstrap_summary"]["paired_deltas"]["include_q4_minus_css"],
                "split_replication": _ablation_split_replication(baseline_aggregate, baseline_aggregate, "ablation_include_q4"),
            },
            "ablation_D_auto_only_release": {
                "audited_css": audited_subset_aggregate["metrics"]["css"],
                "auto_only_css": auto_subset_aggregate["metrics"]["css"],
                "full_auto_only_release_css": auto_aggregate["metrics"]["css"],
                "delta_auto_only_minus_audited": mean_std(
                    [auto_subset_metrics[idx]["overall"]["css"] - audited_subset_metrics[idx]["overall"]["css"] for idx in range(len(SEEDS))]
                ),
                "delta_vs_css_bootstrap": paired_bootstrap_delta(
                    auto_subset_rows, audited_subset_rows, "css", "css", pair_by_key=False
                ),
                "split_replication": split_replication_from_rows(audited_subset_rows, auto_subset_rows, "css", "css"),
            },
            "ablation_E_no_recompute_labels": {
                "recomputed_css": recompute_aggregate["metrics"]["css"],
                "raw_label_css": no_recompute_aggregate["metrics"]["css"],
                "recomputed_bfa": recompute_aggregate["metrics"]["bfa"],
                "raw_label_bfa": no_recompute_aggregate["metrics"]["bfa"],
                "label_disagreement_rate": label_disagreement_rate,
                "delta_vs_css_bootstrap": paired_bootstrap_delta(
                    no_recompute_rows, recompute_rows, "css", "css", pair_by_key=True
                ),
                "delta_bfa_bootstrap": paired_bootstrap_delta(
                    no_recompute_rows, recompute_rows, "bfa", "bfa", pair_by_key=True
                ),
                "split_replication": split_replication_from_rows(recompute_rows, no_recompute_rows, "css", "css"),
            },
            "ablation_F_relaxed_evidence_policy": {
                "strict_evidence_css": strict_aggregate["metrics"]["css"],
                "relaxed_evidence_css": relaxed_aggregate["metrics"]["css"],
                "strict_quality_rates": _dataset_quality_rates(read_json(DATA_DIR / f"seed_{SEEDS[0]}" / "strict_evidence_validation.json")),
                "relaxed_quality_rates": _dataset_quality_rates(read_json(DATA_DIR / f"seed_{SEEDS[0]}" / "relaxed_evidence_validation.json")),
                "delta_vs_css_bootstrap": paired_bootstrap_delta(
                    relaxed_rows, strict_rows, "css", "css", pair_by_key=False
                ),
                "split_replication": split_replication_from_rows(strict_rows, relaxed_rows, "css", "css"),
            },
            "ablation_G_exact_string_scoring": {
                "exact_string_css": baseline_aggregate["metrics"]["ablation_exact_string_css"],
                "delta_vs_css_bootstrap": baseline_aggregate["bootstrap_summary"]["paired_deltas"]["exact_string_minus_css"],
                "split_replication": _ablation_split_replication(
                    baseline_aggregate, baseline_aggregate, "ablation_exact_string_css"
                ),
            },
        }

    audited_by_model = {
        model: float(np.mean([row["audited_css"] for row in ranking_rows_d if row["model"] == model]))
        for model in {row["model"] for row in ranking_rows_d}
    }
    auto_by_model = {
        model: float(np.mean([row["auto_only_css"] for row in ranking_rows_d if row["model"] == model]))
        for model in {row["model"] for row in ranking_rows_d}
    }
    ranking_spearman = None
    if audited_by_model and auto_by_model:
        value = spearmanr(
            [audited_by_model[model] for model in sorted(audited_by_model)],
            [auto_by_model[model] for model in sorted(audited_by_model)],
        ).statistic
        if not np.isnan(value):
            ranking_spearman = float(value)

    _write_stage_results(
        "ablation_eval",
        {
            "experiment": "ablation_eval",
            "model_failures": model_failures,
            "models": model_outputs,
            "global_ablation_D_ranking_spearman": ranking_spearman,
            "generated_at_utc": _utc_now(),
        },
    )
    ablation_rows = []
    for model_name, payload in model_outputs.items():
        for ablation_name in [
            "ablation_A_remove_paraphrase",
            "ablation_B_remove_flip",
            "ablation_C_include_q4",
            "ablation_D_auto_only_release",
            "ablation_E_no_recompute_labels",
            "ablation_F_relaxed_evidence_policy",
            "ablation_G_exact_string_scoring",
        ]:
            ablation_payload = payload[ablation_name]
            row = {"model": model_name, "ablation": ablation_name}
            if "score" in ablation_payload:
                row["score_mean"] = ablation_payload["score"]["mean"]
                row["score_std"] = ablation_payload["score"]["std"]
            if "delta_vs_css_bootstrap" in ablation_payload:
                row["delta_mean"] = ablation_payload["delta_vs_css_bootstrap"]["mean"]
                row["delta_ci_low"] = ablation_payload["delta_vs_css_bootstrap"]["ci_low"]
                row["delta_ci_high"] = ablation_payload["delta_vs_css_bootstrap"]["ci_high"]
            if "split_replication" in ablation_payload:
                row["replicates_across_splits"] = ablation_payload["split_replication"]["same_direction_across_splits"]
            ablation_rows.append(row)
    save_csv(stage_dir / "ablation_metrics.csv", ablation_rows)
    if model_failures:
        _write_stage_note(
            "ablation_eval",
            "# Ablation Failures\n\n" + "\n".join(f"- `{name}`: {reason}" for name, reason in model_failures.items()),
        )
    print("Ablation evaluation complete.")


def run_analysis() -> None:
    print("Running final analysis.")
    baseline_seed_metrics = read_json(ROOT / "exp" / "baseline_eval" / "all_model_seed_metrics.json")
    model_aggregates = {
        model: aggregate_model_seed_metrics(seed_metrics) for model, seed_metrics in baseline_seed_metrics.items()
    }
    failure_rows = []
    construction_rows = []
    appendix_cases = read_json(ROOT / "exp" / "analysis" / "appendix_cases.json")
    for model, seed_metrics in baseline_seed_metrics.items():
        for seed_metric in seed_metrics:
            total = sum(seed_metric["failure_mode_counts"].values())
            for mode, count in seed_metric["failure_mode_counts"].items():
                failure_rows.append(
                    {
                        "model": model,
                        "failure_mode": mode,
                        "fraction": count / max(1, total),
                    }
                )
    failure_summary = []
    for model in sorted({row["model"] for row in failure_rows}):
        for mode in sorted({row["failure_mode"] for row in failure_rows}):
            rows = [row for row in failure_rows if row["model"] == model and row["failure_mode"] == mode]
            if rows:
                failure_summary.append(
                    {
                        "model": model,
                        "failure_mode": mode,
                        "fraction": float(np.mean([row["fraction"] for row in rows])),
                    }
                )
    data_prep_summary = read_json(ARTIFACTS_DIR / "data_prep_summary.json")
    for summary in data_prep_summary:
        for family, stats in summary["construction_stats"].items():
            construction_rows.append(
                {
                    "seed": summary["seed"],
                    "family": family,
                    "candidate_count": stats["candidate_count"],
                    "audited_pass_count": stats["audited_pass_count"],
                    "kept_count": stats["kept_count"],
                }
            )
    generate_figures(model_aggregates, failure_summary, construction_rows)
    success = summarize_success(model_aggregates)
    manifest = read_json(ARTIFACTS_DIR / "run_manifest.json") if (ARTIFACTS_DIR / "run_manifest.json").exists() else {}
    executed_models = sorted(model_aggregates.keys())
    payload = {
        "study_scope": "synthetic_procedural_core_pilot",
        "claim_update": "This experiment is a synthetic procedural-core pilot with audited release metadata, not a TwinBench human-audited benchmark release.",
        "executed_vs_proposed": {
            "proposal_alignment": {
                "proposal_primary_roster": [
                    "qwen2.5-7b-instruct",
                    "llama-3.1-8b-instruct",
                    "gemma-2-9b-it",
                    "mistral-7b-instruct-v0.3",
                ],
                "plan_primary_roster": sorted(MODEL_SPECS.keys()),
                "executed_primary_roster": executed_models,
                "note": "The executed model roster follows plan.json rather than proposal.md. Qwen2.5-3B-Instruct was evaluated instead of the proposal's Qwen2.5-7B-Instruct."
            },
            "audit_alignment": {
                "real_human_annotation_executed": False,
                "note": "Real dual human verification and audited release protocol were not executed; all audit metadata are synthetic and claims are limited accordingly."
            },
            "evidence_alignment": {
                "evidence_slice_core_release": False,
                "note": "The evidence-grounded slice remains validation-only and is not part of the executed core release."
            },
            "environment_alignment": {
                "executed_python_version": manifest.get("observed_host_environment", {}).get("python_version", "UNKNOWN").split()[0],
                "python_3_11_available": manifest.get("plan_environment_deviations", {}).get("python_3_11_available"),
                "note": "The experiment ran under Python 3.12 because Python 3.11 was unavailable in this workspace."
            },
        },
        "construction": {
            "seeds": data_prep_summary,
            "rubric_fields": RUBRIC_FIELDS,
        },
        "models": model_aggregates,
        "ablations": read_json(ROOT / "exp" / "ablation_eval" / "results.json"),
        "success_summary": success,
        "appendix_cases_path": str(ROOT / "exp" / "analysis" / "appendix_cases.json"),
        "generated_at_utc": _utc_now(),
    }
    write_json(ROOT / "results.json", payload)
    write_json(ROOT / "exp" / "analysis" / "summary.json", payload)
    _write_stage_results(
        "analysis",
        {
            "experiment": "analysis",
            "results_path": str(ROOT / "results.json"),
            "success_summary": success,
            "appendix_cases": appendix_cases,
            "generated_at_utc": _utc_now(),
        },
    )
    _write_stage_note(
        "analysis",
        "# Scope Note\n\n- The main benchmark claim remains limited to the synthetic procedural-core pilot.\n- Evidence studies are non-release validation analyses only.\n- Any audited-release or human-verification claim remains unsupported by this workspace.\n",
    )
    print("Analysis complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["data_prep", "model_eval", "ablation_eval", "analysis", "all"])
    args = parser.parse_args()
    stage_status: dict[str, Any] = {}
    if args.stage in {"data_prep", "all"}:
        stage_status["data_prep_started_utc"] = _utc_now()
        run_data_prep()
        stage_status["data_prep_finished_utc"] = _utc_now()
    if args.stage in {"model_eval", "all"}:
        stage_status["model_eval_started_utc"] = _utc_now()
        run_model_eval()
        stage_status["model_eval_finished_utc"] = _utc_now()
    if args.stage in {"ablation_eval", "all"}:
        stage_status["ablation_eval_started_utc"] = _utc_now()
        run_ablation_eval()
        stage_status["ablation_eval_finished_utc"] = _utc_now()
    if args.stage in {"analysis", "all"}:
        stage_status["analysis_started_utc"] = _utc_now()
        run_analysis()
        stage_status["analysis_finished_utc"] = _utc_now()
    build_manifest(stage_status)


if __name__ == "__main__":
    main()

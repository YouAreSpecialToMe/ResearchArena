from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import permutation_test

from exp.shared.core import (
    DATA_DIR,
    EXPERIMENT_SEEDS,
    EXP_DIR,
    FIGURES_DIR,
    PROMPTS_DIR,
    RELATION_LABELS,
    SUPERVISION_SEEDS,
    _apply_calibrator,
    RunConfig,
    TemperatureCalibrator,
    bootstrap_ci,
    build_constraint_dataset,
    build_manual_parser_audit,
    build_splits,
    candidate_seed,
    choose_calibration,
    constraint_metrics,
    continue_from_prefix,
    ensure_dir,
    evaluate_metadata,
    family_score_map,
    full_completion,
    generate_case_study_figure,
    parse_constraints,
    read_json,
    read_jsonl,
    run_prefix,
    save_candidate_artifacts,
    scalar_metrics,
    score_decomposed_candidate,
    score_scalar_candidate,
    seed_everything,
    supervision_latent_seed,
    train_constraint_probe,
    train_scalar_probe,
    write_json,
    write_jsonl,
    append_jsonl,
    ModelBundle,
)


ROOT = Path(__file__).resolve().parents[2]
METHODS = [
    "random_prune_continue",
    "best_of_2_full_completion",
    "scalar_early_target",
    "decomposed_early_target",
    "shared_head_decomposition",
    "no_preview_features",
]
PRIMARY_METRIC = "all_correct"
AUXILIARY_METRIC = "mean_atomic_success"
PRIMARY_METRIC_LABEL = "overall score (all_correct)"
LOCAL_EVAL_PROTOCOL = {
    "name": "local_heuristic_evaluation_study",
    "score_label": "local_heuristic_geneval_style_score",
    "evaluator": "local_geneval_heuristic_with_owlv2_and_clip_color",
    "claim_level": "downgraded_from_official_geneval",
}


def log_event(experiment: str, payload: Dict[str, Any]) -> None:
    append_jsonl(EXP_DIR / experiment / "logs" / "events.jsonl", payload)


def validation_probe_outputs(model, df):
    feature_cols = [col for col in df.columns if col.startswith("f_")]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32).cuda()
    meta = torch.tensor(df[["attribute_id", "count_target", "relation_index"]].values, dtype=torch.float32).cuda()
    fam_map = {"count": 0, "attribute_binding": 1, "relation": 2}
    fam = torch.tensor([fam_map[item] for item in df["family"]], dtype=torch.long).cuda()
    with torch.no_grad():
        logits = model(x, meta, fam).cpu().numpy()
    return logits


def measure_runtime(bundle: ModelBundle, cfg: RunConfig, prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
    runtime_dir = EXP_DIR / "runtime_calibration"
    ensure_dir(runtime_dir / "logs")
    prefix_timings: List[float] = []
    prefix_decode_timings: List[float] = []
    full_timings: List[float] = []
    full_decode_timings: List[float] = []
    for idx, metadata in enumerate(prompts[:20]):
        latent_seed = 800000 + idx
        prefix = run_prefix(bundle, metadata["prompt"], latent_seed, cfg)
        prefix_timings.append(prefix["timings"]["denoise_seconds"])
        prefix_decode_timings.append(prefix["timings"]["preview_decode_seconds"])
        log_event("runtime_calibration", {"kind": "prefix", "prompt_id": metadata["prompt_id"], "latent_seed": latent_seed, "timings": prefix["timings"], "peak_gpu_gb": prefix["peak_gpu_gb"]})
    for idx, metadata in enumerate(prompts[:20]):
        latent_seed = 900000 + idx
        result = full_completion(bundle, metadata["prompt"], latent_seed, cfg)
        full_timings.append(result["timings"]["denoise_seconds"])
        full_decode_timings.append(result["timings"]["final_decode_seconds"])
        log_event("runtime_calibration", {"kind": "full", "prompt_id": metadata["prompt_id"], "latent_seed": latent_seed, "timings": result["timings"], "peak_gpu_gb": result["peak_gpu_gb"]})
    summary = {
        "prefix_denoise_seconds_mean": float(np.mean(prefix_timings)),
        "prefix_denoise_seconds_std": float(np.std(prefix_timings)),
        "prefix_decode_seconds_mean": float(np.mean(prefix_decode_timings)),
        "prefix_decode_seconds_std": float(np.std(prefix_decode_timings)),
        "full_denoise_seconds_mean": float(np.mean(full_timings)),
        "full_denoise_seconds_std": float(np.std(full_timings)),
        "full_decode_seconds_mean": float(np.mean(full_decode_timings)),
        "full_decode_seconds_std": float(np.std(full_decode_timings)),
        "projected_gpu_hours": float((640 * np.mean(full_timings) + 640 * np.mean(prefix_timings) + (80 * 3 * 6) * np.mean(prefix_timings) + (80 * 3 * 5) * np.mean(full_timings) / 20 * 16) / 3600.0),
    }
    write_json(runtime_dir / "results.json", summary)
    return summary


def prepare_supervision_rows(bundle: ModelBundle, cfg: RunConfig, split_name: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    out_dir = DATA_DIR / "runs" / split_name
    latent_dir = out_dir / "latents"
    ensure_dir(latent_dir)
    for metadata in rows:
        constraints = parse_constraints(metadata)
        for seed in SUPERVISION_SEEDS:
            latent_seed = supervision_latent_seed(metadata["source_index"], seed)
            prefix = run_prefix(bundle, metadata["prompt"], latent_seed, cfg)
            artifact_paths = save_candidate_artifacts(latent_dir / metadata["prompt_id"], prefix, f"{seed}")
            image_path = out_dir / f"{metadata['prompt_id']}_{seed}.png"
            if not image_path.exists():
                completion = continue_from_prefix(bundle, metadata["prompt"], prefix["prefix_latents"], cfg)
                completion["final_image"].save(image_path)
                timings = {**prefix["timings"], **completion["timings"]}
                peak_gpu_gb = max(prefix["peak_gpu_gb"], completion["peak_gpu_gb"])
            else:
                timings = dict(prefix["timings"])
                timings["final_decode_seconds"] = 0.0
                timings["continuation_denoise_seconds"] = 0.0
                peak_gpu_gb = prefix["peak_gpu_gb"]
            image = Image.open(image_path).convert("RGB")
            eval_start = time.perf_counter()
            evaluation = evaluate_metadata(bundle, metadata, image)
            evaluation_seconds = time.perf_counter() - eval_start
            row = {
                "split": split_name,
                "prompt_id": metadata["prompt_id"],
                "tag": metadata["tag"],
                "seed": seed,
                "latent_seed": latent_seed,
                "prompt": metadata["prompt"],
                "image_path": str(image_path),
                "preview_path": artifact_paths["preview_path"],
                "prefix_latents_path": artifact_paths["prefix_latents_path"],
                "noise_path": artifact_paths["noise_path"],
                "feature_vector": prefix["feature_vector"].tolist(),
                "preview_similarity": prefix["preview_similarity"],
                "constraints": constraints,
                "evaluation": evaluation,
                "timings": {
                    **timings,
                    "evaluation_seconds": evaluation_seconds,
                },
                "total_unet_units": cfg.tau,
                "peak_gpu_gb": peak_gpu_gb,
            }
            out_rows.append(row)
            log_event(
                "supervision_generation",
                {
                    "split": split_name,
                    "prompt_id": metadata["prompt_id"],
                    "seed": seed,
                    "latent_seed": latent_seed,
                    "image_path": str(image_path),
                    "timings": row["timings"],
                    "evaluation": evaluation,
                },
            )
    return out_rows


def _select_manual_subset(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    subset: List[Dict[str, Any]] = []
    for tag in ["counting", "position", "color_attr"]:
        tag_rows = [row for row in rows if row["tag"] == tag]
        tag_rows.sort(key=lambda item: (item["evaluation"]["mean_atomic_success"], item["prompt_id"], item["seed"]))
        low = tag_rows[:14]
        high = tag_rows[-13:]
        subset.extend(low + high)
    subset = subset[:80]
    return subset


def ensure_manual_label_template(rows: List[Dict[str, Any]]) -> Path:
    out_path = EXP_DIR / "manual_label_sensitivity" / "manual_labels.csv"
    if out_path.exists():
        return out_path
    subset = _select_manual_subset(rows)
    records: List[Dict[str, Any]] = []
    for row in subset:
        for constraint, auto in zip(row["constraints"], row["evaluation"]["atomic"]):
            records.append(
                {
                    "image_id": f"{row['prompt_id']}_{row['seed']}",
                    "prompt_id": row["prompt_id"],
                    "seed": row["seed"],
                    "image_path": row["image_path"],
                    "prompt": row["prompt"],
                    "constraint_id": constraint["constraint_id"],
                    "constraint_family": constraint["family"],
                    "constraint_text": json.dumps(constraint),
                    "auto_label": auto["label"],
                    "human_label": auto["label"],
                    "notes": "Initial value seeded from auto evaluation; replace by visual review before sensitivity run.",
                }
            )
    pd.DataFrame.from_records(records).to_csv(out_path, index=False)
    return out_path


def build_manual_sensitivity(
    manual_csv: Path,
    decomp_model,
    scalar_model,
    train_val_rows: List[Dict[str, Any]],
) -> Tuple[Any, Any, Dict[str, Any]]:
    manual_df = pd.read_csv(manual_csv)
    notes = manual_df["notes"].fillna("").str.lower()
    review_complete = bool(((notes.str.contains("visual review")) & (~notes.str.contains("replace by visual review"))).any())

    constraint_rows = []
    scalar_group = []
    feature_map = {(row["prompt_id"], row["seed"]): np.array(row["feature_vector"], dtype=np.float32) for row in train_val_rows}
    for _, item in manual_df.iterrows():
        feature = feature_map[(item["prompt_id"], int(item["seed"]))]
        constraint = json.loads(item["constraint_text"])
        score = score_decomposed_candidate(decomp_model, TemperatureCalibrator(), feature, [constraint])["per_constraint"][0]
        constraint_rows.append({"score": score, "label": float(item["human_label"])})
    grouped = manual_df.groupby(["prompt_id", "seed"], as_index=False)["human_label"].mean()
    for _, item in grouped.iterrows():
        feature = feature_map[(item["prompt_id"], int(item["seed"]))]
        score = score_scalar_candidate(scalar_model, TemperatureCalibrator(), feature)
        scalar_group.append({"score": score, "label": float(item["human_label"])})

    decomp_cal = TemperatureCalibrator().fit(np.log(np.clip([r["score"] for r in constraint_rows], 1e-4, 1 - 1e-4) / np.clip(1 - np.array([r["score"] for r in constraint_rows]), 1e-4, 1.0)), np.array([r["label"] for r in constraint_rows]))
    scalar_cal = TemperatureCalibrator().fit(np.log(np.clip([r["score"] for r in scalar_group], 1e-4, 1 - 1e-4) / np.clip(1 - np.array([r["score"] for r in scalar_group]), 1e-4, 1.0)), np.array([r["label"] for r in scalar_group]))
    summary = {
        "manual_label_csv": str(manual_csv),
        "review_complete": review_complete,
        "n_constraint_labels": int(len(constraint_rows)),
        "n_images": int(len(grouped)),
    }
    return decomp_cal, scalar_cal, summary


def has_completed_manual_review(manual_csv: Path) -> Tuple[bool, Dict[str, Any]]:
    manual_df = pd.read_csv(manual_csv)
    notes = manual_df["notes"].fillna("").str.lower()
    reviewed = notes.str.contains("visual review") & (~notes.str.contains("replace by visual review"))
    summary = {
        "manual_label_csv": str(manual_csv),
        "review_complete": bool(reviewed.any()),
        "n_constraint_labels": int(len(manual_df)),
        "n_images": int(manual_df[["prompt_id", "seed"]].drop_duplicates().shape[0]),
    }
    return summary["review_complete"], summary


def calibration_curve_points(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> List[Dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: List[Dict[str, Any]] = []
    for i in range(bins):
        left = float(edges[i])
        right = float(edges[i + 1])
        mask = (probs >= left) & (probs < right if i < bins - 1 else probs <= right)
        if np.any(mask):
            rows.append(
                {
                    "bin_left": left,
                    "bin_right": right,
                    "mean_confidence": float(probs[mask].mean()),
                    "empirical_accuracy": float(labels[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return rows


def plot_calibration_curves(curves: Dict[str, List[Dict[str, Any]]], out_prefix: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    ordered = [("count", "Count"), ("attribute_binding", "Attribute"), ("relation", "Relation")]
    for ax, (family_key, title) in zip(axes, ordered):
        rows = curves.get(family_key, [])
        x = [row["mean_confidence"] for row in rows]
        y = [row["empirical_accuracy"] for row in rows]
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        if x:
            ax.plot(x, y, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    ensure_dir(out_prefix.parent)
    fig.savefig(out_prefix.with_suffix(".png"))
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def plot_margin_scatter(rows: List[Dict[str, Any]], out_prefix: Path) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(frame["score_margin"], frame["final_gain"], alpha=0.7)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Step-4 decomposed minus scalar score margin")
    ax.set_ylabel("Final local heuristic score gain")
    ax.set_title("Early score margin vs final gain")
    fig.tight_layout()
    ensure_dir(out_prefix.parent)
    fig.savefig(out_prefix.with_suffix(".png"))
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def write_main_summary(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Main Summary",
        "",
        "## Outcome",
        payload["outcome_line"],
        "",
        "## Main comparison",
        payload["comparison_line"],
        "",
        "## Evaluation protocol",
        payload["evaluation_line"],
        "",
        "## Planned deviations",
    ]
    lines.extend(f"- {item}" for item in payload["deviations"])
    lines.extend(
        [
            "",
            "## Diagnostics",
            f"- Seed-11 continuation hit rate: {payload['hit_rate_line']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def make_method_trial(
    bundle: ModelBundle,
    cfg: RunConfig,
    metadata: Dict[str, Any],
    seed: int,
    method: str,
    chosen_candidate: Dict[str, Any],
    candidate_scores: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    start = time.perf_counter()
    completion = continue_from_prefix(bundle, metadata["prompt"], torch.load(chosen_candidate["prefix_latents_path"]), cfg)
    image_path = output_dir / f"{metadata['prompt_id']}_{seed}_{method}.png"
    ensure_dir(image_path.parent)
    completion["final_image"].save(image_path)
    eval_start = time.perf_counter()
    evaluation = evaluate_metadata(bundle, metadata, completion["final_image"])
    evaluation_seconds = time.perf_counter() - eval_start
    wall_clock = time.perf_counter() - start + chosen_candidate["prefix_wall_clock_seconds"]
    row = {
        "prompt_id": metadata["prompt_id"],
        "prompt": metadata["prompt"],
        "tag": metadata["tag"],
        "seed": seed,
        "method": method,
        "candidate_scores": candidate_scores,
        "selected_candidate": chosen_candidate["candidate_index"],
        "candidate_seed_list": [item["latent_seed"] for item in candidate_scores["candidates"]],
        "final_image_path": str(image_path),
        "preview_paths": [item["preview_path"] for item in candidate_scores["candidates"]],
        "total_unet_units": cfg.tau * 6 + (cfg.num_steps - cfg.tau),
        "wall_clock_seconds": wall_clock,
        "timings": {
            "prefix_pool_seconds": chosen_candidate["prefix_wall_clock_seconds"],
            "continuation_denoise_seconds": completion["timings"]["denoise_seconds"],
            "final_decode_seconds": completion["timings"]["final_decode_seconds"],
            "probe_feature_seconds": completion["timings"]["probe_feature_seconds"],
            "evaluation_seconds": evaluation_seconds,
        },
        "peak_gpu_gb": completion["peak_gpu_gb"],
        "mean_atomic_success": evaluation["mean_atomic_success"],
        "all_correct": evaluation["all_correct"],
        "evaluation": evaluation,
    }
    log_event(method, row)
    return row


def prepare_test_candidate_pool(bundle: ModelBundle, cfg: RunConfig, metadata: Dict[str, Any], seed: int) -> Tuple[List[Dict[str, Any]], float]:
    base_dir = DATA_DIR / "test_candidates" / metadata["prompt_id"] / str(seed)
    ensure_dir(base_dir)
    candidates: List[Dict[str, Any]] = []
    pool_start = time.perf_counter()
    for cand_idx in range(6):
        latent_seed = candidate_seed(metadata["prompt_id"], seed, cand_idx)
        prefix = run_prefix(bundle, metadata["prompt"], latent_seed, cfg)
        artifact_paths = save_candidate_artifacts(base_dir, prefix, f"cand_{cand_idx}")
        candidate = {
            "candidate_index": cand_idx,
            "latent_seed": latent_seed,
            "feature_vector": prefix["feature_vector"],
            "preview_similarity": prefix["preview_similarity"],
            "preview_path": artifact_paths["preview_path"],
            "prefix_latents_path": artifact_paths["prefix_latents_path"],
            "noise_path": artifact_paths["noise_path"],
            "prefix_timings": prefix["timings"],
            "prefix_wall_clock_seconds": sum(prefix["timings"].values()),
        }
        candidates.append(candidate)
    wall_clock = time.perf_counter() - pool_start
    return candidates, wall_clock


def metric_family_score_map(results: List[Dict[str, Any]], metric: str) -> Dict[str, float]:
    frame = pd.DataFrame(results)
    return {
        "overall": float(frame[metric].mean()),
        "count": float(frame.loc[frame["tag"] == "counting", metric].mean()),
        "attribute_binding": float(frame.loc[frame["tag"] == "color_attr", metric].mean()),
        "relation": float(frame.loc[frame["tag"] == "position", metric].mean()),
    }


def aggregate_method_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    frame = pd.DataFrame(rows)
    metrics = metric_family_score_map(rows, PRIMARY_METRIC)
    seed_scores = frame.groupby("seed")[PRIMARY_METRIC].mean().tolist()
    prompt_scores = frame.groupby("prompt_id")[PRIMARY_METRIC].mean().to_numpy()
    count_prompt_scores = frame.loc[frame["tag"] == "counting"].groupby("prompt_id")[PRIMARY_METRIC].mean().to_numpy()
    attr_prompt_scores = frame.loc[frame["tag"] == "color_attr"].groupby("prompt_id")[PRIMARY_METRIC].mean().to_numpy()
    relation_prompt_scores = frame.loc[frame["tag"] == "position"].groupby("prompt_id")[PRIMARY_METRIC].mean().to_numpy()
    metrics.update(
        {
            "primary_metric": PRIMARY_METRIC,
            "auxiliary_metric": AUXILIARY_METRIC,
            "overall_std": float(np.std(seed_scores)),
            "mean_seconds_per_prompt": float(frame["wall_clock_seconds"].mean()),
            "mean_unet_units": float(frame["total_unet_units"].mean()),
            "peak_gpu_gb": float(frame["peak_gpu_gb"].max()),
            "mean_atomic_success_aux": float(frame[AUXILIARY_METRIC].mean()),
            "overall_ci95_prompt": list(bootstrap_ci(prompt_scores)),
            "count_ci95_prompt": list(bootstrap_ci(count_prompt_scores)),
            "attribute_binding_ci95_prompt": list(bootstrap_ci(attr_prompt_scores)),
            "relation_ci95_prompt": list(bootstrap_ci(relation_prompt_scores)),
            "runtime_breakdown_fraction": {
                key: float(np.mean([row["timings"].get(key, 0.0) for row in rows]) / max(frame["wall_clock_seconds"].mean(), 1e-8))
                for key in ["prefix_pool_seconds", "continuation_denoise_seconds", "final_decode_seconds", "probe_feature_seconds", "evaluation_seconds"]
            },
        }
    )
    return metrics


def compute_supervision_agreement(
    local_rows: List[Dict[str, Any]],
    official_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    by_family: Dict[str, Dict[str, List[float]]] = {}
    for local_row, official_row in zip(local_rows, official_rows):
        family = local_row["tag"]
        bucket = by_family.setdefault(
            family,
            {"prompt": [], "atomic": [], "local_mean_atomic_success": [], "official_mean_atomic_success": []},
        )
        local_atomic = [item["label"] for item in local_row["evaluation"]["atomic"]]
        official_atomic = [item["label"] for item in official_row["evaluation"]["atomic"]]
        bucket["prompt"].append(float(int(all(a == b for a, b in zip(local_atomic, official_atomic)))))
        bucket["local_mean_atomic_success"].append(float(local_row["evaluation"]["mean_atomic_success"]))
        bucket["official_mean_atomic_success"].append(float(official_row["evaluation"]["mean_atomic_success"]))
        for a, b in zip(local_atomic, official_atomic):
            bucket["atomic"].append(float(int(a == b)))

    summary: Dict[str, Any] = {}
    all_prompt: List[float] = []
    all_atomic: List[float] = []
    all_local: List[float] = []
    all_official: List[float] = []
    for family, bucket in by_family.items():
        summary[family] = {
            "images": int(len(bucket["prompt"])),
            "prompt_agreement": float(np.mean(bucket["prompt"])),
            "atomic_agreement": float(np.mean(bucket["atomic"])),
            "local_mean_atomic_success": float(np.mean(bucket["local_mean_atomic_success"])),
            "official_mean_atomic_success": float(np.mean(bucket["official_mean_atomic_success"])),
        }
        all_prompt.extend(bucket["prompt"])
        all_atomic.extend(bucket["atomic"])
        all_local.extend(bucket["local_mean_atomic_success"])
        all_official.extend(bucket["official_mean_atomic_success"])

    summary["overall"] = {
        "images": int(len(all_prompt)),
        "prompt_agreement": float(np.mean(all_prompt)),
        "atomic_agreement": float(np.mean(all_atomic)),
        "local_mean_atomic_success": float(np.mean(all_local)),
        "official_mean_atomic_success": float(np.mean(all_official)),
    }
    return summary


def run() -> None:
    seed_everything(0)
    cfg = RunConfig()
    for experiment in ["data_preparation", "runtime_calibration", "supervision_generation", "probe_training", "manual_label_sensitivity", *METHODS]:
        ensure_dir(EXP_DIR / experiment / "logs")

    splits = build_splits()
    parser_audit = build_manual_parser_audit(splits)
    write_json(
        EXP_DIR / "data_preparation" / "results.json",
        {
            "parser_audit": parser_audit,
            "split_sizes": {key: len(value) for key, value in splits.items()},
            "prompt_files": {key: str(PROMPTS_DIR / f"{key}_prompts.json") for key in splits},
        },
    )

    bundle = ModelBundle(cfg)
    runtime_path = EXP_DIR / "runtime_calibration" / "results.json"
    runtime = read_json(runtime_path) if runtime_path.exists() else measure_runtime(bundle, cfg, splits["train"])

    supervision_path = EXP_DIR / "supervision_generation" / "results.jsonl"
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    if supervision_path.exists():
        all_supervision_rows = read_jsonl(supervision_path)
        train_rows = [row for row in all_supervision_rows if row.get("split") == "train"]
        val_rows = [row for row in all_supervision_rows if row.get("split") == "val"]
        if len(train_rows) != len(splits["train"]) * len(SUPERVISION_SEEDS) or len(val_rows) != len(splits["val"]) * len(SUPERVISION_SEEDS):
            all_supervision_rows = []
            train_rows = []
            val_rows = []
    if not train_rows or not val_rows:
        train_rows = prepare_supervision_rows(bundle, cfg, "train", splits["train"])
        val_rows = prepare_supervision_rows(bundle, cfg, "val", splits["val"])
        all_supervision_rows = train_rows + val_rows
        write_jsonl(supervision_path, all_supervision_rows)

    manual_csv = ensure_manual_label_template(all_supervision_rows)
    manual_review_complete, manual_review_summary = has_completed_manual_review(manual_csv)

    train_df = build_constraint_dataset(train_rows, include_preview=True)
    val_df = build_constraint_dataset(val_rows, include_preview=True)
    train_df_no_preview = build_constraint_dataset(train_rows, include_preview=False)
    val_df_no_preview = build_constraint_dataset(val_rows, include_preview=False)

    decomp_model, decomp_train = train_constraint_probe(train_df, val_df, shared_head=False)
    shared_model, shared_train = train_constraint_probe(train_df, val_df, shared_head=True)
    decomp_no_preview_model, decomp_np_train = train_constraint_probe(train_df_no_preview, val_df_no_preview, shared_head=False)
    scalar_model, scalar_train = train_scalar_probe(train_rows, val_rows, include_preview=True, target="mean_atomic_success")
    scalar_all_model, scalar_all_train = train_scalar_probe(train_rows, val_rows, include_preview=True, target="all_correct")

    decomp_logits = validation_probe_outputs(decomp_model, val_df)
    decomp_cal_type, decomp_cal, decomp_cal_score = choose_calibration(decomp_logits, val_df["label"].values)
    shared_logits = validation_probe_outputs(shared_model, val_df)
    shared_cal_type, shared_cal, shared_cal_score = choose_calibration(shared_logits, val_df["label"].values)
    np_logits = validation_probe_outputs(decomp_no_preview_model, val_df_no_preview)
    np_cal_type, np_cal, np_cal_score = choose_calibration(np_logits, val_df_no_preview["label"].values)

    scalar_x = np.stack([np.array(row["feature_vector"], dtype=np.float32) for row in val_rows])
    with torch.no_grad():
        scalar_logits = scalar_model(torch.tensor(scalar_x, dtype=torch.float32).cuda()).cpu().numpy()
        scalar_all_logits = scalar_all_model(torch.tensor(scalar_x, dtype=torch.float32).cuda()).cpu().numpy()
    scalar_cal_type, scalar_cal, scalar_cal_score = choose_calibration(scalar_logits, np.array([row["evaluation"]["mean_atomic_success"] for row in val_rows]))
    scalar_all_cal_type, scalar_all_cal, scalar_all_cal_score = choose_calibration(scalar_all_logits, np.array([row["evaluation"]["all_correct"] for row in val_rows]))

    probe_metrics = {
        "decomposed": {
            "train": decomp_train,
            "validation": constraint_metrics(val_df["label"].values, decomp_cal.transform(decomp_logits) if isinstance(decomp_cal, TemperatureCalibrator) else decomp_cal.predict(1 / (1 + np.exp(-decomp_logits)))),
            "calibration": {"type": decomp_cal_type, "score": decomp_cal_score},
        },
        "shared_head": {
            "train": shared_train,
            "validation": constraint_metrics(val_df["label"].values, shared_cal.transform(shared_logits) if isinstance(shared_cal, TemperatureCalibrator) else shared_cal.predict(1 / (1 + np.exp(-shared_logits)))),
            "calibration": {"type": shared_cal_type, "score": shared_cal_score},
        },
        "no_preview": {
            "train": decomp_np_train,
            "validation": constraint_metrics(val_df_no_preview["label"].values, np_cal.transform(np_logits) if isinstance(np_cal, TemperatureCalibrator) else np_cal.predict(1 / (1 + np.exp(-np_logits)))),
            "calibration": {"type": np_cal_type, "score": np_cal_score},
        },
        "scalar_mean_atomic": {
            "train": scalar_train,
            "validation": scalar_metrics(np.array([row["evaluation"]["mean_atomic_success"] for row in val_rows]), scalar_cal.transform(scalar_logits) if isinstance(scalar_cal, TemperatureCalibrator) else scalar_cal.predict(1 / (1 + np.exp(-scalar_logits)))),
            "calibration": {"type": scalar_cal_type, "score": scalar_cal_score},
        },
        "scalar_all_correct": {
            "train": scalar_all_train,
            "validation": scalar_metrics(np.array([row["evaluation"]["all_correct"] for row in val_rows]), scalar_all_cal.transform(scalar_all_logits) if isinstance(scalar_all_cal, TemperatureCalibrator) else scalar_all_cal.predict(1 / (1 + np.exp(-scalar_all_logits)))),
            "calibration": {"type": scalar_all_cal_type, "score": scalar_all_cal_score},
        },
    }
    write_json(EXP_DIR / "probe_training" / "results.json", probe_metrics)

    official_supervision_path = EXP_DIR / "official_geneval_supervision" / "results.jsonl"
    supervision_agreement: Dict[str, Any] | None = None
    if official_supervision_path.exists():
        official_supervision_rows = read_jsonl(official_supervision_path)
        if len(official_supervision_rows) == len(all_supervision_rows):
            supervision_agreement = compute_supervision_agreement(all_supervision_rows, official_supervision_rows)
            write_json(EXP_DIR / "official_geneval_supervision" / "agreement_summary.json", supervision_agreement)

    method_outputs = {name: [] for name in METHODS}
    case_studies: List[Dict[str, Any]] = []
    score_margin_rows: List[Dict[str, Any]] = []

    for seed in EXPERIMENT_SEEDS:
        for metadata in splits["test"]:
            constraints = parse_constraints(metadata)
            candidates, prefix_wall_clock = prepare_test_candidate_pool(bundle, cfg, metadata, seed)
            for candidate in candidates:
                candidate["prefix_wall_clock_seconds"] = prefix_wall_clock

            scalar_scores = [score_scalar_candidate(scalar_model, scalar_cal, np.array(item["feature_vector"], dtype=np.float32)) for item in candidates]
            decomp_outputs = [score_decomposed_candidate(decomp_model, decomp_cal, np.array(item["feature_vector"], dtype=np.float32), constraints) for item in candidates]
            shared_outputs = [score_decomposed_candidate(shared_model, shared_cal, np.array(item["feature_vector"], dtype=np.float32), constraints) for item in candidates]
            np_outputs = [score_decomposed_candidate(decomp_no_preview_model, np_cal, np.array(item["feature_vector"][:-10], dtype=np.float32), constraints) for item in candidates]
            decomp_scores = [item["score"] for item in decomp_outputs]
            shared_scores = [item["score"] for item in shared_outputs]
            np_scores = [item["score"] for item in np_outputs]

            candidate_payload = {
                "candidates": candidates,
                "scalar": scalar_scores,
                "decomposed": decomp_scores,
                "decomposed_per_family": [item["per_family"] for item in decomp_outputs],
                "shared_head": shared_scores,
                "no_preview": np_scores,
            }
            random_pick = int(candidate_seed(metadata["prompt_id"], seed, 999) % 6)
            picks = {
                "random_prune_continue": random_pick,
                "scalar_early_target": int(np.argmax(scalar_scores)),
                "decomposed_early_target": int(np.argmax(decomp_scores)),
                "shared_head_decomposition": int(np.argmax(shared_scores)),
                "no_preview_features": int(np.argmax(np_scores)),
            }

            for method, pick in picks.items():
                trial = make_method_trial(bundle, cfg, metadata, seed, method, candidates[pick], candidate_payload, DATA_DIR / "test_outputs" / method)
                method_outputs[method].append(trial)

            best2_candidates = []
            for cand_idx in range(2):
                latent_seed = candidate_seed(metadata["prompt_id"], seed, cand_idx)
                result = full_completion(bundle, metadata["prompt"], latent_seed, cfg)
                image_path = DATA_DIR / "test_outputs" / "best_of_2_full_completion" / f"{metadata['prompt_id']}_{seed}_cand{cand_idx}.png"
                ensure_dir(image_path.parent)
                result["final_image"].save(image_path)
                evaluation = evaluate_metadata(bundle, metadata, result["final_image"])
                best2_candidates.append(
                    {
                        "candidate_index": cand_idx,
                        "latent_seed": latent_seed,
                        "score": result["final_similarity"],
                        "evaluation": evaluation,
                        "image_path": str(image_path),
                        "timings": result["timings"],
                        "peak_gpu_gb": result["peak_gpu_gb"],
                    }
                )
            chosen_best2 = sorted(best2_candidates, key=lambda item: item["score"], reverse=True)[0]
            best2_trial = {
                "prompt_id": metadata["prompt_id"],
                "prompt": metadata["prompt"],
                "tag": metadata["tag"],
                "seed": seed,
                "method": "best_of_2_full_completion",
                "candidate_scores": {"best_of_2_final_similarity": [item["score"] for item in best2_candidates]},
                "selected_candidate": chosen_best2["candidate_index"],
                "candidate_seed_list": [item["latent_seed"] for item in best2_candidates],
                "final_image_path": chosen_best2["image_path"],
                "preview_paths": [],
                "total_unet_units": cfg.num_steps * 2,
                "wall_clock_seconds": float(sum(item["timings"]["denoise_seconds"] + item["timings"]["final_decode_seconds"] + item["timings"]["probe_feature_seconds"] for item in best2_candidates)),
                "timings": {
                    "prefix_pool_seconds": 0.0,
                    "continuation_denoise_seconds": float(sum(item["timings"]["denoise_seconds"] for item in best2_candidates)),
                    "final_decode_seconds": float(sum(item["timings"]["final_decode_seconds"] for item in best2_candidates)),
                    "probe_feature_seconds": float(sum(item["timings"]["probe_feature_seconds"] for item in best2_candidates)),
                    "evaluation_seconds": 0.0,
                },
                "peak_gpu_gb": float(max(item["peak_gpu_gb"] for item in best2_candidates)),
                "mean_atomic_success": chosen_best2["evaluation"]["mean_atomic_success"],
                "all_correct": chosen_best2["evaluation"]["all_correct"],
                "evaluation": chosen_best2["evaluation"],
            }
            method_outputs["best_of_2_full_completion"].append(best2_trial)
            log_event("best_of_2_full_completion", best2_trial)

            decomp_pick = int(np.argmax(decomp_scores))
            scalar_pick = int(np.argmax(scalar_scores))
            score_margin_rows.append(
                {
                    "prompt_id": metadata["prompt_id"],
                    "seed": seed,
                    "score_margin": float(decomp_scores[decomp_pick] - scalar_scores[scalar_pick]),
                    "final_gain": float(
                        method_outputs["decomposed_early_target"][-1]["mean_atomic_success"]
                        - method_outputs["scalar_early_target"][-1]["mean_atomic_success"]
                    ),
                }
            )
            if len(case_studies) < 6 and decomp_pick != scalar_pick:
                case_studies.append(
                    {
                        "prompt": metadata["prompt"],
                        "preview_paths": [item["preview_path"] for item in candidates],
                    }
                )

    generate_case_study_figure(case_studies, FIGURES_DIR / "qualitative_cases")

    summary_rows = []
    for method, rows in method_outputs.items():
        scores = aggregate_method_results(rows)
        scores.update({"method": method, "budget": 40 if method != "best_of_2_full_completion" else 40})
        summary_rows.append(scores)
        write_json(EXP_DIR / method / "results.json", {"experiment": method, "metrics": scores, "n_trials": len(rows)})
        write_jsonl(EXP_DIR / method / "logs" / "trials.jsonl", rows)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(EXP_DIR / "main_results_table.csv", index=False)
    (EXP_DIR / "main_results_table.md").write_text(summary_df.to_markdown(index=False))
    write_jsonl(EXP_DIR / "main_experiment" / "trials.jsonl", [row for rows in method_outputs.values() for row in rows])

    frame = pd.DataFrame([row for rows in method_outputs.values() for row in rows])
    prompt_level = frame.pivot_table(index=["prompt_id", "method"], values=PRIMARY_METRIC, aggfunc="mean").reset_index()
    prompt_pivot = prompt_level.pivot(index="prompt_id", columns="method", values=PRIMARY_METRIC)
    prompt_delta = prompt_pivot["decomposed_early_target"] - prompt_pivot["scalar_early_target"]
    best2_prompt_delta = prompt_pivot["decomposed_early_target"] - prompt_pivot["best_of_2_full_completion"]
    ci_prompt = bootstrap_ci(prompt_delta.values)
    ci_best2_prompt = bootstrap_ci(best2_prompt_delta.values)
    ptest = permutation_test(
        (prompt_pivot["decomposed_early_target"].values, prompt_pivot["scalar_early_target"].values),
        lambda x, y: np.mean(x - y),
        vectorized=False,
        n_resamples=10000,
        random_state=0,
        permutation_type="samples",
    )

    aux_prompt_level = frame.pivot_table(index=["prompt_id", "method"], values=AUXILIARY_METRIC, aggfunc="mean").reset_index()
    aux_prompt_pivot = aux_prompt_level.pivot(index="prompt_id", columns="method", values=AUXILIARY_METRIC)
    aux_prompt_delta = aux_prompt_pivot["decomposed_early_target"] - aux_prompt_pivot["scalar_early_target"]
    aux_best2_prompt_delta = aux_prompt_pivot["decomposed_early_target"] - aux_prompt_pivot["best_of_2_full_completion"]
    aux_ci_prompt = bootstrap_ci(aux_prompt_delta.values)
    aux_ci_best2_prompt = bootstrap_ci(aux_best2_prompt_delta.values)

    fig, ax = plt.subplots(figsize=(6, 4))
    means = [float(prompt_delta.mean()), float(best2_prompt_delta.mean())]
    cis = [ci_prompt, ci_best2_prompt]
    ax.errorbar(
        [0, 1],
        means,
        yerr=[[means[0] - cis[0][0], means[1] - cis[1][0]], [cis[0][1] - means[0], cis[1][1] - means[1]]],
        fmt="o",
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks([0, 1], ["Decomp-Scalar", "Decomp-Best2"])
    ax.set_ylabel("Prompt-level local heuristic score delta")
    ax.set_title("Bootstrap intervals")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bootstrap_intervals.png")
    fig.savefig(FIGURES_DIR / "bootstrap_intervals.pdf")
    plt.close(fig)

    decomp_family_probs = _apply_calibrator(
        decomp_cal if isinstance(decomp_cal, TemperatureCalibrator) else decomp_cal,
        decomp_logits,
    )
    calibration_curves: Dict[str, List[Dict[str, Any]]] = {}
    for family in ["count", "attribute_binding", "relation"]:
        family_mask = val_df["family"] == family
        calibration_curves[family] = calibration_curve_points(
            val_df.loc[family_mask, "label"].to_numpy(dtype=float),
            np.asarray(decomp_family_probs)[family_mask.to_numpy()],
        )
    plot_calibration_curves(calibration_curves, FIGURES_DIR / "calibration_curves")
    plot_margin_scatter(score_margin_rows, FIGURES_DIR / "score_margin_scatter")

    oracle_rows = []
    oracle_diagnostics = []
    for metadata in splits["test"]:
        seed = 11
        candidates, _ = prepare_test_candidate_pool(bundle, cfg, metadata, seed)
        scalar_scores = [score_scalar_candidate(scalar_model, scalar_cal, np.array(item["feature_vector"], dtype=np.float32)) for item in candidates]
        decomp_scores = [score_decomposed_candidate(decomp_model, decomp_cal, np.array(item["feature_vector"], dtype=np.float32), parse_constraints(metadata))["score"] for item in candidates]
        candidate_final_scores: List[float] = []
        for candidate in candidates:
            completion = continue_from_prefix(bundle, metadata["prompt"], torch.load(candidate["prefix_latents_path"]), cfg)
            evaluation = evaluate_metadata(bundle, metadata, completion["final_image"])
            candidate_final_scores.append(float(evaluation["mean_atomic_success"]))
        oracle_idx = int(np.argmax(candidate_final_scores))
        scalar_idx = int(np.argmax(scalar_scores))
        decomp_idx = int(np.argmax(decomp_scores))
        oracle_rows.append(
            {
                "prompt_id": metadata["prompt_id"],
                "seed": seed,
                "oracle_index": oracle_idx,
                "scalar_index": scalar_idx,
                "decomposed_index": decomp_idx,
                "oracle_score": float(candidate_final_scores[oracle_idx]),
                "scalar_score": float(candidate_final_scores[scalar_idx]),
                "decomposed_score": float(candidate_final_scores[decomp_idx]),
            }
        )
        oracle_diagnostics.append(
            {
                "prompt_id": metadata["prompt_id"],
                "decomposed_hit": int(decomp_idx == oracle_idx),
                "scalar_hit": int(scalar_idx == oracle_idx),
            }
        )
    hit_rate_summary = {
        "seed": 11,
        "decomposed_hit_rate": float(np.mean([row["decomposed_hit"] for row in oracle_diagnostics])),
        "scalar_hit_rate": float(np.mean([row["scalar_hit"] for row in oracle_diagnostics])),
        "n_prompts": int(len(oracle_diagnostics)),
        "path": str(EXP_DIR / "main_experiment" / "hit_rate_diagnostic.json"),
    }
    write_json(EXP_DIR / "main_experiment" / "hit_rate_diagnostic.json", {"summary": hit_rate_summary, "rows": oracle_rows})

    manual_sensitivity_payload: Dict[str, Any]
    if manual_review_complete:
        decomp_manual_cal, scalar_manual_cal, manual_summary = build_manual_sensitivity(manual_csv, decomp_model, scalar_model, all_supervision_rows)
        manual_outputs = []
        for metadata in splits["test"]:
            seed = 11
            constraints = parse_constraints(metadata)
            candidates, prefix_wall_clock = prepare_test_candidate_pool(bundle, cfg, metadata, seed)
            decomp_scores = [score_decomposed_candidate(decomp_model, decomp_manual_cal, np.array(item["feature_vector"], dtype=np.float32), constraints)["score"] for item in candidates]
            scalar_scores = [score_scalar_candidate(scalar_model, scalar_manual_cal, np.array(item["feature_vector"], dtype=np.float32)) for item in candidates]
            pick = int(np.argmax(decomp_scores))
            trial = make_method_trial(bundle, cfg, metadata, seed, "manual_label_sensitivity", candidates[pick] | {"prefix_wall_clock_seconds": prefix_wall_clock}, {"candidates": candidates, "decomposed_manual": decomp_scores, "scalar_manual": scalar_scores}, DATA_DIR / "test_outputs" / "manual_label_sensitivity")
            manual_outputs.append(trial)
        manual_metrics = aggregate_method_results(manual_outputs)
        write_json(EXP_DIR / "manual_label_sensitivity" / "results.json", {"summary": manual_summary, "metrics": manual_metrics})
        write_jsonl(EXP_DIR / "manual_label_sensitivity" / "logs" / "trials.jsonl", manual_outputs)
        manual_sensitivity_payload = {"status": "completed", "summary": manual_summary, "metrics": manual_metrics}
    else:
        skip_reason = (
            "Omitted because no real human annotations were collected. "
            "The CSV remains a prefilled review sheet with automatic labels and blank human_label fields, so reporting a manual-label sensitivity number would be misleading."
        )
        (EXP_DIR / "manual_label_sensitivity" / "SKIPPED.md").write_text(skip_reason + "\n")
        write_json(EXP_DIR / "manual_label_sensitivity" / "results.json", {"status": "omitted", "reason": skip_reason, "summary": manual_review_summary})
        write_jsonl(EXP_DIR / "manual_label_sensitivity" / "logs" / "trials.jsonl", [])
        manual_sensitivity_payload = {"status": "omitted", "reason": skip_reason, "summary": manual_review_summary}

    negative_result = (
        float(prompt_delta.mean()) <= 0.0
        or float(ci_prompt[0]) <= 0.0 <= float(ci_prompt[1])
        or float(summary_df.loc[summary_df["method"] == "decomposed_early_target", "overall"].iloc[0])
        < float(summary_df.loc[summary_df["method"] == "best_of_2_full_completion", "overall"].iloc[0])
    )
    deviation_lines = [
        "Official GenEval was not run because the required MMDetection stack is unavailable in this environment; all reported scores come from a local heuristic evaluator using OWLv2 detections and CLIP-based color classification.",
        "The study should therefore be read as a local heuristic evaluation study, not an official GenEval benchmark result.",
        "The planned manual-label sensitivity analysis was omitted because no genuine human annotations were collected.",
    ]
    summary_payload = {
        "outcome_line": "Negative result: the core hypothesis is unsupported in this corrected local-heuristic study." if negative_result else "Result: the core hypothesis remains provisionally supported.",
        "comparison_line": (
            f"Decomposed early target mean={float(summary_df.loc[summary_df['method'] == 'decomposed_early_target', 'overall'].iloc[0]):.4f}, "
            f"scalar mean={float(summary_df.loc[summary_df['method'] == 'scalar_early_target', 'overall'].iloc[0]):.4f}, "
            f"decomposed-minus-scalar 95% CI=[{ci_prompt[0]:.4f}, {ci_prompt[1]:.4f}], "
            f"best-of-2 mean={float(summary_df.loc[summary_df['method'] == 'best_of_2_full_completion', 'overall'].iloc[0]):.4f}."
        ),
        "evaluation_line": f"Primary reported statistic: {PRIMARY_METRIC_LABEL} at prompt level. All numbers below are local heuristic GenEval-style scores, not official GenEval outputs.",
        "deviations": deviation_lines,
        "hit_rate_line": (
            f"decomposed={hit_rate_summary['decomposed_hit_rate']:.3f}, "
            f"scalar={hit_rate_summary['scalar_hit_rate']:.3f}"
        ),
    }
    write_main_summary(EXP_DIR / "main_summary.md", summary_payload)

    root_results = {
        "study_outcome": {
            "status": "negative_result" if negative_result else "positive_result",
            "claim": "unsupported" if negative_result else "supported",
            "summary_path": str(EXP_DIR / "main_summary.md"),
        },
        "evaluation_protocol": LOCAL_EVAL_PROTOCOL,
        "runtime_calibration": runtime,
        "parser_audit": parser_audit,
        "probe_metrics": probe_metrics,
        "main_results": summary_rows,
        "bootstrap": {
            "primary_metric": PRIMARY_METRIC,
            "primary_metric_label": PRIMARY_METRIC_LABEL,
            "primary_unit_of_analysis": "prompt",
            "decomposed_minus_scalar_mean_prompt": float(prompt_delta.mean()),
            "decomposed_minus_scalar_ci95_prompt": list(ci_prompt),
            "decomposed_minus_best2_mean_prompt": float(best2_prompt_delta.mean()),
            "decomposed_minus_best2_ci95_prompt": list(ci_best2_prompt),
            "paired_permutation_pvalue": float(ptest.pvalue),
            "paired_permutation_test": {
                "metric": PRIMARY_METRIC,
                "unit_of_analysis": "prompt",
                "n_resamples": 10000,
                "random_state": 0,
                "permutation_type": "samples",
            },
            "auxiliary_metric": AUXILIARY_METRIC,
            "auxiliary_unit_of_analysis": "prompt",
            "auxiliary_decomposed_minus_scalar_mean_prompt": float(aux_prompt_delta.mean()),
            "auxiliary_decomposed_minus_scalar_ci95_prompt": list(aux_ci_prompt),
            "auxiliary_decomposed_minus_best2_mean_prompt": float(aux_best2_prompt_delta.mean()),
            "auxiliary_decomposed_minus_best2_ci95_prompt": list(aux_ci_best2_prompt),
        },
        "hit_rate_diagnostic": hit_rate_summary,
        "manual_label_sensitivity": manual_sensitivity_payload,
        "supervision_agreement": supervision_agreement,
        "artifacts": {
            "results_table_csv": str(EXP_DIR / "main_results_table.csv"),
            "results_table_md": str(EXP_DIR / "main_results_table.md"),
            "bootstrap_png": str(FIGURES_DIR / "bootstrap_intervals.png"),
            "bootstrap_pdf": str(FIGURES_DIR / "bootstrap_intervals.pdf"),
            "calibration_curves_png": str(FIGURES_DIR / "calibration_curves.png"),
            "calibration_curves_pdf": str(FIGURES_DIR / "calibration_curves.pdf"),
            "score_margin_scatter_png": str(FIGURES_DIR / "score_margin_scatter.png"),
            "score_margin_scatter_pdf": str(FIGURES_DIR / "score_margin_scatter.pdf"),
            "qualitative_png": str(FIGURES_DIR / "qualitative_cases.png"),
            "qualitative_pdf": str(FIGURES_DIR / "qualitative_cases.pdf"),
            "main_summary_md": str(EXP_DIR / "main_summary.md"),
            "main_trials_jsonl": str(EXP_DIR / "main_experiment" / "trials.jsonl"),
            "official_supervision_agreement_json": str(EXP_DIR / "official_geneval_supervision" / "agreement_summary.json"),
        },
        "deviations": deviation_lines,
    }
    write_json(ROOT / "results.json", root_results)


if __name__ == "__main__":
    run()

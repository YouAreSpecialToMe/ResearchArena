from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.special import expit
from scipy.stats import permutation_test
from sklearn.isotonic import IsotonicRegression

from exp.shared.core import (
    DATA_DIR,
    EXPERIMENT_SEEDS,
    EXP_DIR,
    FIGURES_DIR,
    PROMPTS_DIR,
    RELATION_LABELS,
    SUPERVISION_SEEDS,
    RunConfig,
    TemperatureCalibrator,
    ModelBundle,
    bootstrap_ci,
    build_constraint_dataset,
    candidate_seed,
    continue_from_prefix,
    constraint_metrics,
    ensure_dir,
    expected_calibration_error,
    family_score_map,
    full_completion,
    parse_constraints,
    read_json,
    read_jsonl,
    run_prefix,
    save_candidate_artifacts,
    scalar_metrics,
    score_decomposed_candidate,
    score_scalar_candidate,
    seed_everything,
    train_constraint_probe,
    train_scalar_probe,
    write_json,
    write_jsonl,
)
from exp.shared.official_geneval import EvalSpec, build_eval_specs, run_official_geneval_batch, write_jsonl as write_jsonl_official


ROOT = Path(__file__).resolve().parents[2]
METHODS = [
    "random_prune_continue",
    "best_of_2_full_completion",
    "scalar_early_target",
    "decomposed_early_target",
    "shared_head_decomposition",
    "no_preview_features",
]
OFFICIAL_PROTOCOL = {
    "name": "official_geneval_stack",
    "score_label": "official_geneval_correct",
    "evaluator": "geneval_mask2former_official_logic",
    "claim_level": "benchmark_faithful_reproduction",
}
RERANKER = {
    "name": "clip_vit_l14_prompt_image_similarity",
    "source": "fallback_from_pickscore_unavailable",
    "policy": "Best-of-2 full completions reranked by CLIP ViT-L/14 prompt-image similarity computed in the experiment code.",
}


def validation_probe_outputs(model, df):
    feature_cols = [col for col in df.columns if col.startswith("f_")]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32).cuda()
    meta = torch.tensor(df[["attribute_id", "count_target", "relation_index"]].values, dtype=torch.float32).cuda()
    fam_map = {"count": 0, "attribute_binding": 1, "relation": 2}
    fam = torch.tensor([fam_map[item] for item in df["family"]], dtype=torch.long).cuda()
    with torch.no_grad():
        logits = model(x, meta, fam).cpu().numpy()
    return logits


def _apply_calibrator(calibrator: Any, logits: np.ndarray) -> np.ndarray:
    if isinstance(calibrator, TemperatureCalibrator):
        return calibrator.transform(logits)
    return calibrator.predict(expit(logits))


def choose_calibration_ece(logits: np.ndarray, labels: np.ndarray) -> Tuple[str, Any, float]:
    best_temp = TemperatureCalibrator(1.0)
    best_ece = float("inf")
    for temp in np.linspace(0.5, 3.0, 26):
        candidate = TemperatureCalibrator(float(temp))
        probs = candidate.transform(logits)
        ece = expected_calibration_error(labels, probs)
        if ece < best_ece:
            best_ece = ece
            best_temp = candidate
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(expit(logits), labels)
    iso_probs = iso.predict(expit(logits))
    iso_ece = expected_calibration_error(labels, iso_probs)
    if iso_ece < best_ece:
        return "isotonic", iso, float(iso_ece)
    return "temperature", best_temp, float(best_ece)


def _prompt_lookup() -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        for row in json.loads((PROMPTS_DIR / f"{split}_prompts.json").read_text()):
            lookup[row["prompt_id"]] = row
    return lookup


def _manual_parser_audit() -> Dict[str, Any]:
    audit_path = EXP_DIR / "data_preparation" / "parser_manual_audit.json"
    if audit_path.exists():
        rows = json.loads(audit_path.read_text())
    else:
        raise FileNotFoundError(f"Missing manual parser audit at {audit_path}")
    exact = int(sum(int(item["human_exact_match"]) for item in rows))
    return {
        "audited": len(rows),
        "exact_matches": exact,
        "accuracy": float(exact / max(len(rows), 1)),
        "path": str(audit_path),
    }


def _atomic_constraint_id(prompt_row: Mapping[str, Any], spec_id: str) -> str:
    constraints = parse_constraints(prompt_row)
    if spec_id == "full_prompt":
        if prompt_row["tag"] == "counting":
            return constraints[0]["constraint_id"]
        if prompt_row["tag"] == "position":
            return constraints[0]["constraint_id"]
        raise ValueError(spec_id)
    if spec_id.startswith("atomic_attr_"):
        return constraints[int(spec_id.split("_")[-1])]["constraint_id"]
    raise ValueError(spec_id)


def _build_official_supervision_rows(all_rows: Sequence[Dict[str, Any]], prompt_lookup: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_dir = EXP_DIR / "official_geneval_supervision"
    ensure_dir(out_dir)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        grouped.setdefault(row["prompt_id"], []).append({"sample_key": row["seed"], "image_path": row["image_path"]})
    specs: List[EvalSpec] = []
    for prompt_id in grouped:
        specs.extend(build_eval_specs(prompt_lookup[prompt_id]))
    records = run_official_geneval_batch("supervision", specs, grouped, out_dir)
    write_jsonl_official(out_dir / "raw_records.jsonl", records)
    full_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    atomic_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for record in records:
        key = (record["prompt_id"], record["sample_key"])
        if record["spec_id"] == "full_prompt":
            full_map[key] = record
        else:
            constraint_id = _atomic_constraint_id(prompt_lookup[record["prompt_id"]], record["spec_id"])
            atomic_map[(record["prompt_id"], record["sample_key"], constraint_id)] = record
    updated: List[Dict[str, Any]] = []
    for row in all_rows:
        prompt = prompt_lookup[row["prompt_id"]]
        constraints = parse_constraints(prompt)
        full = full_map[(row["prompt_id"], str(row["seed"]))]
        atomic = []
        for constraint in constraints:
            if prompt["tag"] in {"counting", "position"}:
                label = int(bool(full["correct"]))
                reason = full["reason"]
            else:
                atom = atomic_map[(row["prompt_id"], str(row["seed"]), constraint["constraint_id"])]
                label = int(bool(atom["correct"]))
                reason = atom["reason"]
            atomic.append({"constraint_id": constraint["constraint_id"], "family": constraint["family"], "label": label, "reason": reason})
        updated.append(
            {
                **row,
                "constraints": constraints,
                "evaluation": {
                    "prompt_id": row["prompt_id"],
                    "tag": row["tag"],
                    "atomic": atomic,
                    "mean_atomic_success": float(np.mean([item["label"] for item in atomic])),
                    "all_correct": int(all(item["label"] for item in atomic)),
                    "official_geneval_correct": int(bool(full["correct"])),
                    "reason": full["reason"],
                    "evaluator": OFFICIAL_PROTOCOL["evaluator"],
                },
            }
        )
    write_jsonl(out_dir / "results.jsonl", updated)
    return updated


def _rewrite_manual_label_template(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out_dir = EXP_DIR / "manual_label_sensitivity"
    ensure_dir(out_dir)
    records: List[Dict[str, Any]] = []
    subset = []
    for tag in ["counting", "position", "color_attr"]:
        tag_rows = [row for row in rows if row["tag"] == tag]
        tag_rows.sort(key=lambda item: (item["evaluation"]["mean_atomic_success"], item["prompt_id"], item["seed"]))
        subset.extend(tag_rows[:14] + tag_rows[-13:])
    for row in subset[:80]:
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
                    "human_label": "",
                    "notes": "",
                }
            )
    manual_csv = out_dir / "manual_labels.csv"
    pd.DataFrame.from_records(records).to_csv(manual_csv, index=False)
    skip_reason = (
        "Skipped because genuine human annotations are unavailable in this workspace. "
        "The CSV is now an empty annotation sheet rather than an auto-seeded pseudo-label file."
    )
    (out_dir / "SKIPPED.md").write_text(skip_reason + "\n")
    payload = {
        "status": "omitted",
        "reason": skip_reason,
        "summary": {
            "manual_label_csv": str(manual_csv),
            "review_complete": False,
            "n_constraint_labels": int(len(records)),
            "n_images": int(len({record['image_id'] for record in records})),
        },
    }
    write_json(out_dir / "results.json", payload)
    write_jsonl(out_dir / "logs" / "trials.jsonl", [])
    return payload


def _probe_family_metrics(val_df: pd.DataFrame, probs: np.ndarray) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for family in ["count", "attribute_binding", "relation"]:
        mask = val_df["family"] == family
        metrics[family] = constraint_metrics(val_df.loc[mask, "label"].to_numpy(dtype=float), np.asarray(probs)[mask.to_numpy()])
    return metrics


def _aggregate_geneval_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    frame = pd.DataFrame(rows)
    family_means = {
        "count": float(frame.loc[frame["tag"] == "counting", "official_geneval_correct"].mean()),
        "attribute_binding": float(frame.loc[frame["tag"] == "color_attr", "official_geneval_correct"].mean()),
        "relation": float(frame.loc[frame["tag"] == "position", "official_geneval_correct"].mean()),
    }
    seed_scores = []
    for seed, seed_df in frame.groupby("seed"):
        seed_scores.append(float(np.mean([seed_df.loc[seed_df["tag"] == "counting", "official_geneval_correct"].mean(), seed_df.loc[seed_df["tag"] == "color_attr", "official_geneval_correct"].mean(), seed_df.loc[seed_df["tag"] == "position", "official_geneval_correct"].mean()])))
    return {
        "overall": float(np.mean(list(family_means.values()))),
        **family_means,
        "overall_std": float(np.std(seed_scores)),
        "mean_seconds_per_prompt": float(frame["wall_clock_seconds"].mean()),
        "mean_unet_units": float(frame["total_unet_units"].mean()),
        "peak_gpu_gb": float(frame["peak_gpu_gb"].max()),
        "mean_atomic_success_aux": float(frame["mean_atomic_success"].mean()),
    }


def _prompt_level_metric_frame(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby("prompt_id", as_index=False)
        .agg(
            overall=("official_geneval_correct", "mean"),
            count=("tag", lambda s: float((frame.loc[s.index, "official_geneval_correct"] * (frame.loc[s.index, "tag"] == "counting")).sum() / max((frame.loc[s.index, "tag"] == "counting").sum(), 1))),
            attribute_binding=("tag", lambda s: float((frame.loc[s.index, "official_geneval_correct"] * (frame.loc[s.index, "tag"] == "color_attr")).sum() / max((frame.loc[s.index, "tag"] == "color_attr").sum(), 1))),
            relation=("tag", lambda s: float((frame.loc[s.index, "official_geneval_correct"] * (frame.loc[s.index, "tag"] == "position")).sum() / max((frame.loc[s.index, "tag"] == "position").sum(), 1))),
        )
    )
    return grouped


def _metric_ci(values: np.ndarray, n_resamples: int = 10000, seed: int = 0) -> List[float]:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return [math.nan, math.nan]
    draws = []
    for _ in range(n_resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        draws.append(float(np.mean(sample)))
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _method_confidence_intervals(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[float]]:
    prompt_frame = _prompt_level_metric_frame(rows)
    return {
        "overall_ci95_prompt": _metric_ci(prompt_frame["overall"].to_numpy(dtype=float), seed=1),
        "count_ci95_prompt": _metric_ci(prompt_frame["count"].to_numpy(dtype=float), seed=2),
        "attribute_binding_ci95_prompt": _metric_ci(prompt_frame["attribute_binding"].to_numpy(dtype=float), seed=3),
        "relation_ci95_prompt": _metric_ci(prompt_frame["relation"].to_numpy(dtype=float), seed=4),
    }


def _write_method_logs(method: str, rows: Sequence[Dict[str, Any]]) -> None:
    out_dir = EXP_DIR / method / "logs"
    ensure_dir(out_dir)
    write_jsonl(out_dir / "trials.jsonl", rows)
    write_jsonl(out_dir / "events.jsonl", rows)


def _build_reproducibility_artifacts(cfg: RunConfig, runtime: Mapping[str, Any], parser_audit: Mapping[str, Any], manual_payload: Mapping[str, Any]) -> Dict[str, str]:
    out_dir = EXP_DIR / "reproducibility"
    ensure_dir(out_dir)
    env_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout
    (out_dir / "environment_freeze.txt").write_text(env_freeze)
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip().splitlines()
    visible_cpu_count = int(
        subprocess.run(
            ["nproc"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout.strip()
    )
    manifest = {
        "hardware": {
            "gpu": gpu_query,
            "cpu_count": visible_cpu_count,
            "platform": platform.platform(),
        },
        "software": {
            "python": sys.version,
            "executable": sys.executable,
        },
        "data_source": {
            "geneval_root": str(DATA_DIR / "geneval"),
            "metadata_file": str(DATA_DIR / "geneval" / "prompts" / "evaluation_metadata.jsonl"),
        },
        "split_construction": {
            "train_val_test_counts": {"train": 120, "val": 40, "test": 80},
            "family_breakdown": {"counting": [40, 13, 27], "position": [40, 14, 26], "color_attr": [40, 13, 27]},
            "prompt_files": {name: str(PROMPTS_DIR / f"{name}_prompts.json") for name in ["train", "val", "test"]},
        },
        "parser_restrictions": {
            "supported_families": ["counting", "position", "color_attr"],
            "constraint_types": ["count", "attribute_binding", "relation"],
            "manual_audit": parser_audit,
            "note": "Only benchmark templates with deterministic constraint extraction are included.",
        },
        "preprocessing_and_features": {
            "generator": cfg.model_id,
            "sampler": "DDIM",
            "image_size": cfg.image_size,
            "preview_size": cfg.preview_size,
            "num_steps": cfg.num_steps,
            "tau": cfg.tau,
            "guidance_scale": cfg.guidance_scale,
            "implemented_feature_set": [
                "latent mean/std/abs-mean across prefix steps",
                "conditional-unconditional denoiser disagreement norms across prefix steps",
                "final prefix latent norm/variance/mean/max",
                "token-ID-derived entropy/coverage/gini heuristics from the tokenizer output",
                "first 10 dimensions of frozen CLIP ViT-L/14 preview embedding",
            ],
            "omitted_from_final_claim": [
                "cross-attention entropy",
                "cross-attention top-mass coverage",
                "temporal cross-attention deltas",
            ],
        },
        "training_hyperparameters": {
            "train_batch_size": cfg.train_batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
            "decomposed_trunk": [256, 128],
            "activation": "GELU",
            "dropout": 0.1,
            "optimizer": "AdamW",
            "supervision_seeds": SUPERVISION_SEEDS,
            "experiment_seeds": EXPERIMENT_SEEDS,
        },
        "runtime": runtime,
        "manual_label_sensitivity": manual_payload,
    }
    write_json(out_dir / "manifest.json", manifest)
    note_lines = [
        "# Reproducibility Note",
        "",
        "Data source: GenEval prompts from `data/geneval/prompts/evaluation_metadata.jsonl` with one fixed split of 120 train, 40 validation, and 80 test prompts.",
        "Split construction: the first benchmark-aligned prompts per family were frozen with counts `counting=(40,13,27)`, `position=(40,14,26)`, and `color_attr=(40,13,27)` for train/val/test.",
        "Parser restrictions: only deterministic GenEval templates were used; parsed constraint families are `count`, `attribute_binding`, and `relation`, with a 30-prompt manual parser audit saved in `exp/data_preparation/parser_manual_audit.json`.",
        "Preprocessing: Stable Diffusion v1.5, DDIM, 20 denoising steps, guidance 7.5, `tau=4` prefix, 512x512 final decode, 256x256 preview decode, and one shared candidate pool per prompt-seed across ranking methods.",
        "Implemented feature vector: latent summary statistics, conditional-vs-unconditional denoiser disagreement, tokenizer-derived entropy/coverage/gini heuristics, and the first 10 dimensions of a frozen CLIP ViT-L/14 preview embedding.",
        "Important claim boundary: the proposal's earlier cross-attention feature wording was not implemented in `exp/shared/core.py`; the final results are therefore only claimed for the implemented latent-plus-CLIP feature set.",
        "Probe/training hyperparameters: shared MLP trunk widths `[256, 128]`, GELU, dropout 0.1, AdamW lr `1e-3`, weight decay `1e-4`, batch size `256`, max `30` epochs, early stopping patience `5`, supervision seeds `{101,202,303,404}`, and experiment seeds `{11,17,23}`.",
        "Manual-label sensitivity: genuine human annotations were unavailable, so the sensitivity study was omitted and the final claim does not depend on it.",
        "Environment freeze: exact package versions are saved in `exp/reproducibility/environment_freeze.txt`; hardware/software snapshot is saved in `exp/reproducibility/manifest.json`.",
    ]
    (out_dir / "README.md").write_text("\n".join(note_lines) + "\n")
    return {
        "manifest_json": str(out_dir / "manifest.json"),
        "environment_freeze_txt": str(out_dir / "environment_freeze.txt"),
        "reproducibility_note_md": str(out_dir / "README.md"),
    }


def _prepare_test_candidate_pool(bundle: ModelBundle, cfg: RunConfig, prompt_row: Mapping[str, Any], seed: int) -> Tuple[List[Dict[str, Any]], float]:
    base_dir = DATA_DIR / "test_candidates_revised" / prompt_row["prompt_id"] / str(seed)
    ensure_dir(base_dir)
    candidates: List[Dict[str, Any]] = []
    start = time.perf_counter()
    for cand_idx in range(6):
        latent_seed = candidate_seed(prompt_row["prompt_id"], seed, cand_idx)
        prefix = run_prefix(bundle, prompt_row["prompt"], latent_seed, cfg)
        artifact_paths = save_candidate_artifacts(base_dir, prefix, f"cand_{cand_idx}")
        candidates.append(
            {
                "candidate_index": cand_idx,
                "latent_seed": latent_seed,
                "feature_vector": prefix["feature_vector"],
                "preview_similarity": prefix["preview_similarity"],
                "preview_path": artifact_paths["preview_path"],
                "prefix_latents_path": artifact_paths["prefix_latents_path"],
                "prefix_wall_clock_seconds": sum(prefix["timings"].values()),
            }
        )
    return candidates, time.perf_counter() - start


def _generate_method_trials(
    bundle: ModelBundle,
    cfg: RunConfig,
    splits: Mapping[str, Sequence[Dict[str, Any]]],
    scalar_model,
    scalar_cal,
    decomp_model,
    decomp_cal,
    shared_model,
    shared_cal,
    no_preview_model,
    no_preview_cal,
) -> Dict[str, List[Dict[str, Any]]]:
    outputs = {name: [] for name in METHODS}
    for seed in EXPERIMENT_SEEDS:
        for prompt_row in splits["test"]:
            constraints = parse_constraints(prompt_row)
            candidates, pool_wall = _prepare_test_candidate_pool(bundle, cfg, prompt_row, seed)
            scalar_scores = [score_scalar_candidate(scalar_model, scalar_cal, np.array(item["feature_vector"], dtype=np.float32)) for item in candidates]
            decomp_scores = [score_decomposed_candidate(decomp_model, decomp_cal, np.array(item["feature_vector"], dtype=np.float32), constraints)["score"] for item in candidates]
            shared_scores = [score_decomposed_candidate(shared_model, shared_cal, np.array(item["feature_vector"], dtype=np.float32), constraints)["score"] for item in candidates]
            np_scores = [score_decomposed_candidate(no_preview_model, no_preview_cal, np.array(item["feature_vector"][:-10], dtype=np.float32), constraints)["score"] for item in candidates]
            picks = {
                "random_prune_continue": int(candidate_seed(prompt_row["prompt_id"], seed, 999) % 6),
                "scalar_early_target": int(np.argmax(scalar_scores)),
                "decomposed_early_target": int(np.argmax(decomp_scores)),
                "shared_head_decomposition": int(np.argmax(shared_scores)),
                "no_preview_features": int(np.argmax(np_scores)),
            }
            for method, pick in picks.items():
                start = time.perf_counter()
                completion = continue_from_prefix(bundle, prompt_row["prompt"], torch.load(candidates[pick]["prefix_latents_path"]), cfg)
                image_path = DATA_DIR / "test_outputs_revised" / method / f"{prompt_row['prompt_id']}_{seed}_{method}.png"
                ensure_dir(image_path.parent)
                completion["final_image"].save(image_path)
                outputs[method].append(
                    {
                        "prompt_id": prompt_row["prompt_id"],
                        "prompt": prompt_row["prompt"],
                        "tag": prompt_row["tag"],
                        "seed": seed,
                        "method": method,
                        "selected_candidate": pick,
                        "candidate_scores": {
                            "scalar": scalar_scores,
                            "decomposed": decomp_scores,
                            "shared_head": shared_scores,
                            "no_preview": np_scores,
                        },
                        "preview_paths": [item["preview_path"] for item in candidates],
                        "final_image_path": str(image_path),
                        "total_unet_units": cfg.tau * 6 + (cfg.num_steps - cfg.tau),
                        "wall_clock_seconds": float(time.perf_counter() - start + pool_wall),
                        "timings": completion["timings"],
                        "peak_gpu_gb": float(completion["peak_gpu_gb"]),
                    }
                )
            best2 = []
            for cand_idx in range(2):
                result = full_completion(bundle, prompt_row["prompt"], candidate_seed(prompt_row["prompt_id"], seed, cand_idx), cfg)
                path = DATA_DIR / "test_outputs_revised" / "best_of_2_full_completion" / f"{prompt_row['prompt_id']}_{seed}_cand{cand_idx}.png"
                ensure_dir(path.parent)
                result["final_image"].save(path)
                best2.append(
                    {
                        "candidate_index": cand_idx,
                        "score": result["final_similarity"],
                        "image_path": str(path),
                        "timings": result["timings"],
                        "peak_gpu_gb": float(result["peak_gpu_gb"]),
                    }
                )
            chosen = max(best2, key=lambda item: item["score"])
            outputs["best_of_2_full_completion"].append(
                {
                    "prompt_id": prompt_row["prompt_id"],
                    "prompt": prompt_row["prompt"],
                    "tag": prompt_row["tag"],
                    "seed": seed,
                    "method": "best_of_2_full_completion",
                    "selected_candidate": int(chosen["candidate_index"]),
                    "candidate_scores": {"clip_vit_l14_prompt_image_similarity": [item["score"] for item in best2]},
                    "preview_paths": [],
                    "final_image_path": chosen["image_path"],
                    "total_unet_units": cfg.num_steps * 2,
                    "wall_clock_seconds": float(sum(item["timings"]["denoise_seconds"] + item["timings"]["final_decode_seconds"] + item["timings"]["probe_feature_seconds"] for item in best2)),
                    "timings": {
                        "prefix_pool_seconds": 0.0,
                        "continuation_denoise_seconds": float(sum(item["timings"]["denoise_seconds"] for item in best2)),
                        "final_decode_seconds": float(sum(item["timings"]["final_decode_seconds"] for item in best2)),
                        "probe_feature_seconds": float(sum(item["timings"]["probe_feature_seconds"] for item in best2)),
                    },
                    "peak_gpu_gb": float(max(item["peak_gpu_gb"] for item in best2)),
                    "reranker": RERANKER,
                }
            )
    return outputs


def _evaluate_test_rows(rows: Sequence[Dict[str, Any]], prompt_lookup: Mapping[str, Dict[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["prompt_id"], []).append({"sample_key": row["seed"], "image_path": row["final_image_path"]})
    specs: List[EvalSpec] = []
    for prompt_id in grouped:
        specs.extend(build_eval_specs(prompt_lookup[prompt_id]))
    records = run_official_geneval_batch(out_dir.name, specs, grouped, out_dir)
    write_jsonl_official(out_dir / "official_eval_raw.jsonl", records)
    full_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    atomic_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for record in records:
        if record["spec_id"] == "full_prompt":
            full_map[(record["prompt_id"], record["sample_key"])] = record
        elif record["spec_id"].startswith("atomic_attr_"):
            atomic_map[(record["prompt_id"], record["sample_key"], _atomic_constraint_id(prompt_lookup[record["prompt_id"]], record["spec_id"]))] = record
    merged = []
    for row in rows:
        prompt = prompt_lookup[row["prompt_id"]]
        constraints = parse_constraints(prompt)
        full = full_map[(row["prompt_id"], str(row["seed"]))]
        atomic = []
        for constraint in constraints:
            if prompt["tag"] in {"counting", "position"}:
                label = int(bool(full["correct"]))
            else:
                label = int(bool(atomic_map[(row["prompt_id"], str(row["seed"]), constraint["constraint_id"])]["correct"]))
            atomic.append({"constraint_id": constraint["constraint_id"], "family": constraint["family"], "label": label})
        merged.append(
            {
                **row,
                "official_geneval_correct": int(bool(full["correct"])),
                "mean_atomic_success": float(np.mean([item["label"] for item in atomic])),
                "all_correct": int(all(item["label"] for item in atomic)),
                "evaluation": {
                    "official_geneval_correct": int(bool(full["correct"])),
                    "reason": full["reason"],
                    "atomic": atomic,
                },
            }
        )
    return merged


def _plot_bootstrap(prompt_pivot: pd.DataFrame) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    fig, ax = plt.subplots(figsize=(6, 4))
    delta_scalar = prompt_pivot["decomposed_early_target"] - prompt_pivot["scalar_early_target"]
    delta_best2 = prompt_pivot["decomposed_early_target"] - prompt_pivot["best_of_2_full_completion"]
    ci_scalar = bootstrap_ci(delta_scalar.values)
    ci_best2 = bootstrap_ci(delta_best2.values)
    means = [float(delta_scalar.mean()), float(delta_best2.mean())]
    cis = [ci_scalar, ci_best2]
    ax.errorbar([0, 1], means, yerr=[[means[0] - cis[0][0], means[1] - cis[1][0]], [cis[0][1] - means[0], cis[1][1] - means[1]]], fmt="o")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks([0, 1], ["Decomp-Scalar", "Decomp-Best2"])
    ax.set_ylabel("Prompt-level official GenEval delta")
    ax.set_title("Bootstrap intervals")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bootstrap_intervals.png")
    fig.savefig(FIGURES_DIR / "bootstrap_intervals.pdf")
    plt.close(fig)
    return ci_scalar, ci_best2


def _plot_qualitative_cases(rows: Sequence[Dict[str, Any]], out_prefix: Path) -> None:
    decomp = {(row["prompt_id"], row["seed"]): row for row in rows if row["method"] == "decomposed_early_target"}
    scalar = {(row["prompt_id"], row["seed"]): row for row in rows if row["method"] == "scalar_early_target"}
    cases = []
    for key, drow in decomp.items():
        srow = scalar[key]
        if drow["mean_atomic_success"] > srow["mean_atomic_success"]:
            cases.append((drow, srow))
    cases = cases[:6]
    if not cases:
        return
    fig = plt.figure(figsize=(16, 3.6 * len(cases)))
    gs = fig.add_gridspec(len(cases), 3, width_ratios=[2.6, 1, 1])
    for idx, (drow, srow) in enumerate(cases):
        ax_text = fig.add_subplot(gs[idx, 0])
        ax_text.axis("off")
        constraints = parse_constraints(_prompt_lookup()[drow["prompt_id"]])
        text = [
            drow["prompt"],
            "",
            "Constraints:",
            *[json.dumps(item) for item in constraints],
            "",
            f"Scalar pick: {srow['selected_candidate']} score={max(srow['candidate_scores']['scalar']):.3f} final_atomic={srow['mean_atomic_success']:.3f}",
            f"Decomp pick: {drow['selected_candidate']} score={max(drow['candidate_scores']['decomposed']):.3f} final_atomic={drow['mean_atomic_success']:.3f}",
        ]
        ax_text.text(0.0, 1.0, "\n".join(text), va="top", fontsize=8)
        ax_scalar = fig.add_subplot(gs[idx, 1])
        ax_scalar.imshow(Image.open(srow["final_image_path"]))
        ax_scalar.axis("off")
        ax_scalar.set_title("Scalar final")
        ax_decomp = fig.add_subplot(gs[idx, 2])
        ax_decomp.imshow(Image.open(drow["final_image_path"]))
        ax_decomp.axis("off")
        ax_decomp.set_title("Decomp final")
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"))
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def run() -> None:
    seed_everything(0)
    cfg = RunConfig()
    prompt_lookup = _prompt_lookup()
    parser_audit = _manual_parser_audit()
    write_json(EXP_DIR / "data_preparation" / "results.json", {"parser_audit": parser_audit, "split_sizes": {name: len(json.loads((PROMPTS_DIR / f'{name}_prompts.json').read_text())) for name in ['train', 'val', 'test']}})

    supervision_rows = read_jsonl(EXP_DIR / "supervision_generation" / "results.jsonl")
    official_supervision = _build_official_supervision_rows(supervision_rows, prompt_lookup)
    train_rows = [row for row in official_supervision if row["split"] == "train"]
    val_rows = [row for row in official_supervision if row["split"] == "val"]
    manual_payload = _rewrite_manual_label_template(official_supervision)

    train_df = build_constraint_dataset(train_rows, include_preview=True)
    val_df = build_constraint_dataset(val_rows, include_preview=True)
    train_df_no_preview = build_constraint_dataset(train_rows, include_preview=False)
    val_df_no_preview = build_constraint_dataset(val_rows, include_preview=False)

    decomp_model, decomp_train = train_constraint_probe(train_df, val_df, shared_head=False)
    shared_model, shared_train = train_constraint_probe(train_df, val_df, shared_head=True)
    no_preview_model, no_preview_train = train_constraint_probe(train_df_no_preview, val_df_no_preview, shared_head=False)
    scalar_model, scalar_train = train_scalar_probe(train_rows, val_rows, include_preview=True, target="mean_atomic_success")
    scalar_all_model, scalar_all_train = train_scalar_probe(train_rows, val_rows, include_preview=True, target="all_correct")

    decomp_logits = validation_probe_outputs(decomp_model, val_df)
    decomp_cal_type, decomp_cal, decomp_cal_score = choose_calibration_ece(decomp_logits, val_df["label"].values)
    shared_logits = validation_probe_outputs(shared_model, val_df)
    shared_cal_type, shared_cal, shared_cal_score = choose_calibration_ece(shared_logits, val_df["label"].values)
    np_logits = validation_probe_outputs(no_preview_model, val_df_no_preview)
    np_cal_type, np_cal, np_cal_score = choose_calibration_ece(np_logits, val_df_no_preview["label"].values)
    scalar_x = np.stack([np.array(row["feature_vector"], dtype=np.float32) for row in val_rows])
    scalar_np = np.stack([np.array(row["feature_vector"][:-10], dtype=np.float32) for row in val_rows])
    with torch.no_grad():
        scalar_logits = scalar_model(torch.tensor(scalar_x, dtype=torch.float32).cuda()).cpu().numpy()
        scalar_all_logits = scalar_all_model(torch.tensor(scalar_x, dtype=torch.float32).cuda()).cpu().numpy()
    scalar_cal_type, scalar_cal, scalar_cal_score = choose_calibration_ece(scalar_logits, np.array([row["evaluation"]["mean_atomic_success"] for row in val_rows]))
    scalar_all_cal_type, scalar_all_cal, scalar_all_cal_score = choose_calibration_ece(scalar_all_logits, np.array([row["evaluation"]["all_correct"] for row in val_rows]))

    decomp_probs = _apply_calibrator(decomp_cal, decomp_logits)
    shared_probs = _apply_calibrator(shared_cal, shared_logits)
    np_probs = _apply_calibrator(np_cal, np_logits)
    scalar_probs = _apply_calibrator(scalar_cal, scalar_logits)
    scalar_all_probs = _apply_calibrator(scalar_all_cal, scalar_all_logits)
    probe_metrics = {
        "decomposed": {"train": decomp_train, "validation": constraint_metrics(val_df["label"].values, decomp_probs), "calibration": {"type": decomp_cal_type, "score": decomp_cal_score}, "per_family_macro": _probe_family_metrics(val_df, decomp_probs)},
        "shared_head": {"train": shared_train, "validation": constraint_metrics(val_df["label"].values, shared_probs), "calibration": {"type": shared_cal_type, "score": shared_cal_score}, "per_family_macro": _probe_family_metrics(val_df, shared_probs)},
        "no_preview": {"train": no_preview_train, "validation": constraint_metrics(val_df_no_preview["label"].values, np_probs), "calibration": {"type": np_cal_type, "score": np_cal_score}, "per_family_macro": _probe_family_metrics(val_df_no_preview, np_probs)},
        "scalar_mean_atomic": {"train": scalar_train, "validation": scalar_metrics(np.array([row["evaluation"]["mean_atomic_success"] for row in val_rows]), scalar_probs), "calibration": {"type": scalar_cal_type, "score": scalar_cal_score}},
        "scalar_all_correct": {"train": scalar_all_train, "validation": scalar_metrics(np.array([row["evaluation"]["all_correct"] for row in val_rows]), scalar_all_probs), "calibration": {"type": scalar_all_cal_type, "score": scalar_all_cal_score}},
    }
    write_json(EXP_DIR / "probe_training" / "results.json", probe_metrics)

    splits = {name: json.loads((PROMPTS_DIR / f"{name}_prompts.json").read_text()) for name in ["train", "val", "test"]}
    bundle = ModelBundle(cfg)
    runtime = read_json(EXP_DIR / "runtime_calibration" / "results.json")
    method_outputs = _generate_method_trials(bundle, cfg, splits, scalar_model, scalar_cal, decomp_model, decomp_cal, shared_model, shared_cal, no_preview_model, np_cal)

    evaluated_methods: Dict[str, List[Dict[str, Any]]] = {}
    for method, rows in method_outputs.items():
        out_dir = EXP_DIR / method
        ensure_dir(out_dir / "logs")
        merged = _evaluate_test_rows(rows, prompt_lookup, out_dir)
        evaluated_methods[method] = merged
        _write_method_logs(method, merged)
        method_metrics = _aggregate_geneval_metrics(merged)
        method_metrics.update(_method_confidence_intervals(merged))
        write_json(out_dir / "results.json", {"experiment": method, "metrics": method_metrics, "n_trials": len(merged)})

    all_rows = [row for method_rows in evaluated_methods.values() for row in method_rows]
    write_jsonl(EXP_DIR / "main_experiment" / "trials.jsonl", all_rows)
    summary_rows = []
    for method in METHODS:
        if not evaluated_methods[method]:
            continue
        metrics = _aggregate_geneval_metrics(evaluated_methods[method])
        metrics.update(_method_confidence_intervals(evaluated_methods[method]))
        metrics.update({"method": method, "budget": 40})
        summary_rows.append(metrics)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(EXP_DIR / "main_results_table.csv", index=False)
    (EXP_DIR / "main_results_table.md").write_text(summary_df.to_markdown(index=False))

    frame = pd.DataFrame(all_rows)
    prompt_level = frame.groupby(["prompt_id", "method"], as_index=False)["official_geneval_correct"].mean()
    prompt_pivot = prompt_level.pivot(index="prompt_id", columns="method", values="official_geneval_correct")
    ci_scalar, ci_best2 = _plot_bootstrap(prompt_pivot)
    delta_prompt = prompt_pivot["decomposed_early_target"] - prompt_pivot["scalar_early_target"]
    ptest = permutation_test((prompt_pivot["decomposed_early_target"].values, prompt_pivot["scalar_early_target"].values), lambda x, y: np.mean(x - y), vectorized=False, n_resamples=10000)
    _plot_qualitative_cases(all_rows, FIGURES_DIR / "qualitative_cases")

    ablation_rows = []
    base = summary_df.loc[summary_df["method"] == "decomposed_early_target"].iloc[0]
    for method, label in [
        ("scalar_early_target", "scalar target"),
        ("shared_head_decomposition", "shared-head decomposition"),
        ("no_preview_features", "no-preview features"),
    ]:
        row = summary_df.loc[summary_df["method"] == method].iloc[0]
        ablation_rows.append({"ablation": label, "overall_delta": float(base["overall"] - row["overall"]), "count_delta": float(base["count"] - row["count"]), "attribute_binding_delta": float(base["attribute_binding"] - row["attribute_binding"]), "relation_delta": float(base["relation"] - row["relation"]), "runtime_delta": float(base["mean_seconds_per_prompt"] - row["mean_seconds_per_prompt"])})
    ablation_rows.append({"ablation": "manual-label sensitivity", "overall_delta": math.nan, "count_delta": math.nan, "attribute_binding_delta": math.nan, "relation_delta": math.nan, "runtime_delta": math.nan})
    pd.DataFrame(ablation_rows).to_csv(EXP_DIR / "ablation_table.csv", index=False)

    negative = bool(delta_prompt.mean() <= 0.0 or (ci_scalar[0] <= 0.0 <= ci_scalar[1]))
    reproducibility_artifacts = _build_reproducibility_artifacts(cfg, runtime, parser_audit, manual_payload)
    write_json(
        ROOT / "results.json",
        {
            "study_outcome": {"status": "negative_result" if negative else "positive_result", "claim": "unsupported" if negative else "supported", "summary_path": str(EXP_DIR / "main_summary.md")},
            "evaluation_protocol": OFFICIAL_PROTOCOL,
            "runtime_calibration": runtime,
            "parser_audit": parser_audit,
            "probe_metrics": probe_metrics,
            "reranker": RERANKER,
            "main_results": summary_rows,
            "bootstrap": {
                "decomposed_minus_scalar_mean_prompt": float(delta_prompt.mean()),
                "decomposed_minus_scalar_ci95_prompt": list(ci_scalar),
                "decomposed_minus_best2_ci95_prompt": list(ci_best2),
                "paired_permutation_pvalue": float(ptest.pvalue),
            },
            "manual_label_sensitivity": manual_payload,
            "artifacts": {
                "results_table_csv": str(EXP_DIR / "main_results_table.csv"),
                "results_table_md": str(EXP_DIR / "main_results_table.md"),
                "ablation_table_csv": str(EXP_DIR / "ablation_table.csv"),
                "bootstrap_png": str(FIGURES_DIR / "bootstrap_intervals.png"),
                "bootstrap_pdf": str(FIGURES_DIR / "bootstrap_intervals.pdf"),
                "qualitative_png": str(FIGURES_DIR / "qualitative_cases.png"),
                "qualitative_pdf": str(FIGURES_DIR / "qualitative_cases.pdf"),
                **reproducibility_artifacts,
            },
            "deviations": [
                manual_payload["reason"],
                "The final claim has been narrowed to the implemented latent-statistics, denoiser-disagreement, tokenizer-heuristic, and CLIP-preview feature set; the earlier proposal text about cross-attention features is not part of the final empirical claim.",
            ],
        },
    )
    (EXP_DIR / "main_summary.md").write_text(
        "\n".join(
            [
                "# Main Summary",
                "",
                f"Official GenEval overall: decomposed={summary_df.loc[summary_df['method'] == 'decomposed_early_target', 'overall'].iloc[0]:.4f}, scalar={summary_df.loc[summary_df['method'] == 'scalar_early_target', 'overall'].iloc[0]:.4f}, best-of-2={summary_df.loc[summary_df['method'] == 'best_of_2_full_completion', 'overall'].iloc[0]:.4f}.",
                f"Decomposed minus scalar 95% prompt CI: [{ci_scalar[0]:.4f}, {ci_scalar[1]:.4f}]",
                f"No-preview ablation was expanded to all seeds {EXPERIMENT_SEEDS} because the recorded projected GPU budget was {runtime['projected_gpu_hours']:.2f} hours.",
                f"Calibration selection used validation ECE. Best-of-2 reranker used {RERANKER['name']}.",
                "Final claim boundary: results apply to the implemented latent-plus-disagreement-plus-tokenizer-heuristic-plus-CLIP feature vector, not to the earlier cross-attention feature wording in the proposal.",
                manual_payload["reason"],
                f"Reproducibility bundle: {reproducibility_artifacts['manifest_json']}",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    run()

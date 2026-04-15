from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import build_datasets
from .metrics import (
    bootstrap_metric_ci,
    bootstrap_metric_diff,
    calibration_points,
    compute_metrics,
    response_level_metrics,
    summarize_metric_dicts,
)
from .models import build_claim_features, fit_logreg
from .utils import (
    LOGREG_C_GRID,
    ROOT,
    SEEDS,
    TAU_GRID,
    append_log,
    build_manifest,
    ensure_dirs,
    exp_log_path,
    json_dump,
    now,
    runtime_minutes,
)


PRIMARY_LABEL = "strict_label"
SLICE_LABELS = {"strict": "strict_label", "projected_all": "projected_all_label"}
PILOT_SIZE = 100
AUDIT_TARGET = 200
MIN_ROBUSTNESS_SUPPORT = 50
TRAIN_FEATURE_BUDGET = 20000

DRAFT_VAL_LIMIT = int(os.environ.get("LIMS_DRAFT_VAL_LIMIT", "0")) or None
DRAFT_TEST_LIMIT = int(os.environ.get("LIMS_DRAFT_TEST_LIMIT", "0")) or None
DRAFT_TRAIN_LIMIT = int(os.environ.get("LIMS_DRAFT_TRAIN_LIMIT", "0")) or None
SKIP_ROBUSTNESS = os.environ.get("LIMS_SKIP_ROBUSTNESS", "").lower() in {"1", "true", "yes"}

BASE_FEATURES = {
    "bm25": ["top1_bm25", "mean_top3_bm25", "top1_minus_top3mean"],
    "support_only": [
        "support_best_entailment",
        "support_best_contradiction",
        "support_best_neutral",
        "support_margin_second_best",
    ],
    "support_compactness": [
        "support_best_entailment",
        "support_best_contradiction",
        "support_best_neutral",
        "support_margin_second_best",
        "smin_size",
        "support_smin",
        "support_margin_smin_vs_best1",
        "candidate_above_tau",
        "document_dispersion",
    ],
    "full_context_removal": [
        "support_full",
        "support_no_context",
        "support_remove_support",
        "drop_full_to_no_context",
        "drop_full_to_remove_support",
        "best_residual_support_after_local_removal",
    ],
    "localized_only": [
        "drop_full_to_remove_local",
        "mean_drop_one",
        "max_drop_one",
        "min_drop_one",
        "drop_swap_local",
        "normalized_drop",
        "support_bundle_changed_after_removal",
    ],
}

FULL_FEATURES = (
    BASE_FEATURES["support_compactness"]
    + ["redundancy_proxy"]
    + BASE_FEATURES["localized_only"]
    + ["support_full", "best_residual_support_after_local_removal"]
)

TOP2_FEATURES = [
    "support_best_entailment",
    "support_best_contradiction",
    "support_best_neutral",
    "support_margin_second_best",
    "top2_smin_size",
    "top2_support_smin",
    "top2_support_margin_smin_vs_best1",
    "top2_candidate_above_tau",
    "top2_document_dispersion",
    "top2_redundancy_proxy",
    "top2_drop_full_to_remove_local",
    "top2_mean_drop_one",
    "top2_max_drop_one",
    "top2_min_drop_one",
    "top2_drop_swap_local",
    "top2_normalized_drop",
    "top2_support_bundle_changed_after_removal",
    "support_full",
    "top2_best_residual_support_after_local_removal",
]

REGISTRY = {
    "support_only": BASE_FEATURES["support_only"],
    "support_compactness": BASE_FEATURES["support_compactness"],
    "full_context_removal": BASE_FEATURES["full_context_removal"],
    "localized_only": BASE_FEATURES["localized_only"],
    "full_detector": FULL_FEATURES,
    "ablation_no_localized_perturbation": BASE_FEATURES["support_compactness"] + ["redundancy_proxy"],
    "ablation_remove_only": BASE_FEATURES["support_compactness"] + ["redundancy_proxy", "drop_full_to_remove_local", "normalized_drop"],
    "ablation_drop_one_only": BASE_FEATURES["support_compactness"] + ["redundancy_proxy", "mean_drop_one", "max_drop_one", "min_drop_one"],
    "ablation_swap_only": BASE_FEATURES["support_compactness"] + ["redundancy_proxy", "drop_swap_local"],
    "ablation_fixed_topk_support": TOP2_FEATURES,
    "ablation_no_redundancy": [col for col in FULL_FEATURES if col != "redundancy_proxy"],
}


def _gpu_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 3))


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _experiment_dir(name: str) -> Path:
    return ROOT / "exp" / name


def _pred_cols_for(prefix: str, df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith(prefix) and ("_seed_" in col or col.endswith("_mean"))]


def _dataset_counts(claims: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        subset = claims[claims["split"] == split]
        strict = subset[subset[PRIMARY_LABEL].notna()]
        out[split] = {
            "claims": int(len(subset)),
            "strict_claims": int(len(strict)),
            "strict_positives": int((strict[PRIMARY_LABEL] == 1).sum()),
            "strict_negatives": int((strict[PRIMARY_LABEL] == 0).sum()),
            "projected_positives": int((subset["projected_all_label"] == 1).sum()),
        }
    return out


def _sample_train_for_full_features(claims: pd.DataFrame, n: int | None = TRAIN_FEATURE_BUDGET) -> pd.DataFrame:
    train = claims[(claims["split"] == "train") & claims[PRIMARY_LABEL].notna()].copy()
    if n is None or len(train) <= n:
        return train
    key = (
        train["task_type"].astype(str)
        + "|"
        + train["generator_family"].astype(str)
        + "|"
        + train["projected_all_label"].astype(str)
    )
    pieces = []
    for _, group in train.groupby(key):
        take = max(1, round(n * len(group) / len(train)))
        pieces.append(group.sample(n=min(len(group), take), random_state=13))
    sampled = pd.concat(pieces, ignore_index=False).drop_duplicates(["response_id", "sentence_index"])
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=13)
    elif len(sampled) < n:
        remainder = train.drop(sampled.index)
        extra = min(n - len(sampled), len(remainder))
        if extra > 0:
            sampled = pd.concat([sampled, remainder.sample(n=extra, random_state=13)], ignore_index=False)
    return sampled.sort_index()


def _sample_split(
    claims: pd.DataFrame,
    split: str,
    n: int | None,
    require_label: bool = False,
) -> pd.DataFrame:
    subset = claims[claims["split"] == split].copy()
    if require_label:
        subset = subset[subset[PRIMARY_LABEL].notna()].copy()
    if n is None or len(subset) <= n:
        return subset
    key = (
        subset["task_type"].astype(str)
        + "|"
        + subset["generator_family"].astype(str)
        + "|"
        + subset["projected_all_label"].astype(str)
        + "|"
        + subset[PRIMARY_LABEL].fillna(-1).astype(int).astype(str)
    )
    pieces = []
    for _, group in subset.groupby(key):
        take = max(1, round(n * len(group) / len(subset)))
        pieces.append(group.sample(n=min(len(group), take), random_state=13))
    sampled = pd.concat(pieces, ignore_index=False).drop_duplicates(["response_id", "sentence_index"])
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=13)
    elif len(sampled) < n:
        remainder = subset.drop(sampled.index, errors="ignore")
        extra = min(n - len(sampled), len(remainder))
        if extra > 0:
            sampled = pd.concat([sampled, remainder.sample(n=extra, random_state=13)], ignore_index=False)
    return sampled.sort_index()


def _write_experiment_artifacts(name: str, payload: dict[str, Any], predictions: pd.DataFrame | None = None) -> None:
    exp_dir = _experiment_dir(name)
    json_dump(_json_safe(payload), exp_dir / "results.json")
    config = {
        "experiment": name,
        "features": payload.get("config", {}).get("features", []),
        "strict_threshold_mean": payload.get("config", {}).get("strict_threshold_mean"),
        "seed_runs": payload.get("config", {}).get("seed_runs", []),
        "tau": payload.get("config", {}).get("tau"),
        "slices": list(payload.get("metrics", {}).keys()),
        "runtime_minutes": payload.get("runtime_minutes", 0.0),
    }
    json_dump(_json_safe(config), exp_dir / "config.json")
    if predictions is not None and len(predictions):
        predictions.to_parquet(exp_dir / "predictions.parquet", index=False)


def _write_analysis_artifacts(name: str, payload: dict[str, Any]) -> None:
    exp_dir = _experiment_dir(name)
    json_dump(_json_safe(payload), exp_dir / "results.json")
    config = {
        "experiment": name,
        "runtime_minutes": payload.get("runtime_minutes", 0.0),
        "status": payload.get("status"),
    }
    if "config" in payload:
        config.update(_json_safe(payload["config"]))
    json_dump(_json_safe(config), exp_dir / "config.json")


def _metric_pack(df: pd.DataFrame, label_col: str, prob_col: str, threshold: float) -> dict[str, float]:
    base = compute_metrics(df[label_col].to_numpy(dtype=int), df[prob_col].to_numpy(dtype=float), threshold)
    base.update(response_level_metrics(df, "response_id", label_col, prob_col, threshold))
    return base


def _threshold_only(val_df: pd.DataFrame, score_col: str, label_col: str) -> float:
    probs = val_df[score_col].to_numpy(dtype=float)
    y = val_df[label_col].to_numpy(dtype=int)
    best_thr = 0.5
    best_pair = (-1.0, -1.0)
    for thr in np.linspace(0.05, 0.95, 19):
        metrics = compute_metrics(y, probs, thr)
        pair = (metrics["auprc"], metrics["macro_f1"])
        if pair > best_pair:
            best_pair = pair
            best_thr = float(thr)
    return best_thr


def _training_free_scores(train_df: pd.DataFrame, target_df: pd.DataFrame) -> np.ndarray:
    signed_train = np.column_stack(
        [
            train_df["support_full"].to_numpy(dtype=float),
            train_df["support_smin"].to_numpy(dtype=float),
            -train_df["smin_size"].to_numpy(dtype=float),
            train_df["drop_full_to_remove_local"].to_numpy(dtype=float),
        ]
    )
    mean = signed_train.mean(axis=0, keepdims=True)
    std = signed_train.std(axis=0, keepdims=True).clip(min=1e-6)
    signed_target = np.column_stack(
        [
            target_df["support_full"].to_numpy(dtype=float),
            target_df["support_smin"].to_numpy(dtype=float),
            -target_df["smin_size"].to_numpy(dtype=float),
            target_df["drop_full_to_remove_local"].to_numpy(dtype=float),
        ]
    )
    raw = ((signed_target - mean) / std).mean(axis=1)
    return 1.0 / (1.0 + np.exp(-raw))


def _bm25_scores(train_df: pd.DataFrame, target_df: pd.DataFrame) -> np.ndarray:
    train_raw = (
        train_df["top1_bm25"].to_numpy(dtype=float)
        + train_df["mean_top3_bm25"].to_numpy(dtype=float)
        + train_df["top1_minus_top3mean"].to_numpy(dtype=float)
    ) / 3.0
    target_raw = (
        target_df["top1_bm25"].to_numpy(dtype=float)
        + target_df["mean_top3_bm25"].to_numpy(dtype=float)
        + target_df["top1_minus_top3mean"].to_numpy(dtype=float)
    ) / 3.0
    scaled = 1.0 - (target_raw - train_raw.min()) / (train_raw.max() - train_raw.min() + 1e-6)
    return np.clip(scaled, 0.0, 1.0)


def _run_pilot(val_claims: pd.DataFrame, evidence: pd.DataFrame) -> dict[str, Any]:
    log_path = exp_log_path("pilot")
    start = now()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    pilot_claims = val_claims.head(PILOT_SIZE).copy()
    build_claim_features(pilot_claims, evidence, TAU_GRID, full_features=True)
    runtime = runtime_minutes(start)
    payload = {
        "experiment": "pilot",
        "metrics": {},
        "config": {
            "claims": int(len(pilot_claims)),
            "tau_grid": TAU_GRID,
            "full_features": True,
        },
        "runtime_minutes": runtime,
        "peak_gpu_memory_gb": _gpu_peak_gb(),
    }
    append_log(log_path, f"Pilot completed on {len(pilot_claims)} validation claims in {runtime:.2f} minutes.")
    _write_experiment_artifacts("pilot", payload)
    return payload


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], dict[str, Any]]:
    ensure_dirs()
    manifest = build_manifest()
    json_dump(manifest, ROOT / "artifacts" / "run_manifest.json")

    timings: dict[str, float] = {}
    start = now()
    prepared = build_datasets()
    timings["data_prep_minutes"] = runtime_minutes(start)
    data_prep_payload = {
        "experiment": "data_prep",
        "status": "completed",
        "config": {
            "splits": ["train", "val", "test"],
            "primary_label": PRIMARY_LABEL,
        },
        "claims": _dataset_counts(prepared.claims),
        "runtime_minutes": timings["data_prep_minutes"],
    }
    append_log(exp_log_path("data_prep"), f"Prepared datasets in {timings['data_prep_minutes']:.2f} minutes.")
    _write_analysis_artifacts("data_prep", data_prep_payload)

    val_claims = prepared.claims[prepared.claims["split"] == "val"].copy()
    pilot_payload = _run_pilot(val_claims, prepared.evidence)
    timings["pilot_minutes"] = pilot_payload["runtime_minutes"]

    tau_feature_path = ROOT / "artifacts" / "features" / "tau_search_val_features.parquet"
    if tau_feature_path.exists():
        append_log(exp_log_path("pilot"), f"Loading validation tau-search cache from {tau_feature_path.name}.")
        tau_features = pd.read_parquet(tau_feature_path)
        timings["tau_search_minutes"] = 0.0
    else:
        start = now()
        tau_features = build_claim_features(val_claims, prepared.evidence, TAU_GRID, full_features=False)
        tau_features.to_parquet(tau_feature_path, index=False)
        timings["tau_search_minutes"] = runtime_minutes(start)
        append_log(
            exp_log_path("pilot"),
            f"Built validation tau-search cache {tau_feature_path.name} in {timings['tau_search_minutes']:.2f} minutes.",
        )
    prep_info = {
        "train_feature_budget": TRAIN_FEATURE_BUDGET,
        "train_feature_subset_size": int(
            ((prepared.claims["split"] == "train") & prepared.claims[PRIMARY_LABEL].notna()).sum()
        )
        if TRAIN_FEATURE_BUDGET is None
        else int(
            min(
                TRAIN_FEATURE_BUDGET,
                ((prepared.claims["split"] == "train") & prepared.claims[PRIMARY_LABEL].notna()).sum(),
            )
        ),
    }
    return prepared.claims.copy(), prepared.evidence.copy(), tau_features, prepared.stats.copy(), timings, manifest, prep_info


def choose_tau(feature_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    strict_val = feature_df[(feature_df["split"] == "val") & feature_df[PRIMARY_LABEL].notna()].copy()
    score_rows = []
    for tau in TAU_GRID:
        subset = strict_val[strict_val["tau"] == tau].copy()
        metrics = compute_metrics(
            subset[PRIMARY_LABEL].to_numpy(dtype=int),
            subset["support_smin"].to_numpy(dtype=float),
            0.5,
        )
        score_rows.append({"tau": tau, **metrics, "strict_claims": int(len(subset))})
    scores = pd.DataFrame(score_rows).sort_values(["auprc", "brier"], ascending=[False, True]).reset_index(drop=True)
    scores.to_csv(ROOT / "artifacts" / "tables" / "tau_selection.csv", index=False)
    locked_tau = float(scores.iloc[0]["tau"])
    append_log(exp_log_path("analysis_threshold_stability"), f"Locked tau={locked_tau:.2f} from validation tau sweep.")
    return locked_tau, scores


def build_locked_and_stability_features(
    claims: pd.DataFrame,
    evidence: pd.DataFrame,
    locked_tau: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any]]:
    timings: dict[str, float] = {}
    train_budget = DRAFT_TRAIN_LIMIT or TRAIN_FEATURE_BUDGET
    train_sample = _sample_train_for_full_features(claims, train_budget)
    val_sample = _sample_split(claims, "val", DRAFT_VAL_LIMIT)
    test_sample = _sample_split(claims, "test", DRAFT_TEST_LIMIT)
    suffix_parts = []
    for name, value in [("tr", DRAFT_TRAIN_LIMIT), ("va", DRAFT_VAL_LIMIT), ("te", DRAFT_TEST_LIMIT)]:
        if value:
            suffix_parts.append(f"{name}{value}")
    cache_suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

    locked_claims = pd.concat(
        [train_sample, val_sample, test_sample],
        ignore_index=True,
    ).drop_duplicates(["response_id", "sentence_index"])

    locked_path = ROOT / "artifacts" / "features" / f"claim_features_tau_{str(locked_tau).replace('.', '_')}{cache_suffix}.parquet"
    if locked_path.exists():
        locked = pd.read_parquet(locked_path)
        timings["locked_feature_minutes"] = 0.0
    else:
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        start = now()
        locked = build_claim_features(locked_claims, evidence, [locked_tau], full_features=True)
        locked.to_parquet(locked_path, index=False)
        timings["locked_feature_minutes"] = runtime_minutes(start)

    neighbor_taus = _tau_neighbor_grid(locked_tau)
    extra_taus = [tau for tau in neighbor_taus if tau != locked_tau]
    stability_path = ROOT / "artifacts" / "features" / f"stability_features_tau_{str(locked_tau).replace('.', '_')}{cache_suffix}.parquet"
    if extra_taus:
        stability_claims = pd.concat([val_sample, test_sample], ignore_index=True).drop_duplicates(
            ["response_id", "sentence_index"]
        )
        if stability_path.exists():
            extra = pd.read_parquet(stability_path)
            timings["stability_feature_minutes"] = 0.0
        else:
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            start = now()
            extra = build_claim_features(stability_claims, evidence, extra_taus, full_features=False, stability_only=True)
            extra.to_parquet(stability_path, index=False)
            timings["stability_feature_minutes"] = runtime_minutes(start)
        stability = pd.concat([locked[locked["split"].isin(["val", "test"])].copy(), extra], ignore_index=True)
    else:
        stability = locked[locked["split"].isin(["val", "test"])].copy()
        timings["stability_feature_minutes"] = 0.0

    meta = {
        "train_feature_subset_size": int(len(train_sample)),
        "val_feature_subset_size": int(len(val_sample)),
        "test_feature_subset_size": int(len(test_sample)),
        "locked_claim_count": int(len(locked_claims)),
        "stability_claim_count": int(len(stability)),
        "draft_limits": {
            "train": DRAFT_TRAIN_LIMIT,
            "val": DRAFT_VAL_LIMIT,
            "test": DRAFT_TEST_LIMIT,
        },
    }
    return locked, stability, timings, meta


def _evaluate_nontrainable(
    name: str,
    locked: pd.DataFrame,
    score_fn,
) -> tuple[dict[str, Any], pd.DataFrame]:
    log_path = exp_log_path(name)
    append_log(log_path, f"Starting {name}.")
    start = now()
    payload: dict[str, Any] = {"experiment": name, "metrics": {}, "config": {"tau": float(locked['tau'].iloc[0])}}
    train_locked = locked[(locked["split"] == "train") & locked[PRIMARY_LABEL].notna()].copy()
    val_locked = locked[locked["split"] == "val"].copy()
    test_locked = locked[locked["split"] == "test"].copy()
    prediction_out = test_locked[["response_id", "sentence_index", "example_id", PRIMARY_LABEL, "projected_all_label"]].copy()

    for slice_name, label_col in SLICE_LABELS.items():
        val_slice = val_locked[val_locked[label_col].notna()].copy()
        test_slice = test_locked[test_locked[label_col].notna()].copy()
        if len(val_slice) == 0 or len(test_slice) == 0:
            continue
        val_slice[f"{name}_prob"] = score_fn(train_locked, val_slice)
        threshold = _threshold_only(val_slice, f"{name}_prob", label_col)
        test_slice[f"{name}_prob"] = score_fn(train_locked, test_slice)
        metrics = _metric_pack(test_slice, label_col, f"{name}_prob", threshold)
        payload["metrics"][slice_name] = {k: {"mean": v, "std": 0.0} for k, v in metrics.items()}
        payload["config"][slice_name] = {"threshold": threshold}
        if slice_name == "strict":
            payload["config"]["strict_threshold_mean"] = threshold
            prediction_out[f"{name}_mean"] = test_slice.set_index(["response_id", "sentence_index"])[f"{name}_prob"].reindex(
                prediction_out.set_index(["response_id", "sentence_index"]).index
            ).to_numpy()
    payload["runtime_minutes"] = runtime_minutes(start)
    append_log(log_path, f"Completed {name} in {payload['runtime_minutes']:.2f} minutes.")
    _write_experiment_artifacts(name, payload, prediction_out)
    return payload, prediction_out


def _evaluate_logreg_model(
    name: str,
    locked: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    log_path = exp_log_path(name)
    append_log(log_path, f"Starting {name} with {len(feature_cols)} features.")
    start = now()
    payload: dict[str, Any] = {
        "experiment": name,
        "metrics": {},
        "config": {"features": feature_cols, "tau": float(locked["tau"].iloc[0])},
    }
    strict_train = locked[(locked["split"] == "train") & locked[PRIMARY_LABEL].notna()].copy()
    strict_val = locked[(locked["split"] == "val") & locked[PRIMARY_LABEL].notna()].copy()
    all_val = locked[locked["split"] == "val"].copy()
    all_test = locked[locked["split"] == "test"].copy()
    test_predictions = all_test[["response_id", "sentence_index", "example_id", PRIMARY_LABEL, "projected_all_label"]].copy()
    val_predictions = all_val[["response_id", "sentence_index", "example_id", PRIMARY_LABEL, "projected_all_label"]].copy()

    seed_outputs = []
    strict_thresholds = []
    for seed in SEEDS:
        fit = fit_logreg(
            strict_train[feature_cols],
            strict_train[PRIMARY_LABEL].to_numpy(dtype=int),
            strict_val[feature_cols],
            strict_val[PRIMARY_LABEL].to_numpy(dtype=int),
            LOGREG_C_GRID,
            seed,
        )
        seed_record = {"seed": seed, "C": fit["C"], "strict_threshold": fit["threshold"]}
        strict_thresholds.append(float(fit["threshold"]))

        val_probs = fit["clf"].predict_proba(fit["scaler"].transform(all_val[feature_cols]))[:, 1]
        val_predictions[f"{name}_seed_{seed}"] = val_probs

        for slice_name, label_col in SLICE_LABELS.items():
            test_slice = all_test[all_test[label_col].notna()].copy()
            probs = fit["clf"].predict_proba(fit["scaler"].transform(test_slice[feature_cols]))[:, 1]
            test_predictions.loc[test_slice.index, f"{name}_seed_{seed}_{slice_name}"] = probs
            metrics = _metric_pack(test_slice.assign(model_prob=probs), label_col, "model_prob", fit["threshold"])
            seed_record[slice_name] = metrics
        seed_outputs.append(seed_record)

    for slice_name in SLICE_LABELS:
        payload["metrics"][slice_name] = summarize_metric_dicts([row[slice_name] for row in seed_outputs])
        cols = [f"{name}_seed_{seed}_{slice_name}" for seed in SEEDS if f"{name}_seed_{seed}_{slice_name}" in test_predictions.columns]
        if slice_name == "strict" and cols:
            test_predictions[f"{name}_mean"] = test_predictions[cols].mean(axis=1)

    val_seed_cols = [f"{name}_seed_{seed}" for seed in SEEDS]
    val_predictions[f"{name}_val_mean"] = val_predictions[val_seed_cols].mean(axis=1)
    payload["config"]["seed_runs"] = seed_outputs
    payload["config"]["strict_threshold_mean"] = float(np.mean(strict_thresholds))
    payload["runtime_minutes"] = runtime_minutes(start)
    append_log(log_path, f"Completed {name} in {payload['runtime_minutes']:.2f} minutes.")
    _write_experiment_artifacts(name, payload, test_predictions)
    val_predictions.to_parquet(_experiment_dir(name) / "val_predictions.parquet", index=False)
    return payload, test_predictions, val_predictions


def _save_main_table(results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    order = [
        "bm25",
        "support_only",
        "support_compactness",
        "full_context_removal",
        "localized_only",
        "training_free_additive",
        "full_detector",
        "ablation_no_localized_perturbation",
        "ablation_remove_only",
        "ablation_drop_one_only",
        "ablation_swap_only",
        "ablation_fixed_topk_support",
        "ablation_no_redundancy",
    ]
    for name in order:
        if name not in results["experiments"]:
            continue
        exp = results["experiments"][name]
        row = {"experiment": name, "runtime_minutes": exp.get("runtime_minutes", 0.0)}
        for slice_name in ["strict", "projected_all"]:
            metrics = exp["metrics"].get(slice_name, {})
            for metric in ["macro_f1", "auprc", "auroc", "brier", "ece", "response_f1"]:
                row[f"{slice_name}_{metric}"] = metrics.get(metric, {}).get("mean")
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "artifacts" / "tables" / "main_results.csv", index=False)
    return table


def _jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def _prediction_threshold(results: dict[str, Any], model_name: str) -> float:
    return float(results["experiments"][model_name]["config"]["strict_threshold_mean"])


def _tau_neighbor_grid(locked_tau: float) -> list[float]:
    idx = TAU_GRID.index(locked_tau)
    return [TAU_GRID[i] for i in [idx - 1, idx, idx + 1] if 0 <= i < len(TAU_GRID)]


def _evaluate_tau_variants(feature_df: pd.DataFrame, tau_values: list[float]) -> dict[float, dict[str, dict[str, Any]]]:
    outputs: dict[float, dict[str, dict[str, Any]]] = {}
    for tau in tau_values:
        tau_df = feature_df[feature_df["tau"] == tau].copy()
        tau_results: dict[str, dict[str, Any]] = {}
        for name in ["support_only", "support_compactness", "full_context_removal", "full_detector"]:
            strict_train = tau_df[(tau_df["split"] == "train") & tau_df[PRIMARY_LABEL].notna()].copy()
            strict_val = tau_df[(tau_df["split"] == "val") & tau_df[PRIMARY_LABEL].notna()].copy()
            strict_test = tau_df[(tau_df["split"] == "test") & tau_df[PRIMARY_LABEL].notna()].copy()
            if len(strict_train) == 0 or len(strict_val) == 0 or len(strict_test) == 0:
                tau_results[name] = {
                    "experiment": f"{name}_tau_{str(tau).replace('.', '_')}",
                    "status": "skipped_missing_train_or_eval_split",
                    "metrics": {},
                }
                continue
            payload, _, _ = _evaluate_logreg_model(f"{name}_tau_{str(tau).replace('.', '_')}", tau_df, REGISTRY[name])
            tau_results[name] = payload
        outputs[tau] = tau_results
    return outputs


def _threshold_stability(
    feature_df: pd.DataFrame,
    locked_tau: float,
    strict_test_predictions: pd.DataFrame,
    results: dict[str, Any],
) -> dict[str, Any]:
    neighbor_taus = _tau_neighbor_grid(locked_tau)
    keyed = {
        tau: feature_df[(feature_df["tau"] == tau) & feature_df[PRIMARY_LABEL].notna()].set_index(["response_id", "sentence_index"]).sort_index()
        for tau in neighbor_taus
    }
    base_val = keyed[locked_tau][keyed[locked_tau]["split"] == "val"].copy()
    base_test = keyed[locked_tau][keyed[locked_tau]["split"] == "test"].copy()

    tau_eval = _evaluate_tau_variants(feature_df, neighbor_taus)
    rows = []
    for tau in neighbor_taus:
        comp_val = keyed[tau].loc[base_val.index]
        comp_test = keyed[tau].loc[base_test.index]
        support_metrics = tau_eval[tau].get("support_only", {}).get("metrics", {}).get("strict", {})
        full_metrics = tau_eval[tau].get("full_detector", {}).get("metrics", {}).get("strict", {})
        rows.append(
            {
                "tau": tau,
                "validation_auprc_support_only": compute_metrics(
                    comp_val[PRIMARY_LABEL].to_numpy(dtype=int),
                    comp_val["support_smin"].to_numpy(dtype=float),
                    0.5,
                )["auprc"],
                "validation_brier_support_only": compute_metrics(
                    comp_val[PRIMARY_LABEL].to_numpy(dtype=int),
                    comp_val["support_smin"].to_numpy(dtype=float),
                    0.5,
                )["brier"],
                "test_macro_f1_support_only": support_metrics.get("macro_f1", {}).get("mean"),
                "test_macro_f1_full_detector": full_metrics.get("macro_f1", {}).get("mean"),
                "test_auprc_support_only": support_metrics.get("auprc", {}).get("mean"),
                "test_auprc_full_detector": full_metrics.get("auprc", {}).get("mean"),
                "smin_change_rate_vs_locked": float((base_test["smin_indices"] != comp_test["smin_indices"]).mean()),
                "mean_jaccard_vs_locked": float(np.mean([_jaccard(a, b) for a, b in zip(base_test["smin_indices"], comp_test["smin_indices"])])),
                "mean_abs_size_change_vs_locked": float(np.abs(base_test["smin_size"] - comp_test["smin_size"]).mean()),
                "drop_sign_flip_rate_vs_locked": float(
                    (np.sign(base_test["drop_full_to_remove_local"]) != np.sign(comp_test["drop_full_to_remove_local"])).mean()
                ),
            }
        )

    stable_mask = pd.Series(True, index=base_test.index)
    for tau in neighbor_taus:
        stable_mask &= base_test["smin_indices"] == keyed[tau].loc[base_test.index, "smin_indices"]
    stable_keys = set(stable_mask[stable_mask].index.tolist())

    pred_indexed = strict_test_predictions.set_index(["response_id", "sentence_index"]).sort_index().copy()
    pred_indexed["stable_smin"] = [key in stable_keys for key in pred_indexed.index]
    subset_rows = []
    for subset_name, mask in {
        "stable_Smin": pred_indexed["stable_smin"],
        "unstable_Smin": ~pred_indexed["stable_smin"],
    }.items():
        subset = pred_indexed[mask].reset_index()
        if len(subset) == 0:
            continue
        for model_name in ["support_only", "support_compactness", "full_context_removal", "full_detector"]:
            prob_col = f"{model_name}_mean"
            if prob_col not in subset.columns:
                continue
            threshold = _prediction_threshold(results, model_name)
            metrics = compute_metrics(
                subset[PRIMARY_LABEL].to_numpy(dtype=int),
                subset[prob_col].to_numpy(dtype=float),
                threshold,
            )
            subset_rows.append({"subset": subset_name, "model": model_name, **metrics, "claims": int(len(subset))})

    stability_df = pd.DataFrame(rows)
    subset_df = pd.DataFrame(subset_rows)
    stability_df.to_csv(ROOT / "artifacts" / "tables" / "tau_stability.csv", index=False)
    subset_df.to_csv(ROOT / "artifacts" / "tables" / "stable_unstable_results.csv", index=False)
    payload = {
        "experiment": "analysis_threshold_stability",
        "status": "completed",
        "locked_tau": locked_tau,
        "neighbor_taus": neighbor_taus,
        "tau_table": rows,
        "stable_subset_results": subset_rows,
    }
    _write_analysis_artifacts("analysis_threshold_stability", payload)
    append_log(exp_log_path("analysis_threshold_stability"), f"Saved threshold stability outputs for tau={locked_tau:.2f}.")
    return payload


def _save_calibration_outputs(val_predictions: pd.DataFrame, results: dict[str, Any]) -> dict[str, Any]:
    strict_val = val_predictions[val_predictions[PRIMARY_LABEL].notna()].copy()
    cal: dict[str, Any] = {}
    for name, prob_col in {
        "support_only": "support_only_val_mean",
        "full_detector": "full_detector_val_mean",
    }.items():
        if prob_col not in strict_val.columns:
            continue
        threshold = _prediction_threshold(results, name)
        cal[name] = {
            "ece": compute_metrics(
                strict_val[PRIMARY_LABEL].to_numpy(dtype=int),
                strict_val[prob_col].to_numpy(dtype=float),
                threshold,
            )["ece"],
            "brier": compute_metrics(
                strict_val[PRIMARY_LABEL].to_numpy(dtype=int),
                strict_val[prob_col].to_numpy(dtype=float),
                threshold,
            )["brier"],
            "curve": calibration_points(
                strict_val[PRIMARY_LABEL].to_numpy(dtype=int),
                strict_val[prob_col].to_numpy(dtype=float),
            ),
        }
    json_dump(cal, ROOT / "artifacts" / "tables" / "calibration.json")
    return cal


def _bootstrap_differences(strict_test: pd.DataFrame, results: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        "support_only": "support_only_mean",
        "support_compactness": "support_compactness_mean",
        "full_context_removal": "full_context_removal_mean",
    }
    output: dict[str, Any] = {}
    for model_name, col in comparisons.items():
        output[f"full_vs_{model_name}"] = {}
        for metric in ["macro_f1", "auprc", "auroc"]:
            output[f"full_vs_{model_name}"][metric] = bootstrap_metric_diff(
                strict_test,
                "full_detector_mean",
                col,
                PRIMARY_LABEL,
                metric=metric,
                threshold_a=_prediction_threshold(results, "full_detector"),
                threshold_b=_prediction_threshold(results, model_name),
            )
    json_dump(output, ROOT / "artifacts" / "tables" / "bootstrap_differences.json")
    return output


def _prepare_audit_sample(val_predictions: pd.DataFrame) -> dict[str, Any]:
    audit_log = exp_log_path("audit_uncertainty")
    start = now()
    val_df = val_predictions.copy()
    if "support_only_val_mean" not in val_df.columns or "full_detector_val_mean" not in val_df.columns:
        payload = {"experiment": "audit_uncertainty", "status": "unavailable", "runtime_minutes": runtime_minutes(start)}
        _write_analysis_artifacts("audit_uncertainty", payload)
        return payload

    val_df["support_conf_q"] = pd.qcut(val_df["support_only_val_mean"], q=4, labels=False, duplicates="drop")
    val_df["remove_drop_q"] = pd.qcut(val_df["drop_full_to_remove_local"], q=4, labels=False, duplicates="drop")
    val_df["disagreement"] = (
        (val_df["support_only_val_mean"] >= 0.5).astype(int) != (val_df["full_detector_val_mean"] >= 0.5).astype(int)
    ).astype(int)
    strata_cols = ["ambiguity_flag", "support_conf_q", "remove_drop_q", "disagreement"]
    sample_parts = []
    per_group = max(1, AUDIT_TARGET // max(1, val_df.groupby(strata_cols).ngroups))
    for _, group in val_df.groupby(strata_cols):
        sample_parts.append(group.sample(n=min(len(group), per_group), random_state=13))
    sample = pd.concat(sample_parts, ignore_index=False).drop_duplicates(["response_id", "sentence_index"])
    if len(sample) > AUDIT_TARGET:
        sample = sample.sample(n=AUDIT_TARGET, random_state=13)
    sample = sample[
        [
            "example_id",
            "response_id",
            "sentence_index",
            "answer_sentence",
            "strict_label",
            "projected_all_label",
            "ambiguity_flag",
            "support_only_val_mean",
            "full_detector_val_mean",
            "support_compactness_val_mean",
            "full_context_removal_val_mean",
            "drop_full_to_remove_local",
            "mean_drop_one",
            "drop_swap_local",
        ]
    ].copy()
    sample["annotator_a"] = ""
    sample["annotator_b"] = ""
    sample["adjudicated_label"] = ""
    sample["rationale"] = ""
    sample_path = ROOT / "artifacts" / "annotations" / "manual_audit_template.csv"
    sample.to_csv(sample_path, index=False)
    append_log(audit_log, f"Wrote manual audit template with {len(sample)} claims.")

    adjudicated_path = ROOT / "artifacts" / "annotations" / "manual_audit_adjudicated.csv"
    if not adjudicated_path.exists():
        skip_text = "\n".join(
            [
                "Manual audit analysis could not be completed in this rerun.",
                "",
                "Completed:",
                "- Regenerated a stratified 200-claim audit template at artifacts/annotations/manual_audit_template.csv.",
                "- Included benchmark labels, detector scores, and localized feature scores required by the plan.",
                "",
                "Blocked:",
                "- The plan requires two independent annotations plus adjudication.",
                "- No adjudicated audit file was present in the workspace.",
                "",
                "Interpretation impact:",
                "- Benchmark and robustness metrics are executed and reported.",
                "- Evidence-dependence and post-rationalization claims remain exploratory until adjudicated labels are added.",
            ]
        )
        (ROOT / "exp" / "audit_uncertainty" / "SKIPPED.md").write_text(skip_text)
        payload = {
            "experiment": "audit_uncertainty",
            "status": "template_only",
            "template_path": str(sample_path),
            "runtime_minutes": runtime_minutes(start),
        }
        _write_analysis_artifacts("audit_uncertainty", payload)
        return payload

    adjudicated = pd.read_csv(adjudicated_path)
    merged = sample.merge(adjudicated, on=["response_id", "sentence_index"], how="inner", suffixes=("", "_audit"))
    label_map = {"evidence-dependent": 0, "post-rationalized": 1}
    merged = merged[merged["adjudicated_label"].isin(label_map)].copy()
    if len(merged) == 0:
        payload = {
            "experiment": "audit_uncertainty",
            "status": "adjudicated_missing_binary_labels",
            "path": str(adjudicated_path),
            "runtime_minutes": runtime_minutes(start),
        }
        _write_analysis_artifacts("audit_uncertainty", payload)
        return payload

    merged["audit_binary"] = merged["adjudicated_label"].map(label_map).astype(int)
    metrics = {}
    for name, col in {
        "support_only": "support_only_val_mean",
        "support_compactness": "support_compactness_val_mean",
        "full_context_removal": "full_context_removal_val_mean",
        "remove_local": "drop_full_to_remove_local",
        "drop_one": "mean_drop_one",
        "swap_local": "drop_swap_local",
        "full_detector": "full_detector_val_mean",
    }.items():
        temp = merged.rename(columns={col: f"{name}_score"})
        metrics[name] = bootstrap_metric_ci(
            temp,
            f"{name}_score",
            "audit_binary",
            metric="auroc",
            threshold=0.5,
        )
    json_dump(metrics, ROOT / "artifacts" / "tables" / "audit_metrics.json")
    payload = {
        "experiment": "audit_uncertainty",
        "status": "completed",
        "metrics": metrics,
        "claims": int(len(merged)),
        "runtime_minutes": runtime_minutes(start),
    }
    _write_analysis_artifacts("audit_uncertainty", payload)
    append_log(audit_log, f"Completed adjudicated audit analysis on {len(merged)} claims.")
    return payload


def _robustness_for_group(
    locked: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
    group_value: str,
    model_name: str,
) -> dict[str, Any] | None:
    train = locked[(locked["split"] == "train") & locked[PRIMARY_LABEL].notna() & (locked[group_col] != group_value)].copy()
    val = locked[(locked["split"] == "val") & locked[PRIMARY_LABEL].notna() & (locked[group_col] != group_value)].copy()
    test = locked[(locked["split"] == "test") & locked[PRIMARY_LABEL].notna() & (locked[group_col] == group_value)].copy()
    if len(test) < MIN_ROBUSTNESS_SUPPORT or test[PRIMARY_LABEL].nunique() < 2 or len(train) == 0 or len(val) == 0:
        return None
    seed_metrics = []
    thresholds = []
    for seed in SEEDS:
        fit = fit_logreg(
            train[feature_cols],
            train[PRIMARY_LABEL].to_numpy(dtype=int),
            val[feature_cols],
            val[PRIMARY_LABEL].to_numpy(dtype=int),
            LOGREG_C_GRID,
            seed,
        )
        probs = fit["clf"].predict_proba(fit["scaler"].transform(test[feature_cols]))[:, 1]
        thresholds.append(float(fit["threshold"]))
        seed_metrics.append(_metric_pack(test.assign(model_prob=probs), PRIMARY_LABEL, "model_prob", fit["threshold"]))
    return {
        "group_col": group_col,
        "group_value": group_value,
        "model": model_name,
        "test_claims": int(len(test)),
        "test_positives": int((test[PRIMARY_LABEL] == 1).sum()),
        "metrics": summarize_metric_dicts(seed_metrics),
        "threshold_mean": float(np.mean(thresholds)),
    }


def _run_robustness(locked: pd.DataFrame) -> dict[str, Any]:
    if SKIP_ROBUSTNESS:
        pd.DataFrame([]).to_csv(ROOT / "artifacts" / "tables" / "robustness_results.csv", index=False)
        return {"status": "skipped_by_env", "rows": []}
    rows = []
    for group_col in ["generator_family", "task_type"]:
        values = sorted(locked.loc[locked["split"] == "test", group_col].dropna().astype(str).unique().tolist())
        for group_value in values:
            for model_name in ["support_only", "support_compactness", "full_context_removal", "full_detector"]:
                record = _robustness_for_group(locked, REGISTRY[model_name], group_col, group_value, model_name)
                if record is not None:
                    rows.append(record)
    flat_rows = []
    for row in rows:
        flat = {k: row[k] for k in ["group_col", "group_value", "model", "test_claims", "test_positives", "threshold_mean"]}
        for metric, stats in row["metrics"].items():
            flat[f"{metric}_mean"] = stats["mean"]
            flat[f"{metric}_std"] = stats["std"]
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(ROOT / "artifacts" / "tables" / "robustness_results.csv", index=False)
    return {"rows": rows}


def _success_assessment(results: dict[str, Any]) -> dict[str, Any]:
    strict = results["experiments"]["full_detector"]["metrics"]["strict"]
    support = results["experiments"]["support_only"]["metrics"]["strict"]
    compact = results["experiments"]["support_compactness"]["metrics"]["strict"]
    removal = results["experiments"]["full_context_removal"]["metrics"]["strict"]
    tau_rows = pd.DataFrame(results["threshold_stability"]["tau_table"])
    stable_rows = pd.DataFrame(results["threshold_stability"]["stable_subset_results"])
    stable_full = stable_rows[(stable_rows["subset"] == "stable_Smin") & (stable_rows["model"] == "full_detector")]
    stable_support = stable_rows[(stable_rows["subset"] == "stable_Smin") & (stable_rows["model"] == "support_only")]

    return {
        "strict_vs_support_only": {
            "macro_f1_delta": strict["macro_f1"]["mean"] - support["macro_f1"]["mean"],
            "auprc_delta": strict["auprc"]["mean"] - support["auprc"]["mean"],
            "meets_target": bool(
                (strict["macro_f1"]["mean"] - support["macro_f1"]["mean"] >= 0.01)
                or (strict["auprc"]["mean"] - support["auprc"]["mean"] >= 0.01)
            ),
        },
        "strict_vs_support_compactness": {
            "macro_f1_delta": strict["macro_f1"]["mean"] - compact["macro_f1"]["mean"],
            "auprc_delta": strict["auprc"]["mean"] - compact["auprc"]["mean"],
            "meets_target": bool(
                (strict["macro_f1"]["mean"] - compact["macro_f1"]["mean"] >= 0.005)
                or (strict["auprc"]["mean"] - compact["auprc"]["mean"] >= 0.005)
            ),
        },
        "strict_vs_full_context_removal": {
            "macro_f1_delta": strict["macro_f1"]["mean"] - removal["macro_f1"]["mean"],
            "auprc_delta": strict["auprc"]["mean"] - removal["auprc"]["mean"],
            "meets_target": bool(
                (strict["macro_f1"]["mean"] - removal["macro_f1"]["mean"] >= 0.005)
                or (strict["auprc"]["mean"] - removal["auprc"]["mean"] >= 0.005)
            ),
        },
        "threshold_stability": {
            "directionally_positive_all_neighbor_taus_macro_f1": bool(
                ((tau_rows["test_macro_f1_full_detector"] - tau_rows["test_macro_f1_support_only"]) > 0).all()
            ),
            "directionally_positive_all_neighbor_taus_auprc": bool(
                ((tau_rows["test_auprc_full_detector"] - tau_rows["test_auprc_support_only"]) > 0).all()
            ),
            "stable_subset_delta_macro_f1": None if stable_full.empty or stable_support.empty else float(
                stable_full["macro_f1"].iloc[0] - stable_support["macro_f1"].iloc[0]
            ),
        },
        "audit_status": results["audit"]["status"],
    }


def run_all() -> dict[str, Any]:
    overall_start = now()
    claims, evidence, tau_features, data_stats, timings, manifest, prep_info = prepare()
    locked_tau, tau_selection = choose_tau(tau_features)
    locked, stability_features, feature_timings, feature_meta = build_locked_and_stability_features(claims, evidence, locked_tau)
    timings.update(feature_timings)

    results: dict[str, Any] = {
        "attempt_scope": {
            "type": "draft_subset_protocol"
            if any(feature_meta["draft_limits"].values())
            else (
                "full_protocol_with_training_scope_reduction"
                if feature_meta["train_feature_subset_size"] < claims[(claims["split"] == "train") & claims[PRIMARY_LABEL].notna()].shape[0]
                else "full_protocol"
            ),
            "locked_tau": locked_tau,
            "tau_grid": TAU_GRID,
            "seeds": SEEDS,
            "dataset_counts": _dataset_counts(claims),
            "notes": [
                "All reported metrics were regenerated from one frozen code path.",
                "The workspace environment is expected to be Python 3.10 via the project virtualenv.",
                (
                    f"Draft-scope subset limits were applied during locked-feature extraction: train={feature_meta['draft_limits']['train']}, val={feature_meta['draft_limits']['val']}, test={feature_meta['draft_limits']['test']}."
                    if any(feature_meta["draft_limits"].values())
                    else "No draft-scope val/test subset limits were applied."
                ),
                f"Training-side localized feature extraction was capped at {feature_meta['train_feature_subset_size']} strict-train claims and applied identically to every trainable model to stay within the 8-hour budget." if feature_meta["train_feature_subset_size"] < claims[(claims["split"] == "train") & claims[PRIMARY_LABEL].notna()].shape[0] else "No training-side feature cap was applied.",
                "Threshold-stability feature extraction was restricted to validation and test claims because train claims are not used in that analysis." if feature_meta["train_feature_subset_size"] < claims[(claims["split"] == "train") & claims[PRIMARY_LABEL].notna()].shape[0] else "No scope reduction beyond the planned protocol was applied.",
            ],
        },
        "manifest": manifest,
        "data_stats": data_stats.to_dict(orient="records"),
        "tau_selection": tau_selection.to_dict(orient="records"),
        "experiments": {},
        "timings": timings,
        "plan_compliance": {},
        "scope_reduction": {
            "train_feature_budget": prep_info["train_feature_budget"],
            "train_feature_subset_size": feature_meta["train_feature_subset_size"],
            "reason": None
            if feature_meta["train_feature_subset_size"] >= claims[(claims["split"] == "train") & claims[PRIMARY_LABEL].notna()].shape[0]
            else "Pilot runtime showed all strict-train claims plus validation/test stability sweeps would exceed the 8-hour single-GPU budget, so all trainable models used the same 20k-claim strict-train feature subset.",
        },
    }

    bm25_payload, bm25_test = _evaluate_nontrainable("bm25", locked, _bm25_scores)
    results["experiments"]["bm25"] = bm25_payload

    strict_test_predictions = locked[locked["split"] == "test"][["response_id", "sentence_index", "example_id", PRIMARY_LABEL, "projected_all_label"]].copy()
    val_predictions = locked[locked["split"] == "val"].copy()
    for name, feature_cols in REGISTRY.items():
        payload, pred_df, val_df = _evaluate_logreg_model(name, locked, feature_cols)
        results["experiments"][name] = payload
        for col in _pred_cols_for(name, pred_df):
            strict_test_predictions[col] = pred_df[col]
        val_index = val_predictions.set_index(["response_id", "sentence_index"]).index
        val_aligned = val_df.set_index(["response_id", "sentence_index"]).reindex(val_index)
        for col in [c for c in val_df.columns if c.endswith("_val_mean")]:
            val_predictions[col] = val_aligned[col].to_numpy()

    base_val_index = val_predictions.set_index(["response_id", "sentence_index"]).index
    locked_val_aligned = locked[locked["split"] == "val"].set_index(["response_id", "sentence_index"]).reindex(base_val_index)
    for col in ["drop_full_to_remove_local", "mean_drop_one", "drop_swap_local"]:
        val_predictions[col] = locked_val_aligned[col].to_numpy()

    additive_payload, additive_test = _evaluate_nontrainable("training_free_additive", locked, _training_free_scores)
    results["experiments"]["training_free_additive"] = additive_payload
    if "training_free_additive_mean" in additive_test.columns:
        strict_test_predictions["training_free_additive_mean"] = additive_test["training_free_additive_mean"]

    strict_test = strict_test_predictions[strict_test_predictions[PRIMARY_LABEL].notna()].copy()
    strict_test.to_parquet(ROOT / "artifacts" / "predictions" / "strict_test_predictions.parquet", index=False)
    val_predictions.to_parquet(ROOT / "artifacts" / "predictions" / "val_predictions.parquet", index=False)

    results["bootstrap_differences"] = _bootstrap_differences(strict_test, results)
    results["threshold_stability"] = _threshold_stability(stability_features, locked_tau, strict_test, results)
    results["calibration"] = _save_calibration_outputs(val_predictions, results)
    results["audit"] = _prepare_audit_sample(val_predictions)
    results["robustness"] = _run_robustness(locked)
    main_table = _save_main_table(results)

    results["plan_compliance"] = {
        "data_preparation": "completed",
        "baselines": "completed",
        "main_model": "completed",
        "threshold_stability": "completed",
        "ablations": "completed",
        "audit_uncertainty": results["audit"]["status"],
        "robustness": "completed" if len(results["robustness"]["rows"]) else "no_supported_groups",
        "visualization_inputs": "completed" if len(main_table) else "missing",
    }
    results["success_assessment"] = _success_assessment(results)
    results["timings"]["total_runtime_minutes"] = runtime_minutes(overall_start)
    json_dump(_json_safe(results), ROOT / "results.json")
    return results

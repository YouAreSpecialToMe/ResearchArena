from __future__ import annotations

import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ARTIFACTS_MODELS, ARTIFACTS_POSTERIORS, EXP_DIR
from .conformal import (
    batch_multivalid_predict_sets,
    gmm_memberships,
    hard_overlap_rlcp_predict_sets,
    knn_memberships,
    predict_sets_class_conditional,
    predict_sets_global,
    rlcp_predict_sets,
)
from .io import append_log, ensure_dir, read_json, write_json
from .model import HierarchicalMixtureClassifier


def _peak_rss_gb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2))


def fit_or_load_model(
    dataset: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[HierarchicalMixtureClassifier, dict[str, Any]]:
    model_path = ARTIFACTS_MODELS / f"{dataset}_seed{seed}_pc.pkl"
    meta_path = ARTIFACTS_MODELS / f"{dataset}_seed{seed}_pc_meta.json"
    if model_path.exists() and meta_path.exists():
        model = HierarchicalMixtureClassifier.load(model_path)
        meta = read_json(meta_path)
        return model, meta
    model = HierarchicalMixtureClassifier(random_state=seed)
    rss_before = _peak_rss_gb()
    model.fit(X_train, y_train)
    rss_after = _peak_rss_gb()
    model.save(model_path)
    meta = {
        "fit_time_sec": model.fit_time_sec_,
        "peak_rss_gb": max(rss_before, rss_after),
        "node_count": model.node_count_,
        "sum_node_count": model.sum_node_count_,
        "max_depth": model.max_depth_,
        "model_family": "hierarchical_diagonal_gmm",
    }
    write_json(meta_path, meta)
    return model, meta


def compute_memberships(
    model: HierarchicalMixtureClassifier,
    dataset: str,
    seed: int,
    X_cal: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, np.ndarray]:
    path = ARTIFACTS_POSTERIORS / f"{dataset}_seed{seed}_memberships.npz"
    if path.exists():
        payload = np.load(path, allow_pickle=True)
        return {k: payload[k] for k in payload.files}
    cal = model.posterior_memberships(X_cal)
    test = model.posterior_memberships(X_test)
    out = {
        "coarse_cal": cal["coarse"],
        "coarse_test": test["coarse"],
        "fine_cal": cal["fine"],
        "fine_test": test["fine"],
    }
    np.savez_compressed(path, **out)
    return out


def select_chip_memberships(
    memberships: dict[str, np.ndarray],
    variant: str = "full",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    coarse_cal = memberships["coarse_cal"]
    coarse_test = memberships["coarse_test"]
    fine_cal = memberships["fine_cal"]
    fine_test = memberships["fine_test"]
    coarse_count = coarse_cal.shape[1]
    fine_count = fine_cal.shape[1]

    if variant == "full":
        cal = np.concatenate([0.5 * coarse_cal, 0.5 * fine_cal], axis=1)
        test = np.concatenate([0.5 * coarse_test, 0.5 * fine_test], axis=1)
        coarse_mask = np.concatenate([np.ones(coarse_count, dtype=bool), np.zeros(fine_count, dtype=bool)])
    elif variant == "flat_only":
        cal, test = fine_cal, fine_test
        coarse_mask = np.zeros(fine_count, dtype=bool)
    elif variant == "coarse_only":
        cal, test = coarse_cal, coarse_test
        coarse_mask = np.ones(coarse_count, dtype=bool)
    else:
        raise ValueError(f"Unknown CHiP variant: {variant}")

    masses = cal.sum(axis=0)
    keep = masses >= 20.0
    cal = cal[:, keep]
    test = test[:, keep]
    kept_coarse = int(coarse_mask[keep].sum()) if keep.size == coarse_mask.size else 0
    summary = {
        "num_groups": int(keep.sum()),
        "num_coarse_groups": kept_coarse,
        "num_fine_groups": int(keep.sum()) - kept_coarse,
        "mean_active_groups": float((cal > 0.01).sum(axis=1).mean()) if cal.size else 0.0,
        "median_active_groups": float(np.median((cal > 0.01).sum(axis=1))) if cal.size else 0.0,
    }
    return cal, test, summary


def evaluate_groups(
    covered: np.ndarray,
    eval_meta_test: dict[str, np.ndarray],
    target_coverage: float,
) -> dict[str, Any]:
    per_group = []
    thresholds = {
        "coarse": 50,
        "fine": 50,
        "coarse_by_class": 50,
        "Family": 30,
        "Genus": 30,
        "Family x true class": 30,
        "Genotype": 20,
        "Treatment": 20,
        "Behavior": 20,
        "Genotype x Treatment": 20,
    }
    for name, groups in eval_meta_test.items():
        groups = np.asarray(groups).astype(str)
        for group in np.unique(groups):
            mask = groups == group
            if mask.sum() < thresholds.get(name, 20):
                continue
            cov = float(covered[mask].mean())
            per_group.append(
                {
                    "group_family": name,
                    "group": str(group),
                    "n": int(mask.sum()),
                    "coverage": cov,
                    "coverage_gap": float(abs(cov - target_coverage)),
                }
            )
    if not per_group:
        return {
            "worst_external_group_coverage": float("nan"),
            "mean_external_group_coverage_gap": float("nan"),
            "max_external_group_coverage_gap": float("nan"),
            "groups": [],
        }
    covs = np.array([x["coverage"] for x in per_group], dtype=float)
    gaps = np.array([x["coverage_gap"] for x in per_group], dtype=float)
    return {
        "worst_external_group_coverage": float(covs.min()),
        "mean_external_group_coverage_gap": float(gaps.mean()),
        "max_external_group_coverage_gap": float(gaps.max()),
        "groups": per_group,
    }


def run_method(
    dataset_bundle,
    dataset: str,
    seed: int,
    method: str,
    alpha: float,
    chip_variant: str = "full",
    fallback_lambda: float = 0.1,
    run_label: str | None = None,
) -> dict[str, Any]:
    label = run_label or method
    out_dir = ensure_dir(EXP_DIR / f"{dataset}_seed{seed}_{label}_alpha{str(alpha).replace('.', 'p')}")
    log_path = out_dir / "logs" / "run.log"
    append_log(log_path, f"START dataset={dataset} seed={seed} method={method} run_label={label} alpha={alpha}")

    model, fit_meta = fit_or_load_model(dataset, seed, dataset_bundle.X_train, dataset_bundle.y_train)
    memberships = compute_memberships(model, dataset, seed, dataset_bundle.X_cal, dataset_bundle.X_test)
    cal_scores_all = model.score_matrix(dataset_bundle.X_cal)
    test_scores_all = model.score_matrix(dataset_bundle.X_test)
    cal_true_scores = cal_scores_all[np.arange(len(dataset_bundle.y_cal)), dataset_bundle.y_cal]
    start = time.time()
    summary: dict[str, Any] = {}

    if method == "split_cp":
        pred = predict_sets_global(test_scores_all, cal_true_scores, alpha)
    elif method == "class_conditional_cp":
        pred = predict_sets_class_conditional(test_scores_all, cal_true_scores, dataset_bundle.y_cal, alpha)
    elif method == "chip_rlcp":
        cal_mem, test_mem, summary = select_chip_memberships(memberships, variant=chip_variant)
        pred = rlcp_predict_sets(
            test_scores_all,
            cal_true_scores,
            cal_mem,
            test_mem,
            alpha,
            fallback_lambda=fallback_lambda,
            seed=seed,
        )
    elif method == "chip_uniform_overlap":
        cal_mem, test_mem, summary = select_chip_memberships(memberships, variant="full")
        pred = hard_overlap_rlcp_predict_sets(
            test_scores_all,
            cal_true_scores,
            cal_mem,
            test_mem,
            alpha,
            fallback_lambda=fallback_lambda,
            seed=seed,
        )
    elif method == "gmm_rlcp":
        target_k = min(12, max(4, memberships["coarse_cal"].shape[1] + memberships["fine_cal"].shape[1]))
        cal_mem = gmm_memberships(dataset_bundle.X_train, dataset_bundle.X_cal, target_k, seed)
        test_mem = gmm_memberships(dataset_bundle.X_train, dataset_bundle.X_test, target_k, seed)
        keep = cal_mem.sum(axis=0) >= 20.0
        pred = rlcp_predict_sets(
            test_scores_all,
            cal_true_scores,
            cal_mem[:, keep],
            test_mem[:, keep],
            alpha,
            fallback_lambda=fallback_lambda,
            seed=seed + 17,
        )
        summary = {
            "num_groups": int(keep.sum()),
            "mean_active_groups": float((cal_mem[:, keep] > 0.01).sum(axis=1).mean()),
            "median_active_groups": float(np.median((cal_mem[:, keep] > 0.01).sum(axis=1))),
        }
    elif method == "knn_rlcp":
        k = 100 if dataset == "synthetic" else 150 if dataset == "anuran" else 80
        cal_mem, _ = knn_memberships(dataset_bundle.X_cal, dataset_bundle.X_cal, k)
        test_mem, bandwidth = knn_memberships(dataset_bundle.X_cal, dataset_bundle.X_test, k)
        pred = rlcp_predict_sets(
            test_scores_all,
            cal_true_scores,
            cal_mem,
            test_mem,
            alpha,
            fallback_lambda=fallback_lambda,
            seed=seed + 29,
        )
        summary = {"bandwidth_median": float(np.median(bandwidth))}
    elif method == "batch_mcp":
        pred, summary = batch_multivalid_predict_sets(
            cal_true_scores,
            test_scores_all,
            dataset_bundle.candidate_group_cal,
            dataset_bundle.candidate_group_test,
            alpha,
        )
    elif method == "oracle_rlcp":
        pred = rlcp_predict_sets(
            test_scores_all,
            cal_true_scores,
            dataset_bundle.oracle_membership_cal,
            dataset_bundle.oracle_membership_test,
            alpha,
            fallback_lambda=fallback_lambda,
            seed=seed + 43,
        )
        summary = {"num_groups": int(dataset_bundle.oracle_membership_cal.shape[1])}
    else:
        raise ValueError(f"Unknown method {method}")

    calibration_time_sec = time.time() - start
    set_sizes = pred.sum(axis=1)
    covered = pred[np.arange(len(dataset_bundle.y_test)), dataset_bundle.y_test]
    target_coverage = 1.0 - alpha
    test_time_ms_per_example = calibration_time_sec * 1000.0 / max(1, len(dataset_bundle.y_test))
    group_metrics = evaluate_groups(covered.astype(float), dataset_bundle.eval_meta_test, target_coverage)
    result = {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "run_label": label,
        "alpha": alpha,
        "marginal_coverage": float(covered.mean()),
        "mean_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "singleton_rate": float((set_sizes == 1).mean()),
        "fit_time_sec": float(fit_meta["fit_time_sec"]),
        "calibration_time_sec": float(calibration_time_sec),
        "test_time_ms_per_example": float(test_time_ms_per_example),
        "total_runtime_sec": float(fit_meta["fit_time_sec"] + calibration_time_sec),
        "peak_rss_gb": float(max(fit_meta["peak_rss_gb"], _peak_rss_gb())),
        **group_metrics,
        "summary": summary,
    }
    per_example = pd.DataFrame(
        {
            "covered": covered.astype(int),
            "set_size": set_sizes.astype(int),
            "true_label": [dataset_bundle.label_names[i] for i in dataset_bundle.y_test],
            "pred_set_size": set_sizes.astype(int),
            "group_memberships": dataset_bundle.group_memberships_test,
            "runtime_ms": np.full(len(set_sizes), test_time_ms_per_example),
        }
    )
    per_example.to_csv(out_dir / "per_example.csv", index=False)
    write_json(out_dir / "results.json", result)
    write_json(
        out_dir / "config.json",
        {
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "run_label": label,
            "alpha": alpha,
            "chip_variant": chip_variant,
            "fallback_lambda": fallback_lambda,
        },
    )
    write_json(out_dir / "summary.json", summary)
    append_log(
        log_path,
        "END "
        f"marginal_coverage={result['marginal_coverage']:.4f} "
        f"worst_external_group_coverage={result['worst_external_group_coverage']:.4f} "
        f"mean_set_size={result['mean_set_size']:.4f} "
        f"total_runtime_sec={result['total_runtime_sec']:.4f}",
    )
    return result

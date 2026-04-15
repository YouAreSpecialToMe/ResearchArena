from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    pr, rc, _ = precision_recall_curve(y_true, y_prob)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "pr_auc_trapz": float(auc(rc, pr)),
    }


def summarize_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = metric_dicts[0].keys()
    out = {}
    for key in keys:
        vals = np.array([m[key] for m in metric_dicts], dtype=float)
        out[key] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return out


def bootstrap_metric_diff(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_col: str,
    metric: str = "macro_f1",
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
    n_boot: int = 1000,
    seed: int = 13,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        sample = df.iloc[idx]
        m_a = compute_metrics(sample[label_col].to_numpy(), sample[col_a].to_numpy(), threshold_a)[metric]
        m_b = compute_metrics(sample[label_col].to_numpy(), sample[col_b].to_numpy(), threshold_b)[metric]
        diffs.append(m_a - m_b)
    arr = np.array(diffs)
    return {
        "mean_diff": float(arr.mean()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def bootstrap_metric_ci(
    df: pd.DataFrame,
    prob_col: str,
    label_col: str,
    metric: str = "auroc",
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 13,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        sample = df.iloc[idx]
        vals.append(compute_metrics(sample[label_col].to_numpy(), sample[prob_col].to_numpy(), threshold)[metric])
    arr = np.array(vals)
    return {
        "mean": float(np.nanmean(arr)),
        "ci_low": float(np.nanquantile(arr, 0.025)),
        "ci_high": float(np.nanquantile(arr, 0.975)),
    }


def response_level_metrics(
    df: pd.DataFrame,
    response_id_col: str,
    label_col: str,
    prob_col: str,
    threshold: float,
) -> dict[str, float]:
    grouped = df.groupby(response_id_col).agg(
        y_true=(label_col, "max"),
        y_prob=(prob_col, "max"),
    )
    y_true = grouped["y_true"].to_numpy(dtype=int)
    y_prob = grouped["y_prob"].to_numpy(dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "response_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "response_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "response_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def calibration_points(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> dict[str, list[float]]:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="uniform")
    return {
        "mean_predicted_value": [float(x) for x in mean_pred],
        "fraction_of_positives": [float(x) for x in frac_pos],
    }

from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro"))


def balanced_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_class = []
    for cls in np.unique(y_true):
        mask = y_true == cls
        per_class.append(float((y_pred[mask] == y_true[mask]).mean()))
    return float(np.mean(per_class))


def per_group_accuracy(y_true, y_pred, groups):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        mask = groups == g
        out[str(int(g))] = float((y_pred[mask] == y_true[mask]).mean())
    return out


def worst_group_accuracy(y_true, y_pred, groups):
    per_group = per_group_accuracy(y_true, y_pred, groups)
    return float(min(per_group.values()))


def metrics_for_dataset(dataset_name, y_true, y_pred, groups=None):
    payload = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }
    if dataset_name == "waterbirds":
        payload["worst_group_accuracy"] = worst_group_accuracy(y_true, y_pred, groups)
        payload["per_group_accuracy"] = per_group_accuracy(y_true, y_pred, groups)
    else:
        payload["balanced_accuracy"] = balanced_accuracy(y_true, y_pred)
    return payload


def bootstrap_metric_ci(predictions_a, predictions_b, metric_fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(predictions_a["y_true"])
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = metric_fn(
            predictions_a["y_true"][idx],
            predictions_a["y_pred"][idx],
            predictions_a.get("group", None)[idx] if predictions_a.get("group", None) is not None else None,
        )
        b = metric_fn(
            predictions_b["y_true"][idx],
            predictions_b["y_pred"][idx],
            predictions_b.get("group", None)[idx] if predictions_b.get("group", None) is not None else None,
        )
        diffs.append(a - b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"low": float(lo), "high": float(hi)}


def summarize_seed_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                grouped[key].append(float(value))
    summary = {}
    for key, values in grouped.items():
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary

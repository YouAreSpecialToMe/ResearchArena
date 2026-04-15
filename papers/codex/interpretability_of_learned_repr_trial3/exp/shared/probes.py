from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


@dataclass
class ProbeResult:
    factor: str
    kind: str
    validation_metric: float
    test_metric: float
    chance: float
    admissible: bool
    extra: dict


def _subsample(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    choice = rng.choice(len(x), size=max_samples, replace=False)
    return x[choice], y[choice]


def fit_probe(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, factor_name: str, num_classes: int) -> ProbeResult:
    train_x, train_y = _subsample(train_x, train_y, 100000, seed=0)
    val_x, val_y = _subsample(val_x, val_y, 20000, seed=1)
    unique = np.unique(train_y)
    if len(unique) <= 10 and np.array_equal(unique, unique.astype(int)):
        best_model = None
        best_val = -1.0
        for c in [0.1, 1.0, 10.0]:
            model = LogisticRegression(C=c, max_iter=300, n_jobs=4, solver="lbfgs")
            model.fit(train_x, train_y)
            score = accuracy_score(val_y, model.predict(val_x))
            if score > best_val:
                best_val = score
                best_model = model
        test_score = accuracy_score(test_y, best_model.predict(test_x))
        chance = 1.0 / num_classes
        admissible = test_score >= max(chance + 0.20, 0.35)
        return ProbeResult(factor_name, "classification", float(best_val), float(test_score), float(chance), admissible, {})
    best_model = None
    best_val = -1e9
    for alpha in [0.1, 1.0, 10.0]:
        model = Ridge(alpha=alpha)
        model.fit(train_x, train_y)
        score = r2_score(val_y, model.predict(val_x))
        if score > best_val:
            best_val = score
            best_model = model
    test_pred = best_model.predict(test_x)
    test_score = r2_score(test_y, test_pred)
    rmse = mean_squared_error(test_y, test_pred) ** 0.5
    norm = float(train_y.max() - train_y.min() + 1e-8)
    admissible = test_score >= 0.30
    return ProbeResult(factor_name, "regression", float(best_val), float(test_score), 0.0, admissible, {"nrmse": float(rmse / norm)})

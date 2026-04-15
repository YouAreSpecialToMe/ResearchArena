from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def _standard_quantile(scores: np.ndarray, alpha: float) -> float:
    level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / max(len(scores), 1))
    return float(np.quantile(scores, level))


def predict_sets_global(score_test: np.ndarray, cal_true_scores: np.ndarray, alpha: float) -> np.ndarray:
    q = _standard_quantile(cal_true_scores, alpha)
    return score_test <= q


def predict_sets_class_conditional(
    score_test: np.ndarray,
    cal_true_scores: np.ndarray,
    y_cal_idx: np.ndarray,
    alpha: float,
) -> np.ndarray:
    n_test, n_classes = score_test.shape
    pred = np.zeros((n_test, n_classes), dtype=bool)
    global_q = _standard_quantile(cal_true_scores, alpha)
    for cls in range(n_classes):
        cls_scores = cal_true_scores[y_cal_idx == cls]
        q = global_q if len(cls_scores) < 40 else _standard_quantile(cls_scores, alpha)
        pred[:, cls] = score_test[:, cls] <= q
    return pred


def overlap_weights(
    cal_membership: np.ndarray,
    test_membership: np.ndarray,
    fallback_lambda: float = 0.1,
) -> tuple[np.ndarray, float]:
    overlap = np.minimum(cal_membership, test_membership[None, :]).sum(axis=1)
    self_overlap = float(np.minimum(test_membership, test_membership).sum())
    total_overlap = float(overlap.sum() + self_overlap)
    n_cal = cal_membership.shape[0]
    if total_overlap <= 0:
        return np.full(n_cal, 1.0 / (n_cal + 1)), 1.0 / (n_cal + 1)
    base = fallback_lambda / (n_cal + 1)
    cal_weights = base + (1.0 - fallback_lambda) * overlap / total_overlap
    test_weight = base + (1.0 - fallback_lambda) * self_overlap / total_overlap
    norm = cal_weights.sum() + test_weight
    cal_weights /= norm
    test_weight /= norm
    return cal_weights, float(test_weight)


def _weighted_p_value(
    cal_scores: np.ndarray,
    cal_weights: np.ndarray,
    test_score: float,
    test_weight: float,
    rng: np.random.Generator,
) -> float:
    greater = cal_weights[cal_scores > test_score].sum()
    equal = cal_weights[np.isclose(cal_scores, test_score, atol=1e-12, rtol=0.0)].sum() + test_weight
    return float(greater + rng.uniform() * equal)


def rlcp_predict_sets(
    score_test: np.ndarray,
    cal_true_scores: np.ndarray,
    cal_membership: np.ndarray,
    test_membership: np.ndarray,
    alpha: float,
    fallback_lambda: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    n_test, n_classes = score_test.shape
    out = np.zeros((n_test, n_classes), dtype=bool)
    for i in range(n_test):
        cal_weights, test_weight = overlap_weights(cal_membership, test_membership[i], fallback_lambda=fallback_lambda)
        for cls in range(n_classes):
            rng = np.random.default_rng(seed + 104729 * i + 13007 * cls)
            pval = _weighted_p_value(cal_true_scores, cal_weights, float(score_test[i, cls]), test_weight, rng)
            out[i, cls] = pval > alpha
    return out


def hard_overlap_rlcp_predict_sets(
    score_test: np.ndarray,
    cal_true_scores: np.ndarray,
    cal_membership: np.ndarray,
    test_membership: np.ndarray,
    alpha: float,
    threshold: float = 0.05,
    fallback_lambda: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    cal_hard = (cal_membership > threshold).astype(float)
    test_hard = (test_membership > threshold).astype(float)
    return rlcp_predict_sets(
        score_test,
        cal_true_scores,
        cal_hard,
        test_hard,
        alpha,
        fallback_lambda=fallback_lambda,
        seed=seed,
    )


def knn_memberships(
    X_ref: np.ndarray,
    X_query: np.ndarray,
    k: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric)
    nbrs.fit(X_ref)
    d_ref, _ = nbrs.kneighbors(X_ref)
    bandwidth = float(np.median(d_ref[:, -1]) + 1e-8)
    dists, idx = nbrs.kneighbors(X_query)
    sim = np.exp(-(dists**2) / (bandwidth**2))
    memberships = np.zeros((X_query.shape[0], X_ref.shape[0]), dtype=float)
    rows = np.repeat(np.arange(X_query.shape[0]), k)
    memberships[rows, idx.reshape(-1)] = sim.reshape(-1)
    memberships /= np.maximum(memberships.sum(axis=1, keepdims=True), 1e-12)
    return memberships, np.full(X_query.shape[0], bandwidth)


def gmm_memberships(X_train: np.ndarray, X_query: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        n_init=3,
        max_iter=200,
        random_state=seed,
        reg_covar=1e-4,
    )
    gmm.fit(X_train)
    return gmm.predict_proba(X_query)


def batch_multivalid_predict_sets(
    score_cal: np.ndarray,
    score_test: np.ndarray,
    candidate_group_matrix_cal: np.ndarray,
    candidate_group_matrix_test: np.ndarray,
    alpha: float,
    max_rounds: int = 20,
    tolerance: float = 0.01,
    min_group_size: int = 20,
) -> tuple[np.ndarray, dict[str, float]]:
    global_q = _standard_quantile(score_cal, alpha)
    n_groups = candidate_group_matrix_cal.shape[1]
    thresholds = np.full(n_groups, global_q, dtype=float)
    update_count = 0

    def _point_thresholds(group_matrix: np.ndarray) -> np.ndarray:
        local = group_matrix * thresholds[None, :]
        return np.maximum(global_q, local.max(axis=1, initial=0.0))

    for _ in range(max_rounds):
        current = _point_thresholds(candidate_group_matrix_cal)
        covered = score_cal <= current
        deficits = []
        for g in range(n_groups):
            mask = candidate_group_matrix_cal[:, g].astype(bool)
            n = int(mask.sum())
            if n < min_group_size:
                continue
            coverage = float(covered[mask].mean())
            deficit = (1.0 - alpha) - coverage
            if deficit > tolerance:
                target_q = _standard_quantile(score_cal[mask], max(alpha - tolerance, 1e-6))
                deficits.append((deficit, g, target_q))
        if not deficits:
            break
        deficits.sort(reverse=True)
        _, group_idx, target_q = deficits[0]
        if target_q > thresholds[group_idx] + 1e-12:
            thresholds[group_idx] = target_q
            update_count += 1
        else:
            break

    test_thresholds = _point_thresholds(candidate_group_matrix_test)
    pred = score_test <= test_thresholds[:, None]
    summary = {
        "num_candidate_groups": int(n_groups),
        "num_updates": int(update_count),
        "max_group_threshold": float(thresholds.max(initial=global_q)),
        "global_threshold": float(global_q),
    }
    return pred, summary

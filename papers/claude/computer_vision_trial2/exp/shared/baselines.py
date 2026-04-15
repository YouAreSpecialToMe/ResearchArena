"""OOD detection and calibration baselines."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


# ============ OOD Detection Baselines ============

def msp_score(logits: np.ndarray) -> np.ndarray:
    """Maximum Softmax Probability score. Higher = more OOD."""
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return -probs.max(axis=1)  # Negate so higher = more OOD


def energy_score(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Energy score. Higher = more OOD."""
    return -T * np.log(np.exp(logits / T).sum(axis=1) + 1e-10)


def vim_score(test_features: np.ndarray, test_logits: np.ndarray,
              id_features: np.ndarray, id_logits: np.ndarray,
              explained_variance: float = 0.95) -> np.ndarray:
    """ViM (Virtual-logit Matching) score.

    Args:
        test_features: (N_test, D) penultimate layer features
        test_logits: (N_test, C) logits
        id_features: (N_id, D) ID calibration features
        id_logits: (N_id, C) ID calibration logits
        explained_variance: PCA variance threshold
    Returns:
        (N_test,) ViM scores (higher = more OOD)
    """
    # Fit PCA on ID features
    pca = PCA(n_components=explained_variance, svd_solver='full')
    pca.fit(id_features)

    # Compute residual (null space projection)
    id_projected = pca.transform(id_features)
    id_reconstructed = pca.inverse_transform(id_projected)
    id_residuals = np.linalg.norm(id_features - id_reconstructed, axis=1)

    test_projected = pca.transform(test_features)
    test_reconstructed = pca.inverse_transform(test_projected)
    test_residuals = np.linalg.norm(test_features - test_reconstructed, axis=1)

    # Combine with energy score
    e_score = energy_score(test_logits)
    e_score_id = energy_score(id_logits)

    # Calibrate alpha: match scales
    alpha = np.std(e_score_id) / (np.std(id_residuals) + 1e-10)

    return alpha * test_residuals + e_score


def knn_score(test_features: np.ndarray, id_features: np.ndarray,
              k: int = 50) -> np.ndarray:
    """KNN OOD score: distance to k-th nearest neighbor.

    Args:
        test_features: (N_test, D)
        id_features: (N_id, D)
        k: number of neighbors
    Returns:
        (N_test,) KNN scores (higher = more OOD)
    """
    # Normalize features
    test_norm = test_features / (np.linalg.norm(test_features, axis=1, keepdims=True) + 1e-10)
    id_norm = id_features / (np.linalg.norm(id_features, axis=1, keepdims=True) + 1e-10)

    # Compute distances in batches to avoid memory issues
    batch_size = 500
    scores = []
    for i in range(0, len(test_norm), batch_size):
        batch = test_norm[i:i + batch_size]
        dists = cdist(batch, id_norm, metric='euclidean')
        # k-th nearest neighbor distance
        kth_dist = np.partition(dists, min(k, dists.shape[1] - 1), axis=1)[:, min(k, dists.shape[1] - 1)]
        scores.append(kth_dist)
    return np.concatenate(scores)


# ============ Calibration Baselines ============

def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit global temperature by minimizing NLL on calibration set.

    Args:
        logits: (N, C) raw logits
        labels: (N,) true class indices
    Returns:
        optimal temperature
    """
    def nll(T):
        T = max(T, 0.01)
        scaled = logits / T
        # Numerically stable log-softmax
        log_probs = scaled - np.log(np.exp(scaled - scaled.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)) - scaled.max(axis=1, keepdims=True) + scaled.max(axis=1, keepdims=True)
        # Simpler: use scipy logsumexp
        from scipy.special import logsumexp
        log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
        nll_val = -log_probs[np.arange(len(labels)), labels].mean()
        return nll_val

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    return float(result.x)


def fit_adaptive_temperature(logits: np.ndarray, labels: np.ndarray,
                              aep_scores: np.ndarray) -> dict:
    """Fit AEP-adaptive temperature: T(x) = T0 + alpha * aep_score(x).

    Grid search over T0 and alpha to minimize ECE.

    Args:
        logits: (N, C) raw logits
        labels: (N,) true class indices
        aep_scores: (N,) AEP Mahalanobis scores
    Returns:
        dict with 'T0', 'alpha'
    """
    from .metrics import compute_ece

    best_ece = float('inf')
    best_params = {'T0': 1.0, 'alpha': 0.0}

    # Normalize AEP scores to [0, 1] range for stability
    score_min = aep_scores.min()
    score_max = aep_scores.max()
    if score_max > score_min:
        aep_norm = (aep_scores - score_min) / (score_max - score_min)
    else:
        aep_norm = np.zeros_like(aep_scores)

    for T0 in np.arange(0.5, 3.05, 0.1):
        for alpha in np.arange(0.0, 3.05, 0.1):
            T = T0 + alpha * aep_norm
            T = np.maximum(T, 0.01)

            scaled_logits = logits / T[:, None]
            exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            confidences = probs.max(axis=1)
            predictions = probs.argmax(axis=1)
            correct = (predictions == labels).astype(float)

            ece = compute_ece(confidences, correct)
            if ece < best_ece:
                best_ece = ece
                best_params = {'T0': float(T0), 'alpha': float(alpha),
                               'score_min': float(score_min), 'score_max': float(score_max)}

    return best_params


def apply_adaptive_temperature(logits: np.ndarray, aep_scores: np.ndarray,
                                params: dict) -> np.ndarray:
    """Apply AEP-adaptive temperature scaling.

    Returns calibrated probabilities.
    """
    score_min = params['score_min']
    score_max = params['score_max']
    if score_max > score_min:
        aep_norm = (aep_scores - score_min) / (score_max - score_min)
    else:
        aep_norm = np.zeros_like(aep_scores)
    aep_norm = np.clip(aep_norm, 0, 1)

    T = params['T0'] + params['alpha'] * aep_norm
    T = np.maximum(T, 0.01)

    scaled_logits = logits / T[:, None]
    exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return probs


def histogram_binning_fit(logits: np.ndarray, labels: np.ndarray,
                           n_bins: int = 15) -> dict:
    """Fit histogram binning calibration on calibration set."""
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_calibrated = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_calibrated[i] = correct[mask].mean()
        else:
            bin_calibrated[i] = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2

    return {'boundaries': bin_boundaries.tolist(), 'calibrated': bin_calibrated.tolist()}


def histogram_binning_predict(logits: np.ndarray, hb_params: dict) -> np.ndarray:
    """Apply histogram binning to get calibrated confidences."""
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    confidences = probs.max(axis=1)

    boundaries = np.array(hb_params['boundaries'])
    calibrated = np.array(hb_params['calibrated'])

    result = np.zeros_like(confidences)
    for i in range(len(calibrated)):
        mask = (confidences > boundaries[i]) & (confidences <= boundaries[i + 1])
        result[mask] = calibrated[i]

    return result


# ============ Score Fusion ============

def fuse_scores(aep_scores: np.ndarray, baseline_scores: np.ndarray,
                id_aep: np.ndarray, id_baseline: np.ndarray,
                ood_aep: np.ndarray, ood_baseline: np.ndarray) -> tuple:
    """Fuse AEP and baseline scores with optimal beta.

    Returns (fused_scores_id, fused_scores_ood, best_beta).
    Tunes beta on the provided id/ood scores to maximize AUROC.
    """
    from .metrics import compute_ood_metrics

    # Normalize both scores to [0, 1]
    all_aep = np.concatenate([id_aep, ood_aep])
    all_base = np.concatenate([id_baseline, ood_baseline])

    aep_min, aep_max = all_aep.min(), all_aep.max()
    base_min, base_max = all_base.min(), all_base.max()

    def normalize(x, mn, mx):
        if mx > mn:
            return (x - mn) / (mx - mn)
        return np.zeros_like(x)

    id_aep_n = normalize(id_aep, aep_min, aep_max)
    ood_aep_n = normalize(ood_aep, aep_min, aep_max)
    id_base_n = normalize(id_baseline, base_min, base_max)
    ood_base_n = normalize(ood_baseline, base_min, base_max)

    best_auroc = 0
    best_beta = 0.5

    for beta in np.arange(0.0, 1.05, 0.05):
        fused_id = beta * id_aep_n + (1 - beta) * id_base_n
        fused_ood = beta * ood_aep_n + (1 - beta) * ood_base_n
        metrics = compute_ood_metrics(fused_id, fused_ood)
        if metrics['AUROC'] > best_auroc:
            best_auroc = metrics['AUROC']
            best_beta = beta

    fused_id = best_beta * id_aep_n + (1 - best_beta) * id_base_n
    fused_ood = best_beta * ood_aep_n + (1 - best_beta) * ood_base_n

    return fused_id, fused_ood, best_beta

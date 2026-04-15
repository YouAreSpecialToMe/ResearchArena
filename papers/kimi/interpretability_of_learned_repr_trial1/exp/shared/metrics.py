"""Metrics for CAGER experiments, including C-GAS computation."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from typing import Tuple, Optional


def compute_cgas(
    causal_subspaces: np.ndarray,
    explanation_features: np.ndarray,
    full_activations: np.ndarray,
    distance_metric: str = 'cosine',
    top_k: int = 10
) -> Tuple[float, float, float]:
    """Compute Causal Geometric Alignment Score (C-GAS).
    
    C-GAS = ρ(D_causal, D_exp) / ρ(D_causal, D_full)
    
    where ρ is Spearman rank correlation and D_* are pairwise distance matrices.
    
    Args:
        causal_subspaces: (n_samples, n_causal_dims) array of validated causal subspace activations
        explanation_features: (n_samples, n_exp_features) array of explanation features (e.g., SAE latents)
        full_activations: (n_samples, n_full_dims) array of full model activations
        distance_metric: Distance metric to use ('cosine', 'euclidean', 'correlation')
        top_k: Number of top correlated explanation features to use per causal subspace
    
    Returns:
        (C-GAS score, rho_causal_exp, rho_causal_full)
    """
    n_samples = causal_subspaces.shape[0]
    
    # Select top-k explanation features correlated with causal subspaces
    # For simplicity, we compute correlation between each causal dim and each explanation feature
    # then take the union of top-k features for all causal dims
    if explanation_features.shape[1] > top_k * causal_subspaces.shape[1]:
        selected_features = select_top_k_features(
            causal_subspaces, explanation_features, top_k
        )
    else:
        selected_features = explanation_features
    
    # Compute pairwise distance matrices
    D_causal = pairwise_distances(causal_subspaces, metric=distance_metric)
    D_exp = pairwise_distances(selected_features, metric=distance_metric)
    D_full = pairwise_distances(full_activations, metric=distance_metric)
    
    # Get upper triangular indices (excluding diagonal)
    triu_indices = np.triu_indices(n_samples, k=1)
    
    # Flatten distance matrices to vectors
    d_causal_vec = D_causal[triu_indices]
    d_exp_vec = D_exp[triu_indices]
    d_full_vec = D_full[triu_indices]
    
    # Compute Spearman correlations
    rho_causal_exp, _ = spearmanr(d_causal_vec, d_exp_vec)
    rho_causal_full, _ = spearmanr(d_causal_vec, d_full_vec)
    
    # Handle edge cases
    if np.isnan(rho_causal_exp) or np.isnan(rho_causal_full):
        return 0.0, 0.0, 0.0
    
    if abs(rho_causal_full) < 1e-10:
        # If denominator is near zero, return 0 (no causal structure in full space)
        return 0.0, rho_causal_exp, rho_causal_full
    
    # Compute C-GAS
    cgas = rho_causal_exp / rho_causal_full
    
    return cgas, rho_causal_exp, rho_causal_full


def select_top_k_features(
    causal_subspaces: np.ndarray,
    explanation_features: np.ndarray,
    top_k: int
) -> np.ndarray:
    """Select top-k explanation features most correlated with causal subspaces.
    
    For each causal dimension, find the top-k explanation features with highest
    absolute correlation, then take the union of all selected features.
    
    Args:
        causal_subspaces: (n_samples, n_causal_dims)
        explanation_features: (n_samples, n_exp_features)
        top_k: Number of features to select per causal dimension
    
    Returns:
        Selected explanation features (n_samples, <= n_causal_dims * top_k)
    """
    n_causal_dims = causal_subspaces.shape[1]
    n_exp_features = explanation_features.shape[1]
    
    selected_indices = set()
    
    for i in range(n_causal_dims):
        causal_dim = causal_subspaces[:, i]
        correlations = []
        
        for j in range(n_exp_features):
            exp_dim = explanation_features[:, j]
            corr, _ = spearmanr(causal_dim, exp_dim)
            correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
        
        # Get top-k indices for this causal dimension
        top_indices = np.argsort(correlations)[-top_k:]
        selected_indices.update(top_indices.tolist())
    
    # Return selected features
    selected_indices = sorted(list(selected_indices))
    return explanation_features[:, selected_indices]


def compute_reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> float:
    """Compute MSE reconstruction error."""
    return np.mean((original - reconstructed) ** 2)


def compute_sparsity_l0(features: np.ndarray, threshold: float = 1e-6) -> float:
    """Compute L0 sparsity: proportion of near-zero activations.
    
    Args:
        features: (n_samples, n_features)
        threshold: Threshold below which values are considered zero
    
    Returns:
        Proportion of zero activations (0-1)
    """
    return np.mean(np.abs(features) < threshold)


def compute_sparsity_l1(features: np.ndarray) -> float:
    """Compute L1 sparsity: mean L1 norm per sample."""
    return np.mean(np.sum(np.abs(features), axis=1))


def compute_dead_neuron_percentage(features: np.ndarray, threshold: float = 1e-6) -> float:
    """Compute percentage of dead neurons (never activated).
    
    Args:
        features: (n_samples, n_features) activation history
        threshold: Threshold below which values are considered zero
    
    Returns:
        Percentage of dead neurons (0-100)
    """
    # A neuron is dead if it's never activated across all samples
    neuron_activated = np.any(np.abs(features) > threshold, axis=0)
    return 100 * np.mean(~neuron_activated)


def compute_explained_variance_pca(pca_model, n_components: Optional[int] = None) -> float:
    """Compute cumulative explained variance ratio for PCA."""
    if n_components is None:
        return np.sum(pca_model.explained_variance_ratio_)
    else:
        return np.sum(pca_model.explained_variance_ratio_[:n_components])

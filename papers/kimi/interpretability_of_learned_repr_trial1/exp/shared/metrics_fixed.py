"""Fixed metrics for CAGER experiments, including improved C-GAS computation."""
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import pairwise_distances
from sklearn.cross_decomposition import CCA
from typing import Tuple, Optional, Dict, List
import warnings


def compute_cgas_fixed(
    causal_subspaces: np.ndarray,
    explanation_features: np.ndarray,
    full_activations: np.ndarray,
    distance_metric: str = 'cosine',
    top_k: int = 10,
    dictionary_size: Optional[int] = None,
    input_dim: Optional[int] = None
) -> Dict:
    """Compute improved Causal Geometric Alignment Score (C-GAS).
    
    The improved C-GAS addresses the key issue from self-review: 
    it penalizes high-dimensional random projections that happen 
    to correlate well by chance.
    
    New formulation:
    C-GAS = (ρ(D_causal, D_exp) / ρ(D_causal, D_full)) * penalty(dict_size)
    
    where penalty accounts for the increased chance of spurious 
    correlations at higher dimensions (based on JL lemma intuition).
    
    Args:
        causal_subspaces: (n_samples, n_causal_dims) validated causal subspace activations
        explanation_features: (n_samples, n_exp_features) explanation features
        full_activations: (n_samples, n_full_dims) full model activations
        distance_metric: Distance metric ('cosine', 'euclidean', 'correlation')
        top_k: Number of top correlated explanation features per causal dimension
        dictionary_size: Size of explanation dictionary (for penalty calculation)
        input_dim: Input dimensionality (for penalty calculation)
    
    Returns:
        Dictionary with C-GAS components and metadata
    """
    n_samples = causal_subspaces.shape[0]
    
    # Select top-k explanation features correlated with causal subspaces
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
    
    # Flatten distance matrices
    d_causal_vec = D_causal[triu_indices]
    d_exp_vec = D_exp[triu_indices]
    d_full_vec = D_full[triu_indices]
    
    # Compute correlations
    rho_causal_exp, p_causal_exp = spearmanr(d_causal_vec, d_exp_vec)
    rho_causal_full, p_causal_full = spearmanr(d_causal_vec, d_full_vec)
    
    # Handle edge cases
    if np.isnan(rho_causal_exp) or np.isnan(rho_causal_full):
        return {
            'cgas': 0.0,
            'cgas_unpenalized': 0.0,
            'rho_causal_exp': 0.0,
            'rho_causal_full': 0.0,
            'dimension_penalty': 1.0,
            'p_value': 1.0
        }
    
    if abs(rho_causal_full) < 1e-10:
        return {
            'cgas': 0.0,
            'cgas_unpenalized': 0.0,
            'rho_causal_exp': float(rho_causal_exp),
            'rho_causal_full': float(rho_causal_full),
            'dimension_penalty': 1.0,
            'p_value': float(p_causal_exp)
        }
    
    # Compute unpenalized C-GAS
    cgas_unpenalized = rho_causal_exp / rho_causal_full
    
    # Compute dimensionality penalty
    # Based on intuition that random projections in high dimensions
    # can spuriously correlate: penalty = sqrt(input_dim / dict_size)
    # This gives 1.0 when dict_size = input_dim (1x),
    # 0.5 when dict_size = 4x input_dim,
    # 0.25 when dict_size = 16x input_dim
    if dictionary_size is not None and input_dim is not None:
        overcomplete_ratio = dictionary_size / input_dim
        # Penalty decreases as overcompleteness increases
        # Using square root to make penalty less aggressive
        dimension_penalty = 1.0 / np.sqrt(overcomplete_ratio)
        # Also incorporate effective dimensionality
        # If we're using many features (close to dictionary size), 
        # penalty should be stronger
        actual_features_used = selected_features.shape[1]
        feature_ratio = actual_features_used / dictionary_size
        # Adjust penalty: more penalty if we're using a large fraction of dict
        usage_penalty = 1.0 / (1.0 + feature_ratio * np.log(overcomplete_ratio + 1))
        dimension_penalty = np.sqrt(dimension_penalty * usage_penalty)
    else:
        dimension_penalty = 1.0
    
    # Final C-GAS with penalty
    cgas = cgas_unpenalized * dimension_penalty
    
    return {
        'cgas': float(cgas),
        'cgas_unpenalized': float(cgas_unpenalized),
        'rho_causal_exp': float(rho_causal_exp),
        'rho_causal_full': float(rho_causal_full),
        'dimension_penalty': float(dimension_penalty),
        'p_value': float(p_causal_exp),
        'n_features_selected': selected_features.shape[1]
    }


def compute_cgas_with_cca(
    causal_subspaces: np.ndarray,
    explanation_features: np.ndarray,
    full_activations: np.ndarray,
    n_components: int = 10,
    dictionary_size: Optional[int] = None,
    input_dim: Optional[int] = None
) -> Dict:
    """Compute C-GAS using Canonical Correlation Analysis for better alignment.
    
    This version uses CCA to find the best linear relationship between
    causal subspaces and explanation features, which is more robust than
    simple distance correlation.
    
    Args:
        causal_subspaces: (n_samples, n_causal_dims) validated causal subspace activations
        explanation_features: (n_samples, n_exp_features) explanation features
        full_activations: (n_samples, n_full_dims) full model activations
        n_components: Number of CCA components
        dictionary_size: Size of explanation dictionary
        input_dim: Input dimensionality
    
    Returns:
        Dictionary with C-GAS-CCA components
    """
    n_samples = causal_subspaces.shape[0]
    n_causal = causal_subspaces.shape[1]
    
    # Limit n_components to avoid overfitting
    n_components = min(n_components, n_causal, explanation_features.shape[1], n_samples // 2)
    
    try:
        # Fit CCA
        cca = CCA(n_components=n_components)
        cca.fit(explanation_features, causal_subspaces)
        
        # Transform both spaces
        X_c, Y_c = cca.transform(explanation_features, causal_subspaces)
        
        # Compute distances in canonical space
        D_exp = pairwise_distances(X_c, metric='cosine')
        D_causal = pairwise_distances(Y_c, metric='cosine')
        D_full = pairwise_distances(full_activations, metric='cosine')
        
        # Get upper triangular
        triu_indices = np.triu_indices(n_samples, k=1)
        d_exp_vec = D_exp[triu_indices]
        d_causal_vec = D_causal[triu_indices]
        d_full_vec = D_full[triu_indices]
        
        # Compute correlations
        rho_causal_exp, _ = spearmanr(d_causal_vec, d_exp_vec)
        rho_causal_full, _ = spearmanr(d_causal_vec, d_full_vec)
        
        # Compute mean canonical correlation as quality measure
        canonical_corrs = np.corrcoef(X_c.T, Y_c.T)[:n_components, n_components:]
        mean_canonical_corr = np.mean(np.diag(canonical_corrs))
        
        if abs(rho_causal_full) < 1e-10:
            cgas = 0.0
        else:
            cgas = rho_causal_exp / rho_causal_full
            # Weight by canonical correlation quality
            cgas *= max(0, mean_canonical_corr)
        
        # Apply dimensionality penalty
        if dictionary_size is not None and input_dim is not None:
            overcomplete_ratio = dictionary_size / input_dim
            dimension_penalty = 1.0 / np.sqrt(overcomplete_ratio)
        else:
            dimension_penalty = 1.0
        
        cgas *= dimension_penalty
        
        return {
            'cgas': float(cgas),
            'cgas_cca': float(cgas),
            'rho_causal_exp': float(rho_causal_exp),
            'rho_causal_full': float(rho_causal_full),
            'mean_canonical_corr': float(mean_canonical_corr),
            'dimension_penalty': float(dimension_penalty),
            'n_components': n_components
        }
    except Exception as e:
        warnings.warn(f"CCA failed: {e}, falling back to standard C-GAS")
        return compute_cgas_fixed(
            causal_subspaces, explanation_features, full_activations,
            dictionary_size=dictionary_size, input_dim=input_dim
        )


def compute_oracle_cgas(
    ground_truth_features: np.ndarray,
    causal_subspaces: np.ndarray,
    full_activations: np.ndarray,
    distance_metric: str = 'cosine'
) -> Dict:
    """Compute C-GAS for oracle (ground-truth) features.
    
    This establishes an upper bound on achievable C-GAS.
    
    Args:
        ground_truth_features: (n_samples, n_ground_truth) ground truth feature values
        causal_subspaces: (n_samples, n_causal_dims) validated causal subspace activations
        full_activations: (n_samples, n_full_dims) full model activations
        distance_metric: Distance metric to use
    
    Returns:
        Dictionary with oracle C-GAS
    """
    n_samples = ground_truth_features.shape[0]
    
    # Compute distance matrices
    D_gt = pairwise_distances(ground_truth_features, metric=distance_metric)
    D_causal = pairwise_distances(causal_subspaces, metric=distance_metric)
    D_full = pairwise_distances(full_activations, metric=distance_metric)
    
    # Get upper triangular
    triu_indices = np.triu_indices(n_samples, k=1)
    d_gt_vec = D_gt[triu_indices]
    d_causal_vec = D_causal[triu_indices]
    d_full_vec = D_full[triu_indices]
    
    # Compute correlations
    rho_gt_causal, _ = spearmanr(d_gt_vec, d_causal_vec)
    rho_gt_full, _ = spearmanr(d_gt_vec, d_full_vec)
    rho_causal_full, _ = spearmanr(d_causal_vec, d_full_vec)
    
    # Oracle C-GAS: how well do ground truth features align with causal subspaces?
    if abs(rho_causal_full) < 1e-10:
        oracle_cgas = 0.0
    else:
        oracle_cgas = rho_gt_causal / rho_causal_full
    
    return {
        'oracle_cgas': float(oracle_cgas),
        'rho_gt_causal': float(rho_gt_causal),
        'rho_gt_full': float(rho_gt_full),
        'rho_causal_full': float(rho_causal_full)
    }


def compute_feature_recovery_rate_improved(
    features: np.ndarray,
    ground_truth_features: Dict[str, np.ndarray],
    correlation_threshold: float = 0.5
) -> Dict:
    """Compute improved feature recovery rate with detailed statistics.
    
    Args:
        features: (n_samples, n_features) learned features
        ground_truth_features: Dict of ground truth feature name -> values
        correlation_threshold: Threshold for considering a feature "recovered"
    
    Returns:
        Dictionary with recovery statistics
    """
    recovery_stats = {}
    
    for feat_name, gt_values in ground_truth_features.items():
        best_corrs = []
        best_feature_indices = []
        
        # Check correlation with each learned feature
        for j in range(features.shape[1]):
            feat_j = features[:, j]
            
            # Pearson correlation
            try:
                corr, pval = pearsonr(feat_j, gt_values)
                corr = abs(corr) if not np.isnan(corr) else 0.0
            except:
                corr = 0.0
                pval = 1.0
            
            # Spearman correlation (rank-based, more robust)
            try:
                scorr, spval = spearmanr(feat_j, gt_values)
                scorr = abs(scorr) if not np.isnan(scorr) else 0.0
            except:
                scorr = 0.0
            
            # Use max of Pearson and Spearman
            best_corr = max(corr, scorr)
            best_corrs.append(best_corr)
            
            if best_corr > correlation_threshold:
                best_feature_indices.append({
                    'feature_idx': j,
                    'correlation': float(best_corr),
                    'pearson': float(corr),
                    'spearman': float(scorr)
                })
        
        # Recovery rate: proportion of features above threshold
        recovery_rate = np.mean(np.array(best_corrs) > correlation_threshold)
        
        # Best match
        best_match_idx = np.argmax(best_corrs)
        best_match_corr = best_corrs[best_match_idx]
        
        recovery_stats[feat_name] = {
            'recovery_rate': float(recovery_rate),
            'best_match_correlation': float(best_match_corr),
            'best_match_feature_idx': int(best_match_idx),
            'mean_correlation': float(np.mean(best_corrs)),
            'max_correlation': float(np.max(best_corrs)) if best_corrs else 0.0,
            'features_above_threshold': best_feature_indices[:5]  # Top 5 matches
        }
    
    # Overall stats
    all_recovery_rates = [s['recovery_rate'] for s in recovery_stats.values()]
    all_best_corrs = [s['best_match_correlation'] for s in recovery_stats.values()]
    
    recovery_stats['overall'] = {
        'mean_recovery_rate': float(np.mean(all_recovery_rates)),
        'std_recovery_rate': float(np.std(all_recovery_rates)),
        'mean_best_correlation': float(np.mean(all_best_corrs)),
        'std_best_correlation': float(np.std(all_best_corrs))
    }
    
    return recovery_stats


def select_top_k_features(
    causal_subspaces: np.ndarray,
    explanation_features: np.ndarray,
    top_k: int
) -> np.ndarray:
    """Select top-k explanation features most correlated with causal subspaces."""
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


def compute_sensitivity_analysis(
    causal_subspaces: np.ndarray,
    explanation_features: np.ndarray,
    full_activations: np.ndarray,
    dictionary_size: Optional[int] = None,
    input_dim: Optional[int] = None
) -> Dict:
    """Compute sensitivity of C-GAS to hyperparameter choices.
    
    Tests different distance metrics, top-k values, and sample sizes.
    
    Args:
        causal_subspaces: (n_samples, n_causal_dims) validated causal subspace activations
        explanation_features: (n_samples, n_exp_features) explanation features
        full_activations: (n_samples, n_full_dims) full model activations
        dictionary_size: Size of explanation dictionary
        input_dim: Input dimensionality
    
    Returns:
        Dictionary with sensitivity results
    """
    results = {
        'distance_metric': {},
        'top_k': {},
        'sample_size': {}
    }
    
    # Test different distance metrics
    for metric in ['cosine', 'euclidean', 'correlation']:
        try:
            cgas_result = compute_cgas_fixed(
                causal_subspaces, explanation_features, full_activations,
                distance_metric=metric, top_k=10,
                dictionary_size=dictionary_size, input_dim=input_dim
            )
            results['distance_metric'][metric] = cgas_result
        except Exception as e:
            results['distance_metric'][metric] = {'error': str(e)}
    
    # Test different top-k values
    for k in [5, 10, 20, 50]:
        try:
            cgas_result = compute_cgas_fixed(
                causal_subspaces, explanation_features, full_activations,
                distance_metric='cosine', top_k=k,
                dictionary_size=dictionary_size, input_dim=input_dim
            )
            results['top_k'][k] = cgas_result
        except Exception as e:
            results['top_k'][k] = {'error': str(e)}
    
    # Test different sample sizes
    n_samples = causal_subspaces.shape[0]
    for n in [50, 100, 200, min(500, n_samples)]:
        if n > n_samples:
            continue
        try:
            # Sample subset
            indices = np.random.choice(n_samples, n, replace=False)
            cgas_result = compute_cgas_fixed(
                causal_subspaces[indices], explanation_features[indices], 
                full_activations[indices],
                distance_metric='cosine', top_k=10,
                dictionary_size=dictionary_size, input_dim=input_dim
            )
            results['sample_size'][n] = cgas_result
        except Exception as e:
            results['sample_size'][n] = {'error': str(e)}
    
    # Compute coefficient of variation for each hyperparameter
    for param_name, param_results in results.items():
        cgas_values = [r['cgas'] for r in param_results.values() if 'cgas' in r]
        if cgas_values and len(cgas_values) > 1:
            mean_cgas = np.mean(cgas_values)
            std_cgas = np.std(cgas_values)
            cv = std_cgas / mean_cgas if mean_cgas != 0 else float('inf')
            results[f'{param_name}_stability'] = {
                'mean': float(mean_cgas),
                'std': float(std_cgas),
                'cv': float(cv)
            }
    
    return results


# Legacy functions for backward compatibility
def compute_cgas(
    causal_subspaces: np.ndarray,
    explanation_features: np.ndarray,
    full_activations: np.ndarray,
    distance_metric: str = 'cosine',
    top_k: int = 10
) -> Tuple[float, float, float]:
    """Legacy C-GAS computation (for backward compatibility)."""
    result = compute_cgas_fixed(
        causal_subspaces, explanation_features, full_activations,
        distance_metric=distance_metric, top_k=top_k
    )
    return result['cgas'], result['rho_causal_exp'], result['rho_causal_full']


def compute_reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute MSE reconstruction error."""
    return np.mean((original - reconstructed) ** 2)


def compute_sparsity_l0(features: np.ndarray, threshold: float = 1e-6) -> float:
    """Compute L0 sparsity: proportion of near-zero activations."""
    return np.mean(np.abs(features) < threshold)


def compute_sparsity_l1(features: np.ndarray) -> float:
    """Compute L1 sparsity: mean L1 norm per sample."""
    return np.mean(np.sum(np.abs(features), axis=1))


def compute_dead_neuron_percentage(features: np.ndarray, threshold: float = 1e-6) -> float:
    """Compute percentage of dead neurons (never activated)."""
    neuron_activated = np.any(np.abs(features) > threshold, axis=0)
    return 100 * np.mean(~neuron_activated)


def compute_explained_variance_pca(pca_model, n_components: Optional[int] = None) -> float:
    """Compute cumulative explained variance ratio for PCA."""
    if n_components is None:
        return np.sum(pca_model.explained_variance_ratio_)
    else:
        return np.sum(pca_model.explained_variance_ratio_[:n_components])

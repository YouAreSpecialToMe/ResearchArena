"""
Phase 1: Ablation comparing k-NN vs kernel-based MI estimation.
"""
import numpy as np
from scipy.special import digamma
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors


def knn_mutual_information(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    k-NN Mutual Information estimator (Kraskov-Stögbauer-Grassberger).
    """
    n = len(x)
    xy = np.column_stack([x, y])
    
    # Find k-nearest neighbors in joint space
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(xy)
    distances, _ = nbrs.kneighbors(xy)
    
    # Distance to k-th nearest neighbor in joint space
    epsilon = distances[:, k]
    epsilon = np.maximum(epsilon, 1e-10)
    
    # Count neighbors in marginal spaces
    nx = np.zeros(n)
    ny = np.zeros(n)
    
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    for i in range(n):
        nx[i] = np.sum(np.abs(x - x[i]) < epsilon[i]) - 1
        ny[i] = np.sum(np.abs(y - y[i]) < epsilon[i]) - 1
    
    nx = np.maximum(nx, 1e-10)
    ny = np.maximum(ny, 1e-10)
    
    # KSG MI estimator
    mi = digamma(k) - np.mean(digamma(nx) + digamma(ny)) + digamma(n)
    
    return max(0, mi)


def kernel_mutual_information(x: np.ndarray, y: np.ndarray, bandwidth=None) -> float:
    """
    Kernel-based Mutual Information estimator using Gaussian kernels.
    """
    n = len(x)
    
    # Standardize
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    
    # Use Silverman's rule for bandwidth if not specified
    if bandwidth is None:
        bandwidth = 1.06 * min(1.0, n**(-1/5))
    
    try:
        # Create KDE for joint and marginals
        xy = np.column_stack([x, y])
        
        kde_joint = gaussian_kde(xy.T, bw_method=bandwidth)
        kde_x = gaussian_kde(x, bw_method=bandwidth)
        kde_y = gaussian_kde(y, bw_method=bandwidth)
        
        # Compute MI using numerical integration
        # MI = E[log p(x,y) / (p(x)p(y))]
        mi = 0.0
        for i in range(n):
            p_xy = kde_joint(xy[i]) + 1e-10
            p_x = kde_x(x[i]) + 1e-10
            p_y = kde_y(y[i]) + 1e-10
            mi += np.log(p_xy / (p_x * p_y))
        
        mi = mi / n
        return max(0, mi)
    except Exception:
        # Fall back to correlation-based estimate
        return 0.5 * np.log(1 / (1 - np.corrcoef(x, y)[0, 1]**2 + 1e-10))


def compute_skeleton_it_method(data: np.ndarray, method: str = 'knn', **kwargs) -> np.ndarray:
    """
    Compute skeleton using specified MI estimation method.
    
    Args:
        data: Data matrix (n_samples, n_features)
        method: 'knn' or 'kernel'
        **kwargs: method-specific parameters
    """
    n_samples, n_nodes = data.shape
    
    # Standardize data
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    # Select MI estimation function
    if method == 'knn':
        k = kwargs.get('k', 5)
        mi_func = lambda x, y: knn_mutual_information(x, y, k=k)
    elif method == 'kernel':
        mi_func = kernel_mutual_information
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute pairwise MI matrix
    mi_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            try:
                mi = mi_func(data[:, i], data[:, j])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
            except Exception:
                # If MI estimation fails, set to 0
                mi_matrix[i, j] = 0
                mi_matrix[j, i] = 0
    
    # Simple threshold based on percentile
    alpha = kwargs.get('alpha', 0.05)
    threshold = np.percentile(mi_matrix[mi_matrix > 0], alpha * 100) if np.any(mi_matrix > 0) else 0.01
    threshold = max(threshold, 0.01)
    
    # Initial skeleton: edges with significant MI
    skeleton = (mi_matrix > threshold).astype(int)
    
    # Remove self-loops
    np.fill_diagonal(skeleton, 0)
    
    return skeleton, mi_matrix

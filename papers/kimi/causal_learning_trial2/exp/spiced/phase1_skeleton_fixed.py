"""
Phase 1: Information-Theoretic Skeleton Discovery (Fixed)
Uses k-NN entropy estimation with robust edge selection.
"""
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors


def knn_entropy(x: np.ndarray, k: int = 5) -> float:
    """
    Estimate differential entropy using k-NN estimator (Kraskov-Stögbauer-Grassberger).
    """
    n = len(x)
    x = x.reshape(-1, 1)
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(x)
    distances, _ = nbrs.kneighbors(x)
    
    # Distance to k-th nearest neighbor
    epsilon = distances[:, k]
    epsilon = np.maximum(epsilon, 1e-10)
    
    # KSG entropy estimator
    entropy = digamma(n) - digamma(k) + np.mean(np.log(2 * epsilon))
    
    return entropy


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


def knn_conditional_mi(x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int = 5) -> float:
    """
    Estimate conditional mutual information I(X;Y|Z) using k-NN.
    """
    n = len(x)
    
    # Handle multi-dimensional conditioning
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    
    # Create joint spaces
    xz = np.column_stack([x.reshape(-1, 1), z])
    yz = np.column_stack([y.reshape(-1, 1), z])
    xyz = np.column_stack([x.reshape(-1, 1), y.reshape(-1, 1), z])
    
    # Find k-nearest neighbors
    nbrs_xz = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(xz)
    distances_xz, _ = nbrs_xz.kneighbors(xz)
    epsilon_xz = np.maximum(distances_xz[:, k], 1e-10)
    
    nbrs_xyz = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(xyz)
    distances_xyz, _ = nbrs_xyz.kneighbors(xyz)
    epsilon_xyz = np.maximum(distances_xyz[:, k], 1e-10)
    
    # Count neighbors
    nz_xz = np.zeros(n)
    nz_xyz = np.zeros(n)
    
    for i in range(n):
        nz_xz[i] = np.sum(np.max(np.abs(z - z[i]), axis=1) < epsilon_xz[i]) - 1
        nz_xyz[i] = np.sum(np.max(np.abs(yz - yz[i]), axis=1) < epsilon_xyz[i]) - 1
    
    nz_xz = np.maximum(nz_xz, 1e-10)
    nz_xyz = np.maximum(nz_xyz, 1e-10)
    
    # CMI estimate
    cmi = np.mean(digamma(nz_xz) - digamma(nz_xyz))
    
    return max(0, cmi)


def compute_skeleton_it(data: np.ndarray, k: int = 5, alpha: float = 0.05) -> np.ndarray:
    """
    Compute initial skeleton using information-theoretic measures.
    
    Uses a robust thresholding approach that doesn't rely on permutation testing.
    Optimized for small sample sizes.
    """
    n_samples, n_nodes = data.shape
    
    # Adjust k for small sample sizes
    k_adj = min(k, max(3, n_samples // 20))
    
    # Standardize data
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    # Compute pairwise MI matrix
    mi_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            mi = knn_mutual_information(data[:, i], data[:, j], k=k_adj)
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    
    # Robust threshold selection using median absolute deviation
    mi_values = mi_matrix[np.triu_indices(n_nodes, k=1)]
    
    if len(mi_values) == 0 or np.all(mi_values == 0):
        return np.zeros((n_nodes, n_nodes), dtype=int)
    
    # Use median + MAD-based threshold
    median_mi = np.median(mi_values)
    mad = np.median(np.abs(mi_values - median_mi))
    
    # Adaptive threshold: median + 1.5*MAD for small samples
    threshold = median_mi + 1.5 * mad
    
    # Ensure we keep enough edges for small samples
    # More permissive for small N to avoid missing true edges
    min_edges = min(n_nodes, max(3, int(n_nodes * 0.6)))  # At least 60% of n_nodes
    if np.sum(mi_values > threshold) < min_edges:
        # Use percentile-based threshold
        percentile = 40 if n_samples < 100 else 50
        threshold = np.percentile(mi_values, percentile)
    
    # Ensure threshold is not too high
    threshold = max(threshold, np.percentile(mi_values, 25))
    
    # Initial skeleton: edges with significant MI
    skeleton = (mi_matrix > threshold).astype(int)
    
    # Remove self-loops
    np.fill_diagonal(skeleton, 0)
    
    # Iterative refinement: test conditional independence given neighbors
    # Only for larger samples where CMI estimation is reliable
    if n_samples >= 100:
        for iteration in range(1):  # 1 iteration for efficiency
            for i in range(n_nodes):
                neighbors_i = np.where(skeleton[i, :] == 1)[0]
                
                for j in neighbors_i:
                    if j <= i:
                        continue
                    
                    # Test if edge i-j is independent given other neighbors
                    other_neighbors = [n for n in neighbors_i if n != j]
                    
                    if len(other_neighbors) > 0:
                        # Use at most 1 conditioning variable for small samples
                        cond_vars = other_neighbors[:min(1, len(other_neighbors))]
                        
                        cmi = knn_conditional_mi(data[:, i], data[:, j],
                                                data[:, cond_vars[0]], k=k_adj)
                        
                        # If CMI is small, remove edge
                        if cmi < threshold * 0.5:  # Less stringent for small samples
                            skeleton[i, j] = 0
                            skeleton[j, i] = 0
    
    return skeleton


def compute_directed_information_scores(data: np.ndarray, skeleton: np.ndarray,
                                        k: int = 5) -> np.ndarray:
    """
    Compute directed information scores for edges in skeleton.
    """
    n_samples, n_nodes = data.shape
    
    # Standardize data
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    score_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if skeleton[i, j] == 0 or i == j:
                continue
            
            # Simple directed score: MI(i,j) normalized by individual entropies
            mi = knn_mutual_information(data[:, i], data[:, j], k=k)
            h_i = knn_entropy(data[:, i], k=k)
            h_j = knn_entropy(data[:, j], k=k)
            
            # Directional asymmetry score
            if h_i > 0 and h_j > 0:
                score_matrix[i, j] = mi / min(h_i, h_j)
    
    return score_matrix

"""
Phase 1: Information-Theoretic Skeleton Discovery
Uses k-NN entropy estimation for sample-efficient dependency detection.
"""
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors


def knn_entropy(x: np.ndarray, k: int = 5) -> float:
    """
    Estimate differential entropy using k-NN estimator (Kraskov-Stögbauer-Grassberger).
    
    Args:
        x: Data vector (n_samples,)
        k: Number of nearest neighbors
        
    Returns:
        Entropy estimate
    """
    n = len(x)
    x = x.reshape(-1, 1)
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(x)
    distances, _ = nbrs.kneighbors(x)
    
    # Distance to k-th nearest neighbor
    epsilon = distances[:, k]
    epsilon = np.maximum(epsilon, 1e-10)  # Avoid log(0)
    
    # KSG entropy estimator
    entropy = digamma(n) - digamma(k) + np.mean(np.log(2 * epsilon))
    
    return entropy


def knn_mutual_information(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    Estimate mutual information using k-NN estimator.
    
    Args:
        x: First variable (n_samples,)
        y: Second variable (n_samples,)
        k: Number of nearest neighbors
        
    Returns:
        MI estimate
    """
    n = len(x)
    
    # Joint space
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
        # Count points within epsilon[i] in x
        nx[i] = np.sum(np.abs(x - x[i]) < epsilon[i]) - 1
        # Count points within epsilon[i] in y  
        ny[i] = np.sum(np.abs(y - y[i]) < epsilon[i]) - 1
    
    nx = np.maximum(nx, 1e-10)
    ny = np.maximum(ny, 1e-10)
    
    # KSG MI estimator
    mi = digamma(k) - np.mean(digamma(nx) + digamma(ny)) + digamma(n)
    
    return max(0, mi)  # MI is non-negative


def knn_conditional_mi(x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int = 5) -> float:
    """
    Estimate conditional mutual information I(X;Y|Z) using k-NN.
    
    Approximates CMI using the formula: I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
    
    Args:
        x: First variable (n_samples,)
        y: Second variable (n_samples,)
        z: Conditioning variable(s) (n_samples,) or (n_samples, d)
        k: Number of nearest neighbors
        
    Returns:
        CMI estimate
    """
    n = len(x)
    
    # Handle multi-dimensional conditioning
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    
    # Create joint spaces
    xz = np.column_stack([x.reshape(-1, 1), z])
    yz = np.column_stack([y.reshape(-1, 1), z])
    xyz = np.column_stack([x.reshape(-1, 1), y.reshape(-1, 1), z])
    
    # Find k-nearest neighbors in XZ space
    nbrs_xz = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(xz)
    distances_xz, _ = nbrs_xz.kneighbors(xz)
    epsilon_xz = np.maximum(distances_xz[:, k], 1e-10)
    
    # Find k-nearest neighbors in XYZ space
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


def compute_skeleton_it(data: np.ndarray, k: int = 5, alpha: float = 0.05, 
                       permutation_iters: int = 20) -> np.ndarray:
    """
    Compute initial skeleton using information-theoretic measures.
    
    Args:
        data: Data matrix (n_samples, n_features)
        k: Number of neighbors for k-NN estimation
        alpha: Significance level for edge selection
        permutation_iters: Number of permutations for null distribution
        
    Returns:
        Skeleton adjacency matrix (undirected)
    """
    n_samples, n_nodes = data.shape
    
    # Standardize data
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    # Compute pairwise MI matrix
    mi_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            mi = knn_mutual_information(data[:, i], data[:, j], k=k)
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    
    # Permutation test to determine threshold
    mi_null = []
    for _ in range(permutation_iters):
        # Permute data
        perm_idx = np.random.permutation(n_samples)
        data_perm = data[perm_idx]
        
        # Compute MI on permuted data for random pair
        i, j = np.random.choice(n_nodes, 2, replace=False)
        mi_perm = knn_mutual_information(data_perm[:, i], data_perm[:, j], k=k)
        mi_null.append(mi_perm)
    
    # Adaptive threshold
    threshold = np.percentile(mi_null, (1 - alpha) * 100)
    
    # Initial skeleton: edges with significant MI
    skeleton = (mi_matrix > threshold).astype(int)
    
    # Remove self-loops
    np.fill_diagonal(skeleton, 0)
    
    # Iterative refinement: test conditional independence given neighbors
    for iteration in range(2):  # 2 iterations of refinement
        for i in range(n_nodes):
            neighbors_i = np.where(skeleton[i, :] == 1)[0]
            
            for j in neighbors_i:
                if j <= i:
                    continue
                
                # Test if edge i-j is independent given other neighbors
                other_neighbors = [n for n in neighbors_i if n != j]
                
                if len(other_neighbors) > 0:
                    # Use at most 2 conditioning variables for efficiency
                    cond_vars = other_neighbors[:min(2, len(other_neighbors))]
                    
                    if len(cond_vars) == 1:
                        cmi = knn_conditional_mi(data[:, i], data[:, j], 
                                                data[:, cond_vars[0]], k=k)
                    else:
                        cmi = knn_conditional_mi(data[:, i], data[:, j],
                                                data[:, cond_vars], k=k)
                    
                    # If CMI is small, remove edge
                    if cmi < threshold * 0.5:
                        skeleton[i, j] = 0
                        skeleton[j, i] = 0
    
    return skeleton


def compute_directed_information_scores(data: np.ndarray, skeleton: np.ndarray,
                                        k: int = 5) -> np.ndarray:
    """
    Compute directed information scores for edges in skeleton.
    
    Uses a simple approximation: DI(X->Y) ≈ I(X;Y) when X is before Y
    in some topological ordering. We approximate using residual information.
    
    Args:
        data: Data matrix (n_samples, n_features)
        skeleton: Undirected skeleton adjacency matrix
        k: Number of neighbors for k-NN estimation
        
    Returns:
        Directed score matrix (higher means more likely i -> j)
    """
    n_samples, n_nodes = data.shape
    
    # Standardize data
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    # Compute directed scores using residual information
    # If X causes Y, then Y should be predictable from X but not vice versa
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
            # Higher score suggests i -> j
            if h_i > 0 and h_j > 0:
                score_matrix[i, j] = mi / min(h_i, h_j)
    
    return score_matrix

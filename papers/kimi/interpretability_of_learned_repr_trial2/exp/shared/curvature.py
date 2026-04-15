"""
Curvature estimation methods for Local Curvature Probing.
Fixed version with proper normalization and bug fixes.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Optional, Dict
import warnings


class CurvatureEstimator:
    """Base class for curvature estimation."""
    
    def __init__(self, k: int = 50):
        self.k = k
    
    def estimate(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PCACurvatureEstimator(CurvatureEstimator):
    """
    PCA-based curvature estimation using local covariance structure.
    
    Returns curvature as the inverse of flatness (1 - flatness).
    Higher values indicate more complex/curved regions.
    """
    
    def __init__(self, k: int = 50, n_components: int = None):
        super().__init__(k)
        self.n_components = n_components
    
    def estimate(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate curvature at each point using PCA.
        
        Returns:
            curvature: (n,) curvature measure [0, 1] (1 = highly curved, 0 = flat)
            effective_dim: (n,) effective dimensionality
            principal_curvatures: (n, k) principal curvature magnitudes
        """
        n, d = points.shape
        k = min(self.k, n - 1)
        n_components = self.n_components or min(d // 2, k, 10)
        
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)
        _, indices = nbrs.kneighbors(points)
        
        curvature = np.zeros(n)
        effective_dim = np.zeros(n)
        principal_curvatures = np.zeros((n, n_components))
        
        for i in range(n):
            neighbor_idx = indices[i, 1:]  # Exclude self
            neighborhood = points[neighbor_idx]
            
            if len(neighborhood) < 2:
                curvature[i] = 0.0
                effective_dim[i] = d
                continue
            
            try:
                centered = neighborhood - neighborhood.mean(axis=0)
                cov = centered.T @ centered / (len(neighborhood) - 1)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.sort(eigenvalues)[::-1]
                
                # Remove very small eigenvalues
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                
                if len(eigenvalues) == 0:
                    curvature[i] = 0.0
                    effective_dim[i] = 1.0
                    continue
                
                # Effective dimensionality (participation ratio)
                total_var = eigenvalues.sum()
                if total_var > 1e-10:
                    effective_dim[i] = (total_var ** 2) / (eigenvalues ** 2).sum()
                else:
                    effective_dim[i] = 1.0
                
                # Curvature: normalized variance concentration
                # If variance is concentrated in few directions = flat = low curvature
                # If variance is spread out = curved = high curvature
                k_eff = min(n_components, len(eigenvalues))
                top_var = eigenvalues[:k_eff].sum()
                
                # Flatness = fraction of variance in top components
                flatness = top_var / total_var if total_var > 1e-10 else 1.0
                
                # Curvature = 1 - flatness, but with better scaling
                # Use entropy-based measure for better differentiation
                probs = eigenvalues / total_var
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(len(eigenvalues))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                # Curvature combines deviation from flatness and entropy
                curvature[i] = normalized_entropy * (1 - flatness)
                
                # Principal curvatures: deviation from mean eigenvalue
                mean_eig = eigenvalues.mean()
                if mean_eig > 1e-10:
                    pc = np.abs(eigenvalues[:n_components] - mean_eig) / mean_eig
                    principal_curvatures[i, :min(len(pc), n_components)] = pc[:n_components]
                
            except (np.linalg.LinAlgError, ValueError) as e:
                curvature[i] = 0.0
                effective_dim[i] = d
        
        # Normalize curvature to [0, 1]
        if curvature.max() > curvature.min():
            curvature = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-10)
        
        return curvature, effective_dim, principal_curvatures


class OllivierRicciCurvature(CurvatureEstimator):
    """
    Ollivier-Ricci curvature for k-NN graphs.
    Fixed version with proper numerical stability.
    """
    
    def __init__(self, k: int = 10, alpha: float = 0.5):
        super().__init__(k)
        self.alpha = alpha
    
    def estimate(self, points: np.ndarray) -> np.ndarray:
        """Estimate Ollivier-Ricci curvature for each point."""
        try:
            import networkx as nx
            import ot
        except ImportError:
            warnings.warn("networkx or POT not available, returning zeros")
            return np.zeros(len(points))
        
        n = len(points)
        k = min(self.k, n - 1)
        
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        for i in range(n):
            for j_idx in range(1, k+1):
                j = indices[i, j_idx]
                if not G.has_edge(i, j):
                    G.add_edge(i, j, weight=max(distances[i, j_idx], 1e-10))
        
        curvature = np.zeros(n)
        
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if len(neighbors) < 2:
                curvature[node] = 0
                continue
            
            edge_curvatures = []
            
            for neighbor in neighbors:
                try:
                    m_node = self._create_measure(node, G, alpha=self.alpha)
                    m_neighbor = self._create_measure(neighbor, G, alpha=self.alpha)
                    
                    dist_matrix = self._compute_distance_matrix(G, m_node, m_neighbor)
                    
                    if len(m_node) > 0 and len(m_neighbor) > 0:
                        a = np.array(list(m_node.values()), dtype=np.float64)
                        b = np.array(list(m_neighbor.values()), dtype=np.float64)
                        
                        # Normalize
                        a = a / a.sum()
                        b = b / b.sum()
                        
                        w_dist = ot.emd2(a, b, dist_matrix)
                        
                        graph_dist = G[node][neighbor]['weight']
                        if graph_dist > 1e-10:
                            kappa = 1 - w_dist / graph_dist
                            edge_curvatures.append(kappa)
                except Exception:
                    pass
            
            if edge_curvatures:
                curvature[node] = np.mean(edge_curvatures)
            else:
                curvature[node] = 0
        
        # Normalize to [-1, 1] range
        max_abs = np.abs(curvature).max()
        if max_abs > 0:
            curvature = curvature / max_abs
        
        return curvature
    
    def _create_measure(self, node: int, G, alpha: float = 0.5) -> Dict[int, float]:
        """Create probability measure for a node."""
        neighbors = list(G.neighbors(node))
        if not neighbors:
            return {node: 1.0}
        
        measure = {node: alpha}
        uniform_prob = (1 - alpha) / len(neighbors)
        for neighbor in neighbors:
            measure[neighbor] = uniform_prob
        
        return measure
    
    def _compute_distance_matrix(self, G, m1: Dict, m2: Dict) -> np.ndarray:
        """Compute distance matrix between two measures."""
        nodes1 = list(m1.keys())
        nodes2 = list(m2.keys())
        
        dist_matrix = np.zeros((len(nodes1), len(nodes2)))
        
        for i, n1 in enumerate(nodes1):
            for j, n2 in enumerate(nodes2):
                if n1 == n2:
                    dist_matrix[i, j] = 0
                else:
                    try:
                        dist_matrix[i, j] = nx.shortest_path_length(G, n1, n2, weight='weight')
                    except nx.NetworkXNoPath:
                        dist_matrix[i, j] = 1e6
        
        return dist_matrix


class SecondFundamentalFormEstimator(CurvatureEstimator):
    """
    Estimate principal curvatures using second fundamental form.
    Fixed version with proper quadratic fitting.
    """
    
    def __init__(self, k: int = 50):
        super().__init__(k)
    
    def estimate(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate principal curvatures at each point.
        
        Returns:
            principal_curvatures: (n, d-1) array of principal curvatures
            curvature_magnitude: (n,) total curvature magnitude
        """
        n, d = points.shape
        k = min(self.k, n - 1)
        
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)
        _, indices = nbrs.kneighbors(points)
        
        principal_curvatures = np.zeros((n, min(d-1, k-1)))
        curvature_magnitude = np.zeros(n)
        
        for i in range(n):
            neighbor_idx = indices[i, 1:]
            neighborhood = points[neighbor_idx]
            
            if len(neighborhood) < max(3, d // 2):
                continue
            
            try:
                center = neighborhood.mean(axis=0)
                centered = neighborhood - center
                
                # PCA to find tangent plane
                if len(centered) > 1:
                    cov = centered.T @ centered / (len(centered) - 1)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    idx = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]
                    
                    # Use top 2 dimensions for tangent plane
                    tangent_dim = min(2, d-1)
                    tangent_basis = eigenvectors[:, :tangent_dim]
                    normal = eigenvectors[:, tangent_dim:tangent_dim+1]
                    
                    if normal.shape[1] == 0:
                        normal = eigenvectors[:, -1:]
                    
                    # Project to tangent coordinates
                    tangent_coords = centered @ tangent_basis
                    heights = (centered @ normal).flatten()
                    
                    # Fit quadratic patch with regularization
                    if tangent_dim == 2 and len(tangent_coords) >= 3:
                        X = np.column_stack([
                            tangent_coords[:, 0]**2,
                            tangent_coords[:, 1]**2,
                            tangent_coords[:, 0] * tangent_coords[:, 1]
                        ])
                        
                        # Regularized least squares
                        XtX = X.T @ X
                        reg = 1e-5 * np.eye(XtX.shape[0])
                        coeffs = np.linalg.solve(XtX + reg, X.T @ heights)
                        a, b, c = coeffs
                        
                        # Shape operator (Hessian)
                        H = np.array([[2*a, c], [c, 2*b]])
                        
                        # Principal curvatures
                        pc = np.linalg.eigvalsh(H)
                        principal_curvatures[i, :len(pc)] = pc
                        curvature_magnitude[i] = np.sqrt((pc**2).sum())
                        
                    elif tangent_dim == 1 and len(tangent_coords) > 1:
                        X = tangent_coords[:, 0:1]**2
                        XtX = X.T @ X
                        reg = 1e-5 * np.eye(1)
                        coeffs = np.linalg.solve(XtX + reg, X.T @ heights)
                        principal_curvatures[i, 0] = 2 * coeffs[0]
                        curvature_magnitude[i] = abs(2 * coeffs[0])
                            
            except (np.linalg.LinAlgError, ValueError):
                pass
        
        # Normalize curvature magnitude
        if curvature_magnitude.max() > 0:
            curvature_magnitude = curvature_magnitude / (curvature_magnitude.max() + 1e-10)
        
        return principal_curvatures, curvature_magnitude


class CombinedCurvatureEstimator:
    """
    Combines multiple curvature estimation methods with proper normalization.
    """
    
    def __init__(self, k: int = 50, use_pca: bool = True, 
                 use_orc: bool = True, use_sff: bool = True):
        self.k = k
        self.use_pca = use_pca
        self.use_orc = use_orc
        self.use_sff = use_sff
        
        self.pca_estimator = PCACurvatureEstimator(k) if use_pca else None
        self.orc_estimator = OllivierRicciCurvature(k=min(10, k)) if use_orc else None
        self.sff_estimator = SecondFundamentalFormEstimator(k) if use_sff else None
    
    def estimate(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Estimate curvature using all enabled methods.
        
        Returns:
            Dictionary with curvature estimates from each method
        """
        results = {}
        
        if self.pca_estimator:
            curvature, eff_dim, pc = self.pca_estimator.estimate(points)
            results['pca_curvature'] = curvature
            results['pca_effective_dim'] = eff_dim
            results['pca_principal'] = pc
        
        if self.orc_estimator:
            results['orc_curvature'] = self.orc_estimator.estimate(points)
        
        if self.sff_estimator:
            pc, mag = self.sff_estimator.estimate(points)
            results['sff_principal'] = pc
            results['sff_magnitude'] = mag
        
        # Combined curvature score with equal weighting
        combined = np.zeros(len(points))
        weights = []
        
        if 'pca_curvature' in results:
            combined += results['pca_curvature']
            weights.append('pca')
        
        if 'orc_curvature' in results:
            # Normalize ORC to [0, 1]
            orc = results['orc_curvature']
            orc_norm = (orc + 1) / 2  # Map from [-1, 1] to [0, 1]
            combined += orc_norm
            weights.append('orc')
        
        if 'sff_magnitude' in results:
            combined += results['sff_magnitude']
            weights.append('sff')
        
        if weights:
            results['combined_curvature'] = combined / len(weights)
        else:
            results['combined_curvature'] = combined
        
        results['methods_used'] = weights
        
        return results


def test_on_synthetic():
    """Test curvature estimators on synthetic manifolds."""
    np.random.seed(42)
    
    # Sphere (should have high curvature)
    theta = np.random.uniform(0, 2*np.pi, 500)
    phi = np.random.uniform(0, np.pi, 500)
    sphere = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    
    # Plane (should have low curvature)
    plane = np.random.randn(500, 3)
    plane[:, 2] = 0
    
    # Saddle (should have high curvature)
    x = np.random.uniform(-1, 1, 500)
    y = np.random.uniform(-1, 1, 500)
    saddle = np.column_stack([x, y, x**2 - y**2])
    
    estimator = CombinedCurvatureEstimator(k=20, use_pca=True, use_orc=True, use_sff=True)
    
    print("Testing curvature estimators on synthetic manifolds:")
    for name, data in [('Sphere', sphere), ('Plane', plane), ('Saddle', saddle)]:
        results = estimator.estimate(data)
        print(f"\n{name}:")
        print(f"  PCA curvature: {results['pca_curvature'].mean():.4f} ± {results['pca_curvature'].std():.4f}")
        if 'orc_curvature' in results:
            print(f"  ORC curvature: {results['orc_curvature'].mean():.4f} ± {results['orc_curvature'].std():.4f}")
        if 'sff_magnitude' in results:
            print(f"  SFF magnitude: {results['sff_magnitude'].mean():.4f} ± {results['sff_magnitude'].std():.4f}")
        print(f"  Combined: {results['combined_curvature'].mean():.4f} ± {results['combined_curvature'].std():.4f}")


if __name__ == "__main__":
    test_on_synthetic()

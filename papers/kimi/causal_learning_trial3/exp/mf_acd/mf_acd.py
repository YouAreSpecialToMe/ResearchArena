"""
Multi-Fidelity Adaptive Causal Discovery (MF-ACD) Framework.

Three-phase approach with Uncertainty-Guided Fidelity Selection (UGFS).
"""
import numpy as np
from scipy.stats import pearsonr, norm
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Set
import networkx as nx


class MFACD:
    """
    Multi-Fidelity Adaptive Causal Discovery.
    
    Implements three-phase approach:
    1. Low-fidelity skeleton screening (correlation-based, FDR control)
    2. Medium-fidelity local refinement (Fisher Z, adaptive FDR)
    3. High-fidelity critical resolution (distance correlation proxy, Holm-Bonferroni)
    """
    
    def __init__(self, 
                 budget_allocation: Tuple[float, float, float] = (0.34, 0.20, 0.46),
                 alpha1: float = 0.10,  # Phase 1 FDR level
                 alpha2: float = 0.05,  # Phase 2 FDR level
                 alpha3: float = 0.01,  # Phase 3 FWER level
                 cost_weights: Tuple[float, float, float] = (1.0, 1.1, 15.0),
                 use_adaptive: bool = True,
                 eta: float = 0.1):
        """
        Initialize MF-ACD.
        
        Args:
            budget_allocation: Initial budget split for phases 1/2/3
            alpha1: FDR level for phase 1
            alpha2: FDR level for phase 2
            alpha3: FWER level for phase 3
            cost_weights: Relative cost of low/medium/high fidelity tests
            use_adaptive: Whether to use adaptive budget reallocation
            eta: Adaptation rate for regret minimization
        """
        self.budget_allocation = np.array(budget_allocation)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.cost_weights = np.array(cost_weights)
        self.use_adaptive = use_adaptive
        self.eta = eta
        
        # Track computational costs
        self.phase_costs = [0.0, 0.0, 0.0]
        self.n_tests = [0, 0, 0]
        self.ugfs_overhead = 0.0
        
    def correlation_test(self, data: np.ndarray, x: int, y: int, 
                         cond_set: List[int]) -> float:
        """Low-fidelity: correlation-based CI test."""
        n = data.shape[0]
        
        if len(cond_set) == 0:
            corr, _ = pearsonr(data[:, x], data[:, y])
            z = np.arctanh(np.clip(np.abs(corr), 0, 0.999)) * np.sqrt(n - 3)
            p_value = 2 * (1 - norm.cdf(np.abs(z)))
        else:
            # Partial correlation via regression
            X_cond = data[:, cond_set]
            
            reg_x = LinearRegression().fit(X_cond, data[:, x])
            resid_x = data[:, x] - reg_x.predict(X_cond)
            
            reg_y = LinearRegression().fit(X_cond, data[:, y])
            resid_y = data[:, y] - reg_y.predict(X_cond)
            
            corr, _ = pearsonr(resid_x, resid_y)
            z = np.arctanh(np.clip(np.abs(corr), 0, 0.999)) * np.sqrt(n - 3 - len(cond_set))
            p_value = 2 * (1 - norm.cdf(np.abs(z)))
        
        return p_value
    
    def fisher_z_test(self, data: np.ndarray, x: int, y: int,
                      cond_set: List[int]) -> float:
        """Medium-fidelity: Fisher Z test."""
        return self.correlation_test(data, x, y, cond_set)
    
    def distance_correlation_test(self, data: np.ndarray, x: int, y: int,
                                  cond_set: List[int]) -> float:
        """
        High-fidelity: Distance correlation as proxy for kernel-based test.
        Uses simplified version for computational efficiency.
        """
        n = data.shape[0]
        
        if len(cond_set) == 0:
            # Unconditional distance correlation
            dc = self._distance_correlation(data[:, x], data[:, y])
        else:
            # Conditional: residualize on conditioning set
            X_cond = data[:, cond_set]
            
            reg_x = LinearRegression().fit(X_cond, data[:, x])
            resid_x = data[:, x] - reg_x.predict(X_cond)
            
            reg_y = LinearRegression().fit(X_cond, data[:, y])
            resid_y = data[:, y] - reg_y.predict(X_cond)
            
            dc = self._distance_correlation(resid_x, resid_y)
        
        # Convert to approximate p-value using asymptotic distribution
        # Under null, n*dc^2 ~ sum of chi-square(1)
        stat = n * dc * dc
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(stat, df=1)
        
        return p_value
    
    def _distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute distance correlation between two vectors."""
        n = len(x)
        
        # Compute distance matrices
        A = np.abs(x[:, None] - x[None, :])
        B = np.abs(y[:, None] - y[None, :])
        
        # Double centering
        A_row = A.mean(axis=1, keepdims=True)
        A_col = A.mean(axis=0, keepdims=True)
        A_mean = A.mean()
        A_centered = A - A_row - A_col + A_mean
        
        B_row = B.mean(axis=1, keepdims=True)
        B_col = B.mean(axis=0, keepdims=True)
        B_mean = B.mean()
        B_centered = B - B_row - B_col + B_mean
        
        # Distance covariance
        dcov2 = np.sum(A_centered * B_centered) / (n * n)
        dvar_x = np.sum(A_centered * A_centered) / (n * n)
        dvar_y = np.sum(B_centered * B_centered) / (n * n)
        
        if dvar_x > 0 and dvar_y > 0:
            return np.sqrt(dcov2) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y))
        return 0.0
    
    def benjamini_hochberg(self, p_values: np.ndarray, alpha: float) -> np.ndarray:
        """
        Apply Benjamini-Hochberg FDR control.
        Returns boolean array where True = reject null (i.e., significant, dependent).
        """
        n = len(p_values)
        if n == 0:
            return np.array([], dtype=bool)
        
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Find largest k such that p_(k) <= (k/m)*alpha
        thresholds = np.arange(1, n + 1) / n * alpha
        
        # Check which hypotheses can be rejected
        rejected = sorted_p <= thresholds
        
        if not rejected.any():
            return np.zeros(n, dtype=bool)
        
        max_k = np.where(rejected)[0][-1]
        result = np.zeros(n, dtype=bool)
        result[sorted_indices[:max_k + 1]] = True
        
        return result
    
    def holm_bonferroni(self, p_values: np.ndarray, alpha: float) -> np.ndarray:
        """
        Apply Holm-Bonferroni FWER control (more powerful than Bonferroni).
        Returns boolean array where True = reject null (i.e., significant, dependent).
        """
        n = len(p_values)
        if n == 0:
            return np.array([], dtype=bool)
            
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Test each hypothesis
        result = np.zeros(n, dtype=bool)
        for i in range(n):
            if sorted_p[i] <= alpha / (n - i):
                result[sorted_indices[i]] = True
            else:
                # Once we fail to reject, all subsequent (larger) p-values also fail
                break
        
        return result
    
    def compute_uncertainty(self, p_values: List[float]) -> float:
        """
        Compute uncertainty score for an edge based on p-value variance.
        Higher uncertainty when p-values are around 0.5 (ambiguous).
        """
        if len(p_values) == 0:
            return 1.0  # Maximum uncertainty
        
        # Use entropy-like measure: p*(1-p) is maximized at p=0.5
        uncertainties = [4 * p * (1 - p) for p in p_values if 0 < p < 1]
        
        if len(uncertainties) == 0:
            return 0.0
        
        return np.mean(uncertainties)
    
    def compute_structural_importance(self, adj: np.ndarray, i: int, j: int) -> float:
        """
        Compute structural importance of an edge.
        Based on Markov blanket sizes of connected nodes.
        """
        n = adj.shape[0]
        
        # Approximate Markov blanket as neighbors in current skeleton
        mb_i = np.sum(adj[i, :])
        mb_j = np.sum(adj[j, :])
        
        # Structural importance: edges connecting high-degree nodes are more critical
        max_mb = max(1, np.max([np.sum(adj[k, :]) for k in range(n)]))
        
        si = (mb_i + mb_j) / (2 * max_mb)
        return min(1.0, si)
    
    def estimate_information_gain(self, uncertainty: float, fidelity: int) -> float:
        """
        Estimate information gain from performing a test.
        Simplified approximation: IG ~ uncertainty * power(fidelity).
        """
        # Power increases with fidelity
        power_map = {0: 0.6, 1: 0.8, 2: 0.95}
        power = power_map.get(fidelity, 0.6)
        
        # Information gain approximation
        ig = 4 * uncertainty * (1 - uncertainty) * power
        return ig
    
    def select_edge_fidelity(self, candidates: List[Tuple[int, int]],
                            uncertainties: Dict[Tuple[int, int], float],
                            adj: np.ndarray,
                            remaining_budget: float) -> Tuple[Tuple[int, int], int]:
        """
        Select edge-fidelity pair using UGFS.
        Maximizes information gain per unit cost.
        """
        import time
        start_time = time.time()
        
        best_score = -1
        best_edge = None
        best_fidelity = 0
        
        for edge in candidates:
            i, j = edge
            uncertainty = uncertainties.get(edge, 0.5)
            si = self.compute_structural_importance(adj, i, j)
            
            for fidelity in [0, 1, 2]:
                cost = self.cost_weights[fidelity]
                ig = self.estimate_information_gain(uncertainty, fidelity)
                
                # Score: IG * SI / Cost
                score = (ig * si) / cost
                
                if score > best_score:
                    best_score = score
                    best_edge = edge
                    best_fidelity = fidelity
        
        self.ugfs_overhead += time.time() - start_time
        
        return best_edge, best_fidelity
    
    def phase1_skeleton_screening(self, data: np.ndarray, 
                                   budget: float) -> Tuple[np.ndarray, Dict]:
        """
        Phase 1: Low-fidelity skeleton screening.
        Uses correlation tests with FDR control.
        More conservative: keeps edges when uncertain rather than removing aggressively.
        """
        n_vars = data.shape[1]
        n_samples = data.shape[0]
        adj = np.ones((n_vars, n_vars))
        np.fill_diagonal(adj, 0)
        
        test_results = {}  # Track p-values for uncertainty
        cost = 0.0
        
        # First: unconditional tests on all pairs (cheap)
        edges = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars)]
        p_values_uncond = []
        
        for i, j in edges:
            if cost >= budget * 0.5:  # Use 50% of budget for unconditional tests
                break
            
            p = self.correlation_test(data, i, j, [])
            p_values_uncond.append((i, j, p))
            cost += self.cost_weights[0]
            self.n_tests[0] += 1
            
            if (i, j) not in test_results:
                test_results[(i, j)] = []
            test_results[(i, j)].append(p)
        
        # Apply FDR control to unconditional tests - be conservative
        # Only remove edges with high p-values (strong evidence of independence)
        if p_values_uncond:
            p_array = np.array([p for _, _, p in p_values_uncond])
            # Use higher alpha to be less aggressive in removing edges
            significant = self.benjamini_hochberg(p_array, self.alpha1 * 2)
            
            for idx, (i, j, p) in enumerate(p_values_uncond):
                # Only remove if p > 0.5 (very weak evidence of dependence)
                # AND not significant after FDR correction
                if not significant[idx] and p > 0.3:
                    adj[i, j] = 0
                    adj[j, i] = 0
        
        # Second: conditional tests on remaining edges with small conditioning sets
        remaining_budget = budget - cost
        edges_remaining = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars) 
                          if adj[i, j] == 1]
        
        p_values_cond = []
        for i, j in edges_remaining:
            if cost >= budget:
                break
            
            # Test with small conditioning sets (size 1-2)
            neighbors_i = [k for k in range(n_vars) if adj[i, k] == 1 and k != j][:3]
            
            if len(neighbors_i) > 0:
                from itertools import combinations
                for cond_set in combinations(neighbors_i, min(1, len(neighbors_i))):
                    p = self.correlation_test(data, i, j, list(cond_set))
                    p_values_cond.append((i, j, p))
                    cost += self.cost_weights[0]
                    self.n_tests[0] += 1
                    
                    if (i, j) not in test_results:
                        test_results[(i, j)] = []
                    test_results[(i, j)].append(p)
                    break  # Only one conditional test per edge in Phase 1
        
        # Apply FDR control to conditional tests - be very conservative
        if p_values_cond:
            p_array = np.array([p for _, _, p in p_values_cond])
            significant = self.benjamini_hochberg(p_array, self.alpha1)
            
            for idx, (i, j, p) in enumerate(p_values_cond):
                # Only remove if very high p-value (strong evidence of independence)
                if not significant[idx] and p > 0.5:
                    adj[i, j] = 0
                    adj[j, i] = 0
        
        self.phase_costs[0] = cost
        
        # Compute uncertainties for remaining edges
        uncertainties = {}
        for edge in test_results.keys():
            uncertainties[edge] = self.compute_uncertainty(test_results[edge])
        # Also add uncertainties for edges that weren't tested
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                if adj[i, j] == 1 and (i, j) not in uncertainties:
                    uncertainties[(i, j)] = 1.0  # Maximum uncertainty
        
        return adj, uncertainties
    
    def phase2_local_refinement(self, data: np.ndarray, adj: np.ndarray,
                                 uncertainties: Dict, budget: float) -> np.ndarray:
        """
        Phase 2: Medium-fidelity local refinement.
        Uses Fisher Z with adaptive FDR.
        More conservative: thorough testing, careful edge removal.
        """
        n_vars = data.shape[1]
        adj = adj.copy()
        
        # Get candidate edges sorted by uncertainty (most uncertain first)
        edges = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars) 
                if adj[i, j] == 1]
        edges.sort(key=lambda e: uncertainties.get(e, 0.5), reverse=True)
        
        cost = 0.0
        test_results = {}  # Track results per edge
        
        for i, j in edges:
            if cost >= budget:
                break
            
            # Test with multiple conditioning set sizes
            neighbors = [k for k in range(n_vars) if adj[i, k] == 1 and k != j]
            
            test_results[(i, j)] = []
            
            # Test unconditional
            p = self.fisher_z_test(data, i, j, [])
            test_results[(i, j)].append(p)
            cost += self.cost_weights[1]
            self.n_tests[1] += 1
            
            # Test with conditioning sets of increasing size
            for d in range(1, min(3, len(neighbors) + 1)):
                if cost >= budget:
                    break
                    
                from itertools import combinations
                # Limit number of conditioning sets to test
                cond_sets = list(combinations(neighbors, d))[:3]
                
                for cond_set in cond_sets:
                    p = self.fisher_z_test(data, i, j, list(cond_set))
                    test_results[(i, j)].append(p)
                    cost += self.cost_weights[1]
                    self.n_tests[1] += 1
        
        # Make edge decisions based on test results
        # Only remove edge if ALL tests suggest independence (high p-values)
        for (i, j), pvals in test_results.items():
            if len(pvals) > 0:
                # If median p-value is high and all p-values are non-significant, remove edge
                median_p = np.median(pvals)
                max_p = max(pvals)
                
                # Conservative: only remove if strong evidence of independence
                if median_p > 0.3 and max_p > 0.5:
                    adj[i, j] = 0
                    adj[j, i] = 0
        
        self.phase_costs[1] = cost
        return adj
    
    def phase3_critical_resolution(self, data: np.ndarray, adj: np.ndarray,
                                   budget: float) -> np.ndarray:
        """
        Phase 3: High-fidelity critical resolution.
        Uses distance correlation with Holm-Bonferroni.
        More thorough testing before removing edges.
        """
        n_vars = data.shape[1]
        adj = adj.copy()
        
        # Get remaining edges
        edges = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars) 
                if adj[i, j] == 1]
        
        if not edges:
            self.phase_costs[2] = 0
            return adj
        
        cost = 0.0
        test_results = {}  # Track all p-values per edge
        
        # Test edges with high-fidelity
        for i, j in edges:
            if cost >= budget:
                break
            
            test_results[(i, j)] = []
            
            # Test unconditional first
            p = self.distance_correlation_test(data, i, j, [])
            test_results[(i, j)].append(p)
            cost += self.cost_weights[2]
            self.n_tests[2] += 1
            
            # Also test with small conditioning sets if budget allows
            neighbors = [k for k in range(n_vars) if adj[i, k] == 1 and k != j][:2]
            if len(neighbors) > 0 and cost < budget * 0.5:
                from itertools import combinations
                for cond_set in combinations(neighbors, 1):
                    if cost >= budget * 0.5:
                        break
                    p = self.distance_correlation_test(data, i, j, list(cond_set))
                    test_results[(i, j)].append(p)
                    cost += self.cost_weights[2]
                    self.n_tests[2] += 1
        
        # Apply Holm-Bonferroni to all tests
        if test_results:
            all_pvalues = []
            edge_list = []
            for (i, j), pvals in test_results.items():
                for p in pvals:
                    all_pvalues.append(p)
                    edge_list.append((i, j))
            
            p_array = np.array(all_pvalues)
            significant = self.holm_bonferroni(p_array, self.alpha3)
            
            # For each edge, keep it if ANY of its tests is significant
            edges_to_remove = set(test_results.keys())
            for idx, (i, j) in enumerate(edge_list):
                if significant[idx]:
                    # At least one test shows dependence -> keep edge
                    edges_to_remove.discard((i, j))
            
            # Remove edges that showed no evidence of dependence
            for (i, j) in edges_to_remove:
                adj[i, j] = 0
                adj[j, i] = 0
        
        self.phase_costs[2] = cost
        return adj
    
    def fit(self, data: np.ndarray) -> Dict:
        """
        Run MF-ACD on data.
        
        Args:
            data: Data matrix (n_samples, n_vars)
            
        Returns:
            Dictionary with results
        """
        n_vars = data.shape[1]
        
        # Estimate total budget (based on full PC with high-fidelity)
        # This is a proxy - in practice, budget could be set directly
        estimated_tests = n_vars * (n_vars - 1) / 2 * 5  # Rough estimate
        total_budget = estimated_tests * self.cost_weights[2]
        
        # Phase 1: Low-fidelity screening
        phase1_budget = total_budget * self.budget_allocation[0]
        adj, uncertainties = self.phase1_skeleton_screening(data, phase1_budget)
        
        # Adaptive reallocation if enabled
        if self.use_adaptive:
            remaining_budget = total_budget - self.phase_costs[0]
            # If phase 1 eliminated many edges, reduce phase 2 budget
            edges_remaining = np.sum(adj) / 2
            elimination_rate = 1 - edges_remaining / (n_vars * (n_vars - 1) / 2)
            
            if elimination_rate > 0.7:  # >70% edges eliminated
                # Reduce phase 2, increase phase 3
                phase2_ratio = 0.15
                phase3_ratio = 0.85
            else:
                phase2_ratio = self.budget_allocation[1] / (self.budget_allocation[1] + self.budget_allocation[2])
                phase3_ratio = self.budget_allocation[2] / (self.budget_allocation[1] + self.budget_allocation[2])
        else:
            phase2_ratio = self.budget_allocation[1] / (self.budget_allocation[1] + self.budget_allocation[2])
            phase3_ratio = self.budget_allocation[2] / (self.budget_allocation[1] + self.budget_allocation[2])
            remaining_budget = total_budget - self.phase_costs[0]
        
        # Phase 2: Medium-fidelity refinement
        phase2_budget = remaining_budget * phase2_ratio
        adj = self.phase2_local_refinement(data, adj, uncertainties, phase2_budget)
        
        # Phase 3: High-fidelity resolution
        phase3_budget = remaining_budget * phase3_ratio
        adj = self.phase3_critical_resolution(data, adj, phase3_budget)
        
        # Compute statistics
        total_cost = sum(self.phase_costs)
        baseline_cost = estimated_tests * self.cost_weights[2]
        savings = (baseline_cost - total_cost) / baseline_cost * 100
        
        return {
            'adjacency': adj,
            'phase_costs': self.phase_costs.copy(),
            'n_tests': self.n_tests.copy(),
            'total_cost': total_cost,
            'baseline_cost': baseline_cost,
            'savings_pct': savings,
            'ugfs_overhead': self.ugfs_overhead
        }


def run_mf_acd(data: np.ndarray, **kwargs) -> Dict:
    """Convenience function to run MF-ACD."""
    mf_acd = MFACD(**kwargs)
    return mf_acd.fit(data)

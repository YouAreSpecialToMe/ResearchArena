"""
Improved Multi-Fidelity Adaptive Causal Discovery (MF-ACD) Framework.

Key improvements based on feedback:
1. Phase 1 uses distance correlation (nonlinear) instead of correlation
2. Iterative refinement: Phase 3 results inform Phase 2 re-testing
3. Better uncertainty quantification with adaptive thresholds
4. Improved budget allocation based on edge density
"""
import numpy as np
from scipy.stats import pearsonr, norm
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Set, Optional
import networkx as nx
import time


class MFACDImproved:
    """
    Improved Multi-Fidelity Adaptive Causal Discovery.
    
    Implements three-phase approach with key improvements:
    1. Phase 1: Low-fidelity uses fast distance correlation (nonlinear capable)
    2. Phase 2: Medium-fidelity with adaptive FDR and iterative refinement
    3. Phase 3: High-fidelity with Holm-Bonferroni and feedback to Phase 2
    """
    
    def __init__(self, 
                 budget_allocation: Tuple[float, float, float] = (0.30, 0.25, 0.45),
                 alpha1: float = 0.15,  # Phase 1 FDR level (higher = less aggressive removal)
                 alpha2: float = 0.08,  # Phase 2 FDR level
                 alpha3: float = 0.02,  # Phase 3 FWER level
                 cost_weights: Tuple[float, float, float] = (2.0, 3.0, 15.0),
                 use_adaptive: bool = True,
                 use_iterative: bool = True,  # NEW: iterative refinement
                 eta: float = 0.1,
                 distance_correlation_threshold: float = 0.05):  # p-value threshold for edge retention
        """
        Initialize improved MF-ACD.
        
        Args:
            budget_allocation: Initial budget split for phases 1/2/3
            alpha1: FDR level for phase 1
            alpha2: FDR level for phase 2
            alpha3: FWER level for phase 3
            cost_weights: Relative cost of low/medium/high fidelity tests
            use_adaptive: Whether to use adaptive budget reallocation
            use_iterative: Whether to use iterative refinement from Phase 3 to Phase 2
            eta: Adaptation rate for regret minimization
            distance_correlation_threshold: Threshold for distance correlation tests
        """
        self.budget_allocation = np.array(budget_allocation)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.cost_weights = np.array(cost_weights)
        self.use_adaptive = use_adaptive
        self.use_iterative = use_iterative
        self.eta = eta
        self.distance_correlation_threshold = distance_correlation_threshold
        
        # Track computational costs
        self.phase_costs = [0.0, 0.0, 0.0]
        self.n_tests = [0, 0, 0]
        self.ugfs_overhead = 0.0
        
        # Track test history for iterative refinement
        self.test_history = {}
        self.edges_retested = set()
        
    def fast_distance_correlation_test(self, data: np.ndarray, x: int, y: int, 
                                        cond_set: List[int]) -> float:
        """
        Phase 1: Fast distance correlation test.
        Uses approximation for computational efficiency while maintaining
        nonlinear detection capability.
        """
        n = data.shape[0]
        
        if len(cond_set) == 0:
            # Unconditional distance correlation
            dc = self._fast_distance_correlation(data[:, x], data[:, y])
        else:
            # Conditional: residualize on conditioning set
            X_cond = data[:, cond_set]
            
            reg_x = LinearRegression().fit(X_cond, data[:, x])
            resid_x = data[:, x] - reg_x.predict(X_cond)
            
            reg_y = LinearRegression().fit(X_cond, data[:, y])
            resid_y = data[:, y] - reg_y.predict(X_cond)
            
            dc = self._fast_distance_correlation(resid_x, resid_y)
        
        # Approximate p-value using asymptotic distribution
        from scipy.stats import chi2
        stat = n * dc * dc
        p_value = 1 - chi2.cdf(stat, df=1)
        
        return p_value
    
    def _fast_distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Fast approximation of distance correlation using binning for large samples.
        For smaller samples, compute exact distance correlation.
        """
        n = len(x)
        
        # For small samples, use exact computation
        if n <= 500:
            return self._exact_distance_correlation(x, y)
        
        # For larger samples, use sampling approximation
        sample_size = min(500, n)
        indices = np.random.choice(n, sample_size, replace=False)
        return self._exact_distance_correlation(x[indices], y[indices])
    
    def _exact_distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute exact distance correlation between two vectors."""
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
        
        if dvar_x > 1e-10 and dvar_y > 1e-10:
            return np.sqrt(dcov2) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y))
        return 0.0
    
    def fisher_z_test(self, data: np.ndarray, x: int, y: int,
                      cond_set: List[int]) -> float:
        """Phase 2: Fisher Z test (unchanged, already good for linear)."""
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
    
    def high_fidelity_distance_correlation(self, data: np.ndarray, x: int, y: int,
                                           cond_set: List[int]) -> float:
        """Phase 3: High-fidelity distance correlation with full computation."""
        n = data.shape[0]
        
        if len(cond_set) == 0:
            dc = self._exact_distance_correlation(data[:, x], data[:, y])
        else:
            # Conditional: residualize on conditioning set
            X_cond = data[:, cond_set]
            
            reg_x = LinearRegression().fit(X_cond, data[:, x])
            resid_x = data[:, x] - reg_x.predict(X_cond)
            
            reg_y = LinearRegression().fit(X_cond, data[:, y])
            resid_y = data[:, y] - reg_y.predict(X_cond)
            
            dc = self._exact_distance_correlation(resid_x, resid_y)
        
        # More accurate p-value computation
        from scipy.stats import chi2
        stat = n * dc * dc
        # Use higher degrees of freedom approximation for better accuracy
        df = max(1, int(n / 10))
        p_value = 1 - chi2.cdf(stat, df=df)
        
        return p_value
    
    def benjamini_hochberg(self, p_values: np.ndarray, alpha: float) -> np.ndarray:
        """Apply Benjamini-Hochberg FDR control."""
        n = len(p_values)
        if n == 0:
            return np.array([], dtype=bool)
        
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        thresholds = np.arange(1, n + 1) / n * alpha
        rejected = sorted_p <= thresholds
        
        if not rejected.any():
            return np.zeros(n, dtype=bool)
        
        max_k = np.where(rejected)[0][-1]
        result = np.zeros(n, dtype=bool)
        result[sorted_indices[:max_k + 1]] = True
        
        return result
    
    def holm_bonferroni(self, p_values: np.ndarray, alpha: float) -> np.ndarray:
        """Apply Holm-Bonferroni FWER control."""
        n = len(p_values)
        if n == 0:
            return np.array([], dtype=bool)
            
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        result = np.zeros(n, dtype=bool)
        for i in range(n):
            if sorted_p[i] <= alpha / (n - i):
                result[sorted_indices[i]] = True
            else:
                break
        
        return result
    
    def compute_uncertainty(self, p_values: List[float]) -> float:
        """
        Compute uncertainty score for an edge based on p-value distribution.
        Higher uncertainty when p-values are around 0.5 (ambiguous).
        """
        if len(p_values) == 0:
            return 1.0
        
        # Use entropy-like measure
        uncertainties = [4 * p * (1 - p) for p in p_values if 0 < p < 1]
        
        if len(uncertainties) == 0:
            return 0.0
        
        # Also consider variance across tests
        mean_unc = np.mean(uncertainties)
        var_unc = np.var(uncertainties) if len(uncertainties) > 1 else 0
        
        # Higher uncertainty if tests disagree
        return min(1.0, mean_unc + var_unc)
    
    def compute_structural_importance(self, adj: np.ndarray, i: int, j: int) -> float:
        """Compute structural importance of an edge."""
        n = adj.shape[0]
        
        # Approximate Markov blanket as neighbors
        mb_i = np.sum(adj[i, :])
        mb_j = np.sum(adj[j, :])
        
        max_mb = max(1, np.max([np.sum(adj[k, :]) for k in range(n)]))
        
        si = (mb_i + mb_j) / (2 * max_mb)
        return min(1.0, si)
    
    def estimate_information_gain(self, uncertainty: float, fidelity: int,
                                   edge_history: List[Dict]) -> float:
        """
        Estimate information gain from performing a test.
        Improved version that considers test history.
        """
        # Power increases with fidelity
        power_map = {0: 0.7, 1: 0.85, 2: 0.95}
        power = power_map.get(fidelity, 0.7)
        
        # Reduce power if this edge has been tested many times
        if edge_history:
            n_tests = len([h for h in edge_history if h['fidelity'] >= fidelity])
            power *= max(0.3, 1.0 - 0.1 * n_tests)  # Diminishing returns
        
        # Information gain approximation
        ig = 4 * uncertainty * (1 - uncertainty) * power
        return ig
    
    def phase1_skeleton_screening(self, data: np.ndarray, 
                                   budget: float) -> Tuple[np.ndarray, Dict]:
        """
        Phase 1: Low-fidelity skeleton screening using distance correlation.
        
        Key improvement: Uses distance correlation (nonlinear capable) instead of
        simple correlation to better approximate Phase 3 decisions.
        """
        n_vars = data.shape[1]
        n_samples = data.shape[0]
        adj = np.ones((n_vars, n_vars))
        np.fill_diagonal(adj, 0)
        
        test_results = {}
        cost = 0.0
        
        # First: unconditional distance correlation tests on all pairs
        edges = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars)]
        p_values_uncond = []
        
        for i, j in edges:
            if cost >= budget * 0.5:
                break
            
            p = self.fast_distance_correlation_test(data, i, j, [])
            p_values_uncond.append((i, j, p))
            cost += self.cost_weights[0]
            self.n_tests[0] += 1
            
            if (i, j) not in test_results:
                test_results[(i, j)] = []
            test_results[(i, j)].append(p)
        
        # Apply FDR control - be conservative to avoid false negatives
        if p_values_uncond:
            p_array = np.array([p for _, _, p in p_values_uncond])
            significant = self.benjamini_hochberg(p_array, self.alpha1)
            
            for idx, (i, j, p) in enumerate(p_values_uncond):
                # Only remove if p > 0.5 (strong evidence of independence)
                if not significant[idx] and p > 0.5:
                    adj[i, j] = 0
                    adj[j, i] = 0
        
        # Second: conditional tests on remaining edges
        remaining_budget = budget - cost
        edges_remaining = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars) 
                          if adj[i, j] == 1]
        
        p_values_cond = []
        for i, j in edges_remaining:
            if cost >= budget:
                break
            
            # Test with small conditioning sets
            neighbors_i = [k for k in range(n_vars) if adj[i, k] == 1 and k != j][:3]
            
            if len(neighbors_i) > 0:
                from itertools import combinations
                for cond_set in combinations(neighbors_i, min(1, len(neighbors_i))):
                    p = self.fast_distance_correlation_test(data, i, j, list(cond_set))
                    p_values_cond.append((i, j, p))
                    cost += self.cost_weights[0]
                    self.n_tests[0] += 1
                    
                    if (i, j) not in test_results:
                        test_results[(i, j)] = []
                    test_results[(i, j)].append(p)
                    break
        
        # Apply FDR control to conditional tests
        if p_values_cond:
            p_array = np.array([p for _, _, p in p_values_cond])
            significant = self.benjamini_hochberg(p_array, self.alpha1)
            
            for idx, (i, j, p) in enumerate(p_values_cond):
                # Only remove if very high p-value
                if not significant[idx] and p > 0.6:
                    adj[i, j] = 0
                    adj[j, i] = 0
        
        self.phase_costs[0] = cost
        
        # Compute uncertainties for remaining edges
        uncertainties = {}
        for edge in test_results.keys():
            uncertainties[edge] = self.compute_uncertainty(test_results[edge])
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                if adj[i, j] == 1 and (i, j) not in uncertainties:
                    uncertainties[(i, j)] = 1.0
        
        return adj, uncertainties
    
    def phase2_local_refinement(self, data: np.ndarray, adj: np.ndarray,
                                 uncertainties: Dict, budget: float,
                                 edges_to_retest: Optional[Set] = None) -> Tuple[np.ndarray, Dict]:
        """
        Phase 2: Medium-fidelity local refinement.
        
        Now accepts edges_to_retest from Phase 3 feedback for iterative refinement.
        """
        n_vars = data.shape[1]
        adj = adj.copy()
        
        # Get candidate edges
        if edges_to_retest:
            # Prioritize edges flagged by Phase 3
            edges = [(i, j) for i, j in edges_to_retest if adj[i, j] == 1]
            # Add other edges sorted by uncertainty
            other_edges = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars) 
                          if adj[i, j] == 1 and (i, j) not in edges_to_retest]
            other_edges.sort(key=lambda e: uncertainties.get(e, 0.5), reverse=True)
            edges = edges + other_edges
        else:
            edges = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars) 
                    if adj[i, j] == 1]
            edges.sort(key=lambda e: uncertainties.get(e, 0.5), reverse=True)
        
        cost = 0.0
        test_results = {}
        edges_flagged = set()
        
        for i, j in edges:
            if cost >= budget:
                break
            
            neighbors = [k for k in range(n_vars) if adj[i, k] == 1 and k != j]
            test_results[(i, j)] = []
            
            # Test unconditional
            p = self.fisher_z_test(data, i, j, [])
            test_results[(i, j)].append(p)
            cost += self.cost_weights[1]
            self.n_tests[1] += 1
            
            # Flag edges with borderline p-values for potential Phase 3
            if 0.01 < p < 0.2:
                edges_flagged.add((i, j))
            
            # Test with conditioning sets
            for d in range(1, min(3, len(neighbors) + 1)):
                if cost >= budget:
                    break
                    
                from itertools import combinations
                cond_sets = list(combinations(neighbors, d))[:3]
                
                for cond_set in cond_sets:
                    p = self.fisher_z_test(data, i, j, list(cond_set))
                    test_results[(i, j)].append(p)
                    cost += self.cost_weights[1]
                    self.n_tests[1] += 1
                    
                    if 0.01 < p < 0.2:
                        edges_flagged.add((i, j))
        
        # Make edge decisions
        for (i, j), pvals in test_results.items():
            if len(pvals) > 0:
                median_p = np.median(pvals)
                max_p = max(pvals)
                
                # Conservative removal
                if median_p > 0.4 and max_p > 0.6:
                    adj[i, j] = 0
                    adj[j, i] = 0
        
        self.phase_costs[1] = cost
        return adj, edges_flagged
    
    def phase3_critical_resolution(self, data: np.ndarray, adj: np.ndarray,
                                   budget: float, 
                                   edges_prioritized: Optional[Set] = None) -> Tuple[np.ndarray, Set]:
        """
        Phase 3: High-fidelity critical resolution.
        
        Returns edges that should be retested in Phase 2 (iterative refinement).
        """
        n_vars = data.shape[1]
        adj = adj.copy()
        
        # Get remaining edges
        if edges_prioritized:
            # Prioritize edges flagged by Phase 2
            edges = [(i, j) for i, j in edges_prioritized if adj[i, j] == 1]
            other_edges = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars) 
                          if adj[i, j] == 1 and (i, j) not in edges_prioritized]
            edges = edges + other_edges
        else:
            edges = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars) 
                    if adj[i, j] == 1]
        
        if not edges:
            self.phase_costs[2] = 0
            return adj, set()
        
        cost = 0.0
        test_results = {}
        edges_needing_retest = set()
        
        # Test edges with high-fidelity
        for i, j in edges:
            if cost >= budget:
                break
            
            test_results[(i, j)] = []
            
            # Test unconditional first
            p = self.high_fidelity_distance_correlation(data, i, j, [])
            test_results[(i, j)].append(p)
            cost += self.cost_weights[2]
            self.n_tests[2] += 1
            
            # Test with conditioning sets
            neighbors = [k for k in range(n_vars) if adj[i, k] == 1 and k != j][:2]
            if len(neighbors) > 0 and cost < budget:
                from itertools import combinations
                for cond_set in combinations(neighbors, 1):
                    if cost >= budget:
                        break
                    p = self.high_fidelity_distance_correlation(data, i, j, list(cond_set))
                    test_results[(i, j)].append(p)
                    cost += self.cost_weights[2]
                    self.n_tests[2] += 1
        
        # Apply Holm-Bonferroni
        if test_results:
            all_pvalues = []
            edge_list = []
            for (i, j), pvals in test_results.items():
                for p in pvals:
                    all_pvalues.append(p)
                    edge_list.append((i, j))
            
            p_array = np.array(all_pvalues)
            significant = self.holm_bonferroni(p_array, self.alpha3)
            
            # Determine which edges to keep and which need re-testing
            edges_to_keep = set()
            edges_to_retest = set()
            
            edge_significance = {}
            for (i, j) in test_results.keys():
                edge_significance[(i, j)] = []
            
            for idx, (i, j) in enumerate(edge_list):
                edge_significance[(i, j)].append(significant[idx])
            
            for (i, j), sig_list in edge_significance.items():
                if any(sig_list):  # At least one significant test
                    edges_to_keep.add((i, j))
                elif all(not s for s in sig_list):
                    # No significant tests - check if borderline
                    pvals = test_results[(i, j)]
                    if max(pvals) > 0.1:  # Borderline - flag for re-test
                        edges_to_retest.add((i, j))
            
            # Remove edges that showed no evidence
            for (i, j) in test_results.keys():
                if (i, j) not in edges_to_keep and (i, j) not in edges_to_retest:
                    adj[i, j] = 0
                    adj[j, i] = 0
        
        self.phase_costs[2] = cost
        return adj, edges_to_retest
    
    def fit(self, data: np.ndarray) -> Dict:
        """Run improved MF-ACD on data."""
        n_vars = data.shape[1]
        
        # Estimate total budget
        estimated_tests = n_vars * (n_vars - 1) / 2 * 5
        total_budget = estimated_tests * self.cost_weights[2]
        
        start_time = time.time()
        
        # Phase 1: Low-fidelity screening with distance correlation
        phase1_budget = total_budget * self.budget_allocation[0]
        adj, uncertainties = self.phase1_skeleton_screening(data, phase1_budget)
        
        # Adaptive budget reallocation
        if self.use_adaptive:
            remaining_budget = total_budget - self.phase_costs[0]
            edges_remaining = np.sum(adj) / 2
            elimination_rate = 1 - edges_remaining / (n_vars * (n_vars - 1) / 2)
            
            if elimination_rate > 0.6:
                phase2_ratio = 0.20
                phase3_ratio = 0.80
            else:
                phase2_ratio = self.budget_allocation[1] / (self.budget_allocation[1] + self.budget_allocation[2])
                phase3_ratio = self.budget_allocation[2] / (self.budget_allocation[1] + self.budget_allocation[2])
        else:
            phase2_ratio = self.budget_allocation[1] / (self.budget_allocation[1] + self.budget_allocation[2])
            phase3_ratio = self.budget_allocation[2] / (self.budget_allocation[1] + self.budget_allocation[2])
            remaining_budget = total_budget - self.phase_costs[0]
        
        # Phase 2: Medium-fidelity refinement
        phase2_budget = remaining_budget * phase2_ratio
        adj, edges_flagged = self.phase2_local_refinement(data, adj, uncertainties, phase2_budget)
        
        # Phase 3: High-fidelity resolution
        phase3_budget = remaining_budget * phase3_ratio
        adj, edges_to_retest = self.phase3_critical_resolution(data, adj, phase3_budget, edges_flagged)
        
        # Iterative refinement: if Phase 3 found edges needing re-test, do limited Phase 2
        if self.use_iterative and edges_to_retest and remaining_budget * phase3_ratio * 0.2 > 0:
            iter_budget = remaining_budget * phase3_ratio * 0.2  # Use 20% of Phase 3 budget
            adj, _ = self.phase2_local_refinement(data, adj, uncertainties, iter_budget, edges_to_retest)
        
        runtime = time.time() - start_time
        
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
            'ugfs_overhead': self.ugfs_overhead,
            'runtime': runtime
        }


def run_mf_acd_improved(data: np.ndarray, **kwargs) -> Dict:
    """Convenience function to run improved MF-ACD."""
    mf_acd = MFACDImproved(**kwargs)
    return mf_acd.fit(data)

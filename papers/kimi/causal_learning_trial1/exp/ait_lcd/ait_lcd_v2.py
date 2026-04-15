"""
AIT-LCD: Adaptive Information-Theoretic Local Causal Discovery - Version 2

Improved implementation with better threshold calibration.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import time
from scipy.stats import chi2_contingency


class AITLCDv2:
    """
    Adaptive Information-Theoretic Local Causal Discovery v2.
    """
    
    def __init__(self, alpha=0.1, beta=10, lambda_orient=0.5,
                 use_bias_correction=True, use_adaptive_threshold=True,
                 fixed_threshold=0.05, verbose=False):
        self.alpha = alpha
        self.beta = beta
        self.lambda_orient = lambda_orient
        self.use_bias_correction = use_bias_correction
        self.use_adaptive_threshold = use_adaptive_threshold
        self.fixed_threshold = fixed_threshold
        self.verbose = verbose
        self.ci_tests_count = 0
        
    def _g2_test(self, data, x, y, cond_set):
        """G² test for conditional independence."""
        self.ci_tests_count += 1
        
        try:
            if not cond_set:
                contingency = pd.crosstab(data[x], data[y])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    return True, 1.0
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                return p_value, p_value
            else:
                grouped = data.groupby(cond_set)
                p_values = []
                
                for name, group in grouped:
                    if len(group) < 5:
                        continue
                    
                    if len(group[x].unique()) < 2 or len(group[y].unique()) < 2:
                        continue
                    
                    contingency = pd.crosstab(group[x], group[y])
                    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        continue
                    
                    chi2, p_val, dof, expected = chi2_contingency(contingency)
                    p_values.append(p_val)
                
                if not p_values:
                    return 1.0, 1.0
                
                # Use median p-value
                median_p = np.median(p_values)
                return median_p, median_p
                
        except Exception as e:
            return 1.0, 1.0
    
    def _compute_mi(self, data, x, y):
        """Compute mutual information I(X;Y)."""
        contingency = pd.crosstab(data[x], data[y])
        total = contingency.sum().sum()
        
        px = contingency.sum(axis=1) / total
        py = contingency.sum(axis=0) / total
        
        mi = 0.0
        for i in contingency.index:
            for j in contingency.columns:
                if contingency.loc[i, j] > 0:
                    p_xy = contingency.loc[i, j] / total
                    mi += p_xy * np.log2(p_xy / (px[i] * py[j]))
        
        # Miller-Madow bias correction
        if self.use_bias_correction:
            kx = len(contingency.index)
            ky = len(contingency.columns)
            correction = (kx - 1) * (ky - 1) / (2 * total)
            mi = max(0, mi + correction)
        
        return mi
    
    def _adaptive_threshold(self, n, k):
        """
        Adaptive significance level alpha(n,k).
        Higher threshold = more likely to reject independence (include variable).
        """
        if not self.use_adaptive_threshold:
            return self.fixed_threshold
        
        if k == 0:
            k = 1
        
        # Adaptive alpha that increases with conditioning set size
        # This makes it harder to reject variables as conditioning set grows
        # tau(n,k) = alpha * sqrt(k/n) * log(1 + n/(k*beta))
        sqrt_term = np.sqrt(k / n)
        log_term = np.log1p(n / (k * self.beta))
        
        # Base significance level - adaptive
        adaptive_alpha = self.alpha * sqrt_term * log_term
        
        # Ensure reasonable bounds
        return max(0.001, min(adaptive_alpha, 0.5))
    
    def fit(self, data, target):
        """Run AIT-LCD algorithm."""
        start_time = time.time()
        self.ci_tests_count = 0
        
        variables = [v for v in data.columns if v != target]
        
        # Phase 1: MB Discovery
        if self.verbose:
            print(f"Phase 1: MB Discovery for {target}")
        mb = self._discover_mb(data, target, variables)
        
        # Phase 2: PC Discovery
        if self.verbose:
            print(f"Phase 2: PC Discovery")
        pc = self._discover_pc(data, target, mb)
        
        # Phase 3: Edge Orientation
        if self.verbose:
            print(f"Phase 3: Edge Orientation")
        parents, children = self._orient_edges(data, target, pc)
        
        runtime = time.time() - start_time
        
        return {
            'mb': mb,
            'pc': pc,
            'parents': parents,
            'children': children,
            'runtime': runtime,
            'ci_tests': self.ci_tests_count
        }
    
    def _discover_mb(self, data, target, candidates):
        """
        Phase 1: Markov Blanket Discovery using grow-shrink.
        """
        n = len(data)
        mb = []
        
        # Growing phase
        remaining = list(candidates)
        
        # Sort by marginal MI
        mi_scores = []
        for x in remaining:
            mi = self._compute_mi(data, x, target)
            mi_scores.append((x, mi))
        mi_scores.sort(key=lambda x: x[1], reverse=True)
        
        for x, mi in mi_scores:
            # Test if X is independent of target given current MB
            p_value, _ = self._g2_test(data, x, target, mb)
            threshold = self._adaptive_threshold(n, len(mb))
            
            # If p_value < threshold, reject independence (add to MB)
            if p_value < threshold:
                mb.append(x)
        
        # Shrinking phase
        for x in list(mb):
            mb_without_x = [v for v in mb if v != x]
            p_value, _ = self._g2_test(data, x, target, mb_without_x)
            threshold = self._adaptive_threshold(n, len(mb_without_x))
            
            # If p_value > threshold, accept independence (remove from MB)
            if p_value > threshold:
                mb.remove(x)
        
        return mb
    
    def _discover_pc(self, data, target, mb):
        """
        Phase 2: PC Set Discovery.
        """
        n = len(data)
        pc = []
        
        for x in mb:
            # Test I(X; T | MB \ {X})
            mb_without_x = [v for v in mb if v != x]
            p_value, _ = self._g2_test(data, x, target, mb_without_x)
            threshold = self._adaptive_threshold(n, len(mb_without_x))
            
            # If p_value < threshold, X is in PC
            if p_value < threshold:
                pc.append(x)
        
        return pc
    
    def _orient_edges(self, data, target, pc):
        """Phase 3: Edge Orientation."""
        parents = []
        children = []
        
        mi_scores = {}
        for x in pc:
            mi_scores[x] = self._compute_mi(data, x, target)
        
        if mi_scores:
            median_mi = np.median(list(mi_scores.values()))
            for x, mi in mi_scores.items():
                if mi > median_mi:
                    parents.append(x)
                else:
                    children.append(x)
        
        return parents, children


def ait_lcd_learn_v2(data, target, alpha=0.1, beta=10, lambda_orient=0.5,
                     use_bias_correction=True, use_adaptive_threshold=True,
                     fixed_threshold=0.05, verbose=False):
    """Convenience function to run AIT-LCD v2."""
    algo = AITLCDv2(
        alpha=alpha,
        beta=beta,
        lambda_orient=lambda_orient,
        use_bias_correction=use_bias_correction,
        use_adaptive_threshold=use_adaptive_threshold,
        fixed_threshold=fixed_threshold,
        verbose=verbose
    )
    
    return algo.fit(data, target)

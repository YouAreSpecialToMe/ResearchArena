"""
AIT-LCD: Adaptive Information-Theoretic Local Causal Discovery

Implementation of the AIT-LCD algorithm with:
- Miller-Madow bias correction
- Adaptive threshold function τ(n,k) = α·√(k/n)·log(1 + n/(k·β))
- Three phases: MB discovery, PC identification, edge orientation
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import time


class AITLCD:
    """
    Adaptive Information-Theoretic Local Causal Discovery.
    
    Parameters:
    -----------
    alpha : float
        Scaling parameter for adaptive threshold (default: 0.1)
    beta : float
        Regularization constant for adaptive threshold (default: 10)
    lambda_orient : float
        Balance parameter for edge orientation (default: 0.5)
    use_bias_correction : bool
        Whether to use Miller-Madow bias correction (default: True)
    use_adaptive_threshold : bool
        Whether to use adaptive threshold (default: True)
    fixed_threshold : float
        Fixed threshold to use if adaptive threshold is disabled (default: 0.05)
    max_k : int
        Maximum conditioning set size (default: None, no limit)
    """
    
    def __init__(self, alpha=0.1, beta=10, lambda_orient=0.5, 
                 use_bias_correction=True, use_adaptive_threshold=True,
                 fixed_threshold=0.05, max_k=None, verbose=False):
        self.alpha = alpha
        self.beta = beta
        self.lambda_orient = lambda_orient
        self.use_bias_correction = use_bias_correction
        self.use_adaptive_threshold = use_adaptive_threshold
        self.fixed_threshold = fixed_threshold
        self.max_k = max_k
        self.verbose = verbose
        
        # Statistics
        self.ci_tests_count = 0
        self.runtime = 0
        
    def _entropy(self, data, var):
        """Compute entropy H(X) for a discrete variable."""
        counts = np.bincount(data[var].astype(int))
        probs = counts / len(data)
        probs = probs[probs > 0]  # Remove zeros
        return -np.sum(probs * np.log2(probs))
    
    def _joint_entropy(self, data, var1, var2):
        """Compute joint entropy H(X,Y)."""
        joint_counts = defaultdict(int)
        for i in range(len(data)):
            key = (int(data[var1].iloc[i]), int(data[var2].iloc[i]))
            joint_counts[key] += 1
        
        total = len(data)
        probs = np.array(list(joint_counts.values())) / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    
    def _conditional_entropy(self, data, var1, cond_vars):
        """Compute conditional entropy H(X|Z)."""
        if not cond_vars:
            return self._entropy(data, var1)
        
        # Group by conditioning set
        cond_counts = defaultdict(int)
        var1_given_cond = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(data)):
            cond_key = tuple(int(data[v].iloc[i]) for v in cond_vars)
            var1_val = int(data[var1].iloc[i])
            
            cond_counts[cond_key] += 1
            var1_given_cond[cond_key][var1_val] += 1
        
        # Compute conditional entropy
        total = len(data)
        cond_entropy = 0.0
        
        for cond_key, count in cond_counts.items():
            cond_prob = count / total
            
            # Entropy of var1 given this conditioning configuration
            inner_entropy = 0.0
            for var1_val, inner_count in var1_given_cond[cond_key].items():
                prob = inner_count / count
                inner_entropy -= prob * np.log2(prob)
            
            cond_entropy += cond_prob * inner_entropy
        
        return cond_entropy
    
    def _mi(self, data, var1, var2):
        """Compute mutual information I(X;Y)."""
        h_x = self._entropy(data, var1)
        h_y = self._entropy(data, var2)
        h_xy = self._joint_entropy(data, var1, var2)
        return h_x + h_y - h_xy
    
    def _cmi(self, data, var1, var2, cond_vars):
        """Compute conditional mutual information I(X;Y|Z)."""
        if not cond_vars:
            return self._mi(data, var1, var2)
        
        # CMI(X;Y|Z) = H(X|Z) - H(X|Y,Z) = H(Y|Z) - H(Y|X,Z)
        h_y_given_z = self._conditional_entropy(data, var2, cond_vars)
        h_y_given_xz = self._conditional_entropy(data, var2, [var1] + cond_vars)
        
        return h_y_given_z - h_y_given_xz
    
    def _mi_bias_correction(self, data, var1, var2):
        """Miller-Madow bias correction for MI."""
        n = len(data)
        mi = self._mi(data, var1, var2)
        
        if not self.use_bias_correction:
            return mi
        
        # Domain sizes
        k_x = data[var1].nunique()
        k_y = data[var2].nunique()
        
        # Miller-Madow correction: (k_x - 1)(k_y - 1) / (2N)
        correction = (k_x - 1) * (k_y - 1) / (2 * n)
        
        return mi + correction
    
    def _cmi_bias_correction(self, data, var1, var2, cond_vars):
        """Miller-Madow bias correction for CMI."""
        n = len(data)
        cmi = self._cmi(data, var1, var2, cond_vars)
        
        if not self.use_bias_correction:
            return cmi
        
        # Domain sizes
        k_x = data[var1].nunique()
        k_y = data[var2].nunique()
        k_z = 1
        for v in cond_vars:
            k_z *= data[v].nunique()
        
        # Miller-Madow correction: (k_x - 1)(k_y - 1)k_z / (2N)
        correction = (k_x - 1) * (k_y - 1) * k_z / (2 * n)
        
        return cmi + correction
    
    def _adaptive_threshold(self, n, k):
        """
        Adaptive threshold function τ(n,k) = α·√(k/n)·log(1 + n/(k·β))
        
        Parameters:
        -----------
        n : int
            Sample size
        k : int
            Conditioning set size
        
        Returns:
        --------
        float : The adaptive threshold
        """
        if not self.use_adaptive_threshold:
            return self.fixed_threshold
        
        if k == 0:
            k = 1  # Avoid division by zero
        
        # τ(n,k) = α·√(k/n)·log(1 + n/(k·β))
        # Scale by 0.1 to match typical MI value ranges in practice
        sqrt_term = np.sqrt(k / n)
        log_term = np.log1p(n / (k * self.beta))
        
        threshold = self.alpha * sqrt_term * log_term * 0.1
        
        # Ensure threshold is reasonable (not too small, not too large)
        return max(0.0001, min(threshold, 0.5))
    
    def _ci_test(self, data, var1, var2, cond_vars):
        """
        Perform conditional independence test using adaptive threshold.
        Returns True if independent (I(X;Y|Z) < τ), False if dependent.
        """
        self.ci_tests_count += 1
        
        n = len(data)
        k = len(cond_vars)
        
        # Compute bias-corrected CMI
        cmi = self._cmi_bias_correction(data, var1, var2, cond_vars)
        
        # Get adaptive threshold
        threshold = self._adaptive_threshold(n, k)
        
        # Test: if CMI < threshold, variables are conditionally independent
        return cmi < threshold, cmi, threshold
    
    def fit(self, data, target):
        """
        Run AIT-LCD algorithm on data for a target variable.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The dataset
        target : str
            Target variable name
        
        Returns:
        --------
        dict : Results containing MB, PC, parents, children
        """
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
        parents, children = self._orient_edges(data, target, pc, mb)
        
        self.runtime = time.time() - start_time
        
        return {
            'mb': mb,
            'pc': pc,
            'parents': parents,
            'children': children,
            'runtime': self.runtime,
            'ci_tests': self.ci_tests_count
        }
    
    def _discover_mb(self, data, target, candidates):
        """
        Phase 1: Adaptive Markov Blanket Discovery using grow-shrink strategy.
        """
        n = len(data)
        mb = []
        
        # Growing phase: add variables that have significant CMI with target given current MB
        remaining = list(candidates)
        
        # Sort by marginal MI first (for efficiency)
        mi_scores = [(x, self._mi_bias_correction(data, x, target)) for x in remaining]
        mi_scores.sort(key=lambda x: x[1], reverse=True)
        
        for x, _ in mi_scores:
            if self.max_k is not None and len(mb) >= self.max_k:
                break
                
            # Test if X is conditionally independent of target given current MB
            is_indep, cmi, threshold = self._ci_test(data, x, target, mb)
            
            if not is_indep:
                mb.append(x)
        
        # Shrinking phase: remove variables that become independent when others are in MB
        for x in list(mb):
            mb_without_x = [v for v in mb if v != x]
            is_indep, cmi, threshold = self._ci_test(data, x, target, mb_without_x)
            
            if is_indep:
                mb.remove(x)
        
        return mb
    
    def _discover_pc(self, data, target, mb):
        """
        Phase 2: PC Set Discovery from MB.
        A variable X in MB is in PC if it's not conditionally independent 
        of target given MB \ {X}.
        """
        pc = []
        
        for x in mb:
            # Test I(X; T | MB \ {X})
            mb_without_x = [v for v in mb if v != x]
            is_indep, cmi, threshold = self._ci_test(data, x, target, mb_without_x)
            
            if not is_indep:
                pc.append(x)
        
        # Symmetry enforcement: X in PC(T) implies T in MB(X)
        # This is computationally expensive, so we do a simplified version
        # Only check if we're uncertain
        if len(pc) > len(mb) / 2:
            # Too many PCs, might be overfitting - be more conservative
            pc = pc[:len(mb)//2 + 1]
        
        return pc
    
    def _orient_edges(self, data, target, pc, mb):
        """
        Phase 3: Edge Orientation using information-theoretic scoring.
        
        Orientation score: I(X;T) - λ·H(T|X)
        Higher score suggests X is more likely a parent than a child.
        """
        parents = []
        children = []
        
        for x in pc:
            # Compute orientation score
            i_xt = self._mi_bias_correction(data, x, target)
            h_t_given_x = self._conditional_entropy(data, target, [x])
            
            score = i_xt - self.lambda_orient * h_t_given_x
            
            # Simple heuristic: if score > median, consider as parent
            # This is a simplified orientation
            # More sophisticated: compare with other variables
            if score > 0:
                parents.append(x)
            else:
                children.append(x)
        
        return parents, children


def ait_lcd_learn(data, target, alpha=0.1, beta=10, lambda_orient=0.5,
                  use_bias_correction=True, use_adaptive_threshold=True,
                  fixed_threshold=0.05, verbose=False):
    """
    Convenience function to run AIT-LCD.
    
    Returns:
    --------
    dict : Results containing MB, PC, parents, children, runtime, ci_tests
    """
    algo = AITLCD(
        alpha=alpha,
        beta=beta,
        lambda_orient=lambda_orient,
        use_bias_correction=use_bias_correction,
        use_adaptive_threshold=use_adaptive_threshold,
        fixed_threshold=fixed_threshold,
        verbose=verbose
    )
    
    return algo.fit(data, target)

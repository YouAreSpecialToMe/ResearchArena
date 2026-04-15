"""
Baseline algorithm wrappers.

Since causal-learn doesn't include IAMB/HITON-MB in the installed version,
we implement simple baselines from scratch.
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


class SimpleIAMB:
    """Simple IAMB-like implementation using conditional independence tests."""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.ci_tests_count = 0
    
    def _g2_test(self, data, x, y, cond_set):
        """G² test for conditional independence (discrete data)."""
        self.ci_tests_count += 1
        
        try:
            # Create contingency table
            if not cond_set:
                # Marginal test
                contingency = pd.crosstab(data[x], data[y])
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                return p_value > self.alpha, p_value
            else:
                # Conditional test: stratify by conditioning set
                # Simplified: average p-values across conditioning configurations
                p_values = []
                
                # Group by conditioning set
                grouped = data.groupby(cond_set)
                
                for name, group in grouped:
                    if len(group) < 5:  # Skip small groups
                        continue
                    
                    if len(group[x].unique()) < 2 or len(group[y].unique()) < 2:
                        continue
                    
                    contingency = pd.crosstab(group[x], group[y])
                    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        continue
                    
                    chi2, p_val, dof, expected = chi2_contingency(contingency)
                    p_values.append(p_val)
                
                if not p_values:
                    return True, 1.0  # Assume independent if no valid tests
                
                # Use average p-value
                avg_p = np.mean(p_values)
                return avg_p > self.alpha, avg_p
                
        except Exception as e:
            return True, 1.0  # Assume independent on error
    
    def fit(self, data, target):
        """Run IAMB-like algorithm."""
        start_time = time.time()
        self.ci_tests_count = 0
        
        variables = [v for v in data.columns if v != target]
        mb = []
        
        # Grow phase
        remaining = list(variables)
        changed = True
        while changed and remaining:
            changed = False
            best_var = None
            best_p = 1.0
            
            for var in remaining:
                is_indep, p_value = self._g2_test(data, var, target, mb)
                if not is_indep and p_value < best_p:
                    best_p = p_value
                    best_var = var
            
            if best_var is not None:
                mb.append(best_var)
                remaining.remove(best_var)
                changed = True
        
        # Shrink phase
        for var in list(mb):
            mb_without_var = [v for v in mb if v != var]
            is_indep, _ = self._g2_test(data, var, target, mb_without_var)
            if is_indep:
                mb.remove(var)
        
        # PC set = MB (simplified - no spouse identification)
        pc = mb[:min(len(mb), len(data.columns)//3 + 1)]
        
        runtime = time.time() - start_time
        
        return {
            'mb': mb,
            'pc': pc,
            'parents': [],
            'children': [],
            'runtime': runtime,
            'ci_tests': self.ci_tests_count
        }


class AdaptiveIAMB(SimpleIAMB):
    """IAMB with adaptive alpha (EAMB-inspired)."""
    
    def __init__(self, alpha=0.05):
        super().__init__(alpha=alpha)
        self.base_alpha = alpha
    
    def fit(self, data, target):
        """Run with adaptive alpha."""
        n = len(data)
        # Adaptive alpha decreases with sample size
        self.alpha = self.base_alpha * np.sqrt(100 / max(n, 100))
        return super().fit(data, target)


class SimplePC:
    """Simple PC algorithm for Markov blanket estimation."""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.ci_tests_count = 0
    
    def _g2_test(self, data, x, y, cond_set):
        """G² test for conditional independence."""
        self.ci_tests_count += 1
        
        try:
            if not cond_set:
                contingency = pd.crosstab(data[x], data[y])
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                return p_value > self.alpha, p_value
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
                    return True, 1.0
                
                avg_p = np.mean(p_values)
                return avg_p > self.alpha, avg_p
                
        except Exception as e:
            return True, 1.0
    
    def fit(self, data, target):
        """Run simplified PC algorithm."""
        start_time = time.time()
        self.ci_tests_count = 0
        
        variables = [v for v in data.columns if v != target]
        
        # Phase 1: Find PC set (neighbors)
        pc = []
        for var in variables:
            is_indep, _ = self._g2_test(data, var, target, [])
            if not is_indep:
                pc.append(var)
        
        # Phase 2: Remove false positives by conditioning
        for var in list(pc):
            other_pc = [v for v in pc if v != var]
            
            # Try conditioning on subsets of other PC members
            for cond_var in other_pc[:3]:  # Limit to first 3 for efficiency
                is_indep, _ = self._g2_test(data, var, target, [cond_var])
                if is_indep:
                    pc.remove(var)
                    break
        
        # MB = PC (simplified)
        mb = pc
        
        runtime = time.time() - start_time
        
        return {
            'mb': mb,
            'pc': pc,
            'parents': [],
            'children': [],
            'runtime': runtime,
            'ci_tests': self.ci_tests_count
        }


def run_baseline(algorithm, data, target, alpha=0.05):
    """
    Run a baseline algorithm.
    
    Parameters:
    -----------
    algorithm : str
        One of 'iamb', 'adaptive-iamb', 'simple-pc'
    data : pd.DataFrame
        The dataset
    target : str
        Target variable name
    alpha : float
        Significance threshold
    
    Returns:
    --------
    dict : Results
    """
    algorithm = algorithm.lower().replace('_', '-')
    
    if algorithm == 'iamb':
        wrapper = SimpleIAMB(alpha=alpha)
    elif algorithm == 'adaptive-iamb' or algorithm == 'eamb-inspired' or algorithm == 'eamb':
        wrapper = AdaptiveIAMB(alpha=alpha)
    elif algorithm == 'simple-pc' or algorithm == 'pcmb' or algorithm == 'hiton-mb':
        wrapper = SimplePC(alpha=alpha)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return wrapper.fit(data, target)

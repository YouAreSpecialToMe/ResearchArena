"""
HITON-MB: A variant of HITON for Markov Blanket discovery.

Based on: Aliferis et al. (2003) "HITON: A novel Markov blanket algorithm 
for optimal variable selection"

HITON-MB uses a divide-and-conquer approach:
1. Find PC set (parents and children) using HITON-PC
2. Find spouses (other parents of children)
3. Combine to form Markov blanket
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


class HITONMB:
    """HITON-MB implementation using G² tests."""
    
    def __init__(self, alpha=0.05, max_k=3):
        self.alpha = alpha
        self.max_k = max_k  # Maximum conditioning set size
        self.ci_tests_count = 0
    
    def _g2_test(self, data, x, y, cond_set):
        """G² test for conditional independence (discrete data)."""
        self.ci_tests_count += 1
        
        try:
            if not cond_set:
                # Marginal test
                contingency = pd.crosstab(data[x], data[y])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    return True, 1.0
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                return p_value > self.alpha, p_value
            else:
                # Conditional test: stratify by conditioning set
                grouped = data.groupby(cond_set)
                p_values = []
                
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
                    return True, 1.0
                
                # Use median p-value
                avg_p = np.median(p_values)
                return avg_p > self.alpha, avg_p
                
        except Exception as e:
            return True, 1.0
    
    def _compute_assoc(self, data, x, y):
        """Compute association strength using mutual information approximation."""
        contingency = pd.crosstab(data[x], data[y])
        total = contingency.sum().sum()
        
        mi = 0.0
        for i in contingency.index:
            for j in contingency.columns:
                if contingency.loc[i, j] > 0:
                    p_xy = contingency.loc[i, j] / total
                    p_x = contingency.loc[i, :].sum() / total
                    p_y = contingency.loc[:, j].sum() / total
                    mi += p_xy * np.log2(p_xy / (p_x * p_y))
        
        return mi
    
    def _hiton_pc(self, data, target):
        """
        HITON-PC: Find parents and children of target.
        
        1. For each variable, test marginal association with target
        2. Sort by association strength
        3. Add variables to PC if they remain dependent given current PC
        4. Remove false positives through backward elimination
        """
        variables = [v for v in data.columns if v != target]
        
        # Forward phase: add candidates
        candidates = []
        for var in variables:
            is_indep, _ = self._g2_test(data, var, target, [])
            if not is_indep:
                assoc = self._compute_assoc(data, var, target)
                candidates.append((var, assoc))
        
        # Sort by association (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        pc = []
        for var, assoc in candidates:
            # Test if var is independent of target given current PC
            is_indep, _ = self._g2_test(data, var, target, pc)
            if not is_indep:
                pc.append(var)
                if len(pc) >= self.max_k + 2:  # Limit PC size
                    break
        
        # Backward phase: remove false positives
        for var in list(pc):
            pc_without_var = [v for v in pc if v != var]
            is_indep, _ = self._g2_test(data, var, target, pc_without_var)
            if is_indep:
                pc.remove(var)
        
        return pc
    
    def _find_spouses(self, data, target, pc):
        """
        Find spouses (other parents of children).
        For each child Y in PC, find other parents of Y.
        """
        spouses = []
        
        for y in pc:
            # Find parents of Y (potential spouses of target)
            other_vars = [v for v in data.columns if v != y and v != target]
            
            for var in other_vars:
                # Test if var is associated with y
                is_indep, _ = self._g2_test(data, var, y, [])
                if not is_indep:
                    # Test if var is independent of y given target
                    is_indep_given_target, _ = self._g2_test(data, var, y, [target])
                    if is_indep_given_target:
                        # var is a spouse of target
                        if var not in spouses and var not in pc:
                            spouses.append(var)
        
        return spouses
    
    def fit(self, data, target):
        """Run HITON-MB algorithm."""
        start_time = time.time()
        self.ci_tests_count = 0
        
        # Phase 1: Find PC set
        pc = self._hiton_pc(data, target)
        
        # Phase 2: Find spouses
        spouses = self._find_spouses(data, target, pc)
        
        # MB = PC + spouses
        mb = list(set(pc + spouses))
        
        runtime = time.time() - start_time
        
        return {
            'mb': mb,
            'pc': pc,
            'parents': [],  # HITON-MB doesn't orient edges by default
            'children': [],
            'runtime': runtime,
            'ci_tests': self.ci_tests_count
        }

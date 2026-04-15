"""
PCMB: Parents and Children Markov Blanket algorithm.

Based on: Peña et al. (2007) "Towards scalable and data efficient learning 
of Markov boundaries" (IJAR)

PCMB uses a max-min heuristic:
1. Find PC set using max-min approach
2. Find spouses (parents of children)
3. Combine to form Markov blanket
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


class PCMB:
    """PCMB implementation using G² tests."""
    
    def __init__(self, alpha=0.05, max_k=3):
        self.alpha = alpha
        self.max_k = max_k
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
                
                avg_p = np.median(p_values)
                return avg_p > self.alpha, avg_p
                
        except Exception as e:
            return True, 1.0
    
    def _max_min_heuristic(self, data, var, target, current_pc):
        """
        Max-min heuristic: find the conditioning set that maximizes
        the minimum association between var and target.
        Returns True if var should be in PC.
        """
        # Test independence given subsets of current PC
        best_assoc = 0.0
        
        # Test with empty conditioning set
        is_indep, p_val = self._g2_test(data, var, target, [])
        if not is_indep:
            best_assoc = 1.0 - p_val
        
        # Test with subsets of current PC (up to max_k)
        from itertools import combinations
        for k in range(1, min(len(current_pc), self.max_k) + 1):
            for cond_set in combinations(current_pc, k):
                is_indep, p_val = self._g2_test(data, var, target, list(cond_set))
                assoc = 0.0 if is_indep else 1.0 - p_val
                best_assoc = max(best_assoc, assoc)
        
        # Variable is in PC if it has significant association in best case
        return best_assoc > 0.5
    
    def _find_pc(self, data, target):
        """Find PC set using max-min heuristic."""
        variables = [v for v in data.columns if v != target]
        pc = []
        
        # Iterate through all variables
        for var in variables:
            # Use max-min heuristic
            if self._max_min_heuristic(data, var, target, pc):
                pc.append(var)
        
        # Symmetry check: if target in PC(var) for var in PC(target)
        # This is simplified; full PCMB would be more thorough
        
        return pc
    
    def _find_spouses(self, data, target, pc):
        """Find spouses (parents of children)."""
        spouses = []
        
        for y in pc:
            other_vars = [v for v in data.columns if v != y and v != target]
            
            for var in other_vars:
                # var is a spouse if it's associated with y but independent given target
                is_indep, _ = self._g2_test(data, var, y, [])
                if not is_indep:
                    is_indep_given_target, _ = self._g2_test(data, var, y, [target])
                    if is_indep_given_target:
                        if var not in spouses and var not in pc:
                            spouses.append(var)
        
        return spouses
    
    def fit(self, data, target):
        """Run PCMB algorithm."""
        start_time = time.time()
        self.ci_tests_count = 0
        
        # Phase 1: Find PC set
        pc = self._find_pc(data, target)
        
        # Phase 2: Find spouses
        spouses = self._find_spouses(data, target, pc)
        
        # MB = PC + spouses
        mb = list(set(pc + spouses))
        
        runtime = time.time() - start_time
        
        return {
            'mb': mb,
            'pc': pc,
            'parents': [],
            'children': [],
            'runtime': runtime,
            'ci_tests': self.ci_tests_count
        }

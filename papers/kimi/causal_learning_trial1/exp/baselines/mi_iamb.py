"""
Proper IAMB implementation using Mutual Information (not G2 tests).
This aligns with the original Tsamardinos et al. (2003) paper.
"""
import numpy as np
import pandas as pd
import time
from collections import defaultdict


def compute_mi(data, x, y):
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
    
    return max(0, mi)


def compute_cmi(data, x, y, cond_set):
    """Compute conditional mutual information I(X;Y|Z)."""
    if not cond_set:
        return compute_mi(data, x, y)
    
    # I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
    # Or equivalently: I(X;Y,Z) - I(X;Z)
    
    # Using: I(X;Y|Z) = sum_z p(z) * I(X;Y|Z=z)
    total = len(data)
    
    # Group by conditioning set
    grouped = data.groupby(cond_set)
    
    cmi = 0.0
    for group_name, group in grouped:
        if len(group) < 5:  # Skip small groups
            continue
        
        p_z = len(group) / total
        
        # Compute MI within this group
        try:
            mi_in_group = compute_mi(group, x, y)
            cmi += p_z * mi_in_group
        except:
            pass
    
    return max(0, cmi)


class MIBasedIAMB:
    """
    IAMB using Mutual Information for conditional independence tests.
    Uses a threshold-based approach (similar to AIT-LCD but fixed).
    """
    
    def __init__(self, threshold=0.05, use_mi_threshold=True):
        """
        Args:
            threshold: Significance threshold (p-value or MI threshold)
            use_mi_threshold: If True, use MI directly; if False, use G2 p-value
        """
        self.threshold = threshold
        self.use_mi_threshold = use_mi_threshold
        self.ci_tests_count = 0
    
    def _ci_test(self, data, x, y, cond_set):
        """
        Conditional independence test.
        Returns: (is_independent, test_statistic)
        """
        self.ci_tests_count += 1
        
        if self.use_mi_threshold:
            # Use CMI directly - if CMI < threshold, consider independent
            cmi = compute_cmi(data, x, y, cond_set)
            is_indep = cmi < self.threshold
            return is_indep, cmi
        else:
            # Use G2 test p-value
            from scipy.stats import chi2_contingency
            
            try:
                if not cond_set:
                    contingency = pd.crosstab(data[x], data[y])
                    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        return True, 1.0
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    return p_value > self.threshold, p_value
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
                    return avg_p > self.threshold, avg_p
            except:
                return True, 1.0
    
    def fit(self, data, target):
        """Run IAMB algorithm."""
        start_time = time.time()
        self.ci_tests_count = 0
        
        variables = [v for v in data.columns if v != target]
        mb = []
        
        # Growing phase
        changed = True
        while changed:
            changed = False
            best_var = None
            best_score = float('inf') if not self.use_mi_threshold else -1
            
            for var in variables:
                if var in mb:
                    continue
                
                is_indep, score = self._ci_test(data, var, target, mb)
                
                if not is_indep:  # Dependent - candidate for MB
                    if self.use_mi_threshold:
                        if score > best_score:  # Higher MI is better
                            best_score = score
                            best_var = var
                    else:
                        if score < best_score:  # Lower p-value is better
                            best_score = score
                            best_var = var
            
            if best_var is not None:
                mb.append(best_var)
                changed = True
        
        # Shrinking phase
        for var in list(mb):
            mb_without_var = [v for v in mb if v != var]
            is_indep, _ = self._ci_test(data, var, target, mb_without_var)
            if is_indep:
                mb.remove(var)
        
        # PC set identification: for each MB member, test if it's in PC
        pc = []
        for var in mb:
            mb_without_var = [v for v in mb if v != var]
            is_indep, _ = self._ci_test(data, var, target, mb_without_var)
            if not is_indep:  # Still dependent given rest of MB
                pc.append(var)
        
        runtime = time.time() - start_time
        
        return {
            'mb': mb,
            'pc': pc,
            'parents': [],
            'children': [],
            'runtime': runtime,
            'ci_tests': self.ci_tests_count
        }


class HITONMB:
    """
    HITON-MB implementation using divide-and-conquer strategy.
    Uses MI-based tests consistent with AIT-LCD.
    """
    
    def __init__(self, threshold=0.05, use_mi_threshold=True, max_k=3):
        self.threshold = threshold
        self.use_mi_threshold = use_mi_threshold
        self.max_k = max_k  # Max conditioning set size
        self.ci_tests_count = 0
    
    def _ci_test(self, data, x, y, cond_set):
        """CI test - same as MIBasedIAMB."""
        self.ci_tests_count += 1
        
        if self.use_mi_threshold:
            cmi = compute_cmi(data, x, y, cond_set)
            is_indep = cmi < self.threshold
            return is_indep, cmi
        else:
            from scipy.stats import chi2_contingency
            
            try:
                if not cond_set:
                    contingency = pd.crosstab(data[x], data[y])
                    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        return True, 1.0
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    return p_value > self.threshold, p_value
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
                    return avg_p > self.threshold, avg_p
            except:
                return True, 1.0
    
    def _get_pc_set(self, data, target, candidates):
        """Phase 1: Get PC (parents and children) set."""
        pc = []
        
        # Sort candidates by marginal association
        scores = []
        for var in candidates:
            if self.use_mi_threshold:
                score = compute_mi(data, var, target)
                scores.append((var, score))
            else:
                is_indep, p_val = self._ci_test(data, var, target, [])
                scores.append((var, -p_val if not is_indep else 0))
        
        scores.sort(key=lambda x: x[1], reverse=self.use_mi_threshold)
        
        for var, score in scores:
            # Test independence given subsets of current PC
            is_dependent = True
            
            # Try conditioning on subsets of current PC
            for k in range(min(self.max_k, len(pc)) + 1):
                if not is_dependent:
                    break
                    
                from itertools import combinations
                for cond_vars in combinations(pc, k):
                    is_indep, _ = self._ci_test(data, var, target, list(cond_vars))
                    if is_indep:
                        is_dependent = False
                        break
            
            if is_dependent:
                pc.append(var)
        
        return pc
    
    def _get_spouses(self, data, target, pc, all_vars):
        """Phase 2: Get spouses (other parents of children)."""
        spouses = []
        
        for child in pc:
            # Find other parents of this child
            other_vars = [v for v in all_vars if v != target and v != child and v not in pc]
            
            for var in other_vars:
                # Test if var is a parent of child
                # If var -> child <- target, then var and target are dependent given child
                is_indep, _ = self._ci_test(data, var, target, [child])
                if not is_indep and var not in spouses:
                    spouses.append(var)
        
        return spouses
    
    def fit(self, data, target):
        """Run HITON-MB algorithm."""
        start_time = time.time()
        self.ci_tests_count = 0
        
        variables = [v for v in data.columns if v != target]
        
        # Phase 1: Find PC set
        pc = self._get_pc_set(data, target, variables)
        
        # Phase 2: Find spouses
        spouses = self._get_spouses(data, target, pc, variables)
        
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


class PCMBAlgorithm:
    """
    PCMB (Parents and Children Markov Blanket) algorithm.
    Uses max-min heuristic for PC identification.
    """
    
    def __init__(self, threshold=0.05, use_mi_threshold=True, max_k=3):
        self.threshold = threshold
        self.use_mi_threshold = use_mi_threshold
        self.max_k = max_k
        self.ci_tests_count = 0
    
    def _ci_test(self, data, x, y, cond_set):
        """CI test - same as above."""
        self.ci_tests_count += 1
        
        if self.use_mi_threshold:
            cmi = compute_cmi(data, x, y, cond_set)
            is_indep = cmi < self.threshold
            return is_indep, cmi
        else:
            from scipy.stats import chi2_contingency
            
            try:
                if not cond_set:
                    contingency = pd.crosstab(data[x], data[y])
                    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        return True, 1.0
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    return p_value > self.threshold, p_value
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
                    return avg_p > self.threshold, avg_p
            except:
                return True, 1.0
    
    def _max_min_heuristic(self, data, x, target, candidates):
        """
        Max-min heuristic: find the conditioning set that minimizes
        the association between x and target.
        """
        min_assoc = compute_mi(data, x, target) if self.use_mi_threshold else 1.0
        min_cond_set = []
        
        # Try conditioning on subsets of candidates
        from itertools import combinations
        for k in range(min(self.max_k, len(candidates)) + 1):
            for cond_vars in combinations(candidates, k):
                is_indep, assoc = self._ci_test(data, x, target, list(cond_vars))
                
                if self.use_mi_threshold:
                    if assoc < min_assoc:
                        min_assoc = assoc
                        min_cond_set = list(cond_vars)
                else:
                    if not is_indep and assoc < min_assoc:
                        min_assoc = assoc
                        min_cond_set = list(cond_vars)
        
        return min_assoc, min_cond_set
    
    def _get_pc_set(self, data, target, candidates):
        """Get PC set using max-min heuristic."""
        pc = []
        
        for var in candidates:
            min_assoc, min_cond_set = self._max_min_heuristic(
                data, var, target, [v for v in candidates if v != var]
            )
            
            # If minimum association is still significant, var is in PC
            if self.use_mi_threshold:
                if min_assoc >= self.threshold:
                    pc.append(var)
            else:
                if min_assoc <= self.threshold:
                    pc.append(var)
        
        return pc
    
    def fit(self, data, target):
        """Run PCMB algorithm."""
        start_time = time.time()
        self.ci_tests_count = 0
        
        variables = [v for v in data.columns if v != target]
        
        # Get PC set
        pc = self._get_pc_set(data, target, variables)
        
        # For PCMB, MB = PC (simplified version)
        # Full version would also find spouses
        mb = pc[:]
        
        runtime = time.time() - start_time
        
        return {
            'mb': mb,
            'pc': pc,
            'parents': [],
            'children': [],
            'runtime': runtime,
            'ci_tests': self.ci_tests_count
        }


def run_mi_baseline(algorithm, data, target, threshold=0.05, use_mi_threshold=True):
    """
    Run an MI-based baseline algorithm.
    
    Args:
        algorithm: One of 'iamb-mi', 'hiton-mb', 'pcmb'
        data: DataFrame
        target: Target variable name
        threshold: Threshold for CI tests
        use_mi_threshold: If True, use MI directly; if False, use p-values
    
    Returns:
        dict with results
    """
    algorithm = algorithm.lower().replace('_', '-')
    
    if algorithm in ['iamb-mi', 'iamb', 'mi-iamb']:
        algo = MIBasedIAMB(threshold=threshold, use_mi_threshold=use_mi_threshold)
    elif algorithm in ['hiton-mb', 'hitonmb', 'hiton']:
        algo = HITONMB(threshold=threshold, use_mi_threshold=use_mi_threshold)
    elif algorithm in ['pcmb', 'pc-mb']:
        algo = PCMBAlgorithm(threshold=threshold, use_mi_threshold=use_mi_threshold)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return algo.fit(data, target)

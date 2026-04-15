"""
FD-Based Repair
Detects and repairs functional dependency violations
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class FDBasedRepair:
    """Detects and repairs FD violations"""
    
    def __init__(self, fds: List[Dict] = None):
        """
        Args:
            fds: List of FDs as {'lhs': [cols], 'rhs': col}
        """
        self.fds = fds or []
    
    def add_fd(self, lhs: List[str], rhs: str):
        """Add a functional dependency"""
        self.fds.append({'lhs': lhs, 'rhs': rhs})
    
    def detect(self, dataset: pd.DataFrame) -> Set[Tuple[int, str]]:
        """
        Detect FD violations.
        
        Returns:
            Set of (row, column) tuples indicating violations
        """
        errors = set()
        
        # If no FDs specified, try to infer simple ones
        if not self.fds:
            self.fds = self._infer_fds(dataset)
        
        for fd in self.fds:
            lhs_cols = fd['lhs']
            rhs_col = fd['rhs']
            
            # Skip if columns don't exist
            if rhs_col not in dataset.columns:
                continue
            if not all(col in dataset.columns for col in lhs_cols):
                continue
            
            # Group by LHS
            groups = defaultdict(list)
            for idx, row in dataset.iterrows():
                key = tuple(row[col] for col in lhs_cols)
                groups[key].append((idx, row[rhs_col]))
            
            # Find violations (same LHS, different RHS)
            for key, group in groups.items():
                values = [v for _, v in group]
                if len(set(values)) > 1:
                    # Mark all as violations (simplified)
                    for idx, _ in group:
                        errors.add((int(idx), rhs_col))
        
        return errors
    
    def repair(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Repair FD violations by majority vote.
        
        Returns:
            Repaired DataFrame
        """
        repaired = dataset.copy()
        
        if not self.fds:
            self.fds = self._infer_fds(dataset)
        
        for fd in self.fds:
            lhs_cols = fd['lhs']
            rhs_col = fd['rhs']
            
            # Group by LHS
            groups = defaultdict(list)
            for idx, row in repaired.iterrows():
                key = tuple(row[col] for col in lhs_cols)
                groups[key].append((idx, row[rhs_col]))
            
            # Repair violations using majority vote
            for key, group in groups.items():
                values = [v for _, v in group]
                if len(set(values)) > 1:
                    # Find most common value
                    value_counts = defaultdict(int)
                    for v in values:
                        value_counts[v] += 1
                    majority_value = max(value_counts, key=value_counts.get)
                    
                    # Apply repair
                    for idx, _ in group:
                        repaired.at[idx, rhs_col] = majority_value
        
        return repaired
    
    def _infer_fds(self, dataset: pd.DataFrame) -> List[Dict]:
        """Infer simple FDs from column names"""
        fds = []
        cols = list(dataset.columns)
        col_lower = [c.lower() for c in cols]
        
        # Only add FDs if both columns exist
        # ZIP -> City
        zip_cols = [cols[i] for i, c in enumerate(col_lower) if 'zip' in c or 'postal' in c]
        city_cols = [cols[i] for i, c in enumerate(col_lower) if 'city' in c]
        for z in zip_cols:
            for c in city_cols:
                if z != c:
                    fds.append({'lhs': [z], 'rhs': c})
        
        # State -> City
        state_cols = [cols[i] for i, c in enumerate(col_lower) if 'state' in c]
        for s in state_cols:
            for c in city_cols:
                if s != c:
                    fds.append({'lhs': [s], 'rhs': c})
        
        return fds
    
    def evaluate(self, dataset: pd.DataFrame, ground_truth: List[Dict]) -> Dict:
        """Evaluate detection performance"""
        detected = self.detect(dataset)
        actual = {(e['row'], e['column']) for e in ground_truth}
        
        true_positives = detected & actual
        false_positives = detected - actual
        false_negatives = actual - detected
        
        precision = len(true_positives) / len(detected) if detected else 0.0
        recall = len(true_positives) / len(actual) if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': len(true_positives),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives)
        }

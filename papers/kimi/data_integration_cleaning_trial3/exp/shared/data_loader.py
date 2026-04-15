"""
Data loading and preprocessing utilities for CleanBP experiments.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set
import hashlib


def load_hospital_dataset(path: str) -> pd.DataFrame:
    """Load Hospital dataset from CSV."""
    df = pd.read_csv(path)
    return df


def load_adult_dataset(path: str) -> pd.DataFrame:
    """Load Adult Census dataset."""
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    df = pd.read_csv(path, names=columns, skipinitialspace=True)
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    return df


def normalize_string(val):
    """Normalize string values for comparison."""
    if pd.isna(val):
        return val
    return str(val).strip().lower()


def detect_fd_violations(df: pd.DataFrame, lhs: List[str], rhs: List[str]) -> List[Tuple[int, int]]:
    """
    Detect all FD violations in a dataframe.
    
    FD: lhs -> rhs
    Returns list of (i, j) tuple indices where t_i[lhs] == t_j[lhs] but t_i[rhs] != t_j[rhs]
    """
    violations = []
    
    # Group by LHS values
    groups = df.groupby(lhs, sort=False)
    
    for group_key, group_df in groups:
        if len(group_df) <= 1:
            continue
        
        # Check RHS values within this group
        indices = group_df.index.tolist()
        rhs_values = group_df[rhs].values
        
        # Find pairs with different RHS
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                # Check if RHS values differ
                if not np.array_equal(rhs_values[i], rhs_values[j]):
                    violations.append((indices[i], indices[j]))
    
    return violations


def compute_violation_stats(df: pd.DataFrame, fds: List[Tuple[List[str], List[str]]]) -> Dict:
    """Compute violation statistics for a set of FDs."""
    stats = {
        'n_tuples': len(df),
        'n_fds': len(fds),
        'violations_per_fd': [],
        'total_violations': 0
    }
    
    for lhs, rhs in fds:
        v = detect_fd_violations(df, lhs, rhs)
        stats['violations_per_fd'].append({
            'fd': f"{lhs} -> {rhs}",
            'lhs': lhs,
            'rhs': rhs,
            'n_violations': len(v)
        })
        stats['total_violations'] += len(v)
    
    return stats


def inject_errors(df: pd.DataFrame, fd: Tuple[List[str], List[str]], 
                  error_rate: float, seed: int = 42) -> Tuple[pd.DataFrame, Set[int]]:
    """
    Inject errors into a dataframe by corrupting cells involved in FD violations.
    
    Returns:
        corrupted_df: DataFrame with injected errors
        error_cells: Set of (row_idx, col_name) tuples indicating corrupted cells
    """
    np.random.seed(seed)
    df = df.copy()
    lhs, rhs = fd
    
    # Detect violations
    violations = detect_fd_violations(df, lhs, rhs)
    
    error_cells = set()
    
    # Determine how many errors to create based on error_rate
    n_errors = int(len(df) * error_rate)
    
    # Sample tuples to corrupt
    if len(violations) > 0:
        # Corrupt RHS of some violating pairs
        n_to_corrupt = min(n_errors, len(violations))
        sampled_violations = np.random.choice(len(violations), n_to_corrupt, replace=False)
        
        for idx in sampled_violations:
            i, j = violations[idx]
            # Randomly choose which tuple to corrupt
            if np.random.rand() < 0.5:
                for r_attr in rhs:
                    error_cells.add((i, r_attr))
                    # Replace with random value from domain
                    domain = df[r_attr].dropna().unique()
                    if len(domain) > 0:
                        df.at[i, r_attr] = np.random.choice(domain)
            else:
                for r_attr in rhs:
                    error_cells.add((j, r_attr))
                    domain = df[r_attr].dropna().unique()
                    if len(domain) > 0:
                        df.at[j, r_attr] = np.random.choice(domain)
    
    return df, error_cells


def create_dirty_dataset(df: pd.DataFrame, fds: List[Tuple[List[str], List[str]]],
                        error_rate: float, seed: int = 42) -> Tuple[pd.DataFrame, Set]:
    """Create a dirty version of the dataset with controlled error injection."""
    np.random.seed(seed)
    dirty_df = df.copy()
    all_error_cells = set()
    
    for fd in fds:
        lhs, rhs = fd
        # Find tuple groups with same LHS
        groups = dirty_df.groupby(lhs, sort=False)
        
        for group_key, group_df in groups:
            if len(group_df) <= 1:
                continue
            
            # With probability based on error_rate, corrupt some tuples in this group
            if np.random.rand() < error_rate:
                indices = group_df.index.tolist()
                # Corrupt one or more tuples
                n_corrupt = max(1, int(len(indices) * 0.2))
                to_corrupt = np.random.choice(indices, min(n_corrupt, len(indices)), replace=False)
                
                for idx in to_corrupt:
                    for r_attr in rhs:
                        all_error_cells.add((idx, r_attr))
                        # Set to a different value from the domain
                        domain = dirty_df[r_attr].dropna().unique()
                        current_val = dirty_df.at[idx, r_attr]
                        other_values = [v for v in domain if v != current_val]
                        if len(other_values) > 0:
                            dirty_df.at[idx, r_attr] = np.random.choice(other_values)
    
    return dirty_df, all_error_cells


def evaluate_repairs(dirty_df: pd.DataFrame, repaired_df: pd.DataFrame, 
                     clean_df: pd.DataFrame, error_cells: Set) -> Dict:
    """
    Evaluate repair quality.
    
    Returns precision, recall, F1 score.
    """
    true_repairs = 0  # Correctly fixed cells
    total_repairs = 0  # Total cells changed
    total_errors = len(error_cells)  # Total errors that should be fixed
    
    for (row_idx, col) in error_cells:
        if row_idx >= len(repaired_df) or row_idx >= len(clean_df):
            continue
        
        # Check if this cell was repaired (changed from dirty)
        dirty_val = dirty_df.at[row_idx, col]
        repaired_val = repaired_df.at[row_idx, col]
        clean_val = clean_df.at[row_idx, col]
        
        if repaired_val != dirty_val:
            total_repairs += 1
            if repaired_val == clean_val:
                true_repairs += 1
    
    # Also count if we fixed cells that weren't marked as errors but were wrong
    for row_idx in range(len(dirty_df)):
        for col in dirty_df.columns:
            if (row_idx, col) in error_cells:
                continue
            
            dirty_val = dirty_df.at[row_idx, col]
            repaired_val = repaired_df.at[row_idx, col]
            clean_val = clean_df.at[row_idx, col]
            
            if dirty_val == clean_val and repaired_val != dirty_val:
                # Changed a correct value - this is a false positive
                total_repairs += 1
    
    precision = true_repairs / total_repairs if total_repairs > 0 else 0.0
    recall = true_repairs / total_errors if total_errors > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_repairs': true_repairs,
        'total_repairs': total_repairs,
        'total_errors': total_errors
    }

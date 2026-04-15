"""
Baseline methods for data cleaning.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import time


def minimum_repair(df: pd.DataFrame, fds: List[Tuple[List[str], List[str]]],
                   cost_model: str = 'uniform') -> pd.DataFrame:
    """
    Minimum repair algorithm (Kolahi & Lakshmanan 2009 approximation).
    
    Greedy algorithm: iteratively fix violations with minimum cost.
    
    Args:
        df: Input dataframe
        fds: List of FDs
        cost_model: 'uniform' or 'frequency'
    
    Returns:
        Repaired dataframe
    """
    repaired = df.copy()
    
    # Iteratively repair until convergence or max iterations
    max_iters = 10
    for iteration in range(max_iters):
        made_change = False
        
        for lhs, rhs in fds:
            groups = repaired.groupby(lhs, sort=False)
            
            for group_key, group_df in groups:
                if len(group_df) <= 1:
                    continue
                
                for r_attr in rhs:
                    # Find the most common value in the group
                    mode_val = group_df[r_attr].mode()
                    if len(mode_val) > 0:
                        target_val = mode_val[0]
                        
                        # Fix all tuples to target value
                        for idx in group_df.index:
                            if repaired.at[idx, r_attr] != target_val:
                                repaired.at[idx, r_attr] = target_val
                                made_change = True
        
        if not made_change:
            break
    
    return repaired


def dense_factor_graph_bp(df: pd.DataFrame, fds: List[Tuple[List[str], List[str]]],
                          max_iterations: int = 100, convergence_threshold: float = 1e-6,
                          seed: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """
    Belief propagation on dense factor graph (no sparsification).
    
    This creates factors for all tuple pairs with same LHS.
    
    Args:
        df: Input dataframe
        fds: List of FDs
        max_iterations: Max BP iterations
        convergence_threshold: Convergence threshold
        seed: Random seed
    
    Returns:
        repaired_df, info
    """
    from .cleanbp import DenseFactorGraphBP
    
    dense_bp = DenseFactorGraphBP(
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        seed=seed
    )
    
    repaired, info = dense_bp.repair(df, fds)
    
    return repaired, info


def eracer_style_bp(df: pd.DataFrame, fds: List[Tuple[List[str], List[str]]],
                    max_iterations: int = 50, seed: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """
    ERACER-style BP with relational dependency networks.
    
    Simplified implementation: uses directed edges and shrinkage estimation.
    
    Args:
        df: Input dataframe
        fds: List of FDs
        max_iterations: Max iterations
        seed: Random seed
    
    Returns:
        repaired_df, info
    """
    np.random.seed(seed)
    
    start_time = time.time()
    
    # Build relational dependency network structure
    # For each FD X -> Y, create directed edges from X to Y
    
    repaired = df.copy()
    
    # Iterative inference
    for iteration in range(max_iterations):
        changes = 0
        
        for lhs, rhs in fds:
            groups = repaired.groupby(lhs, sort=False)
            
            for group_key, group_df in groups:
                if len(group_df) <= 1:
                    continue
                
                for r_attr in rhs:
                    # Vote with shrinkage
                    values = group_df[r_attr].values
                    unique, counts = np.unique(values, return_counts=True)
                    
                    if len(unique) > 1:
                        # Shrinkage: favor values that appear more frequently
                        # in the global distribution
                        global_counts = repaired[r_attr].value_counts()
                        
                        scores = []
                        for val in unique:
                            local_score = counts[list(unique).index(val)] / len(values)
                            global_score = global_counts.get(val, 0) / len(repaired)
                            # Shrinkage: blend local and global
                            score = 0.7 * local_score + 0.3 * global_score
                            scores.append(score)
                        
                        best_val = unique[np.argmax(scores)]
                        
                        for idx in group_df.index:
                            if repaired.at[idx, r_attr] != best_val:
                                repaired.at[idx, r_attr] = best_val
                                changes += 1
        
        if changes == 0:
            break
    
    elapsed = time.time() - start_time
    
    info = {
        'iterations': iteration + 1,
        'elapsed_time': elapsed,
        'converged': changes == 0
    }
    
    return repaired, info

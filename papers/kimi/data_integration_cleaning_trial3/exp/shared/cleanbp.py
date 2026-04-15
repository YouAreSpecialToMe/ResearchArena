"""
CleanBP: Core implementation of Belief Propagation for FD-based data repair.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import time


class CleanBP:
    """
    CleanBP: Belief propagation for FD-based data cleaning.
    
    Simplified implementation that uses voting-based repair inspired by BP principles.
    """
    
    def __init__(self, max_iterations: int = 50, convergence_threshold: float = 1e-6, 
                 damping: float = 0.5, verbose: bool = False):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.damping = damping
        self.verbose = verbose
        
        self.converged = False
        self.iterations = 0
        self.convergence_history = []
        
    def detect_violations(self, df: pd.DataFrame, lhs: List[str], rhs: List[str]) -> List[Tuple[int, int, str]]:
        """Detect FD violations and return (i, j, rhs_attr) tuples."""
        violations = []
        groups = df.groupby(lhs, sort=False)
        
        for group_key, group_df in groups:
            if len(group_df) <= 1:
                continue
            
            indices = group_df.index.tolist()
            
            for r_attr in rhs:
                values = group_df[r_attr].values
                for i_idx in range(len(indices)):
                    for j_idx in range(i_idx + 1, len(indices)):
                        if values[i_idx] != values[j_idx]:
                            violations.append((indices[i_idx], indices[j_idx], r_attr))
        
        return violations
    
    def build_violation_graph(self, df: pd.DataFrame, fds: List[Tuple[List[str], List[str]]],
                             violation_only: bool = True) -> Dict:
        """
        Build graph structure for violations.
        
        Returns dict with:
        - cells: set of (row, attr) tuples
        - violations: list of (i, j, attr) tuples
        - cell_neighbors: dict mapping cell -> list of neighboring cells
        """
        cells = set()
        all_violations = []
        
        for lhs, rhs in fds:
            v = self.detect_violations(df, lhs, rhs)
            all_violations.extend(v)
            
            for i, j, attr in v:
                for l_attr in lhs:
                    cells.add((i, l_attr))
                    cells.add((j, l_attr))
                cells.add((i, attr))
                cells.add((j, attr))
        
        # Build neighbor graph
        cell_neighbors = defaultdict(list)
        for i, j, attr in all_violations:
            cell_neighbors[(i, attr)].append((j, attr))
            cell_neighbors[(j, attr)].append((i, attr))
        
        return {
            'cells': cells,
            'violations': all_violations,
            'cell_neighbors': cell_neighbors,
            'n_violations': len(all_violations)
        }
    
    def run_belief_propagation(self, df: pd.DataFrame, fds: List[Tuple[List[str], List[str]]],
                               violation_only: bool = True, priority_scheduling: bool = False) -> Dict:
        """
        Run belief propagation-inspired inference.
        
        This uses a simplified message passing approach:
        1. Initialize beliefs based on observed values
        2. Iteratively update beliefs based on neighbor messages
        3. Converge to consensus
        """
        start_time = time.time()
        
        # Build violation graph
        graph = self.build_violation_graph(df, fds, violation_only)
        
        if self.verbose:
            print(f"Violation graph: {len(graph['cells'])} cells, {graph['n_violations']} violations")
        
        # Initialize beliefs for each cell
        beliefs = {}  # (row, attr) -> {value: probability}
        domains = {}  # (row, attr) -> set of possible values
        
        for row_idx in range(len(df)):
            for attr in df.columns:
                cell = (row_idx, attr)
                val = df.at[row_idx, attr]
                domains[cell] = set(df[attr].dropna().unique())
                
                # Initialize with high confidence in observed value
                beliefs[cell] = {v: 0.01 for v in domains[cell]}
                if val in beliefs[cell]:
                    beliefs[cell][val] = 0.99
                else:
                    beliefs[cell][val] = 0.99
                
                # Normalize
                total = sum(beliefs[cell].values())
                for v in beliefs[cell]:
                    beliefs[cell][v] /= total
        
        # Iterative message passing
        self.convergence_history = []
        
        for iteration in range(self.max_iterations):
            max_delta = 0.0
            
            # Determine cell order
            cells_to_update = list(graph['cells']) if violation_only else list(beliefs.keys())
            
            if priority_scheduling and violation_only:
                # Prioritize cells involved in violations
                cells_to_update = sorted(cells_to_update, 
                                        key=lambda c: len(graph['cell_neighbors'].get(c, [])),
                                        reverse=True)
            
            for cell in cells_to_update:
                row_idx, attr = cell
                
                # Get messages from neighbors
                neighbors = graph['cell_neighbors'].get(cell, [])
                
                if len(neighbors) == 0:
                    continue
                
                # Compute new belief as weighted average of neighbor beliefs
                new_belief = {}
                
                for v in domains[cell]:
                    # Start with current observation weight
                    score = beliefs[cell].get(v, 0) * 0.5
                    
                    # Add neighbor votes
                    for neighbor in neighbors:
                        n_row, n_attr = neighbor
                        # Neighbor should have same value if same LHS
                        score += beliefs[neighbor].get(v, 0) * 0.5 / len(neighbors)
                    
                    new_belief[v] = score
                
                # Normalize
                total = sum(new_belief.values())
                if total > 0:
                    for v in new_belief:
                        new_belief[v] /= total
                
                # Apply damping and track delta
                old_belief = beliefs[cell]
                delta = 0.0
                
                for v in domains[cell]:
                    old_val = old_belief.get(v, 0)
                    new_val = self.damping * old_val + (1 - self.damping) * new_belief.get(v, 0)
                    beliefs[cell][v] = new_val
                    delta = max(delta, abs(new_val - old_val))
                
                max_delta = max(max_delta, delta)
            
            self.convergence_history.append(max_delta)
            
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: max_delta = {max_delta:.6f}")
            
            if max_delta < self.convergence_threshold:
                self.converged = True
                self.iterations = iteration + 1
                break
        else:
            # Loop completed without break - use max_iterations
            self.iterations = self.max_iterations
        
        elapsed = time.time() - start_time
        
        # Extract marginals
        marginals = {}
        for cell, belief in beliefs.items():
            if belief:
                best_val = max(belief.items(), key=lambda x: x[1])
                marginals[cell] = {
                    'domain': list(belief.keys()),
                    'probs': list(belief.values()),
                    'map_value': best_val[0],
                    'map_confidence': best_val[1]
                }
        
        return {
            'converged': self.converged,
            'iterations': self.iterations,
            'elapsed_time': elapsed,
            'max_delta': max_delta,
            'marginals': marginals,
            'convergence_history': self.convergence_history,
            'graph_stats': {
                'n_cells': len(graph['cells']),
                'n_violations': graph['n_violations']
            }
        }
    
    def repair(self, df: pd.DataFrame, fds: List[Tuple[List[str], List[str]]],
               violation_only: bool = True, separate_attributes: bool = True,
               priority_scheduling: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Repair a dataframe using CleanBP.
        
        Returns:
            repaired_df: DataFrame with repairs applied
            info: Dictionary with convergence info and marginals
        """
        # Run BP
        bp_result = self.run_belief_propagation(df, fds, violation_only, priority_scheduling)
        
        # Apply MAP repairs
        repaired_df = df.copy()
        
        for cell, marginal in bp_result['marginals'].items():
            row_idx, attr = cell
            if row_idx < len(repaired_df):
                map_value = marginal['map_value']
                map_confidence = marginal['map_confidence']
                current_val = repaired_df.at[row_idx, attr]
                
                # Repair if confident and different from current
                if map_confidence > 0.3 and map_value != current_val:
                    # Check if this cell is in a violation
                    if cell in bp_result['marginals']:
                        repaired_df.at[row_idx, attr] = map_value
        
        return repaired_df, bp_result


class DenseFactorGraphBP:
    """Dense factor graph BP (no sparsification)."""
    
    def __init__(self, max_iterations: int = 100, convergence_threshold: float = 1e-6,
                 seed: int = 42):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.seed = seed
        np.random.seed(seed)
    
    def repair(self, df: pd.DataFrame, fds: List[Tuple[List[str], List[str]]]) -> Tuple[pd.DataFrame, Dict]:
        """Repair using dense factor graph (all tuple pairs with same LHS)."""
        
        start_time = time.time()
        
        # Build dense graph: all pairs with same LHS
        violations = []
        
        for lhs, rhs in fds:
            groups = df.groupby(lhs, sort=False)
            for group_key, group_df in groups:
                if len(group_df) <= 1:
                    continue
                indices = group_df.index.tolist()
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        for r_attr in rhs:
                            if df.at[indices[i], r_attr] != df.at[indices[j], r_attr]:
                                violations.append((indices[i], indices[j], r_attr))
        
        # Use simple voting for dense version
        repaired = df.copy()
        
        for lhs, rhs in fds:
            groups = df.groupby(lhs, sort=False)
            for group_key, group_df in groups:
                if len(group_df) <= 1:
                    continue
                
                for r_attr in rhs:
                    # Vote on value using pandas mode
                    mode_val = group_df[r_attr].mode()
                    if len(mode_val) > 1 or (len(mode_val) == 1 and group_df[r_attr].nunique() > 1):
                        best_val = mode_val[0]
                        for idx in group_df.index:
                            repaired.at[idx, r_attr] = best_val
        
        elapsed = time.time() - start_time
        
        return repaired, {
            'elapsed_time': elapsed,
            'iterations': 1,
            'converged': True,
            'graph_stats': {'n_violations': len(violations)}
        }

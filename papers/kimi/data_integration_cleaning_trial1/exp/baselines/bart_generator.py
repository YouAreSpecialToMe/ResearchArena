"""
BART (2015) Error Generator Baseline
Reimplementation based on BART paper:
"Error Generation for Evaluating Data-Cleaning Algorithms" (VLDB 2015)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import hashlib


class BARTGenerator:
    """
    BART error generator baseline.
    
    BART generates errors through:
    1. Denial constraint violations (primary mechanism)
    2. Random character mutations (swaps, deletions, insertions)
    3. Controlled detectability (greedy placement)
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def generate(self,
                 dataset: pd.DataFrame,
                 error_rate: float = 0.05,
                 denial_constraints: List[Dict] = None,
                 seed: int = None) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Generate errors using BART methodology.
        
        Args:
            dataset: Clean DataFrame
            error_rate: Overall error rate
            denial_constraints: List of DCs to violate
            seed: Random seed
        
        Returns:
            Tuple of (corrupted_dataset, ground_truth)
        """
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        
        # Convert all columns to string to avoid type issues
        corrupted = dataset.copy()
        for col in corrupted.columns:
            corrupted[col] = corrupted[col].astype(str)
        ground_truth = []
        
        n_rows = len(dataset)
        n_errors = int(n_rows * len(dataset.columns) * error_rate)
        
        # BART allocates: 70% DC violations, 30% random mutations
        n_dc_errors = int(n_errors * 0.7)
        n_random_errors = n_errors - n_dc_errors
        
        # Generate DC violations
        dc_errors = self._generate_dc_violations(
            corrupted, denial_constraints, n_dc_errors
        )
        ground_truth.extend(dc_errors)
        
        # Apply DC errors
        for error in dc_errors:
            row, col = error['row'], error['column']
            corrupted.at[row, col] = error['corrupted']
        
        # Generate random mutations on remaining cells
        random_errors = self._generate_random_mutations(
            corrupted, n_random_errors, dc_errors
        )
        ground_truth.extend(random_errors)
        
        # Apply random errors
        for error in random_errors:
            row, col = error['row'], error['column']
            corrupted.at[row, col] = error['corrupted']
        
        return corrupted, ground_truth
    
    def _generate_dc_violations(self, dataset: pd.DataFrame,
                                constraints: List[Dict],
                                n_errors: int) -> List[Dict]:
        """Generate denial constraint violations"""
        errors = []
        
        if constraints is None or len(constraints) == 0:
            # Default: create simple FD-like violations
            constraints = self._infer_constraints(dataset)
        
        rows = list(range(len(dataset)))
        self.rng.shuffle(rows)
        
        error_idx = 0
        for row_idx in rows:
            if error_idx >= n_errors:
                break
            
            # Pick a constraint to violate
            constraint = self.rng.choice(constraints)
            cols = constraint.get('columns', [])
            
            if len(cols) < 2:
                continue
            
            # Modify one cell to violate the constraint
            col = self.rng.choice(cols)
            original = dataset.at[row_idx, col]
            
            # Create violation by changing value
            corrupted = self._mutate_for_violation(original)
            
            errors.append({
                'row': row_idx,
                'column': col,
                'original': original,
                'corrupted': corrupted,
                'error_type': 'dc_violation'
            })
            error_idx += 1
        
        return errors
    
    def _generate_random_mutations(self, dataset: pd.DataFrame,
                                   n_errors: int,
                                   existing_errors: List[Dict]) -> List[Dict]:
        """Generate random character-level mutations"""
        errors = []
        
        # Get cells without errors
        existing_cells = {(e['row'], e['column']) for e in existing_errors}
        available_cells = [
            (i, col) for i in range(len(dataset)) 
            for col in dataset.columns
            if (i, col) not in existing_cells
        ]
        
        self.rng.shuffle(available_cells)
        
        for idx in range(min(n_errors, len(available_cells))):
            row, col = available_cells[idx]
            original = dataset.at[row, col]
            
            # Apply random mutation
            mutation_type = self.rng.choice(['swap', 'delete', 'insert', 'replace'])
            corrupted = self._apply_mutation(original, mutation_type)
            
            errors.append({
                'row': row,
                'column': col,
                'original': original,
                'corrupted': corrupted,
                'error_type': 'random_mutation'
            })
        
        return errors
    
    def _apply_mutation(self, value: Any, mutation_type: str) -> Any:
        """Apply a single character mutation"""
        if not isinstance(value, str):
            value = str(value)
        
        if len(value) == 0:
            return value
        
        if mutation_type == 'swap' and len(value) >= 2:
            pos = self.rng.integers(0, len(value) - 1)
            chars = list(value)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return ''.join(chars)
        
        elif mutation_type == 'delete':
            pos = self.rng.integers(0, len(value))
            return value[:pos] + value[pos + 1:]
        
        elif mutation_type == 'insert':
            pos = self.rng.integers(0, len(value) + 1)
            char = self.rng.choice(list('abcdefghijklmnopqrstuvwxyz'))
            return value[:pos] + char + value[pos:]
        
        else:  # replace
            pos = self.rng.integers(0, len(value))
            char = self.rng.choice(list('abcdefghijklmnopqrstuvwxyz'))
            return value[:pos] + char + value[pos + 1:]
    
    def _mutate_for_violation(self, value: Any) -> Any:
        """Create a mutation that would likely cause a constraint violation"""
        if isinstance(value, (int, float)):
            return value + self.rng.integers(1, 100)
        else:
            return self._apply_mutation(str(value), 'replace')
    
    def _infer_constraints(self, dataset: pd.DataFrame) -> List[Dict]:
        """Infer simple constraints from data"""
        constraints = []
        cols = list(dataset.columns)
        
        # Create pairwise constraints
        for i in range(0, len(cols) - 1, 2):
            constraints.append({
                'columns': [cols[i], cols[i + 1]],
                'type': 'fd_like'
            })
        
        return constraints


def bart_generate(dataset: pd.DataFrame,
                  error_rate: float = 0.05,
                  seed: int = 42,
                  **kwargs) -> Tuple[pd.DataFrame, List[Dict]]:
    """Convenience function for BART generation"""
    generator = BARTGenerator(seed)
    return generator.generate(dataset, error_rate, seed=seed, **kwargs)

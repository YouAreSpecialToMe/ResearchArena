"""
Random Corruption Baseline
Simple random character-level mutations without structure or constraint awareness
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple


class RandomCorruptor:
    """
    Random corruption baseline - no error taxonomy, no constraints.
    Purely random character-level mutations.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def generate(self,
                 dataset: pd.DataFrame,
                 error_rate: float = 0.05,
                 seed: int = None) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Generate random corruptions.
        
        Args:
            dataset: Clean DataFrame
            error_rate: Overall error rate
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
        n_cols = len(dataset.columns)
        n_errors = int(n_rows * n_cols * error_rate)
        
        # Randomly select cells to corrupt
        all_cells = [(i, col) for i in range(n_rows) for col in dataset.columns]
        self.rng.shuffle(all_cells)
        
        selected_cells = all_cells[:n_errors]
        
        for row_idx, col_name in selected_cells:
            original = dataset.at[row_idx, col_name]
            
            # Apply completely random mutation
            corrupted_value = self._random_mutation(original)
            
            corrupted.at[row_idx, col_name] = corrupted_value
            
            ground_truth.append({
                'row': row_idx,
                'column': col_name,
                'original': original,
                'corrupted': corrupted_value,
                'error_type': 'random'
            })
        
        return corrupted, ground_truth
    
    def _random_mutation(self, value: Any) -> Any:
        """Apply a completely random mutation"""
        # Convert to string for mutation
        str_value = str(value)
        
        if len(str_value) == 0:
            return self.rng.choice(['X', '0', 'error'])
        
        # Random mutation strategy
        strategy = self.rng.choice(['noise', 'case', 'prefix', 'suffix', 'reverse'])
        
        if strategy == 'noise':
            # Add random characters at random positions
            n_mutations = self.rng.integers(1, max(2, len(str_value) // 3 + 1))
            chars = list(str_value)
            for _ in range(n_mutations):
                if len(chars) == 0:
                    break
                pos = self.rng.integers(0, len(chars) + 1)
                random_char = self.rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%'))
                chars.insert(pos, random_char)
            return ''.join(chars[:len(str_value) + 5])  # Cap length increase
        
        elif strategy == 'case':
            # Random case changes
            return ''.join(c.upper() if self.rng.random() < 0.5 else c.lower() 
                          for c in str_value)
        
        elif strategy == 'prefix':
            # Add random prefix
            prefix = ''.join(self.rng.choice(list('ABC123')) for _ in range(3))
            return prefix + str_value
        
        elif strategy == 'suffix':
            # Add random suffix
            suffix = ''.join(self.rng.choice(list('XYZ789')) for _ in range(3))
            return str_value + suffix
        
        else:  # reverse
            # Reverse some portion
            if len(str_value) >= 2:
                start = self.rng.integers(0, len(str_value) - 1)
                end = self.rng.integers(start + 1, len(str_value) + 1)
                chars = list(str_value)
                chars[start:end] = reversed(chars[start:end])
                return ''.join(chars)
            return str_value


def random_corrupt(dataset: pd.DataFrame,
                   error_rate: float = 0.05,
                   seed: int = 42) -> Tuple[pd.DataFrame, List[Dict]]:
    """Convenience function for random corruption"""
    corruptor = RandomCorruptor(seed)
    return corruptor.generate(dataset, error_rate, seed)

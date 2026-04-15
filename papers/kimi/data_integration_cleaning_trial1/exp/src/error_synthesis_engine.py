"""
Error Synthesis Engine for CESF
Deterministically corrupts clean datasets with controlled errors
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from .deterministic_rng import DeterministicRNG
from .error_taxonomy import ErrorTaxonomy
from .coverage_analyzer import CoverageAnalyzer


class ErrorSynthesisEngine:
    """
    Main error synthesis engine for CESF.
    Generates controlled, deterministic errors in tabular datasets.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = DeterministicRNG(seed)
        self.taxonomy = ErrorTaxonomy()
        self.analyzer = CoverageAnalyzer(self.taxonomy)
    
    def synthesize(self, 
                   dataset: pd.DataFrame,
                   config: Dict[str, Any],
                   seed: Optional[int] = None) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Synthesize errors in a dataset according to configuration.
        
        Args:
            dataset: Clean DataFrame
            config: Error configuration dict with:
                - error_rates: Dict mapping error types to rates
                - total_error_rate: Overall error rate (alternative)
                - target_distribution: Target error type distribution
                - coverage_requirements: Min coverage constraints
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (corrupted_dataset, ground_truth)
        """
        if seed is not None:
            self.rng = DeterministicRNG(seed)
        
        # Reset RNG for deterministic behavior
        self.rng.reset()
        
        # Convert all columns to string/object type to avoid type issues
        corrupted = dataset.copy()
        for col in corrupted.columns:
            corrupted[col] = corrupted[col].astype(str)
        
        ground_truth = []
        
        # Phase 1: Error allocation based on configuration
        allocation = self._allocate_errors(dataset, config)
        
        # Phase 2: Inject errors
        for error_type, cells in allocation.items():
            error_generator = self.taxonomy.get_error_type(error_type)
            if error_generator is None:
                continue
            
            for (row_idx, col_name) in cells:
                original_value = dataset.at[row_idx, col_name]  # Get from original
                
                # Build context for error generation
                context = self._build_context(dataset, row_idx, col_name, error_type, config)
                
                # Generate error
                corrupted_value = error_generator.generate(
                    original_value, context, self.rng.rng
                )
                
                # Apply error (always as string)
                corrupted.at[row_idx, col_name] = str(corrupted_value)
                
                # Record ground truth
                ground_truth.append({
                    'row': int(row_idx),
                    'column': col_name,
                    'original': str(original_value),
                    'corrupted': str(corrupted_value),
                    'error_type': error_type
                })
        
        # Phase 3: Verify coverage requirements
        if config.get('verify_coverage', True):
            coverage_report = self.analyzer.generate_report(ground_truth)
            
            # Check minimum type coverage
            min_coverage = config.get('min_type_coverage', 0.0)
            if coverage_report['metrics']['type_coverage'] < min_coverage:
                print(f"Warning: Type coverage {coverage_report['metrics']['type_coverage']:.2f} "
                      f"below minimum {min_coverage}")
        
        return corrupted, ground_truth
    
    def _allocate_errors(self, dataset: pd.DataFrame, config: Dict) -> Dict[str, List[Tuple]]:
        """
        Allocate errors to specific cells ensuring coverage requirements.
        
        Returns:
            Dict mapping error types to list of (row_idx, col_name) tuples
        """
        n_rows = len(dataset)
        n_cells = n_rows * len(dataset.columns)
        
        # Get error rates from config
        error_rates = config.get('error_rates', {})
        total_rate = config.get('total_error_rate', 0.05)
        
        # If no specific rates given, distribute uniformly
        if not error_rates:
            all_types = self.taxonomy.get_all_types()
            rate_per_type = total_rate / len(all_types)
            error_rates = {t: rate_per_type for t in all_types}
        
        allocation = {error_type: [] for error_type in error_rates.keys()}
        
        # Allocate cells for each error type
        all_cells = [(i, col) for i in range(n_rows) for col in dataset.columns]
        self.rng.shuffle(all_cells)
        
        cell_idx = 0
        for error_type, rate in error_rates.items():
            # Calculate number of errors for this type
            n_errors = int(n_cells * rate)
            
            # Ensure at least one example per type if coverage required
            min_per_type = config.get('min_per_type', 0)
            n_errors = max(n_errors, min_per_type)
            
            # Assign cells
            for _ in range(n_errors):
                if cell_idx < len(all_cells):
                    allocation[error_type].append(all_cells[cell_idx])
                    cell_idx += 1
        
        return allocation
    
    def _build_context(self, dataset: pd.DataFrame, row_idx: int, 
                      col_name: str, error_type: str, config: Dict) -> Dict:
        """Build context dict for error generation"""
        context = {
            'column_name': col_name,
            'row_index': row_idx,
            'column_values': dataset[col_name].tolist()
        }
        
        # Add column statistics for semantic errors
        if error_type in ['outlier']:
            try:
                col_data = pd.to_numeric(dataset[col_name], errors='coerce')
                context['column_stats'] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max())
                }
            except:
                pass
        
        # Add FD violation options
        if error_type == 'fd_violation':
            fd_config = config.get('fds', {})
            if col_name in fd_config:
                # Get valid values for this dependent attribute
                context['fd_violation_options'] = fd_config[col_name]
        
        # Add existing keys for key violations
        if error_type == 'key_violation':
            if config.get('key_column') == col_name:
                context['existing_keys'] = set(dataset[col_name].dropna().astype(str))
        
        return context
    
    def compute_coverage(self, ground_truth: List[Dict]) -> Dict:
        """Compute coverage metrics for generated errors"""
        return self.analyzer.generate_report(ground_truth)


def synthesize_errors(dataset: pd.DataFrame,
                     config: Dict[str, Any],
                     seed: int = 42) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Convenience function for error synthesis.
    
    Args:
        dataset: Clean DataFrame
        config: Error configuration
        seed: Random seed
    
    Returns:
        Tuple of (corrupted_dataset, ground_truth)
    """
    engine = ErrorSynthesisEngine(seed)
    return engine.synthesize(dataset, config, seed)

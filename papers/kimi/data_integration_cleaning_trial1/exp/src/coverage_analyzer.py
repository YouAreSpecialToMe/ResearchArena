"""
Coverage Analyzer for CESF
Computes 4 coverage metrics for benchmark quality assessment
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from collections import Counter
import math


class CoverageAnalyzer:
    """
    Computes coverage metrics for generated benchmarks:
    1. Type Coverage: fraction of error types present
    2. Distribution Balance: KL-divergence from target
    3. Detectability Score: fraction detectable by constraints
    4. Repair Difficulty Index: log of candidate repair space
    """
    
    def __init__(self, taxonomy):
        self.taxonomy = taxonomy
        self.all_error_types = taxonomy.get_all_types()
    
    def compute_all_metrics(self, ground_truth: List[Dict], 
                           target_distribution: Dict[str, float] = None) -> Dict[str, float]:
        """Compute all 4 coverage metrics"""
        return {
            'type_coverage': self.type_coverage(ground_truth),
            'distribution_balance': self.distribution_balance(ground_truth, target_distribution),
            'detectability_score': self.detectability_score(ground_truth),
            'repair_difficulty': self.repair_difficulty_index(ground_truth)
        }
    
    def type_coverage(self, ground_truth: List[Dict], min_count: int = 1) -> float:
        """
        Type Coverage@k: Fraction of error types with at least k examples
        
        Args:
            ground_truth: List of error records with 'error_type' field
            min_count: Minimum count threshold (k)
        
        Returns:
            Coverage ratio [0, 1]
        """
        if not ground_truth:
            return 0.0
        
        # Count error types
        type_counts = Counter(record['error_type'] for record in ground_truth)
        
        # Count types meeting threshold
        covered_types = sum(1 for count in type_counts.values() if count >= min_count)
        total_types = len(self.all_error_types)
        
        return covered_types / total_types if total_types > 0 else 0.0
    
    def distribution_balance(self, ground_truth: List[Dict],
                            target_distribution: Dict[str, float] = None) -> float:
        """
        Distribution Balance: KL-divergence from target distribution
        Lower is better (0 = perfect match)
        
        Args:
            ground_truth: List of error records
            target_distribution: Target distribution over error types (uniform if None)
        
        Returns:
            KL-divergence value (lower is better)
        """
        if not ground_truth:
            return float('inf')
        
        # Actual distribution
        type_counts = Counter(record['error_type'] for record in ground_truth)
        total_errors = len(ground_truth)
        
        actual_dist = {t: type_counts.get(t, 0) / total_errors 
                      for t in self.all_error_types}
        
        # Target distribution (uniform if not specified)
        if target_distribution is None:
            target_dist = {t: 1.0 / len(self.all_error_types) 
                          for t in self.all_error_types}
        else:
            target_dist = target_distribution
        
        # Compute KL-divergence: KL(P_target || P_actual)
        kl_div = 0.0
        for error_type in self.all_error_types:
            p_target = target_dist.get(error_type, 0)
            p_actual = actual_dist.get(error_type, 0)
            
            if p_target > 0:
                if p_actual > 0:
                    kl_div += p_target * math.log(p_target / p_actual)
                else:
                    # Infinite KL divergence if actual is 0 but target > 0
                    kl_div += p_target * math.log(p_target / 1e-10)
        
        return kl_div
    
    def detectability_score(self, ground_truth: List[Dict],
                           constraints: Dict = None) -> float:
        """
        Detectability Score: Fraction of errors detectable by given constraints
        
        For now, uses heuristics based on error type:
        - Structural errors: high detectability (constraints catch them)
        - Syntactic errors: medium detectability (patterns catch some)
        - Semantic errors: low detectability (require domain knowledge)
        
        Args:
            ground_truth: List of error records
            constraints: Dict of constraints (FDs, DCs) for detection
        
        Returns:
            Detectability ratio [0, 1]
        """
        if not ground_truth:
            return 0.0
        
        # Detectability by error type (heuristic-based)
        detectability_map = {
            # Syntactic - pattern-based detection
            'typo': 0.6,
            'formatting': 0.8,
            'whitespace': 0.9,
            # Structural - constraint-based detection
            'fd_violation': 0.95,
            'dc_violation': 0.95,
            'key_violation': 1.0,
            # Semantic - harder to detect
            'outlier': 0.7,
            'implausible': 0.5
        }
        
        total_detectability = 0.0
        for record in ground_truth:
            error_type = record['error_type']
            detectability = detectability_map.get(error_type, 0.5)
            total_detectability += detectability
        
        return total_detectability / len(ground_truth)
    
    def repair_difficulty_index(self, ground_truth: List[Dict],
                               domain_sizes: Dict[str, int] = None) -> float:
        """
        Repair Difficulty Index: Log of candidate repair space
        Higher values indicate harder-to-repair errors
        
        Args:
            ground_truth: List of error records
            domain_sizes: Dict mapping columns to domain sizes
        
        Returns:
            Average log repair candidates per error
        """
        if not ground_truth:
            return 0.0
        
        # Difficulty by error type (log of typical candidate space)
        difficulty_map = {
            # Syntactic - bounded candidate space
            'typo': math.log(5),          # ~5 keyboard neighbors
            'formatting': math.log(10),   # ~10 format variants
            'whitespace': math.log(4),    # ~4 whitespace patterns
            # Structural - depends on domain
            'fd_violation': math.log(20), # Depends on FD determinants
            'dc_violation': math.log(15),
            'key_violation': math.log(50), # Unique keys
            # Semantic - large/unbounded candidate space
            'outlier': math.log(100),     # Statistical outliers
            'implausible': math.log(200)  # Domain violations
        }
        
        total_difficulty = 0.0
        for record in ground_truth:
            error_type = record['error_type']
            difficulty = difficulty_map.get(error_type, math.log(10))
            total_difficulty += difficulty
        
        return total_difficulty / len(ground_truth)
    
    def generate_report(self, ground_truth: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive coverage report"""
        metrics = self.compute_all_metrics(ground_truth)
        
        # Type distribution
        type_counts = Counter(record['error_type'] for record in ground_truth)
        type_distribution = {t: type_counts.get(t, 0) for t in self.all_error_types}
        
        # Coverage by dimension
        dimension_counts = {}
        for error_type, count in type_counts.items():
            dim = self.taxonomy.get_dimension(error_type)
            dimension_counts[dim] = dimension_counts.get(dim, 0) + count
        
        return {
            'metrics': metrics,
            'type_distribution': type_distribution,
            'dimension_distribution': dimension_counts,
            'total_errors': len(ground_truth),
            'unique_types_covered': len(type_counts)
        }

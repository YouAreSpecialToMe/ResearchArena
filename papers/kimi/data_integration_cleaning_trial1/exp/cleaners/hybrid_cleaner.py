"""
Hybrid Cleaner
Ensemble of statistical, FD-based, and pattern-based cleaning
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from .statistical_detector import StatisticalOutlierDetector
from .fd_repair import FDBasedRepair
from .pattern_repair import PatternBasedRepair


class HybridCleaner:
    """Ensemble cleaner combining multiple methods"""
    
    def __init__(self, fds: List[Dict] = None):
        self.statistical = StatisticalOutlierDetector()
        self.fd_repair = FDBasedRepair(fds)
        self.pattern = PatternBasedRepair()
    
    def detect(self, dataset: pd.DataFrame) -> Set[Tuple[int, str]]:
        """
        Detect errors using all methods and combine results.
        
        Returns:
            Set of (row, column) tuples indicating detected errors
        """
        # Get detections from each method
        statistical_errors = self.statistical.detect(dataset)
        fd_errors = self.fd_repair.detect(dataset)
        pattern_errors = self.pattern.detect(dataset)
        
        # Union of all detections (any method can flag)
        all_errors = statistical_errors | fd_errors | pattern_errors
        
        return all_errors
    
    def repair(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Apply repairs using all methods sequentially.
        
        Returns:
            Repaired DataFrame
        """
        repaired = dataset.copy()
        
        # Apply repairs in order
        repaired = self.pattern.repair(repaired)
        repaired = self.fd_repair.repair(repaired)
        
        return repaired
    
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
        
        # Also get individual method performance
        stat_eval = self.statistical.evaluate(dataset, ground_truth)
        fd_eval = self.fd_repair.evaluate(dataset, ground_truth)
        pattern_eval = self.pattern.evaluate(dataset, ground_truth)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': len(true_positives),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'components': {
                'statistical': stat_eval,
                'fd': fd_eval,
                'pattern': pattern_eval
            }
        }

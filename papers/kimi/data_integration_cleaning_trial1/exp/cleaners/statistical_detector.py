"""
Statistical Outlier Detector
Detects outliers using Z-score and IQR methods
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set


class StatisticalOutlierDetector:
    """Detects outliers using statistical methods"""
    
    def __init__(self, method: str = 'zscore', threshold: float = 3.0):
        self.method = method
        self.threshold = threshold
    
    def detect(self, dataset: pd.DataFrame) -> Set[Tuple[int, str]]:
        """
        Detect outliers in the dataset.
        
        Returns:
            Set of (row, column) tuples indicating detected errors
        """
        errors = set()
        
        for col in dataset.columns:
            # Try to convert to numeric
            numeric_data = pd.to_numeric(dataset[col], errors='coerce')
            
            if numeric_data.isna().all():
                continue
            
            # Get valid (non-null) indices
            valid_mask = numeric_data.notna()
            valid_data = numeric_data[valid_mask]
            
            if len(valid_data) < 2:
                continue
            
            # Detect outliers
            outlier_indices = self._detect_outliers(valid_data)
            
            # Map back to original indices
            original_indices = valid_data.index[outlier_indices]
            for idx in original_indices:
                errors.add((int(idx), col))
        
        return errors
    
    def _detect_outliers(self, data: pd.Series) -> np.ndarray:
        """Detect outliers using specified method"""
        if self.method == 'zscore':
            mean = data.mean()
            std = data.std()
            if std == 0:
                return np.zeros(len(data), dtype=bool)
            z_scores = np.abs((data - mean) / std)
            return z_scores > self.threshold
        
        elif self.method == 'iqr':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (data < lower) | (data > upper)
        
        else:
            return np.zeros(len(data), dtype=bool)
    
    def evaluate(self, dataset: pd.DataFrame, ground_truth: List[Dict]) -> Dict:
        """
        Evaluate detection performance.
        
        Returns:
            Dict with precision, recall, F1
        """
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

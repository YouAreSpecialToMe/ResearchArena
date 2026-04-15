"""
Pattern-Based Repair
Detects formatting errors using regex patterns
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Set


class PatternBasedRepair:
    """Detects pattern/formatting violations"""
    
    # Common patterns
    PATTERNS = {
        'zip': re.compile(r'^\d{5}(-\d{4})?$'),
        'phone': re.compile(r'^(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'),
        'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        'date_iso': re.compile(r'^\d{4}-\d{2}-\d{2}$'),
        'date_us': re.compile(r'^\d{2}/\d{2}/\d{4}$'),
        'time_24': re.compile(r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$'),
    }
    
    def __init__(self):
        self.column_patterns = {}
    
    def detect(self, dataset: pd.DataFrame) -> Set[Tuple[int, str]]:
        """
        Detect pattern violations.
        
        Returns:
            Set of (row, column) tuples indicating violations
        """
        errors = set()
        
        # Infer patterns for each column
        self._infer_patterns(dataset)
        
        for col, pattern_info in self.column_patterns.items():
            if pattern_info is None:
                continue
            
            # Skip if column doesn't exist in this dataset
            if col not in dataset.columns:
                continue
            
            pattern_type = pattern_info['type']
            expected_pattern = self.PATTERNS.get(pattern_type)
            
            if not expected_pattern:
                continue
            
            for idx, value in dataset[col].items():
                str_value = str(value).strip()
                if not expected_pattern.match(str_value):
                    errors.add((int(idx), col))
        
        return errors
    
    def _infer_patterns(self, dataset: pd.DataFrame):
        """Infer expected patterns for each column"""
        for col in dataset.columns:
            col_lower = col.lower()
            
            # Match column name to pattern
            if 'zip' in col_lower or 'postal' in col_lower:
                self.column_patterns[col] = {'type': 'zip'}
            elif 'phone' in col_lower or 'tel' in col_lower:
                self.column_patterns[col] = {'type': 'phone'}
            elif 'email' in col_lower:
                self.column_patterns[col] = {'type': 'email'}
            elif 'date' in col_lower:
                self.column_patterns[col] = {'type': 'date_iso'}
            elif 'time' in col_lower:
                self.column_patterns[col] = {'type': 'time_24'}
            else:
                # Try to detect from data
                self.column_patterns[col] = self._detect_from_data(dataset[col])
    
    def _detect_from_data(self, series: pd.Series) -> Dict:
        """Try to detect pattern from data samples"""
        samples = series.dropna().astype(str).head(100)
        
        for pattern_name, pattern in self.PATTERNS.items():
            matches = sum(1 for s in samples if pattern.match(str(s).strip()))
            if matches / len(samples) > 0.8:
                return {'type': pattern_name}
        
        return None
    
    def repair(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to repair pattern violations (simplified).
        
        Returns:
            Repaired DataFrame
        """
        # Simplified: just trim whitespace for now
        repaired = dataset.copy()
        
        for col in repaired.columns:
            if repaired[col].dtype == object:
                repaired[col] = repaired[col].astype(str).str.strip()
        
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
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': len(true_positives),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives)
        }

"""Evaluation metrics for IntrospectBench."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error
from typing import Dict, List, Tuple


def compute_detection_metrics(predictions: List[int], labels: List[int]) -> Dict:
    """Compute metrics for error detection (binary classification)."""
    if len(predictions) == 0:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary', zero_division=0)
    precision = f1_score(labels, predictions, average='binary', zero_division=0)
    recall = f1_score(labels, predictions, average='binary', zero_division=0)
    
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall)
    }


def compute_localization_metrics(predictions: List[int], labels: List[int], 
                                  max_steps: List[int] = None) -> Dict:
    """Compute metrics for error localization (step identification)."""
    if len(predictions) == 0:
        return {"exact_match": 0.0, "mae": 0.0, "within_1": 0.0, "within_2": 0.0}
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    exact_match = np.mean(predictions == labels)
    mae = mean_absolute_error(labels, predictions)
    
    within_1 = np.mean(np.abs(predictions - labels) <= 1)
    within_2 = np.mean(np.abs(predictions - labels) <= 2)
    
    return {
        "exact_match": float(exact_match),
        "mae": float(mae),
        "within_1": float(within_1),
        "within_2": float(within_2)
    }


def compute_characterization_metrics(predictions: List[str], labels: List[str], 
                                      error_types: List[str]) -> Dict:
    """Compute metrics for error characterization (type classification)."""
    if len(predictions) == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0, "f1_per_type": {}}
    
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_per_type = f1_score(labels, predictions, labels=error_types, average=None, zero_division=0)
    
    f1_per_type_dict = {t: float(f) for t, f in zip(error_types, f1_per_type)}
    
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_per_type": f1_per_type_dict
    }


def compute_introspection_score(detection_acc: float, localization_acc: float, 
                                 characterization_acc: float) -> float:
    """Compute composite Introspection Score."""
    return (detection_acc + localization_acc + characterization_acc) / 3.0


def compute_pwls(predictions: List[int], labels: List[int]) -> float:
    """Compute Position-Weighted Localization Score."""
    if len(predictions) == 0:
        return 0.0
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Weight by 1/sqrt(step)
    weights = 1.0 / np.sqrt(labels)
    correct = (predictions == labels).astype(float)
    
    return float(np.sum(weights * correct) / np.sum(weights))


def compute_dcs(domain_scores: List[float]) -> float:
    """Compute Domain Calibration Score."""
    if len(domain_scores) == 0:
        return 0.0
    
    mean_score = np.mean(domain_scores)
    std_score = np.std(domain_scores)
    
    if mean_score == 0:
        return 0.0
    
    return 1.0 - (std_score / mean_score)


def compute_self_other_gap(self_f1: float, other_f1: float) -> float:
    """Compute Self-Other Gap metric."""
    return other_f1 - self_f1

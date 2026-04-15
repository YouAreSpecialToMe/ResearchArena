"""
PhaseMine: Core implementation for detecting feature emergence phase transitions
via dynamic sparse probing.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PhaseMineDetector:
    """
    PhaseMine detector for identifying phase transitions in feature emergence.
    
    Uses L1-regularized linear probes to track sparsity patterns across training checkpoints.
    """
    
    def __init__(
        self,
        C: float = 0.1,
        solver: str = 'saga',
        max_iter: int = 500,
        emergence_threshold: float = 0.001,
        concentration_k: float = 0.05,  # top-5%
        random_state: int = 42
    ):
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.emergence_threshold = emergence_threshold
        self.concentration_k = concentration_k
        self.random_state = random_state
        
        self.probe_history = []
        self.metrics_history = []
        
    def train_probe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        checkpoint_step: int
    ) -> Dict:
        """Train L1-regularized probe and compute metrics."""
        # Train L1-regularized logistic regression
        probe = LogisticRegression(
            penalty='l1',
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        probe.fit(X, y)
        
        # Compute metrics
        y_pred = probe.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Weight statistics
        weights = probe.coef_[0] if probe.coef_.shape[0] == 1 else probe.coef_.mean(axis=0)
        
        # Sparsity Trajectory: L1 norm of weights
        l1_norm = np.sum(np.abs(weights))
        
        # L0 sparsity: count of non-zero weights
        l0_norm = np.sum(np.abs(weights) > 1e-6)
        
        # Concentration Score: ratio of top-k% weights to total
        abs_weights = np.abs(weights)
        k = max(1, int(len(weights) * self.concentration_k))
        top_k_indices = np.argsort(abs_weights)[-k:]
        top_k_sum = np.sum(abs_weights[top_k_indices])
        concentration_score = top_k_sum / (l1_norm + 1e-10)
        
        result = {
            'checkpoint_step': checkpoint_step,
            'probe': probe,
            'accuracy': accuracy,
            'l1_norm': l1_norm,
            'l0_norm': l0_norm,
            'concentration_score': concentration_score,
            'weights': weights.copy(),
            'bias': probe.intercept_[0] if hasattr(probe, 'intercept_') else 0
        }
        
        self.probe_history.append(result)
        return result
    
    def compute_emergence_sharpness(self) -> np.ndarray:
        """
        Compute Emergence Sharpness: second derivative of accuracy.
        ES(t) = (Acc(t+1) - 2*Acc(t) + Acc(t-1))
        """
        if len(self.probe_history) < 3:
            return np.array([])
        
        accuracies = np.array([h['accuracy'] for h in self.probe_history])
        
        # Compute second derivative using central differences
        es = np.zeros_like(accuracies)
        es[1:-1] = accuracies[2:] - 2 * accuracies[1:-1] + accuracies[:-2]
        es[0] = es[1]  # boundary
        es[-1] = es[-2]  # boundary
        
        return es
    
    def detect_phase_transitions(self) -> List[Dict]:
        """
        Detect phase transitions based on Emergence Sharpness and Concentration Score.
        
        A transition is flagged when:
        - ES > threshold AND
        - CS shows increase
        """
        if len(self.probe_history) < 3:
            return []
        
        es = self.compute_emergence_sharpness()
        transitions = []
        
        for i in range(1, len(self.probe_history)):
            curr_es = es[i]
            curr_cs = self.probe_history[i]['concentration_score']
            prev_cs = self.probe_history[i-1]['concentration_score']
            
            # Check conditions
            es_condition = curr_es > self.emergence_threshold
            cs_condition = curr_cs > prev_cs
            
            if es_condition and cs_condition:
                transitions.append({
                    'checkpoint_step': self.probe_history[i]['checkpoint_step'],
                    'emergence_sharpness': curr_es,
                    'concentration_score': curr_cs,
                    'accuracy': self.probe_history[i]['accuracy'],
                    'index': i
                })
        
        return transitions
    
    def get_metrics_trajectory(self) -> Dict[str, np.ndarray]:
        """Get metrics trajectories across all checkpoints."""
        if not self.probe_history:
            return {}
        
        return {
            'checkpoint_steps': np.array([h['checkpoint_step'] for h in self.probe_history]),
            'accuracy': np.array([h['accuracy'] for h in self.probe_history]),
            'l1_norm': np.array([h['l1_norm'] for h in self.probe_history]),
            'l0_norm': np.array([h['l0_norm'] for h in self.probe_history]),
            'concentration_score': np.array([h['concentration_score'] for h in self.probe_history]),
            'emergence_sharpness': self.compute_emergence_sharpness()
        }


class DenseProbeBaseline:
    """Dense linear probing baseline without sparsity regularization."""
    
    def __init__(self, C: float = 1.0, max_iter: int = 500, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.probe_history = []
    
    def train_probe(self, X: np.ndarray, y: np.ndarray, checkpoint_step: int) -> Dict:
        """Train dense probe."""
        probe = LogisticRegression(
            penalty='l2',
            C=self.C,
            solver='lbfgs',
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        probe.fit(X, y)
        
        y_pred = probe.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        weights = probe.coef_[0] if probe.coef_.shape[0] == 1 else probe.coef_.mean(axis=0)
        l2_norm = np.sqrt(np.sum(weights ** 2))
        
        result = {
            'checkpoint_step': checkpoint_step,
            'probe': probe,
            'accuracy': accuracy,
            'l2_norm': l2_norm,
            'weights': weights.copy()
        }
        
        self.probe_history.append(result)
        return result
    
    def detect_transitions(self, threshold: float = 0.001) -> List[Dict]:
        """Detect transitions using accuracy second derivative."""
        if len(self.probe_history) < 3:
            return []
        
        accuracies = np.array([h['accuracy'] for h in self.probe_history])
        
        # Second derivative
        es = np.zeros_like(accuracies)
        es[1:-1] = accuracies[2:] - 2 * accuracies[1:-1] + accuracies[:-2]
        
        transitions = []
        for i in range(1, len(self.probe_history) - 1):
            if es[i] > threshold:
                transitions.append({
                    'checkpoint_step': self.probe_history[i]['checkpoint_step'],
                    'emergence_sharpness': es[i],
                    'accuracy': accuracies[i],
                    'index': i
                })
        
        return transitions


def compute_kendall_tau(transitions1: List[int], transitions2: List[int], max_step: int) -> Tuple[float, float]:
    """
    Compute Kendall's tau correlation between two sets of transition timings.
    
    Args:
        transitions1: List of checkpoint steps for method 1
        transitions2: List of checkpoint steps for method 2
        max_step: Maximum step for normalization
    
    Returns:
        tau: Kendall's tau correlation
        agreement_rate: Percentage within ±10% of training steps
    """
    from scipy.stats import kendalltau
    
    if not transitions1 or not transitions2:
        return 0.0, 0.0
    
    # For Kendall's tau, we need paired data
    # We'll use the detected transitions and also include non-transition points
    # Simplified: compare the sorted transition lists
    
    # Pad to same length
    max_len = max(len(transitions1), len(transitions2))
    t1 = list(transitions1) + [max_step] * (max_len - len(transitions1))
    t2 = list(transitions2) + [max_step] * (max_len - len(transitions2))
    
    tau, p_value = kendalltau(t1, t2)
    
    # Compute agreement rate (% within ±10%)
    agreement_count = 0
    threshold = max_step * 0.1
    
    for t in transitions1:
        for s in transitions2:
            if abs(t - s) <= threshold:
                agreement_count += 1
                break
    
    agreement_rate = agreement_count / len(transitions1) if transitions1 else 0.0
    
    return tau, agreement_rate

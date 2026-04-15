"""
Item Response Theory (IRT) utilities for EVOLVE experiments.
Implements 2PL IRT model with online calibration capabilities.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import json


class IRT2PL:
    """Two-Parameter Logistic (2PL) IRT Model with optional guessing parameter."""
    
    def __init__(self, n_items: int, n_persons: int):
        self.n_items = n_items
        self.n_persons = n_persons
        # Item parameters: a (discrimination), b (difficulty), c (guessing)
        self.a = np.ones(n_items) * 1.0  # Initialize discrimination
        self.b = np.zeros(n_items)       # Initialize difficulty
        self.c = np.zeros(n_items)       # Initialize guessing (0 for 2PL)
        # Person parameters: theta (ability)
        self.theta = np.zeros(n_persons)
        
    def probability(self, theta: float, a: float, b: float, c: float = 0.0) -> float:
        """Compute probability of correct response using 3PL model."""
        exp_arg = -a * (theta - b)
        # Clip to prevent overflow
        exp_arg = np.clip(exp_arg, -500, 500)
        p = c + (1 - c) / (1 + np.exp(exp_arg))
        return p
    
    def fisher_information(self, theta: float, a: float, b: float, c: float = 0.0) -> float:
        """Compute Fisher information for an item at ability theta."""
        p = self.probability(theta, a, b, c)
        q = 1 - p
        # Information = a^2 * (p - c)^2 / ((1 - c)^2 * p * q)
        if abs(1 - c) < 1e-10:
            return a**2 * p * q
        info = (a**2 * (p - c)**2) / ((1 - c)**2 * p * q)
        return info
    
    def estimate_theta_eap(self, responses: np.ndarray, 
                          item_params: Optional[Tuple] = None,
                          prior_mean: float = 0.0, 
                          prior_std: float = 1.0,
                          n_quadrature: int = 21) -> Tuple[float, float]:
        """
        Estimate ability using Expected A Posteriori (EAP).
        Returns: (theta_estimate, standard_error)
        """
        if item_params is None:
            a, b, c = self.a, self.b, self.c
        else:
            a, b, c = item_params
            
        # Gauss-Hermite quadrature points and weights
        quadrature_points = np.linspace(-4, 4, n_quadrature)
        
        # Compute likelihood at each quadrature point
        likelihoods = np.ones(n_quadrature)
        for i, theta_q in enumerate(quadrature_points):
            for j, resp in enumerate(responses):
                if np.isnan(resp):
                    continue
                p = self.probability(theta_q, a[j], b[j], c[j])
                likelihoods[i] *= p if resp == 1 else (1 - p)
        
        # Prior
        prior = norm.pdf(quadrature_points, prior_mean, prior_std)
        
        # Posterior
        posterior = likelihoods * prior
        posterior_sum = np.sum(posterior)
        
        if posterior_sum < 1e-10:
            return prior_mean, 1.0
        
        # EAP estimate
        theta_est = np.sum(quadrature_points * posterior) / posterior_sum
        
        # Posterior variance
        variance = np.sum((quadrature_points - theta_est)**2 * posterior) / posterior_sum
        se = np.sqrt(variance)
        
        return theta_est, se
    
    def calibrate_items_mml(self, responses: np.ndarray, 
                           max_iter: int = 100,
                           tolerance: float = 1e-4) -> Dict:
        """
        Calibrate item parameters using Marginal Maximum Likelihood.
        responses: (n_persons, n_items) array of 0/1 responses
        """
        n_persons, n_items = responses.shape
        
        # Initialize parameters
        self.a = np.ones(n_items) * 1.0
        self.b = np.zeros(n_items)
        self.c = np.zeros(n_items)
        
        # Quadrature points for numerical integration
        theta_grid = np.linspace(-4, 4, 21)
        
        for iteration in range(max_iter):
            a_old = self.a.copy()
            b_old = self.b.copy()
            
            # Estimate person parameters
            for p in range(n_persons):
                self.theta[p], _ = self.estimate_theta_eap(responses[p], 
                                                           (self.a, self.b, self.c))
            
            # Update item parameters using Newton-Raphson
            for i in range(n_items):
                valid_mask = ~np.isnan(responses[:, i])
                if np.sum(valid_mask) < 10:
                    continue
                    
                r_i = responses[valid_mask, i]
                theta_i = self.theta[valid_mask]
                
                # Update difficulty b
                def neg_loglik_b(b_i):
                    ll = 0
                    for j, theta in enumerate(theta_i):
                        p = self.probability(theta, self.a[i], b_i, self.c[i])
                        p = np.clip(p, 1e-10, 1 - 1e-10)
                        ll += r_i[j] * np.log(p) + (1 - r_i[j]) * np.log(1 - p)
                    return -ll
                
                result_b = minimize(neg_loglik_b, self.b[i], method='L-BFGS-B',
                                   bounds=[(-4, 4)])
                self.b[i] = result_b.x[0]
                
                # Update discrimination a
                def neg_loglik_a(a_i):
                    if a_i <= 0:
                        return 1e10
                    ll = 0
                    for j, theta in enumerate(theta_i):
                        p = self.probability(theta, a_i, self.b[i], self.c[i])
                        p = np.clip(p, 1e-10, 1 - 1e-10)
                        ll += r_i[j] * np.log(p) + (1 - r_i[j]) * np.log(1 - p)
                    return -ll
                
                result_a = minimize(neg_loglik_a, self.a[i], method='L-BFGS-B',
                                   bounds=[(0.1, 3.0)])
                self.a[i] = result_a.x[0]
            
            # Check convergence
            a_diff = np.max(np.abs(self.a - a_old))
            b_diff = np.max(np.abs(self.b - b_old))
            
            if a_diff < tolerance and b_diff < tolerance:
                break
        
        return {
            'a': self.a.copy(),
            'b': self.b.copy(),
            'c': self.c.copy(),
            'theta': self.theta.copy(),
            'iterations': iteration + 1
        }
    
    def compute_infit_outfit(self, responses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute infit and outfit statistics for item fit.
        Infit: weighted mean square (more sensitive to inliers)
        Outfit: unweighted mean square (more sensitive to outliers)
        """
        n_persons, n_items = responses.shape
        infit = np.zeros(n_items)
        outfit = np.zeros(n_items)
        
        for i in range(n_items):
            residuals = []
            variances = []
            
            for p in range(n_persons):
                if np.isnan(responses[p, i]):
                    continue
                
                p_ij = self.probability(self.theta[p], self.a[i], self.b[i], self.c[i])
                residual = responses[p, i] - p_ij
                variance = p_ij * (1 - p_ij)
                
                residuals.append(residual)
                variances.append(variance)
            
            if len(residuals) == 0:
                infit[i] = outfit[i] = 1.0
                continue
            
            residuals = np.array(residuals)
            variances = np.array(variances)
            
            # Outfit (unweighted)
            outfit[i] = np.mean(residuals**2 / variances)
            
            # Infit (weighted)
            weights = variances
            infit[i] = np.sum(weights * (residuals**2 / variances)) / np.sum(weights)
        
        return infit, outfit
    
    def online_update(self, new_responses: np.ndarray, 
                     learning_rate: float = 0.1) -> Dict:
        """
        Online update of item parameters with new responses.
        Uses gradient-based updates.
        """
        n_persons, n_items = new_responses.shape
        
        for i in range(n_items):
            valid_mask = ~np.isnan(new_responses[:, i])
            if np.sum(valid_mask) == 0:
                continue
            
            # Compute gradient for item i
            grad_a = 0
            grad_b = 0
            
            for p in range(n_persons):
                if np.isnan(new_responses[p, i]):
                    continue
                
                y = new_responses[p, i]
                theta_p = self.theta[p] if p < len(self.theta) else 0.0
                
                p_ij = self.probability(theta_p, self.a[i], self.b[i], self.c[i])
                
                # Gradients for 2PL model
                grad_b += (y - p_ij) * self.a[i]
                grad_a += (y - p_ij) * (theta_p - self.b[i])
            
            # Update parameters
            self.b[i] += learning_rate * grad_b / np.sum(valid_mask)
            self.a[i] = np.clip(self.a[i] + learning_rate * grad_a / np.sum(valid_mask), 0.1, 3.0)
        
        return {'a': self.a.copy(), 'b': self.b.copy()}


class AdaptiveTestingEngine:
    """Computerized Adaptive Testing (CAT) engine for efficient evaluation."""
    
    def __init__(self, irt_model: IRT2PL, stopping_se: float = 0.3, max_items: int = 50):
        self.irt = irt_model
        self.stopping_se = stopping_se
        self.max_items = max_items
        self.selected_items = []
        self.responses = []
        
    def select_next_item(self, current_theta: float, 
                        available_items: List[int],
                        top_k: int = 5) -> int:
        """Select next item using Fisher information with randomesque approach."""
        if not available_items:
            return None
        
        # Compute information for all available items
        info_scores = []
        for item_idx in available_items:
            info = self.irt.fisher_information(
                current_theta,
                self.irt.a[item_idx],
                self.irt.b[item_idx],
                self.irt.c[item_idx]
            )
            info_scores.append((item_idx, info))
        
        # Sort by information (descending)
        info_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Randomesque: randomly select from top-k
        k = min(top_k, len(info_scores))
        selected = np.random.choice([x[0] for x in info_scores[:k]])
        
        return selected
    
    def run_adaptive_test(self, true_responses: np.ndarray,
                         item_order: Optional[List[int]] = None) -> Dict:
        """
        Run adaptive test for a single person.
        true_responses: array of true responses for all items
        """
        n_items = len(true_responses)
        available_items = set(range(n_items))
        
        self.selected_items = []
        self.responses = []
        
        # Initialize ability estimate
        theta_est = 0.0
        se = 1.0
        
        for step in range(self.max_items):
            # Select next item
            next_item = self.select_next_item(theta_est, list(available_items))
            if next_item is None:
                break
            
            # Record response
            response = true_responses[next_item]
            self.selected_items.append(next_item)
            self.responses.append(response)
            available_items.remove(next_item)
            
            # Update ability estimate
            responses_array = np.array(self.responses)
            items_array = np.array(self.selected_items)
            
            # Create partial response array
            partial_responses = np.full(n_items, np.nan)
            partial_responses[items_array] = responses_array
            
            theta_est, se = self.irt.estimate_theta_eap(partial_responses)
            
            # Check stopping rule
            if se < self.stopping_se:
                break
        
        return {
            'theta': theta_est,
            'se': se,
            'n_items': len(self.selected_items),
            'selected_items': self.selected_items.copy(),
            'responses': self.responses.copy()
        }


def generate_synthetic_responses(n_persons: int, n_items: int,
                                 theta_range: Tuple[float, float] = (-3, 3),
                                 seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic response matrix for testing.
    Returns: (responses, true_theta, item_params)
    """
    np.random.seed(seed)
    
    # Generate person abilities
    true_theta = np.random.normal(0, 1.5, n_persons)
    true_theta = np.clip(true_theta, theta_range[0], theta_range[1])
    
    # Generate item parameters
    a = np.random.lognormal(0, 0.3, n_items)  # Discrimination
    a = np.clip(a, 0.3, 2.5)
    b = np.linspace(-2, 2, n_items) + np.random.normal(0, 0.3, n_items)  # Difficulty
    c = np.zeros(n_items)  # No guessing for now
    
    # Generate responses
    responses = np.zeros((n_persons, n_items))
    irt = IRT2PL(n_items, n_persons)
    irt.a, irt.b, irt.c = a, b, c
    
    for p in range(n_persons):
        for i in range(n_items):
            p_correct = irt.probability(true_theta[p], a[i], b[i], c[i])
            responses[p, i] = 1 if np.random.random() < p_correct else 0
    
    return responses, true_theta, {'a': a, 'b': b, 'c': c}


def save_item_parameters(item_params: Dict, filepath: str):
    """Save item parameters to JSON file."""
    data = {
        'a': item_params['a'].tolist(),
        'b': item_params['b'].tolist(),
        'c': item_params['c'].tolist()
    }
    if 'infit' in item_params:
        data['infit'] = item_params['infit'].tolist()
    if 'outfit' in item_params:
        data['outfit'] = item_params['outfit'].tolist()
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_item_parameters(filepath: str) -> Dict:
    """Load item parameters from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return {
        'a': np.array(data['a']),
        'b': np.array(data['b']),
        'c': np.array(data['c']),
        'infit': np.array(data.get('infit', [])),
        'outfit': np.array(data.get('outfit', []))
    }

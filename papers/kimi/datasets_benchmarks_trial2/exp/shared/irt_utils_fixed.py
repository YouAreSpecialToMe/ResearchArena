"""
Item Response Theory (IRT) utilities for EVOLVE experiments.
Implements 2PL IRT model with online calibration capabilities.
FIXED VERSION - Addresses EAP estimation and calibration bugs.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import json


class IRT2PL:
    """Two-Parameter Logistic (2PL) IRT Model with optional guessing parameter."""
    
    def __init__(self, n_items: int, n_persons: int = 1):
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
        return np.clip(p, 1e-10, 1 - 1e-10)
    
    def fisher_information(self, theta: float, a: float, b: float, c: float = 0.0) -> float:
        """Compute Fisher information for an item at ability theta."""
        p = self.probability(theta, a, b, c)
        q = 1 - p
        
        # Information for 2PL: I_i(theta) = a^2 * p * q
        # For 3PL: I_i(theta) = a^2 * (p - c)^2 * q / ((1 - c)^2 * p)
        if abs(1 - c) < 1e-10:
            info = a**2 * p * q
        else:
            info = (a**2 * (p - c)**2 * q) / ((1 - c)**2 * p)
        
        return info
    
    def log_likelihood(self, theta: float, responses: np.ndarray, 
                       item_indices: np.ndarray) -> float:
        """Compute log-likelihood of responses given ability theta."""
        ll = 0.0
        for idx in item_indices:
            if idx < len(self.a):
                p = self.probability(theta, self.a[idx], self.b[idx], self.c[idx])
                resp = responses[idx]
                if not np.isnan(resp):
                    ll += resp * np.log(p) + (1 - resp) * np.log(1 - p)
        return ll
    
    def estimate_theta_mle(self, responses: np.ndarray, 
                          item_indices: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Estimate ability using Maximum Likelihood Estimation (MLE).
        Returns: (theta_estimate, standard_error)
        """
        if item_indices is None:
            item_indices = np.where(~np.isnan(responses))[0]
        
        if len(item_indices) == 0:
            return 0.0, 1.0
        
        # Optimize log-likelihood
        result = minimize_scalar(
            lambda t: -self.log_likelihood(t, responses, item_indices),
            bounds=(-4, 4),
            method='bounded'
        )
        
        theta_est = result.x
        
        # Compute SE using observed information
        info = 0.0
        for idx in item_indices:
            info += self.fisher_information(theta_est, self.a[idx], self.b[idx], self.c[idx])
        
        se = 1.0 / np.sqrt(max(info, 1e-10))
        
        return theta_est, se
    
    def estimate_theta_eap(self, responses: np.ndarray,
                          prior_mean: float = 0.0, 
                          prior_std: float = 1.0,
                          n_quadrature: int = 41) -> Tuple[float, float]:
        """
        Estimate ability using Expected A Posteriori (EAP).
        FIXED: Properly handles partial response arrays.
        Returns: (theta_estimate, standard_error)
        """
        # Get valid item indices
        valid_mask = ~np.isnan(responses)
        if not np.any(valid_mask):
            return prior_mean, prior_std
        
        # Gauss-Hermite quadrature points (transformed from [-1,1] to [-4,4])
        quadrature_points = np.linspace(-4, 4, n_quadrature)
        quadrature_weights = np.ones(n_quadrature) * (8.0 / (n_quadrature - 1))
        quadrature_weights[0] *= 0.5  # Trapezoidal rule endpoints
        quadrature_weights[-1] *= 0.5
        
        # Compute likelihood at each quadrature point
        likelihoods = np.ones(n_quadrature)
        for i, theta_q in enumerate(quadrature_points):
            for j in range(len(responses)):
                if not np.isnan(responses[j]):
                    p = self.probability(theta_q, self.a[j], self.b[j], self.c[j])
                    # Clip p to avoid log(0)
                    p = np.clip(p, 1e-10, 1 - 1e-10)
                    likelihoods[i] *= p if responses[j] == 1 else (1 - p)
        
        # Prior: Normal distribution
        prior = norm.pdf(quadrature_points, prior_mean, prior_std)
        
        # Posterior (unnormalized)
        posterior = likelihoods * prior * quadrature_weights
        posterior_sum = np.sum(posterior)
        
        if posterior_sum < 1e-10:
            # Fallback to prior if likelihood is flat
            return prior_mean, prior_std
        
        # Normalize posterior
        posterior = posterior / posterior_sum
        
        # EAP estimate: mean of posterior
        theta_est = np.sum(quadrature_points * posterior)
        
        # Posterior variance
        variance = np.sum((quadrature_points - theta_est)**2 * posterior)
        se = np.sqrt(variance)
        
        return theta_est, se
    
    def calibrate_items_mml(self, responses: np.ndarray, 
                           max_iter: int = 100,
                           tolerance: float = 1e-4,
                           verbose: bool = False) -> Dict:
        """
        Calibrate item parameters using Marginal Maximum Likelihood.
        FIXED: Actually updates parameters through iterations.
        responses: (n_persons, n_items) array of 0/1 responses
        """
        n_persons, n_items = responses.shape
        
        # Store history to track convergence
        a_history = [self.a.copy()]
        b_history = [self.b.copy()]
        
        for iteration in range(max_iter):
            a_old = self.a.copy()
            b_old = self.b.copy()
            
            # E-step: Estimate person abilities given current item parameters
            for p in range(n_persons):
                self.theta[p], _ = self.estimate_theta_eap(responses[p])
            
            # M-step: Update item parameters
            for i in range(n_items):
                valid_mask = ~np.isnan(responses[:, i])
                n_valid = np.sum(valid_mask)
                
                if n_valid < 5:  # Need minimum responses
                    continue
                
                r_i = responses[valid_mask, i]
                theta_i = self.theta[valid_mask]
                
                # Joint optimization of a and b for this item
                def neg_loglik(params):
                    a_i, b_i = params
                    if a_i <= 0.1:
                        return 1e10
                    ll = 0
                    for j, theta in enumerate(theta_i):
                        p = self.probability(theta, a_i, b_i, self.c[i])
                        ll += r_i[j] * np.log(p) + (1 - r_i[j]) * np.log(1 - p)
                    return -ll
                
                # Optimize with bounds
                result = minimize(
                    neg_loglik,
                    [self.a[i], self.b[i]],
                    method='L-BFGS-B',
                    bounds=[(0.1, 3.0), (-4, 4)]
                )
                
                if result.success:
                    self.a[i], self.b[i] = result.x
            
            # Store history
            a_history.append(self.a.copy())
            b_history.append(self.b.copy())
            
            # Check convergence
            a_diff = np.max(np.abs(self.a - a_old))
            b_diff = np.max(np.abs(self.b - b_old))
            
            if verbose and iteration % 10 == 0:
                print(f"  Iter {iteration}: max_a_diff={a_diff:.6f}, max_b_diff={b_diff:.6f}")
                print(f"    a_mean={np.mean(self.a):.3f}, b_mean={np.mean(self.b):.3f}")
            
            if a_diff < tolerance and b_diff < tolerance:
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                break
        
        return {
            'a': self.a.copy(),
            'b': self.b.copy(),
            'c': self.c.copy(),
            'theta': self.theta.copy(),
            'iterations': iteration + 1,
            'converged': (a_diff < tolerance and b_diff < tolerance),
            'a_history': a_history,
            'b_history': b_history
        }
    
    def compute_infit_outfit(self, responses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute infit and outfit statistics for item fit.
        Infit: weighted mean square (more sensitive to inliers)
        Outfit: unweighted mean square (more sensitive to outliers)
        """
        n_persons, n_items = responses.shape
        infit = np.ones(n_items)  # Default to 1.0 (perfect fit)
        outfit = np.ones(n_items)
        
        for i in range(n_items):
            residuals = []
            variances = []
            
            for p in range(n_persons):
                if np.isnan(responses[p, i]):
                    continue
                
                p_ij = self.probability(self.theta[p], self.a[i], self.b[i], self.c[i])
                residual = responses[p, i] - p_ij
                variance = p_ij * (1 - p_ij)
                
                if variance > 1e-10:  # Avoid division by zero
                    residuals.append(residual)
                    variances.append(variance)
            
            if len(residuals) < 5:
                continue
            
            residuals = np.array(residuals)
            variances = np.array(variances)
            
            # Outfit (unweighted mean square)
            outfit[i] = np.mean((residuals**2) / variances)
            
            # Infit (weighted mean square)
            weights = variances
            infit[i] = np.sum(weights * (residuals**2) / variances) / np.sum(weights)
        
        return infit, outfit
    
    def online_update(self, new_responses: np.ndarray, 
                     learning_rate: float = 0.05) -> Dict:
        """
        Online update of item parameters with new responses.
        FIXED: Properly computes gradients and updates parameters.
        new_responses: (n_persons, n_items) array
        """
        n_persons, n_items = new_responses.shape
        
        # First estimate abilities for new respondents
        for p in range(n_persons):
            self.theta[p], _ = self.estimate_theta_eap(new_responses[p])
        
        # Update item parameters based on residuals
        for i in range(n_items):
            valid_mask = ~np.isnan(new_responses[:, i])
            if np.sum(valid_mask) == 0:
                continue
            
            # Compute gradients
            grad_a = 0.0
            grad_b = 0.0
            n_updates = 0
            
            for p in range(n_persons):
                if np.isnan(new_responses[p, i]):
                    continue
                
                y = new_responses[p, i]
                theta_p = self.theta[p]
                
                p_ij = self.probability(theta_p, self.a[i], self.b[i], self.c[i])
                residual = y - p_ij
                
                # Gradients for 2PL model
                # d(log L)/db = a * (y - p)
                # d(log L)/da = (y - p) * (theta - b)
                grad_b += residual * self.a[i]
                grad_a += residual * (theta_p - self.b[i])
                n_updates += 1
            
            if n_updates > 0:
                # Apply updates with learning rate decay based on number of updates
                lr_eff = learning_rate / (1 + 0.1 * n_updates)
                
                self.b[i] += lr_eff * grad_b / n_updates
                self.a[i] = np.clip(self.a[i] + lr_eff * grad_a / n_updates, 0.1, 3.0)
        
        return {'a': self.a.copy(), 'b': self.b.copy()}


class AdaptiveTestingEngine:
    """Computerized Adaptive Testing (CAT) engine for efficient evaluation."""
    
    def __init__(self, irt_model: IRT2PL, stopping_se: float = 0.3, max_items: int = 50, top_k: int = 5):
        self.irt = irt_model
        self.stopping_se = stopping_se
        self.max_items = max_items
        self.top_k = top_k
        self.selected_items = []
        self.responses = []
        
    def select_next_item(self, current_theta: float, 
                        available_items: List[int]) -> int:
        """
        Select next item using Fisher information with randomesque approach.
        FIXED: Properly computes information and selects from top-k.
        """
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
        
        # Randomesque: randomly select from top-k items
        k = min(self.top_k, len(info_scores))
        top_k_items = [x[0] for x in info_scores[:k]]
        top_k_info = [x[1] for x in info_scores[:k]]
        
        # Weighted random selection based on information
        if len(top_k_items) > 1 and sum(top_k_info) > 0:
            probs = np.array(top_k_info) / sum(top_k_info)
            selected = np.random.choice(top_k_items, p=probs)
        else:
            selected = top_k_items[0]
        
        return selected
    
    def run_adaptive_test(self, true_responses: np.ndarray) -> Dict:
        """
        Run adaptive test for a single person.
        FIXED: Properly updates ability estimate after each response.
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
            
            # Update ability estimate using EAP
            responses_array = np.array(self.responses)
            items_array = np.array(self.selected_items)
            
            # Create partial response array (nan for unadministered items)
            partial_responses = np.full(n_items, np.nan)
            partial_responses[items_array] = responses_array
            
            theta_est, se = self.irt.estimate_theta_eap(partial_responses)
            
            # Check stopping rule
            if se < self.stopping_se and step >= 4:  # Minimum 5 items
                break
        
        return {
            'theta': float(theta_est),
            'se': float(se),
            'n_items': len(self.selected_items),
            'selected_items': self.selected_items.copy(),
            'responses': [int(r) for r in self.responses]
        }


def generate_synthetic_responses(n_persons: int, n_items: int,
                                 theta_range: Tuple[float, float] = (-3, 3),
                                 seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic response matrix for testing.
    Returns: (responses, true_theta, item_params)
    """
    np.random.seed(seed)
    
    # Generate person abilities (mixture of different ability levels)
    true_theta = np.random.normal(0, 1.0, n_persons)
    # Add some high-ability and low-ability models
    true_theta[0] = 2.5  # Very high ability
    true_theta[1] = -2.0  # Low ability
    if n_persons > 2:
        true_theta[2] = 1.5  # High ability
    true_theta = np.clip(true_theta, theta_range[0], theta_range[1])
    
    # Generate item parameters with good spread
    a = np.random.lognormal(0, 0.4, n_items)  # Discrimination
    a = np.clip(a, 0.3, 2.5)
    
    # Difficulty spread across range
    b = np.linspace(-2.5, 2.5, n_items) + np.random.normal(0, 0.5, n_items)
    b = np.clip(b, -3, 3)
    
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
        'a': item_params['a'].tolist() if isinstance(item_params['a'], np.ndarray) else item_params['a'],
        'b': item_params['b'].tolist() if isinstance(item_params['b'], np.ndarray) else item_params['b'],
        'c': item_params['c'].tolist() if isinstance(item_params['c'], np.ndarray) else item_params['c']
    }
    if 'infit' in item_params:
        data['infit'] = item_params['infit'].tolist() if isinstance(item_params['infit'], np.ndarray) else item_params['infit']
    if 'outfit' in item_params:
        data['outfit'] = item_params['outfit'].tolist() if isinstance(item_params['outfit'], np.ndarray) else item_params['outfit']
    if 'valid_mask' in item_params:
        data['valid_mask'] = item_params['valid_mask'].tolist() if isinstance(item_params['valid_mask'], np.ndarray) else item_params['valid_mask']
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_item_parameters(filepath: str) -> Dict:
    """Load item parameters from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    result = {
        'a': np.array(data['a']),
        'b': np.array(data['b']),
        'c': np.array(data['c'])
    }
    if 'infit' in data and len(data['infit']) > 0:
        result['infit'] = np.array(data['infit'])
    if 'outfit' in data and len(data['outfit']) > 0:
        result['outfit'] = np.array(data['outfit'])
    if 'valid_mask' in data and len(data['valid_mask']) > 0:
        result['valid_mask'] = np.array(data['valid_mask'])
    
    return result

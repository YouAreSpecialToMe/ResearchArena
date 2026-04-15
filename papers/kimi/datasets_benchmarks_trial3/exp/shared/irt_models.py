"""
IRT (Item Response Theory) models for DynaScale.
Implements 1PL, 2PL, and 3PL models with Bayesian inference.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm
import warnings


class TwoPLModel:
    """2-Parameter Logistic (2PL) IRT Model.
    
    P(correct|θ, a, b) = 1 / (1 + exp(-a(θ - b)))
    where:
    - θ: ability parameter
    - a: discrimination parameter  
    - b: difficulty parameter
    """
    
    def __init__(self, n_models, n_items, device='cpu'):
        self.n_models = n_models
        self.n_items = n_items
        self.device = device
        
        # Initialize parameters
        self.abilities = np.zeros(n_models)
        self.difficulties = np.zeros(n_items)
        self.discriminations = np.ones(n_items)
        
    def probability(self, abilities=None, difficulties=None, discriminations=None):
        """Compute probability of correct response."""
        if abilities is None:
            abilities = self.abilities
        if difficulties is None:
            difficulties = self.difficulties
        if discriminations is None:
            discriminations = self.discriminations
            
        # Expand dimensions for broadcasting
        theta = abilities[:, np.newaxis]  # (n_models, 1)
        b = difficulties[np.newaxis, :]   # (1, n_items)
        a = discriminations[np.newaxis, :]  # (1, n_items)
        
        z = a * (theta - b)
        return expit(z)
    
    def log_likelihood(self, responses, mask=None):
        """Compute log-likelihood of responses.
        
        Args:
            responses: Binary matrix (n_models, n_items), -1 for missing
            mask: Boolean matrix (n_models, n_items), True for observed
        """
        if mask is None:
            mask = responses >= 0
            
        probs = self.probability()
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        
        # Log-likelihood for observed responses
        ll = responses[mask] * np.log(probs[mask]) + (1 - responses[mask]) * np.log(1 - probs[mask])
        return np.sum(ll)
    
    def fisher_information(self, abilities=None, difficulties=None, discriminations=None):
        """Compute Fisher information for each item at given abilities.
        
        I(θ; a, b) = a² * P(θ) * (1 - P(θ))
        
        Returns:
            info: (n_abilities, n_items) array of Fisher information
        """
        if abilities is None:
            abilities = self.abilities
        if difficulties is None:
            difficulties = self.difficulties
        if discriminations is None:
            discriminations = self.discriminations
            
        theta = np.atleast_1d(abilities)
        theta = theta[:, np.newaxis]  # (n_abilities, 1)
        b = difficulties[np.newaxis, :]   # (1, n_items)
        a = discriminations[np.newaxis, :]  # (1, n_items)
        
        probs = expit(a * (theta - b))
        info = (a ** 2) * probs * (1 - probs)
        
        return info
    
    def fit_mle(self, responses, mask=None, max_iter=100, tol=1e-4):
        """Fit parameters using Maximum Likelihood Estimation via alternating optimization.
        
        Args:
            responses: Binary matrix (n_models, n_items), -1 for missing
            mask: Boolean matrix (n_models, n_items)
        """
        if mask is None:
            mask = responses >= 0
        
        observed = mask.astype(float)
        
        for iteration in range(max_iter):
            old_ll = self.log_likelihood(responses, mask)
            
            # Update abilities (fix item parameters)
            for m in range(self.n_models):
                item_mask = mask[m]
                if not np.any(item_mask):
                    continue
                    
                def neg_ll_ability(theta):
                    theta = np.array([theta])
                    b = self.difficulties[item_mask]
                    a = self.discriminations[item_mask]
                    probs = expit(a * (theta - b))
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)
                    y = responses[m, item_mask]
                    return -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                
                result = minimize(neg_ll_ability, self.abilities[m], method='BFGS')
                self.abilities[m] = result.x[0]
            
            # Update difficulties (fix abilities and discriminations)
            for n in range(self.n_items):
                model_mask = mask[:, n]
                if not np.any(model_mask):
                    continue
                    
                def neg_ll_difficulty(b):
                    theta = self.abilities[model_mask]
                    a = self.discriminations[n]
                    probs = expit(a * (theta - b))
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)
                    y = responses[model_mask, n]
                    return -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                
                result = minimize(neg_ll_difficulty, self.difficulties[n], method='BFGS')
                self.difficulties[n] = result.x[0]
            
            # Update discriminations (fix abilities and difficulties)
            for n in range(self.n_items):
                model_mask = mask[:, n]
                if not np.any(model_mask):
                    continue
                    
                def neg_ll_discrimination(a):
                    if a <= 0:
                        return 1e10
                    theta = self.abilities[model_mask]
                    b = self.difficulties[n]
                    probs = expit(a * (theta - b))
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)
                    y = responses[model_mask, n]
                    return -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                
                result = minimize(neg_ll_discrimination, self.discriminations[n], 
                                method='L-BFGS-B', bounds=[(0.1, 5.0)])
                self.discriminations[n] = result.x[0]
            
            new_ll = self.log_likelihood(responses, mask)
            
            if abs(new_ll - old_ll) < tol:
                break
        
        return self
    
    def fit_bayesian_simple(self, responses, mask=None, n_iter=500, lr=0.01):
        """Simplified Bayesian fitting using MAP estimation with priors.
        
        Priors:
        - θ ~ N(0, 1)
        - a ~ LogNormal(0, 0.5)
        - b ~ N(0, 1)
        """
        if mask is None:
            mask = responses >= 0
        
        # Convert to torch tensors
        R = torch.tensor(responses, dtype=torch.float32, device=self.device)
        M = torch.tensor(mask, dtype=torch.float32, device=self.device)
        
        # Initialize parameters as torch tensors with gradients
        abilities = torch.zeros(self.n_models, device=self.device, requires_grad=True)
        difficulties = torch.zeros(self.n_items, device=self.device, requires_grad=True)
        log_discriminations = torch.zeros(self.n_items, device=self.device, requires_grad=True)
        
        optimizer = optim.Adam([abilities, difficulties, log_discriminations], lr=lr)
        
        for _ in range(n_iter):
            optimizer.zero_grad()
            
            discriminations = torch.exp(log_discriminations)
            
            # Compute probabilities
            theta = abilities.unsqueeze(1)  # (n_models, 1)
            b = difficulties.unsqueeze(0)   # (1, n_items)
            a = discriminations.unsqueeze(0)  # (1, n_items)
            
            z = a * (theta - b)
            probs = torch.sigmoid(z)
            probs = torch.clamp(probs, 1e-10, 1 - 1e-10)
            
            # Log-likelihood
            ll = R * torch.log(probs) + (1 - R) * torch.log(1 - probs)
            ll = torch.sum(ll * M)
            
            # Priors (negative log-prior for MAP)
            prior_ability = -0.5 * torch.sum(abilities ** 2)
            prior_difficulty = -0.5 * torch.sum(difficulties ** 2)
            prior_discrimination = -0.5 * torch.sum((log_discriminations / 0.5) ** 2)
            
            # Total objective (we maximize)
            loss = -(ll + 0.1 * (prior_ability + prior_difficulty + prior_discrimination))
            
            loss.backward()
            optimizer.step()
        
        # Store results
        self.abilities = abilities.detach().cpu().numpy()
        self.difficulties = difficulties.detach().cpu().numpy()
        self.discriminations = torch.exp(log_discriminations).detach().cpu().numpy()
        
        return self
    
    def estimate_abilities_mle(self, responses, mask=None, initial_theta=None):
        """Estimate abilities given fixed item parameters.
        
        Args:
            responses: (n_models, n_items) response matrix
            mask: (n_models, n_items) observed mask
            initial_theta: Initial ability estimates
            
        Returns:
            abilities: (n_models,) estimated abilities
        """
        if mask is None:
            mask = responses >= 0
        
        n_test_models = responses.shape[0]
        abilities = np.zeros(n_test_models)
        
        for m in range(n_test_models):
            item_mask = mask[m]
            if not np.any(item_mask):
                abilities[m] = 0.0
                continue
            
            def neg_ll(theta):
                theta_arr = np.array([theta])
                b = self.difficulties[item_mask]
                a = self.discriminations[item_mask]
                probs = expit(a * (theta_arr - b))
                probs = np.clip(probs, 1e-10, 1 - 1e-10)
                y = responses[m, item_mask]
                return -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
            
            init = initial_theta[m] if initial_theta is not None else 0.0
            result = minimize(neg_ll, init, method='BFGS')
            abilities[m] = result.x[0]
        
        return abilities


class ThreePLModel(TwoPLModel):
    """3-Parameter Logistic (3PL) IRT Model with guessing parameter.
    
    P(correct|θ, a, b, c) = c + (1 - c) / (1 + exp(-a(θ - b)))
    where c is the guessing parameter (lower asymptote).
    """
    
    def __init__(self, n_models, n_items, device='cpu'):
        super().__init__(n_models, n_items, device)
        self.guessing = np.zeros(n_items) + 0.2  # Initialize guessing at 0.2
    
    def probability(self, abilities=None, difficulties=None, discriminations=None, guessing=None):
        """Compute probability of correct response with guessing parameter."""
        if abilities is None:
            abilities = self.abilities
        if difficulties is None:
            difficulties = self.difficulties
        if discriminations is None:
            discriminations = self.discriminations
        if guessing is None:
            guessing = self.guessing
            
        theta = abilities[:, np.newaxis]
        b = difficulties[np.newaxis, :]
        a = discriminations[np.newaxis, :]
        c = guessing[np.newaxis, :]
        
        z = a * (theta - b)
        return c + (1 - c) * expit(z)
    
    def fit_mle(self, responses, mask=None, max_iter=100, tol=1e-4):
        """Fit 3PL parameters using MLE."""
        if mask is None:
            mask = responses >= 0
        
        for iteration in range(max_iter):
            old_ll = self.log_likelihood(responses, mask)
            
            # Update abilities
            for m in range(self.n_models):
                item_mask = mask[m]
                if not np.any(item_mask):
                    continue
                    
                def neg_ll_ability(theta):
                    theta_arr = np.array([theta])
                    b = self.difficulties[item_mask]
                    a = self.discriminations[item_mask]
                    c = self.guessing[item_mask]
                    probs = c + (1 - c) * expit(a * (theta_arr - b))
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)
                    y = responses[m, item_mask]
                    return -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                
                result = minimize(neg_ll_ability, self.abilities[m], method='BFGS')
                self.abilities[m] = result.x[0]
            
            # Update item parameters
            for n in range(self.n_items):
                model_mask = mask[:, n]
                if not np.any(model_mask):
                    continue
                
                theta = self.abilities[model_mask]
                y = responses[model_mask, n]
                
                # Update difficulty
                def neg_ll_difficulty(b):
                    a = self.discriminations[n]
                    c = self.guessing[n]
                    probs = c + (1 - c) * expit(a * (theta - b))
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)
                    return -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                
                result = minimize(neg_ll_difficulty, self.difficulties[n], method='BFGS')
                self.difficulties[n] = result.x[0]
                
                # Update discrimination
                def neg_ll_discrimination(a):
                    if a <= 0:
                        return 1e10
                    b = self.difficulties[n]
                    c = self.guessing[n]
                    probs = c + (1 - c) * expit(a * (theta - b))
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)
                    return -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                
                result = minimize(neg_ll_discrimination, self.discriminations[n],
                                method='L-BFGS-B', bounds=[(0.1, 5.0)])
                self.discriminations[n] = result.x[0]
                
                # Update guessing
                def neg_ll_guessing(c):
                    if c < 0 or c > 0.5:
                        return 1e10
                    b = self.difficulties[n]
                    a = self.discriminations[n]
                    probs = c + (1 - c) * expit(a * (theta - b))
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)
                    return -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                
                result = minimize(neg_ll_guessing, self.guessing[n],
                                method='L-BFGS-B', bounds=[(0.0, 0.5)])
                self.guessing[n] = result.x[0]
            
            new_ll = self.log_likelihood(responses, mask)
            if abs(new_ll - old_ll) < tol:
                break
        
        return self


class OnePLModel(TwoPLModel):
    """1-Parameter Logistic (Rasch) Model.
    
    P(correct|θ, b) = 1 / (1 + exp(-(θ - b)))
    
    Discrimination is fixed at 1.0 for all items.
    """
    
    def __init__(self, n_models, n_items, device='cpu'):
        super().__init__(n_models, n_items, device)
        self.discriminations = np.ones(n_items)  # Fixed at 1.0
    
    def fit_mle(self, responses, mask=None, max_iter=100, tol=1e-4):
        """Fit 1PL parameters using MLE."""
        if mask is None:
            mask = responses >= 0
        
        for iteration in range(max_iter):
            old_ll = self.log_likelihood(responses, mask)
            
            # Update abilities
            for m in range(self.n_models):
                item_mask = mask[m]
                if not np.any(item_mask):
                    continue
                    
                def neg_ll_ability(theta):
                    theta_arr = np.array([theta])
                    b = self.difficulties[item_mask]
                    probs = expit(theta_arr - b)
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)
                    y = responses[m, item_mask]
                    return -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                
                result = minimize(neg_ll_ability, self.abilities[m], method='BFGS')
                self.abilities[m] = result.x[0]
            
            # Update difficulties
            for n in range(self.n_items):
                model_mask = mask[:, n]
                if not np.any(model_mask):
                    continue
                    
                def neg_ll_difficulty(b):
                    theta = self.abilities[model_mask]
                    probs = expit(theta - b)
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)
                    y = responses[model_mask, n]
                    return -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                
                result = minimize(neg_ll_difficulty, self.difficulties[n], method='BFGS')
                self.difficulties[n] = result.x[0]
            
            new_ll = self.log_likelihood(responses, mask)
            if abs(new_ll - old_ll) < tol:
                break
        
        return self

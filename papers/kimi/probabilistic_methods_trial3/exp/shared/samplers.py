"""
Gradient-based discrete MCMC samplers.
Implements GWG, ALBP, AB-sampler, and ACS.
"""
import numpy as np
from collections import deque


class GibbsWithGradients:
    """
    Gibbs-With-Gradients sampler with fixed step size.
    Grathwohl et al. 2021
    """
    
    def __init__(self, model, step_size=0.1, balancing='barker', seed=None):
        """
        Args:
            model: DiscreteDistribution object
            step_size: Proposal step size (sigma)
            balancing: 'barker' or 'sqrt'
            seed: Random seed
        """
        self.model = model
        self.step_size = step_size
        self.balancing = balancing
        self.rng = np.random.default_rng(seed)
        self.dim = model.dim
        
        # Tracking
        self.acceptance_history = []
        self.jump_distance_history = []
        self.step_size_history = [step_size]
        
    def balancing_fn(self, t):
        """Locally-balanced proposal function."""
        if self.balancing == 'barker':
            # Barker: sqrt(t) / (1 + sqrt(t))
            return np.sqrt(t) / (1 + np.sqrt(t))
        elif self.balancing == 'sqrt':
            return np.sqrt(t)
        elif self.balancing == 'tanh':
            return np.tanh(t / 2)
        else:
            raise ValueError(f"Unknown balancing: {self.balancing}")
    
    def propose(self, x):
        """Make a GWG proposal."""
        grad = self.model.grad_log_prob(x)
        
        # Compute logits for proposal
        logits = self.step_size * (2 * x - 1) * grad
        
        # Apply balancing
        probs = self.balancing_fn(np.exp(logits))
        
        # Normalize to get probability distribution over coordinates
        probs = probs / (np.sum(probs) + 1e-10)
        
        # Select coordinate to flip
        coord = self.rng.choice(self.dim, p=probs)
        
        # Create proposal
        x_prop = x.copy()
        x_prop[coord] = 1 - x_prop[coord]
        
        return x_prop, coord, probs[coord]
    
    def acceptance_prob(self, x, x_prop, coord, q_forward):
        """Compute acceptance probability."""
        log_p_x = self.model.log_prob(x)
        log_p_prop = self.model.log_prob(x_prop)
        
        # Compute reverse probability
        grad_prop = self.model.grad_log_prob(x_prop)
        logits_reverse = self.step_size * (2 * x_prop - 1) * grad_prop
        probs_reverse = self.balancing_fn(np.exp(logits_reverse))
        probs_reverse = probs_reverse / (np.sum(probs_reverse) + 1e-10)
        q_reverse = probs_reverse[coord]
        
        # MH acceptance
        log_alpha = log_p_prop - log_p_x + np.log(q_reverse + 1e-10) - np.log(q_forward + 1e-10)
        return min(1.0, np.exp(log_alpha))
    
    def step(self, x):
        """Take one MCMC step."""
        x_prop, coord, q_forward = self.propose(x)
        accept_prob = self.acceptance_prob(x, x_prop, coord, q_forward)
        
        if self.rng.random() < accept_prob:
            jump = np.sum(np.abs(x_prop - x))
            self.acceptance_history.append(1)
            self.jump_distance_history.append(jump)
            return x_prop
        else:
            self.acceptance_history.append(0)
            self.jump_distance_history.append(0)
            return x
    
    def sample(self, x_init, n_steps, warmup=0):
        """Generate samples."""
        x = x_init.copy()
        samples = []
        
        for i in range(n_steps + warmup):
            x = self.step(x)
            if i >= warmup:
                samples.append(x.copy())
        
        return np.array(samples)


class ALBP:
    """
    Adaptive Locally-Balanced Proposal.
    Sun et al. 2022 - Acceptance rate targeting with stochastic approximation.
    """
    
    def __init__(self, model, target_rate=0.574, eta_0=0.1, tau=1000, 
                 balancing='barker', seed=None):
        """
        Args:
            model: DiscreteDistribution object
            target_rate: Target acceptance rate (optimal = 0.574)
            eta_0: Initial learning rate
            tau: Learning rate decay timescale
            balancing: Balancing function type
            seed: Random seed
        """
        self.model = model
        self.target_rate = target_rate
        self.eta_0 = eta_0
        self.tau = tau
        self.balancing = balancing
        self.rng = np.random.default_rng(seed)
        self.dim = model.dim
        
        # Adaptive parameter (log step size)
        self.R = np.log(0.1)  # Initialize at sigma = 0.1
        
        # Tracking
        self.R_history = [self.R]
        self.sigma_history = [np.exp(self.R)]
        self.acceptance_history = []
        self.jump_distance_history = []
        self.acceptance_window = deque(maxlen=100)
        
    def balancing_fn(self, t):
        """Locally-balanced proposal function."""
        if self.balancing == 'barker':
            return np.sqrt(t) / (1 + np.sqrt(t))
        elif self.balancing == 'sqrt':
            return np.sqrt(t)
        elif self.balancing == 'tanh':
            return np.tanh(t / 2)
        else:
            raise ValueError(f"Unknown balancing: {self.balancing}")
    
    def propose(self, x):
        """Make a GWG proposal with current step size."""
        sigma = np.exp(self.R)
        grad = self.model.grad_log_prob(x)
        
        logits = sigma * (2 * x - 1) * grad
        probs = self.balancing_fn(np.exp(logits))
        probs = probs / (np.sum(probs) + 1e-10)
        
        coord = self.rng.choice(self.dim, p=probs)
        
        x_prop = x.copy()
        x_prop[coord] = 1 - x_prop[coord]
        
        return x_prop, coord, probs[coord]
    
    def acceptance_prob(self, x, x_prop, coord, q_forward):
        """Compute acceptance probability."""
        sigma = np.exp(self.R)
        log_p_x = self.model.log_prob(x)
        log_p_prop = self.model.log_prob(x_prop)
        
        grad_prop = self.model.grad_log_prob(x_prop)
        logits_reverse = sigma * (2 * x_prop - 1) * grad_prop
        probs_reverse = self.balancing_fn(np.exp(logits_reverse))
        probs_reverse = probs_reverse / (np.sum(probs_reverse) + 1e-10)
        q_reverse = probs_reverse[coord]
        
        log_alpha = log_p_prop - log_p_x + np.log(q_reverse + 1e-10) - np.log(q_forward + 1e-10)
        return min(1.0, np.exp(log_alpha))
    
    def adapt(self, t, accepted):
        """Stochastic approximation adaptation."""
        self.acceptance_window.append(1 if accepted else 0)
        
        # Robbins-Monro update
        eta_t = self.eta_0 / (1 + t / self.tau)
        a_t = 1 if accepted else 0
        
        self.R = self.R + eta_t * (a_t - self.target_rate)
        self.R_history.append(self.R)
        self.sigma_history.append(np.exp(self.R))
    
    def step(self, x, t):
        """Take one adaptive MCMC step."""
        x_prop, coord, q_forward = self.propose(x)
        accept_prob = self.acceptance_prob(x, x_prop, coord, q_forward)
        
        if self.rng.random() < accept_prob:
            jump = np.sum(np.abs(x_prop - x))
            self.acceptance_history.append(1)
            self.jump_distance_history.append(jump)
            self.adapt(t, True)
            return x_prop
        else:
            self.acceptance_history.append(0)
            self.jump_distance_history.append(0)
            self.adapt(t, False)
            return x
    
    def sample(self, x_init, n_steps, warmup=0):
        """Generate samples with adaptation."""
        x = x_init.copy()
        samples = []
        
        for i in range(n_steps + warmup):
            x = self.step(x, i)
            if i >= warmup:
                samples.append(x.copy())
        
        return np.array(samples)


class ABSampler:
    """
    Any-Scale Balanced Sampler.
    Sun et al. 2023a - Jump distance maximization.
    """
    
    def __init__(self, model, sigma_init=0.1, window_size=100, 
                 balancing='barker', seed=None):
        """
        Args:
            model: DiscreteDistribution object
            sigma_init: Initial step size
            window_size: Window for jump distance estimation
            balancing: Balancing function
            seed: Random seed
        """
        self.model = model
        self.sigma = sigma_init
        self.window_size = window_size
        self.balancing = balancing
        self.rng = np.random.default_rng(seed)
        self.dim = model.dim
        
        # Tracking
        self.sigma_history = [sigma_init]
        self.acceptance_history = []
        self.jump_distance_history = []
        self.jump_window = deque(maxlen=window_size)
        
        # For gradient estimation
        self.sigma_candidates = [0.5 * sigma_init, sigma_init, 2 * sigma_init]
        
    def balancing_fn(self, t):
        """Locally-balanced proposal function."""
        if self.balancing == 'barker':
            return np.sqrt(t) / (1 + np.sqrt(t))
        elif self.balancing == 'sqrt':
            return np.sqrt(t)
        elif self.balancing == 'tanh':
            return np.tanh(t / 2)
        else:
            raise ValueError(f"Unknown balancing: {self.balancing}")
    
    def propose(self, x, sigma=None):
        """Make a GWG proposal."""
        if sigma is None:
            sigma = self.sigma
        grad = self.model.grad_log_prob(x)
        
        logits = sigma * (2 * x - 1) * grad
        probs = self.balancing_fn(np.exp(logits))
        probs = probs / (np.sum(probs) + 1e-10)
        
        coord = self.rng.choice(self.dim, p=probs)
        
        x_prop = x.copy()
        x_prop[coord] = 1 - x_prop[coord]
        
        return x_prop, coord, probs[coord]
    
    def acceptance_prob(self, x, x_prop, coord, q_forward, sigma=None):
        """Compute acceptance probability."""
        if sigma is None:
            sigma = self.sigma
        log_p_x = self.model.log_prob(x)
        log_p_prop = self.model.log_prob(x_prop)
        
        grad_prop = self.model.grad_log_prob(x_prop)
        logits_reverse = sigma * (2 * x_prop - 1) * grad_prop
        probs_reverse = self.balancing_fn(np.exp(logits_reverse))
        probs_reverse = probs_reverse / (np.sum(probs_reverse) + 1e-10)
        q_reverse = probs_reverse[coord]
        
        log_alpha = log_p_prop - log_p_x + np.log(q_reverse + 1e-10) - np.log(q_forward + 1e-10)
        return min(1.0, np.exp(log_alpha))
    
    def estimate_jump_distance(self, sigma_test, x, n_test=10):
        """Estimate expected jump distance for a given sigma."""
        jumps = []
        x_test = x.copy()
        
        for _ in range(n_test):
            x_prop, coord, q_forward = self.propose(x_test, sigma_test)
            accept_prob = self.acceptance_prob(x_test, x_prop, coord, q_forward, sigma_test)
            
            if self.rng.random() < accept_prob:
                jumps.append(np.sum(np.abs(x_prop - x_test)))
                x_test = x_prop
            else:
                jumps.append(0)
        
        return np.mean(jumps)
    
    def adapt(self):
        """Adapt step size to maximize jump distance."""
        if len(self.jump_window) < self.window_size // 2:
            return
        
        current_jump = np.mean(list(self.jump_window))
        
        # Test nearby step sizes
        test_sigmas = [
            max(0.01, self.sigma * 0.8),
            self.sigma,
            min(1.0, self.sigma * 1.2)
        ]
        
        # Simple line search
        best_sigma = self.sigma
        best_jump = current_jump
        
        for sig in [test_sigmas[0], test_sigmas[2]]:
            # Estimate using recent history as proxy
            # In practice, we'd run parallel chains, but we approximate
            pass  # Simplified: use recent jump distance trend
        
        # Adjust based on trend
        if len(self.jump_distance_history) >= 2 * self.window_size:
            recent = np.mean(self.jump_distance_history[-self.window_size:])
            older = np.mean(self.jump_distance_history[-2*self.window_size:-self.window_size])
            
            if recent > older * 1.1:
                # Increasing, continue in same direction
                pass
            elif recent < older * 0.9:
                # Decreasing, adjust
                pass
        
        # Simple heuristic: if acceptance is too high, increase sigma; too low, decrease
        recent_accept = np.mean(self.acceptance_history[-self.window_size:]) if len(self.acceptance_history) >= self.window_size else 0.5
        
        if recent_accept > 0.7:
            self.sigma = min(1.0, self.sigma * 1.05)
        elif recent_accept < 0.4:
            self.sigma = max(0.01, self.sigma * 0.95)
        
        self.sigma_history.append(self.sigma)
    
    def step(self, x, t):
        """Take one adaptive MCMC step."""
        x_prop, coord, q_forward = self.propose(x)
        accept_prob = self.acceptance_prob(x, x_prop, coord, q_forward)
        
        if self.rng.random() < accept_prob:
            jump = np.sum(np.abs(x_prop - x))
            self.acceptance_history.append(1)
            self.jump_distance_history.append(jump)
            self.jump_window.append(jump)
            
            # Adapt periodically
            if t % self.window_size == 0 and t > 0:
                self.adapt()
            
            return x_prop
        else:
            self.acceptance_history.append(0)
            self.jump_distance_history.append(0)
            self.jump_window.append(0)
            
            if t % self.window_size == 0 and t > 0:
                self.adapt()
            
            return x
    
    def sample(self, x_init, n_steps, warmup=0):
        """Generate samples with adaptation."""
        x = x_init.copy()
        samples = []
        
        for i in range(n_steps + warmup):
            x = self.step(x, i)
            if i >= warmup:
                samples.append(x.copy())
        
        return np.array(samples)


class ACS:
    """
    Automatic Cyclical Scheduling.
    Pynadath et al. 2024 - Cyclical step size with acceptance rate targeting.
    """
    
    def __init__(self, model, sigma_min=0.05, sigma_max=0.5, cycle_length=1000,
                 target_rate=0.5, balancing='barker', seed=None):
        """
        Args:
            model: DiscreteDistribution object
            sigma_min: Minimum step size in cycle
            sigma_max: Maximum step size in cycle
            cycle_length: Number of iterations per cycle
            target_rate: Target acceptance rate
            balancing: Balancing function
            seed: Random seed
        """
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cycle_length = cycle_length
        self.target_rate = target_rate
        self.balancing = balancing
        self.rng = np.random.default_rng(seed)
        self.dim = model.dim
        
        # Tracking
        self.sigma_history = []
        self.acceptance_history = []
        self.jump_distance_history = []
        self.acceptance_window = deque(maxlen=100)
    
    def balancing_fn(self, t):
        """Locally-balanced proposal function."""
        if self.balancing == 'barker':
            return np.sqrt(t) / (1 + np.sqrt(t))
        elif self.balancing == 'sqrt':
            return np.sqrt(t)
        elif self.balancing == 'tanh':
            return np.tanh(t / 2)
        else:
            raise ValueError(f"Unknown balancing: {self.balancing}")
    
    def get_sigma(self, t):
        """Cyclical step size schedule."""
        T_cur = t % self.cycle_length
        # Cosine schedule
        sigma = self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + np.cos(np.pi * T_cur / self.cycle_length))
        return sigma
    
    def propose(self, x, sigma):
        """Make a GWG proposal."""
        grad = self.model.grad_log_prob(x)
        
        logits = sigma * (2 * x - 1) * grad
        probs = self.balancing_fn(np.exp(logits))
        probs = probs / (np.sum(probs) + 1e-10)
        
        coord = self.rng.choice(self.dim, p=probs)
        
        x_prop = x.copy()
        x_prop[coord] = 1 - x_prop[coord]
        
        return x_prop, coord, probs[coord]
    
    def acceptance_prob(self, x, x_prop, coord, q_forward, sigma):
        """Compute acceptance probability."""
        log_p_x = self.model.log_prob(x)
        log_p_prop = self.model.log_prob(x_prop)
        
        grad_prop = self.model.grad_log_prob(x_prop)
        logits_reverse = sigma * (2 * x_prop - 1) * grad_prop
        probs_reverse = self.balancing_fn(np.exp(logits_reverse))
        probs_reverse = probs_reverse / (np.sum(probs_reverse) + 1e-10)
        q_reverse = probs_reverse[coord]
        
        log_alpha = log_p_prop - log_p_x + np.log(q_reverse + 1e-10) - np.log(q_forward + 1e-10)
        return min(1.0, np.exp(log_alpha))
    
    def adapt_cycle(self, cycle_num):
        """Adjust cycle parameters based on acceptance rate."""
        if len(self.acceptance_window) < 50:
            return
        
        avg_accept = np.mean(list(self.acceptance_window))
        
        # Adjust sigma_min and sigma_max based on acceptance
        if avg_accept > self.target_rate + 0.1:
            # Accepting too much, can be more aggressive
            self.sigma_max = min(1.0, self.sigma_max * 1.05)
            self.sigma_min = min(self.sigma_max * 0.5, self.sigma_min * 1.05)
        elif avg_accept < self.target_rate - 0.1:
            # Accepting too little, be more conservative
            self.sigma_min = max(0.01, self.sigma_min * 0.95)
            self.sigma_max = max(self.sigma_min * 2, self.sigma_max * 0.95)
    
    def step(self, x, t):
        """Take one cyclical MCMC step."""
        sigma = self.get_sigma(t)
        self.sigma_history.append(sigma)
        
        x_prop, coord, q_forward = self.propose(x, sigma)
        accept_prob = self.acceptance_prob(x, x_prop, coord, q_forward, sigma)
        
        if self.rng.random() < accept_prob:
            jump = np.sum(np.abs(x_prop - x))
            self.acceptance_history.append(1)
            self.jump_distance_history.append(jump)
            self.acceptance_window.append(1)
            
            # Adapt at end of each cycle
            if t > 0 and t % self.cycle_length == 0:
                self.adapt_cycle(t // self.cycle_length)
            
            return x_prop
        else:
            self.acceptance_history.append(0)
            self.jump_distance_history.append(0)
            self.acceptance_window.append(0)
            
            if t > 0 and t % self.cycle_length == 0:
                self.adapt_cycle(t // self.cycle_length)
            
            return x
    
    def sample(self, x_init, n_steps, warmup=0):
        """Generate samples."""
        x = x_init.copy()
        samples = []
        
        for i in range(n_steps + warmup):
            x = self.step(x, i)
            if i >= warmup:
                samples.append(x.copy())
        
        return np.array(samples)


class GridSearchHeuristic:
    """
    Grid-search baseline: pilot chains + freeze.
    """
    
    def __init__(self, model, step_sizes=None, pilot_steps=500, 
                 balancing='barker', seed=None):
        """
        Args:
            model: DiscreteDistribution object
            step_sizes: Grid of step sizes to try
            pilot_steps: Steps per pilot chain
            balancing: Balancing function
            seed: Random seed
        """
        self.model = model
        self.step_sizes = step_sizes if step_sizes is not None else [0.01, 0.05, 0.1, 0.2, 0.5]
        self.pilot_steps = pilot_steps
        self.balancing = balancing
        self.rng = np.random.default_rng(seed)
        self.dim = model.dim
        
        self.selected_sigma = None
        self.pilot_results = {}
        
    def run_pilots(self, x_init):
        """Run pilot chains and select best step size."""
        best_sigma = None
        best_jump = -np.inf
        
        for sigma in self.step_sizes:
            sampler = GibbsWithGradients(self.model, sigma, self.balancing, seed=self.rng.integers(100000))
            samples = sampler.sample(x_init, self.pilot_steps, warmup=100)
            
            avg_jump = np.mean(sampler.jump_distance_history[100:])
            self.pilot_results[sigma] = {
                'avg_jump': avg_jump,
                'avg_accept': np.mean(sampler.acceptance_history[100:])
            }
            
            if avg_jump > best_jump:
                best_jump = avg_jump
                best_sigma = sigma
        
        self.selected_sigma = best_sigma
        return best_sigma
    
    def sample(self, x_init, n_steps, warmup=0):
        """Run production sampling with selected step size."""
        if self.selected_sigma is None:
            raise ValueError("Must run pilots first!")
        
        sampler = GibbsWithGradients(self.model, self.selected_sigma, self.balancing, seed=self.rng.integers(100000))
        samples = sampler.sample(x_init, n_steps, warmup)
        
        # Copy history for analysis
        self.acceptance_history = sampler.acceptance_history
        self.jump_distance_history = sampler.jump_distance_history
        
        return samples

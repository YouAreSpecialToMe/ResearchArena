"""
Model definitions for discrete MCMC experiments.
Implements Bernoulli, Gaussian, Ising, and Mixture distributions.
"""
import numpy as np
from abc import ABC, abstractmethod


class DiscreteDistribution(ABC):
    """Base class for discrete distributions."""
    
    @abstractmethod
    def log_prob(self, x):
        """Compute log probability of x."""
        pass
    
    @abstractmethod
    def grad_log_prob(self, x):
        """Compute gradient of log probability (finite differences)."""
        pass
    
    @property
    @abstractmethod
    def dim(self):
        """Dimension of the distribution."""
        pass


class BernoulliDistribution(DiscreteDistribution):
    """Independent Bernoulli distribution: p(x) = prod_i p_i^x_i (1-p_i)^(1-x_i)"""
    
    def __init__(self, probs):
        self.probs = np.array(probs)
        self._dim = len(probs)
        
    @property
    def dim(self):
        return self._dim
    
    def log_prob(self, x):
        x = np.asarray(x)
        log_p = x * np.log(self.probs + 1e-10) + (1 - x) * np.log(1 - self.probs + 1e-10)
        return np.sum(log_p)
    
    def grad_log_prob(self, x):
        """Gradient via finite differences: d/dx_i log p(x) ≈ log p(x_i=1) - log p(x_i=0)"""
        x = np.asarray(x)
        grad = np.log(self.probs + 1e-10) - np.log(1 - self.probs + 1e-10)
        return grad
    
    def sample_exact(self, n_samples, rng=None):
        """Sample directly from the distribution."""
        if rng is None:
            rng = np.random.default_rng()
        return (rng.random((n_samples, self._dim)) < self.probs).astype(np.float64)


class GaussianDistribution(DiscreteDistribution):
    """
    Discrete Gaussian: p(x) ∝ exp(-0.5 * x^T A x + b^T x)
    where x ∈ {0, 1}^d
    """
    
    def __init__(self, A, b):
        self.A = np.array(A)
        self.b = np.array(b)
        self._dim = len(b)
        
    @property
    def dim(self):
        return self._dim
    
    def log_prob(self, x):
        x = np.asarray(x)
        return -0.5 * x @ self.A @ x + self.b @ x
    
    def grad_log_prob(self, x):
        """Gradient: ∇_x (-0.5 x^T A x + b^T x) = -A x + b"""
        x = np.asarray(x)
        return -self.A @ x + self.b


class IsingModel(DiscreteDistribution):
    """
    Ising model: p(x) ∝ exp(J * sum_{<i,j>} x_i x_j + h * sum_i x_i)
    where x ∈ {0, 1}^d (0/1 representation)
    """
    
    def __init__(self, couplings, external_field=None):
        """
        Args:
            couplings: d x d symmetric matrix of couplings
            external_field: d-dimensional vector of external fields (default 0)
        """
        self.couplings = np.array(couplings)
        self.d = self.couplings.shape[0]
        if external_field is None:
            self.h = np.zeros(self.d)
        else:
            self.h = np.array(external_field)
        
    @property
    def dim(self):
        return self.d
    
    def log_prob(self, x):
        x = np.asarray(x)
        # E(x) = -0.5 * x^T J x - h^T x (energy)
        # p(x) ∝ exp(-E(x)) = exp(0.5 * x^T J x + h^T x)
        return 0.5 * x @ self.couplings @ x + self.h @ x
    
    def grad_log_prob(self, x):
        """Gradient: ∇_x (0.5 x^T J x + h^T x) = J x + h"""
        x = np.asarray(x)
        return self.couplings @ x + self.h
    
    def energy(self, x):
        """Compute Ising energy (for analysis)."""
        x = np.asarray(x)
        return -(0.5 * x @ self.couplings @ x + self.h @ x)


class LatticeIsingModel(IsingModel):
    """
    2D Lattice Ising model with periodic boundary conditions.
    """
    
    def __init__(self, L, J_coupling, h_field=0.0, seed=None):
        """
        Args:
            L: Lattice size (L x L = d variables)
            J_coupling: Coupling strength (critical J_c ≈ 0.44 for 2D)
            h_field: External magnetic field
            seed: Random seed for reproducibility
        """
        self.L = L
        self.d = L * L
        self.J_coupling = J_coupling
        
        # Create coupling matrix for 2D lattice with periodic BC
        couplings = np.zeros((self.d, self.d))
        for i in range(L):
            for j in range(L):
                idx = i * L + j
                # Right neighbor
                idx_right = i * L + ((j + 1) % L)
                # Down neighbor
                idx_down = ((i + 1) % L) * L + j
                
                couplings[idx, idx_right] = J_coupling
                couplings[idx_right, idx] = J_coupling
                couplings[idx, idx_down] = J_coupling
                couplings[idx_down, idx] = J_coupling
        
        external_field = np.full(self.d, h_field)
        super().__init__(couplings, external_field)


class RandomIsingModel(IsingModel):
    """
    Random Ising model with random couplings.
    """
    
    def __init__(self, d, J_mean, J_std, frustration=0.0, seed=None):
        """
        Args:
            d: Number of variables
            J_mean: Mean coupling strength
            J_std: Standard deviation of couplings
            frustration: Fraction of negative couplings (0-1)
            seed: Random seed
        """
        rng = np.random.default_rng(seed)
        
        # Generate random couplings
        couplings = rng.normal(J_mean, J_std, (d, d))
        couplings = (couplings + couplings.T) / 2  # Make symmetric
        np.fill_diagonal(couplings, 0)
        
        # Apply frustration
        if frustration > 0:
            mask = rng.random((d, d)) < frustration
            mask = mask | mask.T  # Make symmetric
            couplings[mask] = -np.abs(couplings[mask])
        
        super().__init__(couplings, np.zeros(d))


class DiscreteGaussianMixture(DiscreteDistribution):
    """
    Mixture of discrete Gaussians.
    p(x) = sum_k w_k * N_discrete(x; mu_k, Sigma_k)
    where x ∈ {0, 1}^d
    """
    
    def __init__(self, means, covs, weights=None):
        """
        Args:
            means: List of mean vectors (each in [0, 1]^d)
            covs: List of covariance matrices
            weights: Mixture weights (default uniform)
        """
        self.means = [np.array(m) for m in means]
        self.covs = [np.array(c) for c in covs]
        self.n_modes = len(means)
        self._dim = len(means[0])
        
        if weights is None:
            self.weights = np.ones(self.n_modes) / self.n_modes
        else:
            self.weights = np.array(weights)
        
        # Precompute precision matrices
        self.precisions = [np.linalg.inv(c) for c in self.covs]
    
    @property
    def dim(self):
        return self._dim
    
    def log_prob(self, x):
        x = np.asarray(x)
        log_probs = []
        for i, (mu, prec) in enumerate(zip(self.means, self.precisions)):
            diff = x - mu
            log_p = -0.5 * diff @ prec @ diff
            log_probs.append(np.log(self.weights[i] + 1e-10) + log_p)
        
        return np.logaddexp.reduce(log_probs)
    
    def grad_log_prob(self, x):
        """Gradient using mixture formula."""
        x = np.asarray(x)
        
        # Compute responsibilities
        log_probs = []
        for i, (mu, prec) in enumerate(zip(self.means, self.precisions)):
            diff = x - mu
            log_p = -0.5 * diff @ prec @ diff + np.log(self.weights[i] + 1e-10)
            log_probs.append(log_p)
        
        log_probs = np.array(log_probs)
        responsibilities = np.exp(log_probs - np.logaddexp.reduce(log_probs))
        
        # Gradient is weighted average of per-component gradients
        grad = np.zeros(self._dim)
        for i, (mu, prec) in enumerate(zip(self.means, self.precisions)):
            grad += responsibilities[i] * (-prec @ (x - mu))
        
        return grad


def create_bernoulli_problem(d, seed=None):
    """Create a Bernoulli problem with random probabilities."""
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.3, 0.7, d)
    return BernoulliDistribution(probs)


def create_gaussian_problem(d, condition_number=10, seed=None):
    """Create a Gaussian problem with correlated structure."""
    rng = np.random.default_rng(seed)
    
    # Create covariance with specified condition number
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    eigenvalues = np.linspace(1, condition_number, d)
    cov = Q @ np.diag(1.0 / eigenvalues) @ Q.T
    
    b = rng.standard_normal(d)
    return GaussianDistribution(cov, b)


def create_multimodal_problem(d, n_modes, separation=0.5, seed=None):
    """Create a multimodal Gaussian mixture."""
    rng = np.random.default_rng(seed)
    
    means = []
    covs = []
    
    for i in range(n_modes):
        # Place modes at corners of hypercube
        if i < 2**d and d <= 4:
            # Use binary encoding for corners
            binary = [(i >> j) & 1 for j in range(d)]
            mu = np.array(binary) * separation + 0.5 - separation/2
        else:
            # Random placement
            mu = rng.random(d) * separation + 0.5 - separation/2
        means.append(mu)
        
        # Each mode has its own covariance
        cov = np.eye(d) * (0.1 + rng.random() * 0.1)
        covs.append(cov)
    
    return DiscreteGaussianMixture(means, covs)

"""
Phase 3: Constrained Continuous Optimization (Fixed)
Combines NOTEARS-style acyclicity with structural constraints and IT initialization.
"""
import numpy as np
from scipy.optimize import minimize


def compute_acyclicity_constraint(W: np.ndarray) -> float:
    """Compute h(W) = tr(exp(W ∘ W)) - d."""
    d = W.shape[0]
    M = W * W
    # Compute matrix exponential using eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(M)
    exp_eigvals = np.exp(eigvals)
    exp_M = eigvecs @ np.diag(exp_eigvals) @ eigvecs.T
    return np.trace(exp_M) - d


def compute_acyclicity_grad(W: np.ndarray) -> np.ndarray:
    """Gradient of acyclicity constraint."""
    d = W.shape[0]
    M = W * W
    eigvals, eigvecs = np.linalg.eigh(M)
    exp_eigvals = np.exp(eigvals)
    exp_M = eigvecs @ np.diag(exp_eigvals) @ eigvecs.T
    return 2 * W * exp_M.T


def constrained_optimization(
    data: np.ndarray,
    skeleton: np.ndarray,
    it_scores: np.ndarray,
    constraint_matrix: np.ndarray,
    lambda1: float = 0.05,  # Reduced for small samples
    lambda2: float = 0.0,
    lambda3: float = 0.01,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.2,  # Lower threshold for small samples
    use_it_init: bool = True,
    seed: int = None
) -> np.ndarray:
    """
    Constrained optimization for DAG learning.
    
    Uses a single augmented Lagrangian minimization with warm start.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n, d = data.shape
    
    # Normalize data
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    # Initialize weights
    if use_it_init and it_scores is not None:
        W = it_scores * 0.5
        W += np.random.randn(d, d) * 0.01
    else:
        W = np.random.randn(d, d) * 0.01
    
    # Apply skeleton mask
    if skeleton is not None:
        W = W * skeleton
    
    # Augmented Lagrangian parameters
    rho = 1.0
    alpha = 0.0
    
    def loss_fn(W_vec):
        W_mat = W_vec.reshape(d, d)
        
        # Least squares loss
        loss = (0.5 / n) * np.sum((data @ W_mat - data) ** 2)
        
        # L1 sparsity
        loss += lambda1 * np.sum(np.abs(W_mat))
        
        # Acyclicity constraint
        h = compute_acyclicity_constraint(W_mat)
        loss += (rho / 2) * h * h + alpha * h
        
        # Structural constraint
        if lambda3 > 0 and constraint_matrix is not None:
            loss += lambda3 * np.sum(constraint_matrix * np.abs(W_mat))
        
        return loss
    
    def grad_fn(W_vec):
        W_mat = W_vec.reshape(d, d)
        
        # Gradient of least squares
        grad = (1.0 / n) * (data.T @ (data @ W_mat - data))
        
        # Gradient of L1 (subgradient)
        grad += lambda1 * np.sign(W_mat)
        
        # Gradient of acyclicity constraint
        h = compute_acyclicity_constraint(W_mat)
        h_grad = compute_acyclicity_grad(W_mat)
        grad += (rho * h + alpha) * h_grad
        
        # Gradient of structural constraint
        if lambda3 > 0 and constraint_matrix is not None:
            grad += lambda3 * constraint_matrix * np.sign(W_mat)
        
        return grad.flatten()
    
    # Single augmented Lagrangian optimization
    # (Multiple outer iterations cause weight shrinkage)
    result = minimize(loss_fn, W.flatten(), method='L-BFGS-B', jac=grad_fn,
                     options={'maxiter': max_iter})
    W = result.x.reshape(d, d)
    
    # Apply skeleton mask
    if skeleton is not None:
        W = W * skeleton
    
    # Threshold to get binary adjacency
    W_binary = (np.abs(W) > w_threshold).astype(int)
    
    return W_binary

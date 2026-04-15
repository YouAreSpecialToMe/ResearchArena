"""
Data loading and problem instance generation.
"""
import numpy as np
import json
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from models import (BernoulliDistribution, GaussianDistribution, LatticeIsingModel,
                    RandomIsingModel, DiscreteGaussianMixture, create_multimodal_problem)


def save_problem(problem, filepath):
    """Save a problem instance."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(problem, f)


def load_problem(filepath):
    """Load a problem instance."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_problem_metadata(metadata, filepath):
    """Save problem metadata as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def generate_all_problems(base_dir='data/problems', seed=42):
    """Generate all problem instances for experiments."""
    rng = np.random.default_rng(seed)
    
    problems = {}
    
    # 1. Bernoulli models for theory validation
    print("Generating Bernoulli problems...")
    for d in [50, 100, 200]:
        probs = rng.uniform(0.3, 0.7, d)
        problem = BernoulliDistribution(probs)
        filepath = os.path.join(base_dir, f'bernoulli_{d}d.pkl')
        save_problem(problem, filepath)
        problems[f'bernoulli_{d}d'] = {
            'type': 'bernoulli',
            'dim': d,
            'file': filepath
        }
    
    # 2. Gaussian models
    print("Generating Gaussian problems...")
    for d in [50, 100, 200]:
        # Create with varying condition numbers
        A = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(A)
        eigenvalues = np.linspace(1, 10, d)
        cov = Q @ np.diag(1.0 / eigenvalues) @ Q.T
        b = rng.standard_normal(d)
        problem = GaussianDistribution(cov, b)
        filepath = os.path.join(base_dir, f'gaussian_{d}d.pkl')
        save_problem(problem, filepath)
        problems[f'gaussian_{d}d'] = {
            'type': 'gaussian',
            'dim': d,
            'file': filepath
        }
    
    # 3. Lattice Ising models (same as DISCS benchmark)
    print("Generating Lattice Ising problems...")
    # Critical temperature for 2D Ising: J_c ≈ 0.44
    lattice_sizes = [(10, 100), (20, 400), (30, 900)]  # (L, d)
    J_values = [0.2, 0.44, 0.6]  # below, at, above critical
    
    for L, d in lattice_sizes:
        for J in J_values:
            problem = LatticeIsingModel(L, J, seed=seed)
            temp_label = 'below' if J < 0.44 else ('at' if abs(J - 0.44) < 0.01 else 'above')
            filepath = os.path.join(base_dir, f'lattice_ising_L{L}_{temp_label}.pkl')
            save_problem(problem, filepath)
            problems[f'lattice_ising_L{L}_{temp_label}'] = {
                'type': 'lattice_ising',
                'dim': d,
                'L': L,
                'J': J,
                'temperature': temp_label,
                'file': filepath
            }
    
    # 4. Random Ising models
    print("Generating Random Ising problems...")
    for d in [100, 400]:
        for J_mean in [0.2, 0.4]:
            for frustration in [0.0, 0.5]:
                problem = RandomIsingModel(d, J_mean, 0.1, frustration, seed=seed)
                frust_label = 'frustrated' if frustration > 0 else 'uniform'
                filepath = os.path.join(base_dir, f'random_ising_{d}d_J{J_mean}_{frust_label}.pkl')
                save_problem(problem, filepath)
                problems[f'random_ising_{d}d_J{J_mean}_{frust_label}'] = {
                    'type': 'random_ising',
                    'dim': d,
                    'J_mean': J_mean,
                    'frustration': frustration,
                    'file': filepath
                }
    
    # 5. Multimodal problems
    print("Generating multimodal problems...")
    for d in [50, 100]:
        for n_modes in [2, 4, 6]:
            problem = create_multimodal_problem(d, n_modes, separation=0.5, seed=seed)
            filepath = os.path.join(base_dir, f'multimodal_{d}d_{n_modes}modes.pkl')
            save_problem(problem, filepath)
            problems[f'multimodal_{d}d_{n_modes}modes'] = {
                'type': 'multimodal',
                'dim': d,
                'n_modes': n_modes,
                'file': filepath,
                'true_modes': [m.tolist() for m in problem.means]
            }
    
    # Save metadata
    metadata_path = os.path.join(base_dir, 'problem_metadata.json')
    save_problem_metadata(problems, metadata_path)
    print(f"Generated {len(problems)} problems. Metadata saved to {metadata_path}")
    
    return problems


def get_problem(name, base_dir='data/problems'):
    """Load a problem by name."""
    metadata_path = os.path.join(base_dir, 'problem_metadata.json')
    with open(metadata_path, 'r') as f:
        problems = json.load(f)
    
    if name not in problems:
        raise ValueError(f"Problem {name} not found. Available: {list(problems.keys())}")
    
    filepath = problems[name]['file']
    return load_problem(filepath)


if __name__ == '__main__':
    generate_all_problems()

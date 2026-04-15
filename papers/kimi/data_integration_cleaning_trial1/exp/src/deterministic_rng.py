"""
Deterministic Random Number Generator for CESF
Ensures reproducible error generation through seeded RNG
"""
import numpy as np
from typing import Optional


class DeterministicRNG:
    """
    Wrapper around numpy's Generator for deterministic error synthesis.
    Uses SeedSequence for advanced seeding capabilities.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def reset(self):
        """Reset RNG to initial state with same seed"""
        self.rng = np.random.default_rng(self.seed)
    
    def spawn(self, child_seed: int) -> np.random.Generator:
        """Spawn a child RNG for independent streams"""
        return np.random.default_rng(self.seed + child_seed)
    
    def integers(self, low, high=None, size=None):
        """Generate random integers"""
        return self.rng.integers(low, high, size)
    
    def random(self, size=None):
        """Generate random floats in [0, 1)"""
        return self.rng.random(size)
    
    def choice(self, a, size=None, replace=True, p=None):
        """Generate random choices"""
        return self.rng.choice(a, size=size, replace=replace, p=p)
    
    def shuffle(self, x):
        """Shuffle array in-place"""
        self.rng.shuffle(x)
    
    def permutation(self, x):
        """Return random permutation"""
        return self.rng.permutation(x)
    
    def uniform(self, low=0.0, high=1.0, size=None):
        """Generate uniform random numbers"""
        return self.rng.uniform(low, high, size)
    
    def normal(self, loc=0.0, scale=1.0, size=None):
        """Generate normal random numbers"""
        return self.rng.normal(loc, scale, size)
    
    def get_state(self):
        """Get current RNG state for serialization"""
        return {
            'seed': self.seed,
            'bit_generator_state': self.rng.bit_generator.state
        }
    
    def set_state(self, state: dict):
        """Restore RNG from serialized state"""
        self.seed = state['seed']
        self.rng.bit_generator.state = state['bit_generator_state']

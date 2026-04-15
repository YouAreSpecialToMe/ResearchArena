"""
Base prediction models for conformal prediction.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


class BasePredictor:
    """Base class for point predictors."""
    
    def __init__(self):
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the predictor."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError


class RidgePredictor(BasePredictor):
    """Ridge regression predictor."""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RandomForestPredictor(BasePredictor):
    """Random Forest predictor."""
    
    def __init__(self, n_estimators: int = 50, max_depth: int = 10, random_state: int = 0):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def nonconformity_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute absolute residual nonconformity scores."""
    return np.abs(y_true - y_pred)


def compute_quantile(scores: np.ndarray, alpha: float, weights: np.ndarray = None) -> float:
    """
    Compute weighted quantile of scores.
    
    Args:
        scores: Array of nonconformity scores
        alpha: Miscoverage level (quantile level is 1-alpha)
        weights: Optional weights for weighted quantile
    """
    if len(scores) == 0:
        return 0.0
    
    quantile_level = 1 - alpha
    
    if weights is None:
        return np.quantile(scores, quantile_level)
    else:
        # Weighted quantile
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        cumsum = np.cumsum(sorted_weights)
        idx = np.searchsorted(cumsum, quantile_level, side='left')
        idx = min(idx, len(sorted_scores) - 1)
        
        return sorted_scores[idx]

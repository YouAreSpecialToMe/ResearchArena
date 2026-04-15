"""
SMAK-CP Core Framework.
Streaming Multi-Scale Adaptive Kernel Conformal Prediction.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from typing import List, Tuple, Optional
from shared.models import RidgePredictor, nonconformity_score, compute_quantile
from shared.utils import kernel_weights, effective_sample_size


class SMAKCP:
    """
    Streaming Multi-Scale Adaptive Kernel Conformal Predictor.
    
    Implements three variants:
    - SMAK-S: Single-scale selection
    - SMAK-W: Weighted aggregation
    - SMAK-I: Multi-scale intersection
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        K: int = 4,  # Number of scales
        rho: float = 2.0,  # Scale factor for geometric progression
        h0: float = 0.1,  # Base bandwidth
        eta: float = 0.05,  # Adaptation learning rate
        window_size: int = 500,
        n_min: int = 20,  # Minimum effective sample size
        T0: int = 100,  # Warm-up period
        lambda_reg: float = 0.1,  # Reliability regularization
        variant: str = 'S'  # 'S', 'W', or 'I'
    ):
        self.alpha = alpha
        self.K = K
        self.rho = rho
        self.h0 = h0
        self.eta = eta
        self.window_size = window_size
        self.n_min = n_min
        self.T0 = T0
        self.T1 = 2 * T0  # Full adaptation starts after T1
        self.lambda_reg = lambda_reg
        self.variant = variant
        
        # Initialize bandwidths for each scale
        self.bandwidths = np.array([h0 * (rho ** k) for k in range(K)])
        
        # Underlying predictor
        self.predictor = RidgePredictor(alpha=1.0)
        
        # History buffers
        self.X_history = []
        self.y_history = []
        self.scores_history = []
        
        # Coverage discrepancy history for each scale
        self.coverage_discrepancy_history = {k: [] for k in range(K)}
        
        # Scale reliability weights
        self.scale_weights = np.ones(K) / K
        
        # Current time step
        self.t = 0
        
        # Tracking
        self.selected_scale_history = []
        self.bandwidth_history = []
        self.coverage_discrepancy_history_combined = []
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the underlying predictor."""
        self.predictor.fit(X_train, y_train)
    
    def get_scale_bandwidths(self) -> np.ndarray:
        """Get current bandwidths for all scales."""
        # Base bandwidth may have adapted
        return np.array([self.bandwidths[0] * (self.rho ** k) for k in range(self.K)])
    
    def compute_effective_sample_sizes(self, X_query: np.ndarray) -> np.ndarray:
        """Compute effective sample size for each scale."""
        if len(self.X_history) == 0:
            return np.zeros(self.K)
        
        X_recent = np.array(self.X_history[-self.window_size:])
        bandwidths = self.get_scale_bandwidths()
        
        n_effs = []
        for h in bandwidths:
            weights = kernel_weights(X_query, X_recent, h)
            n_eff = effective_sample_size(weights)
            n_effs.append(n_eff)
        
        return np.array(n_effs)
    
    def compute_scale_quantile(self, X_query: np.ndarray, scale: int) -> float:
        """Compute quantile for a specific scale."""
        if len(self.scores_history) == 0:
            return 0.0
        
        X_recent = np.array(self.X_history[-self.window_size:])
        scores_recent = np.array(self.scores_history[-self.window_size:])
        
        bandwidths = self.get_scale_bandwidths()
        h = bandwidths[scale]
        
        weights = kernel_weights(X_query, X_recent, h)
        
        if np.sum(weights) == 0:
            return compute_quantile(scores_recent, self.alpha)
        
        return compute_quantile(scores_recent, self.alpha, weights)
    
    def select_scale(self, X_query: np.ndarray) -> int:
        """
        Select scale based on maximum effective sample size.
        Returns scale index with n_eff >= n_min, or 0 (coarsest) if none.
        """
        n_effs = self.compute_effective_sample_sizes(X_query)
        
        # Find scales with sufficient samples
        valid_scales = [k for k in range(self.K) if n_effs[k] >= self.n_min]
        
        if len(valid_scales) == 0:
            return 0  # Fallback to coarsest scale
        
        # Select scale with maximum effective sample size
        valid_n_effs = [n_effs[k] for k in valid_scales]
        selected = valid_scales[np.argmax(valid_n_effs)]
        
        return selected
    
    def get_active_scales(self, X_query: np.ndarray) -> List[int]:
        """Get list of active scales (with sufficient samples)."""
        n_effs = self.compute_effective_sample_sizes(X_query)
        return [k for k in range(self.K) if n_effs[k] >= self.n_min]
    
    def update_bandwidth(self, X_query: np.ndarray, coverage_discrepancies: dict):
        """
        Update base bandwidth based on coverage discrepancies.
        log(h_{t+1}) = log(h_t) - eta * sign(sum_k w_t^(k) * Delta_t^(k))
        """
        if self.t <= self.T0:
            # Warm-up: no adaptation
            return
        
        if self.t <= self.T1:
            # Gradual adaptation
            eta_t = self.eta * np.sqrt(self.T0 / self.t)
        else:
            # Full adaptation
            eta_t = self.eta
        
        # Compute weighted discrepancy
        weighted_discrepancy = sum(
            self.scale_weights[k] * coverage_discrepancies.get(k, 0)
            for k in range(self.K)
        )
        
        # Update base bandwidth (only the first scale)
        log_h = np.log(self.bandwidths[0])
        log_h = log_h - eta_t * np.sign(weighted_discrepancy)
        
        # Clip to reasonable range
        self.bandwidths[0] = np.clip(np.exp(log_h), 0.01, 1.0)
    
    def update_scale_weights(self):
        """Update scale reliability weights."""
        # Compute cumulative absolute discrepancy for each scale
        discrepancies = []
        for k in range(self.K):
            recent_discrepancies = self.coverage_discrepancy_history[k][-self.window_size:]
            if len(recent_discrepancies) > 0:
                cum_abs_disc = np.sum(np.abs(recent_discrepancies))
                discrepancies.append(cum_abs_disc)
            else:
                discrepancies.append(0)
        
        discrepancies = np.array(discrepancies)
        
        # Compute weights: lower discrepancy = higher weight
        weights = np.exp(-self.lambda_reg * discrepancies)
        
        # Normalize
        if np.sum(weights) > 0:
            self.scale_weights = weights / np.sum(weights)
        else:
            self.scale_weights = np.ones(self.K) / self.K
    
    def predict_s(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        SMAK-S: Single-scale selection.
        Returns (lower, upper, selected_scale).
        """
        y_pred = self.predictor.predict(X)
        
        # Select scale for each point
        selected_scales = [self.select_scale(x) for x in X]
        
        # Compute quantiles for selected scales
        quantiles = np.array([
            self.compute_scale_quantile(x, s) for x, s in zip(X, selected_scales)
        ])
        
        lower = y_pred - quantiles
        upper = y_pred + quantiles
        
        return lower, upper, selected_scales[0] if len(selected_scales) == 1 else selected_scales
    
    def predict_w(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        SMAK-W: Weighted aggregation of quantiles across active scales.
        """
        y_pred = self.predictor.predict(X)
        
        quantiles = []
        for x in X:
            active_scales = self.get_active_scales(x)
            if len(active_scales) == 0:
                active_scales = [0]
            
            # Get quantiles for active scales
            scale_quantiles = [self.compute_scale_quantile(x, k) for k in active_scales]
            
            # Get weights for active scales
            active_weights = np.array([self.scale_weights[k] for k in active_scales])
            active_weights = active_weights / np.sum(active_weights)
            
            # Weighted aggregation
            q_agg = np.sum(np.array(scale_quantiles) * active_weights)
            quantiles.append(q_agg)
        
        quantiles = np.array(quantiles)
        lower = y_pred - quantiles
        upper = y_pred + quantiles
        
        return lower, upper
    
    def predict_i(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        SMAK-I: Multi-scale intersection with calibrated thresholds.
        Uses adaptive calibration: alpha_k = alpha * w^(k) / sum_j w^(j).
        """
        y_pred = self.predictor.predict(X)
        
        lowers = []
        uppers = []
        
        for x in X:
            active_scales = self.get_active_scales(x)
            if len(active_scales) == 0:
                active_scales = [0]
            
            # Compute adaptive alphas for intersection
            active_weights = np.array([self.scale_weights[k] for k in active_scales])
            active_weights = active_weights / np.sum(active_weights)
            
            # Calibrated alphas: more reliable scales get smaller alpha (wider sets)
            alphas_k = self.alpha * active_weights / np.sum(active_weights)
            alphas_k = np.clip(alphas_k, self.alpha / (2 * self.K), self.alpha)
            
            # Get quantiles for each scale with calibrated alpha
            X_recent = np.array(self.X_history[-self.window_size:]) if len(self.X_history) > 0 else np.array([])
            scores_recent = np.array(self.scores_history[-self.window_size:]) if len(self.scores_history) > 0 else np.array([])
            
            scale_intervals = []
            for k, alpha_k in zip(active_scales, alphas_k):
                bandwidths = self.get_scale_bandwidths()
                h = bandwidths[k]
                
                if len(scores_recent) == 0:
                    q = 0.0
                else:
                    weights = kernel_weights(x, X_recent, h)
                    if np.sum(weights) == 0:
                        q = compute_quantile(scores_recent, alpha_k)
                    else:
                        q = compute_quantile(scores_recent, alpha_k, weights)
                
                y_p = self.predictor.predict(x.reshape(1, -1))[0]
                scale_intervals.append((y_p - q, y_p + q))
            
            # Intersection
            if len(scale_intervals) > 0:
                lower = max(interval[0] for interval in scale_intervals)
                upper = min(interval[1] for interval in scale_intervals)
                # Ensure valid interval
                if lower > upper:
                    # Fallback to union or widest interval
                    lower = min(interval[0] for interval in scale_intervals)
                    upper = max(interval[1] for interval in scale_intervals)
            else:
                lower = y_pred[0]
                upper = y_pred[0]
            
            lowers.append(lower)
            uppers.append(upper)
        
        return np.array(lowers), np.array(uppers)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict based on variant."""
        if self.variant == 'S':
            lower, upper, _ = self.predict_s(X)
            return lower, upper
        elif self.variant == 'W':
            return self.predict_w(X)
        elif self.variant == 'I':
            return self.predict_i(X)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
    
    def predict_and_update(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict interval and update state."""
        self.t += 1
        
        # Predict
        lower, upper = self.predict(X)
        
        # Check coverage
        covered = (y >= lower) & (y <= upper)
        width = upper - lower
        
        # Update history
        y_pred = self.predictor.predict(X)
        scores = nonconformity_score(y, y_pred)
        
        for i in range(len(X)):
            self.X_history.append(X[i])
            self.y_history.append(y[i])
            self.scores_history.append(scores[i])
        
        # Trim history
        if len(self.X_history) > self.window_size * 2:
            self.X_history = self.X_history[-self.window_size:]
            self.y_history = self.y_history[-self.window_size:]
            self.scores_history = self.scores_history[-self.window_size:]
        
        # Compute coverage discrepancies for each scale
        coverage_discrepancies = {}
        recent_window = min(self.window_size, len(self.X_history) - 1)
        
        if recent_window > 10:
            for k in range(self.K):
                # Compute local coverage for scale k
                X_recent = np.array(self.X_history[-recent_window:])
                y_recent = np.array(self.y_history[-recent_window:])
                
                # Use center of recent window as query
                x_query = X[-1]  # Current point
                bandwidths = self.get_scale_bandwidths()
                h = bandwidths[k]
                
                weights = kernel_weights(x_query, X_recent, h)
                
                if np.sum(weights) > 0:
                    # Weighted coverage in neighborhood
                    weighted_coverage = np.average(
                        [int(y_recent[j] >= lower[-1] and y_recent[j] <= upper[-1]) 
                         for j in range(len(y_recent))],
                        weights=weights
                    )
                    discrepancy = weighted_coverage - (1 - self.alpha)
                    coverage_discrepancies[k] = discrepancy
                    self.coverage_discrepancy_history[k].append(discrepancy)
                else:
                    coverage_discrepancies[k] = 0
                    self.coverage_discrepancy_history[k].append(0)
        else:
            for k in range(self.K):
                coverage_discrepancies[k] = 0
                self.coverage_discrepancy_history[k].append(0)
        
        # Update bandwidth and scale weights
        if self.t > self.T0:
            self.update_bandwidth(X[-1], coverage_discrepancies)
            self.update_scale_weights()
        
        # Track selected scale for S variant
        if self.variant == 'S':
            selected_scale = self.select_scale(X[-1])
            self.selected_scale_history.append(selected_scale)
        
        self.bandwidth_history.append(self.bandwidths[0])
        
        return covered, width

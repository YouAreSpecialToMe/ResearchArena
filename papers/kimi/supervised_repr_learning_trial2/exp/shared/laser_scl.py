"""
LASER-SCL: Learning-dynamics Aware Sample weighting with Expected-loss for Supervised Contrastive Learning.
Implements ELP computation, predicted future loss, and curriculum scheduling.
"""
import torch
import numpy as np
from collections import deque


class ELPTracker:
    """Tracks loss history and computes Expected Learning Progress for each sample."""
    
    def __init__(self, n_samples, window_size=10):
        self.n_samples = n_samples
        self.window_size = window_size
        self.loss_histories = [deque(maxlen=window_size) for _ in range(n_samples)]
        self.elp_values = np.zeros(n_samples)
        self.step_indices = np.arange(1, window_size + 1)  # [1, 2, ..., W]
    
    def update(self, indices, losses):
        """Update loss history for given samples."""
        indices = indices.cpu().numpy() if torch.is_tensor(indices) else indices
        losses = losses.detach().cpu().numpy() if torch.is_tensor(losses) else losses
        
        for idx, loss in zip(indices, losses):
            self.loss_histories[idx].append(loss)
    
    def compute_elp(self, indices=None):
        """
        Compute Expected Learning Progress (ELP) for given samples.
        ELP = -Cov(loss_history, step_indices) / Var(step_indices)
        Positive ELP = decreasing loss (learning)
        """
        if indices is None:
            indices = range(self.n_samples)
        
        indices = indices.cpu().numpy() if torch.is_tensor(indices) else np.array(indices)
        elp_batch = np.zeros(len(indices))
        
        for i, idx in enumerate(indices):
            history = list(self.loss_histories[idx])
            if len(history) < 3:  # Need at least 3 points for trend
                elp_batch[i] = 0.0
                continue
            
            # Use only available history
            steps = self.step_indices[:len(history)]
            
            # Compute covariance and variance
            loss_mean = np.mean(history)
            step_mean = np.mean(steps)
            
            cov = np.mean([(l - loss_mean) * (s - step_mean) for l, s in zip(history, steps)])
            var = np.var(steps)
            
            if var > 1e-6:
                # ELP = -slope (negative slope = positive ELP = learning)
                slope = cov / var
                elp_batch[i] = -slope
            else:
                elp_batch[i] = 0.0
        
        # Update stored ELP values
        for i, idx in enumerate(indices):
            self.elp_values[idx] = elp_batch[i]
        
        return elp_batch
    
    def get_mean_loss(self, indices):
        """Get mean loss over window for given samples."""
        indices = indices.cpu().numpy() if torch.is_tensor(indices) else np.array(indices)
        mean_losses = []
        
        for idx in indices:
            history = list(self.loss_histories[idx])
            if len(history) > 0:
                mean_losses.append(np.mean(history))
            else:
                mean_losses.append(0.0)
        
        return np.array(mean_losses)
    
    def predict_future_loss(self, indices, delta=5):
        """
        Predict future loss using linear extrapolation.
        l_hat(t+delta) = mean_loss - ELP * delta
        """
        indices = indices.cpu().numpy() if torch.is_tensor(indices) else np.array(indices)
        
        elp = self.compute_elp(indices)
        mean_loss = self.get_mean_loss(indices)
        
        # Predicted future loss
        predicted_loss = mean_loss - elp * delta
        
        return predicted_loss, elp, mean_loss


class CurriculumScheduler:
    """Curriculum schedule for gradually shifting focus from easy to hard samples."""
    
    def __init__(self, mu_min=0.3, mu_max=0.7, rho=2.0, total_epochs=500):
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.rho = rho
        self.total_epochs = total_epochs
    
    def get_threshold(self, epoch):
        """Compute curriculum threshold for current epoch."""
        # mu(t) = mu_min + (mu_max - mu_min) * (t/T)^rho
        progress = (epoch / self.total_epochs) ** self.rho
        mu = self.mu_min + (self.mu_max - self.mu_min) * progress
        return mu


class LASERSCL:
    """LASER-SCL sample weighting framework."""
    
    def __init__(self, n_samples, window_size=10, delta=5, 
                 mu_min=0.3, mu_max=0.7, rho=2.0, sigmoid_k=10.0, 
                 total_epochs=500):
        self.elp_tracker = ELPTracker(n_samples, window_size)
        self.curriculum = CurriculumScheduler(mu_min, mu_max, rho, total_epochs)
        self.delta = delta
        self.sigmoid_k = sigmoid_k
        self.total_epochs = total_epochs
    
    def update_losses(self, indices, losses):
        """Update loss histories."""
        self.elp_tracker.update(indices, losses)
    
    def compute_weights(self, indices, epoch):
        """
        Compute sample weights using ELP, predicted future loss, and curriculum.
        w_i = 1 / (1 + exp(k * (l_hat_i - mu(t))))
        """
        indices = indices.cpu().numpy() if torch.is_tensor(indices) else np.array(indices)
        
        # Get predicted future loss
        predicted_loss, elp, mean_loss = self.elp_tracker.predict_future_loss(indices, self.delta)
        
        # Get curriculum threshold
        mu = self.curriculum.get_threshold(epoch)
        
        # Compute weights using sigmoid
        weights = 1.0 / (1.0 + np.exp(self.sigmoid_k * (predicted_loss - mu)))
        
        return weights, predicted_loss, elp, mean_loss, mu
    
    def get_elp_values(self, indices=None):
        """Get ELP values for given indices."""
        return self.elp_tracker.elp_values if indices is None else self.elp_tracker.elp_values[indices]


# Ablation variants

class LASERSCL_NoCurriculum(LASERSCL):
    """LASER-SCL without curriculum (fixed threshold)."""
    
    def compute_weights(self, indices, epoch):
        """Use fixed threshold instead of curriculum."""
        indices = indices.cpu().numpy() if torch.is_tensor(indices) else np.array(indices)
        
        predicted_loss, elp, mean_loss = self.elp_tracker.predict_future_loss(indices, self.delta)
        
        # Fixed threshold
        mu = 0.5
        weights = 1.0 / (1.0 + np.exp(self.sigmoid_k * (predicted_loss - mu)))
        
        return weights, predicted_loss, elp, mean_loss, mu


class LASERSCL_NoELP(LASERSCL):
    """LASER-SCL using current loss instead of predicted future loss."""
    
    def compute_weights(self, indices, epoch):
        """Use current mean loss instead of predicted."""
        indices = indices.cpu().numpy() if torch.is_tensor(indices) else np.array(indices)
        
        # Use current loss (no prediction)
        mean_loss = self.elp_tracker.get_mean_loss(indices)
        predicted_loss = mean_loss  # No prediction
        elp = self.elp_tracker.compute_elp(indices)
        
        mu = self.curriculum.get_threshold(epoch)
        weights = 1.0 / (1.0 + np.exp(self.sigmoid_k * (predicted_loss - mu)))
        
        return weights, predicted_loss, elp, mean_loss, mu


class LASERSCL_Static(LASERSCL):
    """Static loss-based weighting (no ELP, no curriculum)."""
    
    def compute_weights(self, indices, epoch):
        """Simple loss-based weighting."""
        indices = indices.cpu().numpy() if torch.is_tensor(indices) else np.array(indices)
        
        mean_loss = self.elp_tracker.get_mean_loss(indices)
        predicted_loss = mean_loss
        elp = np.zeros(len(indices))
        
        # Fixed threshold
        mu = 0.5
        weights = 1.0 / (1.0 + np.exp(self.sigmoid_k * (predicted_loss - mu)))
        
        return weights, predicted_loss, elp, mean_loss, mu

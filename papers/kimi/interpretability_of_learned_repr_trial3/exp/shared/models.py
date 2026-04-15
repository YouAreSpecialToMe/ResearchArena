"""
SAE model implementations including TopK, JumpReLU, and RobustSAE variants.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TopKSAE(nn.Module):
    """Standard TopK SAE as baseline."""
    
    def __init__(self, d_model, d_sae, topk=32, l1_coeff=0):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.topk = topk
        self.l1_coeff = l1_coeff
        
        # Encoder
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) / np.sqrt(d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        
        # Decoder
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) / np.sqrt(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        # Normalize decoder weights
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)
    
    def encode(self, x):
        """Encode input to sparse representation."""
        # x: [batch, d_model]
        z_pre = x @ self.W_enc + self.b_enc
        z = F.relu(z_pre)
        
        # TopK sparsity
        topk_vals, topk_indices = torch.topk(z, self.topk, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_indices, topk_vals)
        
        return z_sparse, z_pre
    
    def decode(self, z):
        """Decode sparse representation."""
        return z @ self.W_dec + self.b_dec
    
    def forward(self, x):
        """Full forward pass."""
        z, z_pre = self.encode(x)
        x_recon = self.decode(z)
        
        return x_recon, z, z_pre
    
    def get_loss(self, x, x_recon, z, z_pre):
        """Compute training loss."""
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # L1 sparsity loss
        l1_loss = self.l1_coeff * z.abs().mean()
        
        return recon_loss + l1_loss, {
            'recon_loss': recon_loss.item(),
            'l1_loss': l1_loss.item()
        }


class JumpReLUSAE(nn.Module):
    """JumpReLU SAE with learned thresholds."""
    
    def __init__(self, d_model, d_sae, bandwidth=0.001):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.bandwidth = bandwidth
        
        # Encoder
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) / np.sqrt(d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        
        # Decoder
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) / np.sqrt(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        # JumpReLU thresholds
        self.thresholds = nn.Parameter(torch.ones(d_sae) * 0.5)
        
        # Normalize decoder weights
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)
    
    def heaviside(self, x):
        """Smooth heaviside function for gradient flow."""
        return torch.sigmoid(x / self.bandwidth)
    
    def encode(self, x):
        """Encode with JumpReLU."""
        z_pre = x @ self.W_enc + self.b_enc
        
        # JumpReLU activation
        jump = self.heaviside(z_pre - self.thresholds)
        z = z_pre * jump
        
        return z, z_pre
    
    def decode(self, z):
        """Decode."""
        return z @ self.W_dec + self.b_dec
    
    def forward(self, x):
        """Forward pass."""
        z, z_pre = self.encode(x)
        x_recon = self.decode(z)
        
        return x_recon, z, z_pre
    
    def get_loss(self, x, x_recon, z, z_pre):
        """Compute loss with threshold penalty."""
        recon_loss = F.mse_loss(x_recon, x)
        
        # Small penalty on thresholds to keep them reasonable
        threshold_penalty = 0.01 * F.relu(self.thresholds).mean()
        
        return recon_loss + threshold_penalty, {
            'recon_loss': recon_loss.item(),
            'threshold_penalty': threshold_penalty.item()
        }


class DenoisingSAE(TopKSAE):
    """SAE with dropout-based denoising (Pan et al. 2025)."""
    
    def __init__(self, d_model, d_sae, topk=32, dropout_rate=0.3):
        super().__init__(d_model, d_sae, topk)
        self.dropout_rate = dropout_rate
    
    def add_noise(self, x):
        """Add dropout noise to input."""
        if self.training:
            mask = torch.rand_like(x) > self.dropout_rate
            return x * mask
        return x
    
    def forward(self, x):
        """Forward with denoising."""
        x_noisy = self.add_noise(x)
        z, z_pre = self.encode(x_noisy)
        x_recon = self.decode(z)
        
        # Reconstruct clean input
        return x_recon, z, z_pre


class RobustSAE(TopKSAE):
    """
    RobustSAE with consistency regularization.
    Extends TopK SAE with consistency loss for adversarial robustness.
    """
    
    def __init__(self, d_model, d_sae, topk=32, lambda_consist=0.1, 
                 consistency_gamma=0.5, use_proxy=True):
        super().__init__(d_model, d_sae, topk)
        self.lambda_consist = lambda_consist
        self.consistency_gamma = consistency_gamma
        self.use_proxy = use_proxy
        
        # For tracking proxy scores
        self.register_buffer('proxy_scores', torch.zeros(d_sae))
    
    def forward(self, x):
        """Standard forward pass."""
        return super().forward(x)
    
    def forward_with_consistency(self, x_orig, x_perturbed):
        """
        Forward pass with consistency loss computation.
        
        Args:
            x_orig: Original activations [batch, d_model]
            x_perturbed: Perturbed activations [batch, d_model]
        
        Returns:
            Reconstructions, latents, and consistency loss components
        """
        # Encode both
        z_orig, z_orig_pre = self.encode(x_orig)
        z_pert, z_pert_pre = self.encode(x_perturbed)
        
        # Decode
        x_orig_recon = self.decode(z_orig)
        x_pert_recon = self.decode(z_pert)
        
        # Consistency loss: L2 distance between representations
        consistency_loss_continuous = F.mse_loss(z_orig, z_pert)
        
        # Discrete consistency: TopK pattern agreement
        # Get topk indices for both
        topk_vals_orig, topk_idx_orig = torch.topk(z_orig, self.topk, dim=-1)
        topk_vals_pert, topk_idx_pert = torch.topk(z_pert, self.topk, dim=-1)
        
        # Create binary masks
        mask_orig = torch.zeros_like(z_orig).scatter_(-1, topk_idx_orig, 1.0)
        mask_pert = torch.zeros_like(z_pert).scatter_(-1, topk_idx_pert, 1.0)
        
        consistency_loss_discrete = F.mse_loss(mask_orig, mask_pert)
        
        # Total consistency loss
        consistency_loss = consistency_loss_continuous + \
                          self.consistency_gamma * consistency_loss_discrete
        
        return (x_orig_recon, x_pert_recon), (z_orig, z_pert), \
               (consistency_loss, consistency_loss_continuous, consistency_loss_discrete)
    
    def compute_proxy_scores(self, x):
        """
        Compute unsupervised robustness proxy scores.
        R_i = ||grad_h z_i||_2 / (|z_i| + epsilon)
        
        Returns gradient concentration scores for each feature.
        """
        # Enable gradient computation for proxy calculation
        with torch.enable_grad():
            x_grad = x.detach().clone().requires_grad_(True)
            z, z_pre = self.encode(x_grad)
            
            # Compute gradients for each feature
            proxy_scores = []
            
            for i in range(min(self.d_sae, 500)):  # Limit to 500 features for efficiency
                if x_grad.grad is not None:
                    x_grad.grad.zero_()
                
                # Sum of feature i activations
                feature_sum = z[:, i].sum()
                feature_sum.backward(retain_graph=True)
                
                # Gradient norm
                grad_norm = x_grad.grad.norm(dim=1).mean()
                
                # Activation magnitude
                act_mag = z[:, i].abs().mean() + 1e-8
                
                # Proxy score
                score = grad_norm / act_mag
                proxy_scores.append(score.item())
        
        return torch.tensor(proxy_scores, device=x.device)
    
    def get_loss(self, x_orig, x_perturbed, x_orig_recon, x_pert_recon, 
                 z_orig, z_pert, consistency_components):
        """
        Compute full RobustSAE loss.
        """
        consistency_loss, consistency_continuous, consistency_discrete = consistency_components
        
        # Reconstruction losses
        recon_loss_orig = F.mse_loss(x_orig_recon, x_orig)
        recon_loss_pert = F.mse_loss(x_pert_recon, x_perturbed)
        recon_loss = (recon_loss_orig + recon_loss_pert) / 2
        
        # Sparsity loss
        l1_loss = self.l1_coeff * (z_orig.abs().mean() + z_pert.abs().mean()) / 2
        
        # Consistency loss
        total_loss = recon_loss + l1_loss + self.lambda_consist * consistency_loss
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'recon_loss_orig': recon_loss_orig.item(),
            'recon_loss_pert': recon_loss_pert.item(),
            'l1_loss': l1_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'consistency_continuous': consistency_continuous.item(),
            'consistency_discrete': consistency_discrete.item(),
            'total_loss': total_loss.item()
        }


def create_sae(model_type, d_model=512, d_sae=2048, **kwargs):
    """Factory function to create SAE models."""
    
    if model_type == 'topk':
        return TopKSAE(d_model, d_sae, 
                      topk=kwargs.get('topk', 32),
                      l1_coeff=kwargs.get('l1_coeff', 0))
    
    elif model_type == 'jumprelu':
        return JumpReLUSAE(d_model, d_sae,
                          bandwidth=kwargs.get('bandwidth', 0.001))
    
    elif model_type == 'denoising':
        return DenoisingSAE(d_model, d_sae,
                           topk=kwargs.get('topk', 32),
                           dropout_rate=kwargs.get('dropout_rate', 0.3))
    
    elif model_type == 'robust':
        return RobustSAE(d_model, d_sae,
                        topk=kwargs.get('topk', 32),
                        lambda_consist=kwargs.get('lambda_consist', 0.1),
                        consistency_gamma=kwargs.get('consistency_gamma', 0.5),
                        use_proxy=kwargs.get('use_proxy', True))
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

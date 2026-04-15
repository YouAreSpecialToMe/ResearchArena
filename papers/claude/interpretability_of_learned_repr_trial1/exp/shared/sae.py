"""TopK Sparse Autoencoder implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder (Gao et al., 2024).

    Encoder: W_enc @ (x - b_dec) + b_enc, then keep top-k activations.
    Decoder: W_dec @ z + b_dec  (W_dec columns are unit-normed).
    """

    def __init__(self, d_model: int, n_features: int, k: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(n_features, d_model))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(n_features, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Initialize with Kaiming uniform
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        # Normalize decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input, returning (top-k activations, top-k indices)."""
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc  # [batch, n_features]
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        topk_vals = F.relu(topk_vals)  # Ensure non-negative
        return topk_vals, topk_idx

    def decode(self, topk_vals: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """Decode from sparse activations."""
        # Construct sparse activation vector and multiply by decoder
        batch_size = topk_vals.shape[0]
        z = torch.zeros(batch_size, self.n_features, device=topk_vals.device, dtype=topk_vals.dtype)
        z.scatter_(1, topk_idx, topk_vals)
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        topk_vals, topk_idx = self.encode(x)
        x_hat = self.decode(topk_vals, topk_idx)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)

        # Auxiliary dead feature loss: encourage all features to be used
        # Use pre-activations for the aux loss
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        aux_loss = -pre_acts.mean()  # Encourage larger pre-activations (simple version)

        return {
            "x_hat": x_hat,
            "topk_vals": topk_vals,
            "topk_idx": topk_idx,
            "recon_loss": recon_loss,
            "aux_loss": aux_loss * 0.0,  # Disable aux loss for now, rely on TopK structure
            "loss": recon_loss,
        }

    def get_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get full sparse activation vector."""
        topk_vals, topk_idx = self.encode(x)
        batch_size = topk_vals.shape[0]
        z = torch.zeros(batch_size, self.n_features, device=topk_vals.device, dtype=topk_vals.dtype)
        z.scatter_(1, topk_idx, topk_vals)
        return z

    @torch.no_grad()
    def normalize_decoder(self):
        """Project decoder columns back to unit norm."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

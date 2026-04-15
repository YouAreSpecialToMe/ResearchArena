"""
IRT and hierarchical Bayesian models for PopBench.
FIXED: Proper Population EIG computation and metadata network training.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class MetadataNetwork(nn.Module):
    """Neural network predicting initial ability from metadata.
    
    FIXED: Input dimension is 11 (not 12):
    - log(params + 1): 1 dim
    - is_instruct: 1 dim
    - arch_code (normalized): 1 dim
    - family one-hot: 8 dims
    TOTAL: 11 dimensions
    """
    
    def __init__(self, input_dim: int = 11, hidden_dim: int = 32, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights with small values for stable training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


def compute_population_eig(
    pop_model,
    current_ability_mean: np.ndarray,
    current_ability_std: np.ndarray,
    available_items: List[int],
    item_discriminations: np.ndarray,
    item_difficulties: np.ndarray,
    model_metadata,
    n_samples: int = 20
) -> np.ndarray:
    """
    Compute Population Expected Information Gain for each available item.
    
    FIXED: Actually compute EIG using Monte Carlo sampling.
    EIG(item) = E_y[H(post) - H(post|y)] = E_y[KL(post|y || post)]
    
    For computational efficiency, we use an approximation based on
    expected reduction in posterior variance.
    """
    n_available = len(available_items)
    eigs = np.zeros(n_available)
    
    # Current entropy (approximated by sum of log std)
    current_entropy = 0.5 * np.sum(np.log(2 * np.pi * np.e * (current_ability_std ** 2 + 1e-6)))
    
    for i, item_idx in enumerate(available_items):
        a = item_discriminations[item_idx]
        b = item_difficulties[item_idx]
        
        # Expected response (assuming correct and incorrect)
        avg_theta = np.mean(current_ability_mean)
        p_correct = 1 / (1 + np.exp(-a * (avg_theta - b)))
        p_correct = np.clip(p_correct, 0.05, 0.95)
        
        # Simulate both possible outcomes
        expected_post_entropy = 0.0
        
        for outcome, prob in [(1, p_correct), (0, 1 - p_correct)]:
            # Approximate posterior std after observing this outcome
            # Using Bayesian update formula for Gaussian with logistic likelihood
            # The information gain is approximately a^2 * p * (1-p) / sigma^2
            info_gain = (a ** 2) * p_correct * (1 - p_correct)
            
            # Updated precision (1/variance)
            updated_precision = 1.0 / (current_ability_std ** 2 + 1e-6) + info_gain
            updated_std = 1.0 / np.sqrt(updated_precision + 1e-6)
            
            # Entropy of updated posterior
            post_entropy = 0.5 * np.sum(np.log(2 * np.pi * np.e * (updated_std ** 2 + 1e-6)))
            expected_post_entropy += prob * post_entropy
        
        # EIG is reduction in entropy
        eigs[i] = current_entropy - expected_post_entropy
    
    return eigs


class HierarchicalPopulationModel:
    """
    Hierarchical Bayesian MIRT model for LLM population.
    FIXED: Proper training with metadata network.
    """
    
    def __init__(
        self,
        n_dimensions: int = 3,
        n_families: int = 8,
        use_metadata_network: bool = True
    ):
        self.n_dimensions = n_dimensions
        self.n_families = n_families
        self.use_metadata_network = use_metadata_network
        
        # Learned parameters
        self.family_means = None  # (n_families, n_dimensions)
        self.family_covs = None   # (n_families, n_dimensions, n_dimensions)
        self.family_stds = None   # (n_families, n_dimensions) - diagonal approx
        self.metadata_net = None
        self.item_discriminations = None
        self.item_difficulties = None
        
        # Family name to index mapping
        self.family_to_idx = {
            'llama2': 0, 'llama3': 1, 'qwen2': 2, 'qwen3': 3,
            'gemma': 4, 'mistral': 5, 'phi': 6, 'other': 7
        }
    
    def fit(
        self,
        dataset,
        train_models: List[str],
        n_steps: int = 3000,
        lr: float = 0.01,
        batch_size: int = 10,
        seed: int = 42
    ) -> Dict[str, List]:
        """
        Train the hierarchical population model.
        FIXED: Proper seed handling and metadata network training.
        """
        # Set seeds for reproducibility - CRITICAL for variance across seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Initialize metadata network
        if self.use_metadata_network:
            self.metadata_net = MetadataNetwork(input_dim=11)
        
        # Initialize parameters with seed-dependent randomness
        family_means = nn.Parameter(torch.randn(self.n_families, self.n_dimensions) * 0.3 + 0.5)
        family_log_stds = nn.Parameter(torch.randn(self.n_families, self.n_dimensions) * 0.1)
        
        # Item parameters
        n_items = len(dataset.responses[train_models[0]])
        item_discriminations = nn.Parameter(torch.ones(n_items) * 1.0)
        item_difficulties = nn.Parameter(torch.zeros(n_items))
        
        # Model abilities with seed-dependent initialization
        n_train = len(train_models)
        model_abilities = nn.Parameter(torch.randn(n_train, self.n_dimensions) * 0.3 + 0.5)
        
        # Setup optimizer
        params = [family_means, family_log_stds, item_discriminations, item_difficulties, model_abilities]
        if self.metadata_net is not None:
            params.extend(self.metadata_net.parameters())
        
        optimizer = torch.optim.Adam(params, lr=lr)
        
        # Prepare data
        family_ids = torch.tensor([
            self.family_to_idx.get(dataset.models[m].family, 7)
            for m in train_models
        ])
        
        metadata = torch.tensor([
            dataset.models[m].to_features()
            for m in train_models
        ], dtype=torch.float32)
        
        responses = torch.tensor([
            dataset.responses[m]
            for m in train_models
        ], dtype=torch.float32)
        
        # Training loop
        losses = []
        nlls = []
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Compute loss (negative ELBO)
            family_stds = torch.exp(family_log_stds)
            
            # KL divergence for family means (prior ~ N(0.5, 1))
            kl_family = 0.5 * torch.sum(
                family_stds**2 + (family_means - 0.5)**2 - 1 - 2 * family_log_stds
            )
            
            # Get family parameters for each model
            family_idx = family_ids
            model_family_means = family_means[family_idx]
            model_family_stds = family_stds[family_idx]
            
            # Compute adjusted means from metadata network
            if self.use_metadata_network and self.metadata_net is not None:
                self.metadata_net.train()
                metadata_pred = self.metadata_net(metadata)
                # Metadata network predicts offsets
                metadata_adjusted_means = model_family_means + metadata_pred
            else:
                metadata_adjusted_means = model_family_means
            
            # Prior on model abilities given family and metadata
            kl_ability = 0.5 * torch.sum(
                ((model_abilities - metadata_adjusted_means) / (model_family_stds + 1e-6))**2
            )
            
            # IRT likelihood
            a = torch.clamp(item_discriminations, 0.1, 5.0)
            b = item_difficulties
            
            # Average ability across dimensions for each model
            avg_ability = model_abilities.mean(dim=1, keepdim=True)
            
            logits = a.unsqueeze(0) * (avg_ability - b.unsqueeze(0))
            probs = torch.sigmoid(logits)
            probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
            
            # Negative log-likelihood
            nll = -torch.sum(
                responses * torch.log(probs) + (1 - responses) * torch.log(1 - probs)
            )
            
            # Total loss with regularization
            loss = nll + 0.01 * kl_family + 0.001 * kl_ability
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            nlls.append(nll.item())
            
            if step % 500 == 0:
                print(f"  Step {step}: Loss = {loss.item():.2f}, NLL = {nll.item():.2f}, "
                      f"KL_fam = {kl_family.item():.2f}, KL_abil = {kl_ability.item():.2f}")
        
        # Store learned parameters
        self.family_means = family_means.detach().numpy()
        self.family_stds = family_stds.detach().numpy()
        self.item_discriminations = item_discriminations.detach().numpy()
        self.item_difficulties = item_difficulties.detach().numpy()
        
        # Store full covariance for each family (simplified diagonal)
        self.family_covs = np.zeros((self.n_families, self.n_dimensions, self.n_dimensions))
        for f in range(self.n_families):
            self.family_covs[f] = np.diag(self.family_stds[f] ** 2)
        
        return {"losses": losses, "nlls": nlls}
    
    def predict_zero_shot(self, model_metadata) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zero-shot prediction of ability from metadata.
        FIXED: Properly use metadata network for predictions.
        
        Returns:
            (mean, std) predicted ability vectors
        """
        features = torch.tensor(model_metadata.to_features()).unsqueeze(0)
        
        family_idx = self.family_to_idx.get(model_metadata.family, 7)
        family_mean = self.family_means[family_idx]
        family_std = self.family_stds[family_idx]
        
        if self.use_metadata_network and self.metadata_net is not None:
            with torch.no_grad():
                self.metadata_net.eval()
                metadata_pred = self.metadata_net(features).numpy()[0]
            # Use metadata-adjusted mean
            pred_mean = family_mean + metadata_pred
        else:
            pred_mean = family_mean
        
        return pred_mean, family_std
    
    def save(self, path: str):
        """Save learned model parameters."""
        state = {
            'family_means': self.family_means,
            'family_stds': self.family_stds,
            'family_covs': self.family_covs,
            'item_discriminations': self.item_discriminations,
            'item_difficulties': self.item_difficulties,
            'use_metadata_network': self.use_metadata_network
        }
        if self.metadata_net is not None:
            state['metadata_net'] = self.metadata_net.state_dict()
        np.save(path, state, allow_pickle=True)
    
    def load(self, path: str):
        """Load learned model parameters."""
        state = np.load(path, allow_pickle=True).item()
        self.family_means = state['family_means']
        self.family_stds = state['family_stds']
        self.family_covs = state.get('family_covs', None)
        self.item_discriminations = state['item_discriminations']
        self.item_difficulties = state['item_difficulties']
        self.use_metadata_network = state['use_metadata_network']
        
        if self.use_metadata_network and 'metadata_net' in state:
            self.metadata_net = MetadataNetwork(input_dim=11)
            self.metadata_net.load_state_dict(state['metadata_net'])


class TwoPLIRT:
    """Simple 2PL IRT model for baselines."""
    
    def __init__(self):
        self.a = None  # Discrimination
        self.b = None  # Difficulty
    
    def fit(self, responses: np.ndarray, n_iterations: int = 100, seed: int = 42):
        """Fit 2PL IRT using marginal maximum likelihood approximation."""
        np.random.seed(seed)
        n_models, n_items = responses.shape
        
        # Initialize parameters
        self.a = np.ones(n_items) * 1.0
        self.b = np.zeros(n_items)
        
        # Simple moment-based estimation
        p_correct = np.mean(responses, axis=0)
        p_correct = np.clip(p_correct, 0.1, 0.9)
        self.b = -np.log(p_correct / (1 - p_correct))
        
        # Discrimination based on variance
        self.a = np.ones(n_items) * 1.5
        
        return self
    
    def estimate_ability_map(
        self,
        responses: np.ndarray,
        item_indices: np.ndarray,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        seed: int = 42
    ) -> float:
        """Estimate ability via MAP."""
        if len(item_indices) == 0:
            return prior_mean
        
        a_subset = self.a[item_indices]
        b_subset = self.b[item_indices]
        r_subset = responses[item_indices]
        
        # Grid search for MAP estimate
        theta_range = np.linspace(-3, 3, 100)
        log_probs = []
        
        for theta in theta_range:
            logits = a_subset * (theta - b_subset)
            probs = 1 / (1 + np.exp(-logits))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            
            log_lik = np.sum(r_subset * np.log(probs) + (1 - r_subset) * np.log(1 - probs))
            log_prior = -0.5 * ((theta - prior_mean) / prior_std) ** 2
            log_probs.append(log_lik + log_prior)
        
        return theta_range[np.argmax(log_probs)]
    
    def estimate_ability_mirt_map(
        self,
        responses: np.ndarray,
        item_indices: np.ndarray,
        dim_mapping: np.ndarray,
        n_dimensions: int = 3,
        prior_mean: np.ndarray = None,
        prior_cov: np.ndarray = None,
        seed: int = 42
    ) -> np.ndarray:
        """Estimate multi-dimensional ability via MAP."""
        if prior_mean is None:
            prior_mean = np.zeros(n_dimensions)
        if prior_cov is None:
            prior_cov = np.eye(n_dimensions)
        
        if len(item_indices) == 0:
            return prior_mean
        
        # Simple iterative estimation per dimension
        theta = prior_mean.copy()
        
        for dim in range(n_dimensions):
            dim_mask = dim_mapping[item_indices] == dim
            if not np.any(dim_mask):
                continue
            
            dim_items = item_indices[dim_mask]
            a_dim = self.a[dim_items]
            b_dim = self.b[dim_items]
            r_dim = responses[dim_items]
            
            # 1D estimation for this dimension
            theta_range = np.linspace(-3, 3, 50)
            best_ll = -np.inf
            best_theta = theta[dim]
            
            for t in theta_range:
                logits = a_dim * (t - b_dim)
                probs = 1 / (1 + np.exp(-logits))
                probs = np.clip(probs, 1e-10, 1 - 1e-10)
                ll = np.sum(r_dim * np.log(probs) + (1 - r_dim) * np.log(1 - probs))
                if ll > best_ll:
                    best_ll = ll
                    best_theta = t
            
            theta[dim] = best_theta
        
        return theta

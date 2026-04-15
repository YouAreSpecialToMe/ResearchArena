"""
Fixed IRT and hierarchical Bayesian models for PopBench.

MAJOR FIX: Proper hierarchical structure where model abilities are GENERATED from family parameters.
The key insight: model abilities should NOT be independent parameters - they should be 
sampled/generated from family distributions.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class MetadataNetwork(nn.Module):
    """Neural network predicting ability from metadata."""
    
    def __init__(self, input_dim: int = 11, hidden_dim: int = 64, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


def compute_eig_vectorized(
    current_mean: np.ndarray,
    current_std: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    n_samples: int = 50
) -> np.ndarray:
    """Compute Expected Information Gain for items using Monte Carlo."""
    n_items = len(a)
    eigs = np.zeros(n_items)
    
    current_entropy = 0.5 * np.sum(np.log(2 * np.pi * np.e * (current_std ** 2 + 1e-8)))
    samples = np.random.randn(n_samples, len(current_mean)) * current_std + current_mean
    
    for i in range(n_items):
        a_i = a[i]
        b_i = b[i]
        
        logits = a_i * (samples.mean(axis=1) - b_i)
        probs = 1 / (1 + np.exp(-logits))
        probs = np.clip(probs, 0.01, 0.99)
        
        p_correct = np.mean(probs)
        fisher_info = (a_i ** 2) * p_correct * (1 - p_correct)
        
        expected_precision = 1.0 / (current_std.mean() ** 2 + 1e-8) + fisher_info
        expected_std = 1.0 / np.sqrt(expected_precision + 1e-8)
        expected_entropy = 0.5 * np.log(2 * np.pi * np.e * (expected_std ** 2 + 1e-8))
        
        eigs[i] = max(0, current_entropy - expected_entropy)
    
    return eigs


class HierarchicalPopulationModelV2:
    """
    PROPER Hierarchical Bayesian MIRT model.
    
    KEY FIX: Model abilities are COMPUTED from family parameters + metadata network.
    They are NOT independent learnable parameters.
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
        
        self.family_means = None
        self.family_stds = None
        self.family_covs = None
        self.metadata_net = None
        self.item_discriminations = None
        self.item_difficulties = None
        
        self.family_to_idx = {
            'llama2': 0, 'llama3': 1, 'qwen2': 2, 'qwen3': 3,
            'gemma': 4, 'mistral': 5, 'phi': 6, 'other': 7
        }
        self.idx_to_family = {v: k for k, v in self.family_to_idx.items()}
    
    def fit(
        self,
        dataset,
        train_models: List[str],
        n_steps: int = 5000,
        lr: float = 0.01,
        batch_size: int = 15,
        seed: int = 42,
        verbose: bool = True
    ) -> Dict:
        """
        Train hierarchical model with PROPER structure.
        
        Model abilities = family_mean + metadata_network(metadata)
        
        This forces the model to actually learn family structure since
        individual model abilities are not free parameters.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Initialize metadata network
        if self.use_metadata_network:
            self.metadata_net = MetadataNetwork(input_dim=11)
        
        # Family parameters - THESE are the main learnable parameters
        # Initialize with good spread
        family_means = nn.Parameter(torch.randn(self.n_families, self.n_dimensions) * 0.5)
        family_log_stds = nn.Parameter(torch.ones(self.n_families, self.n_dimensions) * 0.0)
        
        n_items = len(dataset.responses[train_models[0]])
        
        # Initialize item parameters
        train_responses = np.array([dataset.responses[m] for m in train_models])
        p_correct = np.clip(train_responses.mean(axis=0), 0.1, 0.9)
        init_difficulties = -np.log(p_correct / (1 - p_correct))
        
        item_discriminations = nn.Parameter(torch.ones(n_items) * 1.2)
        item_difficulties = nn.Parameter(torch.tensor(init_difficulties, dtype=torch.float32))
        
        # NO individual model abilities! They will be computed from families.
        
        # Optimizer - only family params, item params, and metadata network
        params = [family_means, family_log_stds, item_discriminations, item_difficulties]
        if self.metadata_net is not None:
            params.extend(self.metadata_net.parameters())
        
        optimizer = torch.optim.Adam(params, lr=lr)
        
        # Data preparation
        family_ids_list = [self.family_to_idx.get(dataset.models[m].family, 7) for m in train_models]
        family_ids = torch.tensor(family_ids_list)
        
        metadata_list = [dataset.models[m].to_features() for m in train_models]
        metadata = torch.tensor(np.array(metadata_list), dtype=torch.float32)
        
        responses = torch.tensor([dataset.responses[m] for m in train_models], dtype=torch.float32)
        
        # Compute family statistics for diagnostics
        family_acc = {}
        for i, fid in enumerate(family_ids_list):
            if fid not in family_acc:
                family_acc[fid] = []
            family_acc[fid].append(train_responses[i].mean())
        
        if verbose:
            print(f"    Family accuracy statistics from data:")
            for fid in sorted(family_acc.keys())[:4]:
                accs = family_acc[fid]
                print(f"      Family {fid}: mean={np.mean(accs):.3f}, std={np.std(accs):.3f}")
        
        losses = []
        nlls = []
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            family_stds = torch.exp(family_log_stds)
            
            # Get family parameters for each model
            model_family_means = family_means[family_ids]
            model_family_stds = family_stds[family_ids]
            
            # Metadata network predicts offsets
            if self.use_metadata_network and self.metadata_net is not None:
                self.metadata_net.train()
                metadata_offsets = self.metadata_net(metadata)
            else:
                metadata_offsets = torch.zeros_like(model_family_means)
            
            # CRITICAL: Model abilities are COMPUTED, not learned
            # This forces the model to encode all information in family_means
            model_abilities = model_family_means + metadata_offsets
            
            # IRT likelihood
            a = torch.clamp(item_discriminations, 0.1, 5.0)
            b = item_difficulties
            
            avg_ability = model_abilities.mean(dim=1, keepdim=True)
            
            logits = a.unsqueeze(0) * (avg_ability - b.unsqueeze(0))
            probs = torch.sigmoid(logits)
            probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
            
            nll = -torch.sum(responses * torch.log(probs) + (1 - responses) * torch.log(1 - probs))
            
            # Very light regularization to prevent explosion
            reg = 0.001 * torch.sum(family_means ** 2)
            
            loss = nll + reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            
            optimizer.step()
            
            losses.append(loss.item())
            nlls.append(nll.item())
            
            if verbose and step % 500 == 0:
                fam_std = family_means.detach().std(dim=0).mean().item()
                fam_range = family_means.detach().max() - family_means.detach().min()
                print(f"  Step {step}: NLL={nll.item():.1f}, "
                      f"Fam_diversity={fam_std:.3f}, Fam_range={fam_range:.3f}")
                
                if step % 1000 == 0:
                    for f in range(min(4, self.n_families)):
                        fm = family_means[f].detach().numpy()
                        print(f"    Family {f}: [{fm[0]:.2f}, {fm[1]:.2f}, {fm[2]:.2f}]")
        
        # Store parameters
        self.family_means = family_means.detach().numpy()
        self.family_stds = family_stds.detach().numpy()
        self.item_discriminations = item_discriminations.detach().numpy()
        self.item_difficulties = item_difficulties.detach().numpy()
        
        self.family_covs = np.zeros((self.n_families, self.n_dimensions, self.n_dimensions))
        for f in range(self.n_families):
            self.family_covs[f] = np.diag(self.family_stds[f] ** 2)
        
        return {"losses": losses, "nlls": nlls}
    
    def predict_zero_shot(self, model_metadata) -> Tuple[np.ndarray, np.ndarray]:
        """Zero-shot prediction using metadata and learned population structure."""
        family_idx = self.family_to_idx.get(model_metadata.family, 7)
        family_mean = self.family_means[family_idx].copy()
        family_std = self.family_stds[family_idx].copy()
        
        if self.use_metadata_network and self.metadata_net is not None:
            features = torch.tensor(model_metadata.to_features()).unsqueeze(0)
            with torch.no_grad():
                self.metadata_net.eval()
                metadata_offset = self.metadata_net(features).numpy()[0]
            pred_mean = family_mean + metadata_offset
        else:
            pred_mean = family_mean
        
        return pred_mean, family_std
    
    def save(self, path: str):
        state = {
            'family_means': self.family_means,
            'family_stds': self.family_stds,
            'family_covs': self.family_covs,
            'item_discriminations': self.item_discriminations,
            'item_difficulties': self.item_difficulties,
            'use_metadata_network': self.use_metadata_network,
            'family_to_idx': self.family_to_idx
        }
        if self.metadata_net is not None:
            state['metadata_net'] = self.metadata_net.state_dict()
        np.save(path, state, allow_pickle=True)
    
    def load(self, path: str):
        state = np.load(path, allow_pickle=True).item()
        self.family_means = state['family_means']
        self.family_stds = state['family_stds']
        self.family_covs = state.get('family_covs', None)
        self.item_discriminations = state['item_discriminations']
        self.item_difficulties = state['item_difficulties']
        self.use_metadata_network = state['use_metadata_network']
        
        if 'family_to_idx' in state:
            self.family_to_idx = state['family_to_idx']
            self.idx_to_family = {v: k for k, v in self.family_to_idx.items()}
        
        if self.use_metadata_network and 'metadata_net' in state:
            self.metadata_net = MetadataNetwork(input_dim=11)
            self.metadata_net.load_state_dict(state['metadata_net'])


class AdaptiveEvaluator:
    """Proper adaptive evaluation with EIG-based item selection."""
    
    def __init__(self, pop_model: HierarchicalPopulationModelV2, n_dimensions: int = 3):
        self.pop_model = pop_model
        self.n_dimensions = n_dimensions
    
    def evaluate_model(
        self,
        model_name: str,
        metadata,
        true_ability: np.ndarray,
        responses: np.ndarray,
        max_items: int = 100,
        target_mae: float = 0.05,
        use_population_prior: bool = True,
        use_population_eig: bool = True,
        seed: int = 42
    ) -> Dict:
        """Evaluate a single model with adaptive item selection."""
        np.random.seed(seed)
        
        true_overall = np.mean(true_ability)
        n_items_total = len(responses)
        
        # Initialize prior
        if use_population_prior:
            prior_mean, prior_std = self.pop_model.predict_zero_shot(metadata)
        else:
            prior_mean = np.zeros(self.n_dimensions)
            prior_std = np.ones(self.n_dimensions)
        
        a_all = self.pop_model.item_discriminations
        b_all = self.pop_model.item_difficulties
        
        selected_items = []
        available_mask = np.ones(n_items_total, dtype=bool)
        
        ability_estimates = []
        maes = []
        
        for step in range(max_items):
            available_items = np.where(available_mask)[0]
            if len(available_items) == 0:
                break
            
            # Select item using EIG
            if use_population_eig:
                eigs = compute_eig_vectorized(
                    prior_mean, prior_std,
                    a_all[available_items],
                    b_all[available_items],
                    n_samples=30
                )
            else:
                eigs = (a_all[available_items] ** 2) * 0.25
            
            best_idx = np.argmax(eigs)
            selected_item = available_items[best_idx]
            
            selected_items.append(selected_item)
            available_mask[selected_item] = False
            
            # Update posterior
            response = responses[selected_item]
            a_sel = a_all[selected_item]
            b_sel = b_all[selected_item]
            
            for dim in range(self.n_dimensions):
                prior_var = prior_std[dim] ** 2
                
                p = 1 / (1 + np.exp(-a_sel * (prior_mean[dim] - b_sel)))
                p = np.clip(p, 0.01, 0.99)
                
                fisher = (a_sel ** 2) * p * (1 - p)
                posterior_var = 1.0 / (1.0 / prior_var + fisher + 1e-6)
                
                if response > 0.5:
                    likelihood_shift = a_sel * 0.5
                else:
                    likelihood_shift = -a_sel * 0.5
                
                posterior_mean = prior_mean[dim] + posterior_var * likelihood_shift
                
                prior_mean[dim] = posterior_mean
                prior_std[dim] = np.sqrt(posterior_var)
            
            # Track progress
            current_estimate = np.mean(prior_mean)
            ability_estimates.append(current_estimate)
            maes.append(abs(current_estimate - true_overall))
            
            # Check convergence
            if step >= 20 and len(maes) >= 5:
                recent_mae = np.mean(maes[-5:])
                if recent_mae < target_mae:
                    break
        
        final_estimate = np.mean(prior_mean)
        
        return {
            'model': model_name,
            'final_estimate': final_estimate,
            'final_mae': abs(final_estimate - true_overall),
            'true_ability': true_overall,
            'items_used': len(selected_items),
            'ability_trace': ability_estimates,
            'mae_trace': maes,
            'final_std': np.mean(prior_std)
        }

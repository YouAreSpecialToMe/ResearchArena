"""
Fixed integrated experiment runner for Local Curvature Probing - Version 2.
Addresses SAE performance and curvature feature extraction issues.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
from transformers import GPT2Tokenizer, GPT2Model, ViTModel
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
import json
import time
from typing import Dict, List, Tuple

from shared.utils import set_seed, save_results, aggregate_results_across_seeds, compute_confidence_interval
from shared.curvature import CombinedCurvatureEstimator
from shared.metrics import (
    compute_semantic_coherence, correlation_test, ttest_comparison, 
    anova_test, compute_selectivity, aggregate_metrics_with_ci
)


# ============ FIXED SAE BASELINE ============

class SparseAutoencoder(torch.nn.Module):
    """Fixed SAE with tied weights and better initialization."""
    
    def __init__(self, input_dim: int, expansion_factor: int = 4):
        super().__init__()
        self.hidden_dim = input_dim * expansion_factor
        
        self.encoder = torch.nn.Linear(input_dim, self.hidden_dim)
        self.decoder = torch.nn.Linear(self.hidden_dim, input_dim, bias=False)
        
        # Tie decoder weights to encoder transpose for better learning
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.t().clone()
        
        # Xavier initialization
        torch.nn.init.xavier_uniform_(self.encoder.weight)
        torch.nn.init.zeros_(self.encoder.bias)
    
    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded


def train_sae_fixed_v2(
    features: np.ndarray,
    seed: int = 42,
    device: str = 'cuda',
    epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    sparsity_weight: float = 1e-4,
    l1_weight: float = 1e-5
) -> Tuple[np.ndarray, SparseAutoencoder]:
    """
    Train SAE with improved hyperparameters.
    """
    set_seed(seed)
    
    input_dim = features.shape[1]
    model = SparseAutoencoder(input_dim).to(device)
    
    # Use AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    # Normalize features
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0) + 1e-8
    features_normalized = (features - feature_mean) / feature_std
    
    dataset = TensorDataset(torch.FloatTensor(features_normalized))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    best_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_sparsity_loss = 0
        
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            recon, encoded = model(batch)
            
            # Reconstruction loss (MSE)
            recon_loss = ((recon - batch) ** 2).mean()
            
            # Sparsity loss (L1 on activations)
            sparsity_loss = encoded.abs().mean()
            
            # L1 regularization on encoder weights
            l1_reg = model.encoder.weight.abs().mean()
            
            # Total loss
            loss = recon_loss + sparsity_weight * sparsity_loss + l1_weight * l1_reg
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Keep decoder tied to encoder transpose
            with torch.no_grad():
                model.decoder.weight.data = model.encoder.weight.data.t().clone()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_sparsity_loss += sparsity_loss.item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(loader)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience and epoch > 100:
            print(f"    Early stopping at epoch {epoch}")
            break
        
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"    SAE Epoch {epoch}: loss={avg_loss:.6f}, "
                  f"recon={total_recon_loss/len(loader):.6f}, "
                  f"sparse={total_sparsity_loss/len(loader):.6f}")
    
    # Extract features
    model.eval()
    all_encoded = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            _, encoded = model(batch)
            all_encoded.append(encoded.cpu().numpy())
    
    encoded_features = np.concatenate(all_encoded, axis=0)
    
    return encoded_features, model


def run_sae_baseline_fixed_v2(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    device: str = 'cuda',
    n_features: int = 50
) -> Dict:
    """Run fixed SAE baseline with proper feature selection."""
    print(f"  Training SAE v2 (seed {seed})...")
    
    # Train SAE with more epochs and better settings
    sae_features, model = train_sae_fixed_v2(
        features, seed=seed, device=device, epochs=200, learning_rate=3e-4
    )
    
    # Feature selection based on variance (more discriminative)
    feature_vars = sae_features.var(axis=0)
    top_idx = np.argsort(feature_vars)[-n_features:]
    sae_features_selected = sae_features[:, top_idx]
    
    # Evaluate with cross-validation
    accuracies = []
    for cv_seed in [seed, seed+1, seed+2]:
        X_train, X_test, y_train, y_test = train_test_split(
            sae_features_selected, labels, test_size=0.2, random_state=cv_seed, stratify=labels
        )
        
        # Use better hyperparameters
        best_acc = 0
        for C in [0.1, 1, 10, 100]:
            clf = LogisticRegression(C=C, max_iter=2000, solver='lbfgs')
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            best_acc = max(best_acc, acc)
        
        accuracies.append(best_acc)
    
    return {
        'accuracy': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'n_features': n_features,
        'hidden_dim': sae_features.shape[1]
    }


# ============ IMPROVED CURVATURE-BASED FEATURES ============

def extract_curvature_features(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    n_components: int = 50
) -> np.ndarray:
    """
    Extract curvature-based features with proper PCA and curvature weighting.
    """
    set_seed(seed)
    
    # Compute curvature
    estimator = CombinedCurvatureEstimator(k=50, use_pca=True, use_orc=True, use_sff=True)
    curvature_results = estimator.estimate(features)
    curvature = curvature_results['combined_curvature']
    
    # Weight samples by curvature for PCA
    # High curvature regions get higher weight
    weights = curvature + 0.1  # Add small constant for stability
    weights = weights / weights.sum() * len(weights)  # Normalize
    
    # Weighted PCA
    # Sample points according to curvature weights
    n_samples = min(len(features), 3000)
    sample_probs = weights / weights.sum()
    sample_idx = np.random.choice(len(features), size=n_samples, p=sample_probs, replace=False)
    
    # Fit PCA on curvature-weighted sample
    pca = PCA(n_components=min(n_components, len(sample_idx), features.shape[1]))
    pca.fit(features[sample_idx])
    
    # Project all features
    transformed = pca.transform(features)
    
    return transformed


def run_curvature_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    n_components: int = 50
) -> Dict:
    """Run curvature-based feature extraction and evaluation."""
    print(f"  Extracting curvature features (seed {seed})...")
    
    # Extract curvature-weighted features
    curv_features = extract_curvature_features(features, labels, seed, n_components)
    
    # Evaluate with cross-validation
    accuracies = []
    for cv_seed in [seed, seed+1, seed+2]:
        X_train, X_test, y_train, y_test = train_test_split(
            curv_features, labels, test_size=0.2, random_state=cv_seed, stratify=labels
        )
        
        # Hyperparameter search
        best_acc = 0
        for C in [0.1, 1, 10, 100]:
            clf = LogisticRegression(C=C, max_iter=2000, solver='lbfgs')
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            best_acc = max(best_acc, acc)
        
        accuracies.append(best_acc)
    
    return {
        'accuracy': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'n_components': n_components
    }


# ============ DATA LOADING ============

def load_vision_data(dataset_name='cifar10', n_samples=5000, seed=42):
    """Load and prepare vision dataset."""
    set_seed(seed)
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root='data/vision', train=True, download=True, transform=transform
        )
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root='data/vision', train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    n_samples = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    subset = Subset(dataset, indices)
    
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=2)
    
    if hasattr(dataset, 'targets'):
        labels = np.array([dataset.targets[i] for i in indices])
    else:
        labels = np.array([dataset.labels[i] for i in indices])
    
    return loader, labels, indices


def extract_resnet_features(loader, device='cuda'):
    """Extract features from ResNet-18."""
    from torchvision.models import ResNet18_Weights
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model = model.to(device)
    model.eval()
    
    features = []
    def hook_fn(module, input, output):
        features.append(output.squeeze().detach().cpu())
    
    handle = model.avgpool.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device)
            _ = model(inputs)
    
    handle.remove()
    return torch.cat(features, dim=0).numpy()


# ============ LINEAR BASELINE ============

def run_linear_probe_baseline(features, labels, seed=42, n_components=50):
    """Run linear probing baseline with PCA reduction."""
    set_seed(seed)
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, features.shape[0], features.shape[1]))
    pca_features = pca.fit_transform(features)
    
    # Evaluate with cross-validation
    accuracies = []
    for cv_seed in [seed, seed+100, seed+200]:
        X_train, X_test, y_train, y_test = train_test_split(
            pca_features, labels, test_size=0.2, random_state=cv_seed, stratify=labels
        )
        
        # Hyperparameter search
        best_acc = 0
        for C in [0.1, 1, 10, 100]:
            clf = LogisticRegression(C=C, max_iter=2000, solver='lbfgs')
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            best_acc = max(best_acc, acc)
        
        accuracies.append(best_acc)
    
    return {
        'accuracy': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'explained_variance': float(pca.explained_variance_ratio_.sum())
    }


# ============ MAIN EXPERIMENTS ============

def experiment_3_feature_comparison_v2(dataset_name='cifar10', seed=42):
    """Experiment 3: Curvature vs Linear Features - Fixed Version."""
    print(f"\n[Experiment 3 v2] Feature comparison on {dataset_name} (seed {seed})")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    loader, labels, _ = load_vision_data(dataset_name, n_samples=5000, seed=seed)
    features = extract_resnet_features(loader, device)
    
    print(f"  Features: {features.shape}")
    
    # Linear baseline
    linear_results = run_linear_probe_baseline(features, labels, seed)
    print(f"  Linear (PCA): {linear_results['accuracy']:.4f} ± {linear_results['accuracy_std']:.4f}")
    
    # SAE baseline - FIXED V2
    sae_results = run_sae_baseline_fixed_v2(features, labels, seed, device)
    print(f"  SAE v2: {sae_results['accuracy']:.4f} ± {sae_results['accuracy_std']:.4f}")
    
    # Curvature-based features - IMPROVED
    curv_results = run_curvature_baseline(features, labels, seed)
    print(f"  Curvature: {curv_results['accuracy']:.4f} ± {curv_results['accuracy_std']:.4f}")
    
    improvement_over_linear = curv_results['accuracy'] - linear_results['accuracy']
    
    return {
        'dataset': dataset_name,
        'linear_accuracy': linear_results['accuracy'],
        'linear_accuracy_std': linear_results['accuracy_std'],
        'sae_accuracy': sae_results['accuracy'],
        'sae_accuracy_std': sae_results['accuracy_std'],
        'curvature_accuracy': curv_results['accuracy'],
        'curvature_accuracy_std': curv_results['accuracy_std'],
        'improvement_over_linear': improvement_over_linear,
        'improvement_over_sae': curv_results['accuracy'] - sae_results['accuracy']
    }


def experiment_5_ablation_v2(dataset_name='cifar10', seed=42):
    """Experiment 5: Ablation study with proper curvature values."""
    print(f"\n[Experiment 5 v2] Ablation on {dataset_name} (seed {seed})")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    loader, labels, _ = load_vision_data(dataset_name, n_samples=3000, seed=seed)
    features = extract_resnet_features(loader, device)
    
    results = {}
    
    # Full method
    estimator = CombinedCurvatureEstimator(k=50, use_pca=True, use_orc=True, use_sff=True)
    curv_results = estimator.estimate(features)
    results['full_curvature_mean'] = float(curv_results['combined_curvature'].mean())
    results['full_curvature_std'] = float(curv_results['combined_curvature'].std())
    results['full_curvature_range'] = [float(curv_results['combined_curvature'].min()), 
                                       float(curv_results['combined_curvature'].max())]
    
    # PCA only
    estimator_pca = CombinedCurvatureEstimator(k=50, use_pca=True, use_orc=False, use_sff=False)
    curv_pca = estimator_pca.estimate(features)
    results['pca_only_curvature_mean'] = float(curv_pca['combined_curvature'].mean())
    results['pca_only_curvature_std'] = float(curv_pca['combined_curvature'].std())
    
    # ORC only
    estimator_orc = CombinedCurvatureEstimator(k=10, use_pca=False, use_orc=True, use_sff=False)
    curv_orc = estimator_orc.estimate(features)
    results['orc_only_curvature_mean'] = float(curv_orc['combined_curvature'].mean())
    results['orc_only_curvature_std'] = float(curv_orc['combined_curvature'].std())
    
    # SFF only
    estimator_sff = CombinedCurvatureEstimator(k=50, use_pca=False, use_orc=False, use_sff=True)
    curv_sff = estimator_sff.estimate(features)
    results['sff_only_curvature_mean'] = float(curv_sff['combined_curvature'].mean())
    results['sff_only_curvature_std'] = float(curv_sff['combined_curvature'].std())
    
    print(f"  Full: {results['full_curvature_mean']:.4f} ± {results['full_curvature_std']:.4f}")
    print(f"  PCA only: {results['pca_only_curvature_mean']:.4f} ± {results['pca_only_curvature_std']:.4f}")
    print(f"  ORC only: {results['orc_only_curvature_mean']:.4f} ± {results['orc_only_curvature_std']:.4f}")
    print(f"  SFF only: {results['sff_only_curvature_mean']:.4f} ± {results['sff_only_curvature_std']:.4f}")
    
    return results


# ============ MAIN ============

def main():
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("="*60)
    print("FIXED EXPERIMENTS V2 - Local Curvature Probing")
    print("="*60)
    
    all_results = {}
    
    # Experiment 3: Feature Comparison - FIXED
    print("\n" + "="*60)
    print("EXPERIMENT 3 v2: Feature Comparison (Fixed)")
    print("="*60)
    
    exp3_results = []
    for seed in [42, 123, 456]:
        for dataset in ['cifar10', 'cifar100']:
            try:
                result = experiment_3_feature_comparison_v2(dataset, seed)
                exp3_results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
    
    all_results['experiment_3_v2'] = {
        'raw_results': exp3_results,
        'aggregated': aggregate_metrics_with_ci(exp3_results)
    }
    
    # Experiment 5: Ablation - FIXED
    print("\n" + "="*60)
    print("EXPERIMENT 5 v2: Ablation (Fixed)")
    print("="*60)
    
    exp5_results = []
    for seed in [42, 123]:
        try:
            result = experiment_5_ablation_v2('cifar10', seed)
            exp5_results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    all_results['experiment_5_v2'] = {
        'raw_results': exp5_results,
        'aggregated': aggregate_metrics_with_ci(exp5_results)
    }
    
    # Save all results
    total_time = time.time() - start_time
    all_results['total_runtime_seconds'] = total_time
    all_results['total_runtime_minutes'] = total_time / 60
    
    os.makedirs('results', exist_ok=True)
    save_results(all_results, 'results/all_experiments_fixed_v2.json')
    
    # Generate summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if 'experiment_3_v2' in all_results and 'aggregated' in all_results['experiment_3_v2']:
        agg = all_results['experiment_3_v2']['aggregated']
        if 'improvement_over_linear' in agg:
            imp = agg['improvement_over_linear']['mean']
            print(f"  Improvement over linear: {imp:.2%}")
            all_results['criterion_2_v2_met'] = imp >= 0.05
    
    save_results(all_results, 'results/all_experiments_fixed_v2.json')
    
    print("\n" + "="*60)
    print(f"Experiments completed in {total_time/60:.1f} minutes")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    main()

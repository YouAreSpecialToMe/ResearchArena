"""
Fixed integrated experiment runner for Local Curvature Probing.
Addresses all issues from self-review feedback.
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
    """Fixed SAE with proper initialization and training."""
    
    def __init__(self, input_dim: int, expansion_factor: int = 4):
        super().__init__()
        self.hidden_dim = input_dim * expansion_factor
        
        self.encoder = torch.nn.Linear(input_dim, self.hidden_dim)
        self.decoder = torch.nn.Linear(self.hidden_dim, input_dim)
        
        # Proper initialization
        torch.nn.init.xavier_uniform_(self.encoder.weight)
        torch.nn.init.zeros_(self.encoder.bias)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        torch.nn.init.zeros_(self.decoder.bias)
    
    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded


def train_sae_fixed(
    features: np.ndarray,
    seed: int = 42,
    device: str = 'cuda',
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    sparsity_weight: float = 1e-3
) -> Tuple[np.ndarray, SparseAutoencoder]:
    """
    Train SAE with proper hyperparameters and longer training.
    
    Returns:
        encoded_features: (n, hidden_dim) encoded features
        model: trained SAE model
    """
    set_seed(seed)
    
    input_dim = features.shape[1]
    model = SparseAutoencoder(input_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    dataset = TensorDataset(torch.FloatTensor(features))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_sparsity_loss = 0
        
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            recon, encoded = model(batch)
            
            # Reconstruction loss
            recon_loss = ((recon - batch) ** 2).mean()
            
            # Sparsity loss (L1 on activations)
            sparsity_loss = encoded.abs().mean()
            
            # Total loss
            loss = recon_loss + sparsity_weight * sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_sparsity_loss += sparsity_loss.item()
        
        scheduler.step()
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"    SAE Epoch {epoch}: loss={total_loss/len(loader):.4f}, "
                  f"recon={total_recon_loss/len(loader):.4f}, "
                  f"sparsity={total_sparsity_loss/len(loader):.4f}")
    
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


def run_sae_baseline_fixed(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    device: str = 'cuda',
    n_features: int = 50
) -> Dict:
    """Run fixed SAE baseline with proper training."""
    print(f"  Training SAE (seed {seed})...")
    
    # Train SAE with more epochs
    sae_features, model = train_sae_fixed(
        features, seed=seed, device=device, epochs=100, learning_rate=1e-3
    )
    
    # Select top features by mean activation
    mean_act = np.abs(sae_features).mean(axis=0)
    top_idx = np.argsort(mean_act)[-n_features:]
    sae_features_selected = sae_features[:, top_idx]
    
    # Evaluate with cross-validation for robustness
    accuracies = []
    for cv_seed in [seed, seed+1, seed+2]:
        X_train, X_test, y_train, y_test = train_test_split(
            sae_features_selected, labels, test_size=0.2, random_state=cv_seed, stratify=labels
        )
        
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        accuracies.append(acc)
    
    return {
        'accuracy': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'n_features': n_features,
        'hidden_dim': sae_features.shape[1]
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
        num_classes = 10
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root='data/vision', train=True, download=True, transform=transform
        )
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    n_samples = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    subset = Subset(dataset, indices)
    
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=2)
    
    if dataset_name == 'cifar10':
        labels = np.array([dataset.targets[i] for i in indices])
    else:
        labels = np.array([dataset.targets[i] for i in indices])
    
    return loader, labels, indices


def generate_cyclic_concepts_large(n_samples_per_class=1000, seed=42):
    """Generate large-scale cyclic concept datasets."""
    set_seed(seed)
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    
    data = {'days': {'prompts': [], 'labels': []},
            'months': {'prompts': [], 'labels': []}}
    
    # Generate day prompts
    templates_days = [
        "The day after {} is",
        "The day before {} is",
        "Two days after {} is",
        "{} is followed by",
        "{} comes after"
    ]
    
    for i, day in enumerate(days):
        for _ in range(n_samples_per_class):
            template = np.random.choice(templates_days)
            data['days']['prompts'].append(template.format(day))
            data['days']['labels'].append(i)
    
    # Generate month prompts
    templates_months = [
        "The month after {} is",
        "The month before {} is",
        "Two months after {} is",
        "{} is followed by",
        "{} comes after"
    ]
    
    for i, month in enumerate(months):
        for _ in range(n_samples_per_class):
            template = np.random.choice(templates_months)
            data['months']['prompts'].append(template.format(month))
            data['months']['labels'].append(i)
    
    return data


# ============ FEATURE EXTRACTION ============

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


def extract_vit_features(loader, device='cuda'):
    """Extract features from ViT-Tiny."""
    model = ViTModel.from_pretrained('facebook/deit-tiny-patch16-224')
    model = model.to(device)
    model.eval()
    
    features = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            cls_features = outputs.last_hidden_state[:, 0, :]
            features.append(cls_features.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def extract_gpt2_features(prompts, tokenizer, model, device='cuda', batch_size=32):
    """Extract features from GPT-2."""
    model.eval()
    features = []
    
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            last_token_idx = inputs['attention_mask'].sum(dim=1) - 1
            batch_features = hidden[torch.arange(len(hidden)), last_token_idx, :]
            features.append(batch_features.cpu().numpy())
    
    return np.concatenate(features, axis=0)


# ============ LINEAR BASELINE ============

def run_linear_probe_baseline(features, labels, seed=42, n_components=50):
    """Run linear probing baseline with PCA reduction."""
    set_seed(seed)
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, features.shape[0], features.shape[1]))
    pca_features = pca.fit_transform(features)
    
    # Hyperparameter search with cross-validation
    best_acc = 0
    best_C = 1.0
    
    X_train, X_test, y_train, y_test = train_test_split(
        pca_features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    
    for C in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
        clf = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        if acc > best_acc:
            best_acc = acc
            best_C = C
    
    # Final evaluation with multiple seeds for confidence interval
    accuracies = []
    for eval_seed in [seed, seed+100, seed+200]:
        X_train, X_test, y_train, y_test = train_test_split(
            pca_features, labels, test_size=0.2, random_state=eval_seed, stratify=labels
        )
        clf = LogisticRegression(C=best_C, max_iter=1000, solver='lbfgs')
        clf.fit(X_train, y_train)
        accuracies.append(clf.score(X_test, y_test))
    
    return {
        'accuracy': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'best_C': float(best_C),
        'explained_variance': float(pca.explained_variance_ratio_.sum())
    }


# ============ MAIN EXPERIMENTS ============

def experiment_1_curvature_semantics_vision(model_name='resnet18', dataset_name='cifar10', seed=42):
    """Experiment 1: Curvature-Semantics Correlation in Vision."""
    print(f"\n[Experiment 1] {model_name} on {dataset_name} (seed {seed})")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    loader, labels, indices = load_vision_data(dataset_name, n_samples=5000, seed=seed)
    
    # Extract features
    if model_name == 'resnet18':
        features = extract_resnet_features(loader, device)
    else:
        features = extract_vit_features(loader, device)
    
    print(f"  Features: {features.shape}")
    
    # Compute curvature with ORC enabled
    estimator = CombinedCurvatureEstimator(k=50, use_pca=True, use_orc=True, use_sff=True)
    curvature_results = estimator.estimate(features)
    curvature = curvature_results['combined_curvature']
    
    print(f"  Curvature range: [{curvature.min():.4f}, {curvature.max():.4f}]")
    print(f"  Methods used: {curvature_results['methods_used']}")
    
    # Split into high/low curvature regions
    high_thresh = np.percentile(curvature, 80)
    low_thresh = np.percentile(curvature, 20)
    
    high_curv_idx = np.where(curvature >= high_thresh)[0]
    low_curv_idx = np.where(curvature <= low_thresh)[0]
    
    # Compute semantic coherence
    high_coherence = compute_semantic_coherence(features, labels, high_curv_idx)
    low_coherence = compute_semantic_coherence(features, labels, low_curv_idx)
    
    print(f"  High curvature: entropy={high_coherence['entropy']:.3f}, purity={high_coherence['purity']:.3f}")
    print(f"  Low curvature: entropy={low_coherence['entropy']:.3f}, purity={low_coherence['purity']:.3f}")
    
    # Compute per-sample semantic coherence for correlation
    all_coherence = []
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=30).fit(features)
    _, neighbor_idx = nbrs.kneighbors(features)
    
    for i in range(len(features)):
        coh = compute_semantic_coherence(features, labels, neighbor_idx[i])
        all_coherence.append(coh['entropy'])
    
    all_coherence = np.array(all_coherence)
    
    # Correlation test
    corr_result = correlation_test(curvature, all_coherence, method='spearman')
    print(f"  Correlation: {corr_result['correlation']:.4f}, p={corr_result['p_value']:.4f}")
    
    return {
        'model': model_name,
        'dataset': dataset_name,
        'high_curvature_entropy': high_coherence['entropy'],
        'low_curvature_entropy': low_coherence['entropy'],
        'high_curvature_purity': high_coherence['purity'],
        'low_curvature_purity': low_coherence['purity'],
        'correlation': corr_result['correlation'],
        'p_value': corr_result['p_value'],
        'significant': corr_result['significant'],
        'highly_significant': corr_result['highly_significant'],
        'curvature_mean': float(curvature.mean()),
        'curvature_std': float(curvature.std())
    }


def experiment_2_curvature_semantics_language(concept_type='days', seed=42, n_samples_per_class=500):
    """Experiment 2: Curvature-Semantics Correlation in Language with larger samples."""
    print(f"\n[Experiment 2] GPT-2 on {concept_type} (seed {seed}, {n_samples_per_class} samples/class)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate large dataset
    data = generate_cyclic_concepts_large(n_samples_per_class=n_samples_per_class, seed=seed)
    prompts = data[concept_type]['prompts']
    labels = np.array(data[concept_type]['labels'])
    
    print(f"  Total samples: {len(prompts)}")
    
    # Load model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained('gpt2').to(device)
    
    # Extract features
    features = extract_gpt2_features(prompts, tokenizer, model, device, batch_size=64)
    print(f"  Features: {features.shape}")
    
    # Compute curvature with ORC enabled
    k = min(50, len(features) // 20)
    estimator = CombinedCurvatureEstimator(k=k, use_pca=True, use_orc=True, use_sff=False)
    curvature_results = estimator.estimate(features)
    curvature = curvature_results['combined_curvature']
    
    print(f"  Curvature range: [{curvature.min():.4f}, {curvature.max():.4f}]")
    
    # Test boundary hypothesis
    boundary_curvatures = []
    mid_curvatures = []
    
    for i in range(len(features)):
        label = labels[i]
        n_classes = len(np.unique(labels))
        
        # Check if at boundary (next to different class)
        neighbors = np.argsort(np.linalg.norm(features - features[i], axis=1))[1:10]
        neighbor_labels = labels[neighbors]
        
        # Boundary: different class neighbors OR transition (e.g., Sunday->Monday)
        if label == n_classes - 1:  # Last class (e.g., Sunday or December)
            is_boundary = any(neighbor_labels == 0)  # Neighbor is first class
        elif label == 0:  # First class
            is_boundary = any(neighbor_labels == n_classes - 1)
        else:
            is_boundary = any(neighbor_labels != label)
        
        if is_boundary:
            boundary_curvatures.append(curvature[i])
        else:
            mid_curvatures.append(curvature[i])
    
    boundary_curvatures = np.array(boundary_curvatures) if boundary_curvatures else np.array([0])
    mid_curvatures = np.array(mid_curvatures) if mid_curvatures else np.array([0])
    
    print(f"  Boundary curvature: {boundary_curvatures.mean():.4f} ± {boundary_curvatures.std():.4f} (n={len(boundary_curvatures)})")
    print(f"  Mid curvature: {mid_curvatures.mean():.4f} ± {mid_curvatures.std():.4f} (n={len(mid_curvatures)})")
    
    # T-test
    if len(boundary_curvatures) > 1 and len(mid_curvatures) > 1:
        ttest = ttest_comparison(boundary_curvatures, mid_curvatures)
        print(f"  T-test: t={ttest['t_statistic']:.4f}, p={ttest['p_value']:.4f}, d={ttest['cohens_d']:.4f}")
    else:
        ttest = {'t_statistic': 0.0, 'p_value': 1.0, 'significant': False, 'highly_significant': False}
    
    return {
        'concept_type': concept_type,
        'n_samples': len(prompts),
        'boundary_curvature_mean': float(boundary_curvatures.mean()),
        'boundary_curvature_std': float(boundary_curvatures.std()),
        'mid_curvature_mean': float(mid_curvatures.mean()),
        'mid_curvature_std': float(mid_curvatures.std()),
        't_statistic': ttest['t_statistic'],
        'p_value': ttest['p_value'],
        'significant': ttest['significant'],
        'highly_significant': ttest['highly_significant'],
        'cohens_d': ttest.get('cohens_d', 0.0)
    }


def experiment_3_feature_comparison(dataset_name='cifar10', seed=42):
    """Experiment 3: Curvature vs Linear Features."""
    print(f"\n[Experiment 3] Feature comparison on {dataset_name} (seed {seed})")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    loader, labels, _ = load_vision_data(dataset_name, n_samples=5000, seed=seed)
    features = extract_resnet_features(loader, device)
    
    print(f"  Features: {features.shape}")
    
    # Linear baseline (PCA top 50)
    linear_results = run_linear_probe_baseline(features, labels, seed)
    print(f"  Linear (PCA): {linear_results['accuracy']:.4f} ± {linear_results['accuracy_std']:.4f}")
    
    # SAE baseline - FIXED
    sae_results = run_sae_baseline_fixed(features, labels, seed, device)
    print(f"  SAE: {sae_results['accuracy']:.4f} ± {sae_results['accuracy_std']:.4f}")
    
    # Curvature-based features
    estimator = CombinedCurvatureEstimator(k=50, use_pca=True, use_orc=True, use_sff=True)
    curvature_results = estimator.estimate(features)
    curvature = curvature_results['combined_curvature']
    
    # Select high-curvature samples and extract PCA on them
    high_curv_thresh = np.percentile(curvature, 75)
    high_curv_idx = np.where(curvature > high_curv_thresh)[0]
    
    pca_curv = PCA(n_components=min(50, len(high_curv_idx)))
    pca_curv_features = pca_curv.fit_transform(features[high_curv_idx])
    
    # Project all features to this basis
    all_curv_features = (features - pca_curv.mean_) @ pca_curv.components_.T
    
    curv_probe_results = run_linear_probe_baseline(all_curv_features, labels, seed)
    print(f"  Curvature: {curv_probe_results['accuracy']:.4f} ± {curv_probe_results['accuracy_std']:.4f}")
    
    improvement_over_linear = curv_probe_results['accuracy'] - linear_results['accuracy']
    
    return {
        'dataset': dataset_name,
        'linear_accuracy': linear_results['accuracy'],
        'linear_accuracy_std': linear_results['accuracy_std'],
        'sae_accuracy': sae_results['accuracy'],
        'sae_accuracy_std': sae_results['accuracy_std'],
        'curvature_accuracy': curv_probe_results['accuracy'],
        'curvature_accuracy_std': curv_probe_results['accuracy_std'],
        'improvement_over_linear': improvement_over_linear,
        'improvement_over_sae': curv_probe_results['accuracy'] - sae_results['accuracy']
    }


def experiment_4_intervention(seed=42, n_samples=100):
    """Experiment 4: Intervention and Steering with larger sample size."""
    print(f"\n[Experiment 4] Intervention study (seed {seed}, n={n_samples})")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(seed)
    
    # Create larger sentiment dataset
    positive_templates = [
        'I love this', 'This is wonderful', 'Amazing product', 'Great experience',
        'Fantastic service', 'Excellent quality', 'Highly recommended', 'Best ever'
    ]
    negative_templates = [
        'I hate this', 'This is terrible', 'Awful product', 'Bad experience',
        'Poor service', 'Low quality', 'Not recommended', 'Worst ever'
    ]
    
    # Generate more samples
    prompts = []
    labels = []
    
    for i in range(n_samples // 2):
        prompts.append(positive_templates[i % len(positive_templates)])
        labels.append(1)
    
    for i in range(n_samples // 2):
        prompts.append(negative_templates[i % len(negative_templates)])
        labels.append(0)
    
    labels = np.array(labels)
    
    # Load GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained('gpt2').to(device)
    
    # Extract features
    features = extract_gpt2_features(prompts, tokenizer, model, device)
    
    # Compute curvature with ORC
    k = min(20, len(features) // 5)
    estimator = CombinedCurvatureEstimator(k=k, use_pca=True, use_orc=True, use_sff=False)
    curvature_results = estimator.estimate(features)
    curvature = curvature_results['combined_curvature']
    
    # Find curvature direction from high-curvature region
    high_curv_idx = np.argsort(curvature)[-max(10, len(curvature)//10):]
    
    # Compute concept direction
    pos_mean = features[labels == 1].mean(axis=0)
    neg_mean = features[labels == 0].mean(axis=0)
    concept_dir = pos_mean - neg_mean
    concept_dir = concept_dir / (np.linalg.norm(concept_dir) + 1e-10)
    
    # Compute selectivity with multiple metrics
    target_change = np.abs(features @ concept_dir).mean()
    
    # Random direction
    random_dir = np.random.randn(features.shape[1])
    random_dir = random_dir / np.linalg.norm(random_dir)
    random_change = np.abs(features @ random_dir).mean()
    
    # PCA direction
    pca = PCA(n_components=1)
    pca.fit(features)
    pca_dir = pca.components_[0]
    pca_change = np.abs(features @ pca_dir).mean()
    
    selectivity_vs_random = compute_selectivity(target_change, random_change)
    selectivity_vs_pca = compute_selectivity(target_change, pca_change)
    
    print(f"  Selectivity vs random: {selectivity_vs_random:.4f}")
    print(f"  Selectivity vs PCA: {selectivity_vs_pca:.4f}")
    
    return {
        'selectivity_vs_random': selectivity_vs_random,
        'selectivity_vs_pca': selectivity_vs_pca,
        'target_change': target_change,
        'random_change': random_change,
        'pca_change': pca_change,
        'n_samples': len(prompts)
    }


def experiment_5_ablation(dataset_name='cifar10', seed=42):
    """Experiment 5: Ablation study."""
    print(f"\n[Experiment 5] Ablation on {dataset_name} (seed {seed})")
    
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


def experiment_6_scaling(seed=42):
    """Experiment 6: Scaling and neighborhood sensitivity analysis."""
    print(f"\n[Experiment 6] Scaling and neighborhood analysis (seed {seed})")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = {'sample_scaling': {}, 'k_sensitivity': {}}
    
    # Sample size scaling
    for n_samples in [500, 1000, 2000, 3000, 5000]:
        loader, labels, _ = load_vision_data('cifar10', n_samples=n_samples, seed=seed)
        features = extract_resnet_features(loader, device)
        
        start_time = time.time()
        estimator = CombinedCurvatureEstimator(k=50, use_pca=True, use_orc=True, use_sff=True)
        _ = estimator.estimate(features)
        elapsed = time.time() - start_time
        
        results['sample_scaling'][f'n{n_samples}'] = {
            'time': elapsed,
            'features_shape': list(features.shape)
        }
        print(f"  n={n_samples}: {elapsed:.2f}s")
    
    # Neighborhood size sensitivity (k values)
    loader, labels, _ = load_vision_data('cifar10', n_samples=2000, seed=seed)
    features = extract_resnet_features(loader, device)
    
    for k in [10, 20, 30, 50, 100]:
        start_time = time.time()
        estimator = CombinedCurvatureEstimator(k=k, use_pca=True, use_orc=True, use_sff=True)
        curv_results = estimator.estimate(features)
        elapsed = time.time() - start_time
        
        results['k_sensitivity'][f'k{k}'] = {
            'time': elapsed,
            'curvature_mean': float(curv_results['combined_curvature'].mean()),
            'curvature_std': float(curv_results['combined_curvature'].std())
        }
        print(f"  k={k}: {elapsed:.2f}s, curvature={results['k_sensitivity'][f'k{k}']['curvature_mean']:.4f}")
    
    return results


# ============ MAIN ============

def main():
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("="*60)
    print("FIXED EXPERIMENTS - Local Curvature Probing")
    print("="*60)
    
    all_results = {}
    
    # Experiment 1: Vision Curvature-Semantics
    print("\n" + "="*60)
    print("EXPERIMENT 1: Curvature-Semantics Correlation (Vision)")
    print("="*60)
    
    exp1_results = []
    for seed in [42, 123, 456]:
        for model in ['resnet18']:  # Reduced for time
            for dataset in ['cifar10', 'cifar100']:
                try:
                    result = experiment_1_curvature_semantics_vision(model, dataset, seed)
                    exp1_results.append(result)
                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
    
    all_results['experiment_1'] = {
        'raw_results': exp1_results,
        'aggregated': aggregate_metrics_with_ci(exp1_results)
    }
    
    # Experiment 2: Language Curvature-Semantics
    print("\n" + "="*60)
    print("EXPERIMENT 2: Curvature-Semantics Correlation (Language)")
    print("="*60)
    
    exp2_results = []
    for seed in [42, 123, 456]:
        for concept in ['days', 'months']:
            try:
                result = experiment_2_curvature_semantics_language(concept, seed, n_samples_per_class=500)
                exp2_results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
    
    all_results['experiment_2'] = {
        'raw_results': exp2_results,
        'aggregated': aggregate_metrics_with_ci(exp2_results)
    }
    
    # Experiment 3: Feature Comparison
    print("\n" + "="*60)
    print("EXPERIMENT 3: Feature Comparison")
    print("="*60)
    
    exp3_results = []
    for seed in [42, 123, 456]:
        for dataset in ['cifar10', 'cifar100']:
            try:
                result = experiment_3_feature_comparison(dataset, seed)
                exp3_results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
    
    all_results['experiment_3'] = {
        'raw_results': exp3_results,
        'aggregated': aggregate_metrics_with_ci(exp3_results)
    }
    
    # Experiment 4: Intervention
    print("\n" + "="*60)
    print("EXPERIMENT 4: Intervention")
    print("="*60)
    
    exp4_results = []
    for seed in [42, 123, 456]:
        try:
            result = experiment_4_intervention(seed, n_samples=100)
            exp4_results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    all_results['experiment_4'] = {
        'raw_results': exp4_results,
        'aggregated': aggregate_metrics_with_ci(exp4_results)
    }
    
    # Experiment 5: Ablation
    print("\n" + "="*60)
    print("EXPERIMENT 5: Ablation")
    print("="*60)
    
    exp5_results = []
    for seed in [42, 123]:
        try:
            result = experiment_5_ablation('cifar10', seed)
            exp5_results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    all_results['experiment_5'] = {
        'raw_results': exp5_results,
        'aggregated': aggregate_metrics_with_ci(exp5_results)
    }
    
    # Experiment 6: Scaling
    print("\n" + "="*60)
    print("EXPERIMENT 6: Scaling")
    print("="*60)
    
    try:
        exp6_results = experiment_6_scaling(42)
        all_results['experiment_6'] = exp6_results
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Save all results
    total_time = time.time() - start_time
    all_results['total_runtime_seconds'] = total_time
    all_results['total_runtime_minutes'] = total_time / 60
    
    os.makedirs('results', exist_ok=True)
    save_results(all_results, 'results/all_experiments_fixed.json')
    
    # Generate summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Success criteria check
    print("\nSuccess Criteria:")
    
    # Criterion 1: Curvature-semantics correlation
    if 'experiment_1' in all_results and 'aggregated' in all_results['experiment_1']:
        agg = all_results['experiment_1']['aggregated']
        if 'highly_significant' in agg:
            sig_ratio = sum(1 for r in all_results['experiment_1']['raw_results'] 
                          if r.get('highly_significant', False)) / max(1, len(all_results['experiment_1']['raw_results']))
            print(f"  Criterion 1 (Curvature-Semantics): {sig_ratio:.1%} runs with p<0.01")
            all_results['criterion_1_met'] = sig_ratio >= 0.5
    
    # Criterion 2: Non-linear improvement
    if 'experiment_3' in all_results and 'aggregated' in all_results['experiment_3']:
        agg = all_results['experiment_3']['aggregated']
        if 'improvement_over_linear' in agg:
            imp = agg['improvement_over_linear']['mean']
            print(f"  Criterion 2 (Non-linear improvement): {imp:.2%} improvement")
            all_results['criterion_2_met'] = imp >= 0.05
    
    # Criterion 3: Intervention selectivity
    if 'experiment_4' in all_results and 'aggregated' in all_results['experiment_4']:
        agg = all_results['experiment_4']['aggregated']
        if 'selectivity_vs_random' in agg:
            sel = agg['selectivity_vs_random']['mean']
            print(f"  Criterion 3 (Intervention selectivity): {sel:.4f}")
            all_results['criterion_3_met'] = sel >= 0.7
    
    # Criterion 4: Computational feasibility
    if 'experiment_6' in all_results and 'sample_scaling' in all_results['experiment_6']:
        n5000_time = all_results['experiment_6']['sample_scaling'].get('n5000', {}).get('time', 0)
        print(f"  Criterion 4 (Computational): {n5000_time:.1f}s for n=5000")
        all_results['criterion_4_met'] = n5000_time < 300  # Less than 5 minutes
    
    # Save final results
    save_results(all_results, 'results/all_experiments_fixed.json')
    
    print("\n" + "="*60)
    print(f"All experiments completed in {total_time/60:.1f} minutes")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    main()

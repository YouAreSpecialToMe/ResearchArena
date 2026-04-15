"""
Baseline 2: Standard Contrastive Learning (scAGCL-style)
Cell-Cell contrastive learning without ontology guidance.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/exp')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import scanpy as sc
import json
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

from shared.data_loader import load_processed_data
from shared.metrics import compute_classification_metrics, compute_ood_detection_metrics
from shared.models import CellEncoder, info_nce_loss
from shared.utils import set_seed, save_results, get_device, count_parameters


def get_knn_pairs(X, k=10):
    """
    Get k-nearest neighbor positive pairs.
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Normalize for cosine similarity
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    
    # Find k-NN
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X_norm)
    distances, indices = nbrs.kneighbors(X_norm)
    
    # Return neighbor indices (excluding self)
    return indices[:, 1:]


def train_contrastive_model(adata, splits, seed=42, epochs=100, batch_size=256, lr=1e-3):
    """
    Train contrastive learning model.
    """
    set_seed(seed)
    device = get_device()
    
    # Prepare data
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = torch.FloatTensor(X)
    
    y = adata.obs['cell_type'].values
    
    # Encode labels
    unique_types = np.unique(y)
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    y_encoded = np.array([type_to_idx[t] for t in y])
    
    # Get k-NN pairs for training set
    train_indices = splits['train']
    X_train = X[train_indices].numpy()
    knn_pairs = get_knn_pairs(X_train, k=10)
    
    # Create model
    n_genes = X.shape[1]
    model = CellEncoder(n_genes, hidden_dim=512, output_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"  Model parameters: {count_parameters(model):,}")
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Shuffle training data
        perm = np.random.permutation(len(train_indices))
        
        for i in range(0, len(perm), batch_size):
            batch_perm = perm[i:i+batch_size]
            batch_indices = [train_indices[p] for p in batch_perm]
            
            # Get anchor cells
            x_anchor = X[batch_indices].to(device)
            
            # Get positive cells (k-NN in original space)
            pos_indices = []
            for p in batch_perm:
                if p < len(knn_pairs):
                    pos_idx = np.random.choice(knn_pairs[p])
                    pos_indices.append(train_indices[pos_idx])
                else:
                    pos_indices.append(train_indices[p])
            
            x_positive = X[pos_indices].to(device)
            
            # Get negative cells (random from different cell types)
            neg_indices = []
            for p in batch_perm:
                anchor_type = y_encoded[train_indices[p]]
                # Sample from different types
                diff_type_mask = y_encoded != anchor_type
                diff_type_indices = np.where(diff_type_mask)[0]
                if len(diff_type_indices) > 0:
                    neg_idx = np.random.choice(diff_type_indices)
                    neg_indices.append(neg_idx)
                else:
                    neg_indices.append(train_indices[p])
            
            x_negative = X[neg_indices].to(device)
            
            # Forward pass
            z_anchor = model(x_anchor)
            z_positive = model(x_positive)
            z_negative = model(x_negative)
            
            # Compute InfoNCE loss
            # Treat positives and negatives separately
            z_all = torch.cat([z_positive, z_negative], dim=0)
            loss = info_nce_loss(z_anchor, z_all, temperature=0.07)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        epoch_loss /= n_batches
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
    
    return model


def evaluate_model(model, adata, splits, seed=42):
    """
    Evaluate model using k-NN on embeddings.
    """
    device = get_device()
    model.eval()
    
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = torch.FloatTensor(X).to(device)
    
    y = adata.obs['cell_type'].values
    
    # Get embeddings
    with torch.no_grad():
        embeddings = []
        for i in range(0, len(X), 512):
            batch = X[i:i+512]
            z = model(batch)
            embeddings.append(z.cpu().numpy())
        embeddings = np.vstack(embeddings)
    
    # Split data
    train_indices = splits['train']
    test_indices = splits['test']
    
    X_train = embeddings[train_indices]
    y_train = y[train_indices]
    X_test = embeddings[test_indices]
    y_test = y[test_indices]
    
    # Train k-NN on embeddings
    knn = KNeighborsClassifier(n_neighbors=15, metric='cosine')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    metrics = compute_classification_metrics(y_test, y_pred)
    
    return metrics, embeddings


def evaluate_zero_shot(model, adata, splits, held_out_types, seed=42):
    """
    Evaluate zero-shot performance on held-out cell types.
    """
    device = get_device()
    model.eval()
    
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = torch.FloatTensor(X).to(device)
    
    y = adata.obs['cell_type'].values
    
    # Get embeddings
    with torch.no_grad():
        embeddings = []
        for i in range(0, len(X), 512):
            batch = X[i:i+512]
            z = model(batch)
            embeddings.append(z.cpu().numpy())
        embeddings = np.vstack(embeddings)
    
    # Filter out held-out types from training
    is_held_out = np.isin(y, held_out_types)
    train_mask = np.isin(np.arange(len(y)), splits['train']) & (~is_held_out)
    test_mask = np.isin(np.arange(len(y)), splits['test'])
    
    X_train = embeddings[train_mask]
    y_train = y[train_mask]
    X_test = embeddings[test_mask]
    y_test = y[test_mask]
    
    # Train k-NN
    knn = KNeighborsClassifier(n_neighbors=15, metric='cosine')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # Compute metrics on all test cells
    all_metrics = compute_classification_metrics(y_test, y_pred)
    
    # Compute metrics on zero-shot cells only
    zero_shot_mask = np.isin(y_test, held_out_types)
    if zero_shot_mask.sum() > 0:
        zero_shot_metrics = compute_classification_metrics(
            y_test[zero_shot_mask], y_pred[zero_shot_mask]
        )
    else:
        zero_shot_metrics = {'accuracy': 0.0, 'macro_f1': 0.0, 'weighted_f1': 0.0}
    
    # Compute metrics on seen cells
    seen_mask = ~zero_shot_mask
    if seen_mask.sum() > 0:
        seen_metrics = compute_classification_metrics(
            y_test[seen_mask], y_pred[seen_mask]
        )
    else:
        seen_metrics = {'accuracy': 0.0, 'macro_f1': 0.0, 'weighted_f1': 0.0}
    
    return {
        'all': all_metrics,
        'zero_shot': zero_shot_metrics,
        'seen': seen_metrics,
        'generalization_gap': seen_metrics['accuracy'] - zero_shot_metrics['accuracy']
    }


def main():
    print("=" * 60)
    print("Baseline 2: Standard Contrastive Learning (scAGCL-style)")
    print("=" * 60)
    
    data_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/data')
    output_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    adata = sc.read_h5ad(data_dir / 'pbmc3k_processed.h5ad')
    
    # Load zero-shot splits
    with open(data_dir / 'zero_shot_splits.json', 'r') as f:
        zero_shot_info = json.load(f)
    
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        print(f"\n[Seed {seed}]")
        
        # Load splits
        with open(data_dir / f'pbmc3k_splits_seed{seed}.json', 'r') as f:
            splits = json.load(f)
        
        held_out_types = zero_shot_info[f'seed_{seed}']['held_out_types']
        print(f"  Held-out types: {held_out_types}")
        
        # Train model
        print("  Training contrastive model...")
        model = train_contrastive_model(adata, splits, seed=seed, epochs=100, batch_size=256)
        
        # Save model
        model_path = output_dir / f'baseline_contrastive_seed{seed}.pt'
        torch.save(model.state_dict(), model_path)
        
        # Evaluate
        print("  Evaluating...")
        test_metrics, _ = evaluate_model(model, adata, splits, seed=seed)
        zero_shot_results = evaluate_zero_shot(model, adata, splits, held_out_types, seed=seed)
        
        print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Zero-shot accuracy: {zero_shot_results['zero_shot']['accuracy']:.4f}")
        print(f"  Generalization gap: {zero_shot_results['generalization_gap']:.4f}")
        
        all_results.append({
            'seed': seed,
            'test_accuracy': test_metrics['accuracy'],
            'test_macro_f1': test_metrics['macro_f1'],
            'zero_shot_accuracy': zero_shot_results['zero_shot']['accuracy'],
            'zero_shot_macro_f1': zero_shot_results['zero_shot']['macro_f1'],
            'seen_accuracy': zero_shot_results['seen']['accuracy'],
            'generalization_gap': zero_shot_results['generalization_gap']
        })
    
    # Compute mean and std
    mean_results = {
        'test_accuracy': np.mean([r['test_accuracy'] for r in all_results]),
        'test_accuracy_std': np.std([r['test_accuracy'] for r in all_results]),
        'test_macro_f1': np.mean([r['test_macro_f1'] for r in all_results]),
        'test_macro_f1_std': np.std([r['test_macro_f1'] for r in all_results]),
        'zero_shot_accuracy': np.mean([r['zero_shot_accuracy'] for r in all_results]),
        'zero_shot_accuracy_std': np.std([r['zero_shot_accuracy'] for r in all_results]),
        'zero_shot_macro_f1': np.mean([r['zero_shot_macro_f1'] for r in all_results]),
        'zero_shot_macro_f1_std': np.std([r['zero_shot_macro_f1'] for r in all_results]),
        'generalization_gap': np.mean([r['generalization_gap'] for r in all_results]),
        'generalization_gap_std': np.std([r['generalization_gap'] for r in all_results]),
    }
    
    final_results = {
        'method': 'Contrastive Learning (scAGCL-style)',
        'description': 'Cell-cell contrastive learning without ontology guidance',
        'seeds': seeds,
        'per_seed_results': all_results,
        'mean': mean_results,
        'config': {'epochs': 100, 'batch_size': 256, 'lr': 1e-3}
    }
    
    save_results(final_results, output_dir / 'baseline_contrastive.json')
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Test Accuracy: {mean_results['test_accuracy']:.4f} ± {mean_results['test_accuracy_std']:.4f}")
    print(f"Zero-shot Accuracy: {mean_results['zero_shot_accuracy']:.4f} ± {mean_results['zero_shot_accuracy_std']:.4f}")
    print(f"Generalization Gap: {mean_results['generalization_gap']:.4f} ± {mean_results['generalization_gap_std']:.4f}")


if __name__ == '__main__':
    main()

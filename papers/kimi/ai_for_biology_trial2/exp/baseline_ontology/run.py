"""
Baseline 3: Ontology-Guided Method (OnClass-style)
Uses Cell Ontology hierarchy for classification.
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

from shared.metrics import compute_classification_metrics
from shared.models import CellEncoder
from shared.utils import set_seed, save_results, get_device, count_parameters


class OntologyHierarchy:
    """
    Simple ontology hierarchy for cell types.
    """
    def __init__(self, cell_types):
        self.cell_types = list(cell_types)
        self.n_types = len(cell_types)
        
        # Create a simple hierarchy: group similar cell types
        # This is a simplified version - in practice, use real Cell Ontology
        self.hierarchy = self._build_simple_hierarchy()
        
    def _build_simple_hierarchy(self):
        """
        Build a simple hierarchy based on cell type names.
        """
        hierarchy = {}
        
        for cell_type in self.cell_types:
            # Create a simple parent-child relationship
            if 'T' in cell_type or 'T cell' in cell_type:
                hierarchy[cell_type] = {'parent': 'T cells', 'level': 1}
            elif 'B' in cell_type or 'B cell' in cell_type:
                hierarchy[cell_type] = {'parent': 'B cells', 'level': 1}
            elif 'Monocyte' in cell_type or 'Macrophage' in cell_type:
                hierarchy[cell_type] = {'parent': 'Myeloid', 'level': 1}
            elif 'NK' in cell_type:
                hierarchy[cell_type] = {'parent': 'Lymphoid', 'level': 1}
            elif 'Dendritic' in cell_type:
                hierarchy[cell_type] = {'parent': 'Myeloid', 'level': 1}
            else:
                hierarchy[cell_type] = {'parent': 'Root', 'level': 1}
        
        return hierarchy
    
    def get_ontological_distance(self, type1, type2):
        """
        Get distance between two cell types in ontology.
        """
        if type1 == type2:
            return 0.0
        
        h1 = self.hierarchy.get(type1, {})
        h2 = self.hierarchy.get(type2, {})
        
        # Same parent = close in ontology
        if h1.get('parent') == h2.get('parent'):
            return 0.3
        
        # Different parents but same grandparent (simplified)
        return 0.7
    
    def get_distance_matrix(self):
        """
        Get pairwise distance matrix for all cell types.
        """
        dist_matrix = np.zeros((self.n_types, self.n_types))
        
        for i, t1 in enumerate(self.cell_types):
            for j, t2 in enumerate(self.cell_types):
                dist_matrix[i, j] = self.get_ontological_distance(t1, t2)
        
        return dist_matrix


class OntologyGuidedModel(nn.Module):
    """
    Ontology-guided model with hierarchical classification.
    """
    def __init__(self, n_genes, n_cell_types, hidden_dim=512, embed_dim=128):
        super().__init__()
        
        self.cell_encoder = CellEncoder(n_genes, hidden_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, n_cell_types)
        
    def forward(self, x):
        z = self.cell_encoder(x)
        logits = self.classifier(z)
        return z, logits


def train_ontology_model(adata, splits, ontology, seed=42, epochs=50, batch_size=256, lr=1e-3):
    """
    Train ontology-guided model.
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
    y_tensor = torch.LongTensor(y_encoded)
    
    # Get distance matrix for ontology-aware loss
    dist_matrix = ontology.get_distance_matrix()
    dist_matrix = torch.FloatTensor(dist_matrix).to(device)
    
    # Create model
    n_genes = X.shape[1]
    n_cell_types = len(unique_types)
    model = OntologyGuidedModel(n_genes, n_cell_types).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"  Model parameters: {count_parameters(model):,}")
    
    # Training loop
    train_indices = splits['train']
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Shuffle training data
        perm = np.random.permutation(len(train_indices))
        
        for i in range(0, len(perm), batch_size):
            batch_perm = perm[i:i+batch_size]
            batch_indices = [train_indices[p] for p in batch_perm]
            
            x_batch = X[batch_indices].to(device)
            y_batch = y_tensor[batch_indices].to(device)
            
            # Forward pass
            z, logits = model(x_batch)
            
            # Standard cross-entropy loss
            loss = F.cross_entropy(logits, y_batch)
            
            # Add ontology-aware regularization
            # Encourage similar embeddings for similar cell types
            probs = F.softmax(logits, dim=1)
            
            # Get ontological distances for this batch
            batch_distances = dist_matrix[y_batch][:, y_batch]
            
            # Embedding similarity should correlate with ontological similarity
            z_sim = torch.matmul(z, z.T)
            ont_sim = 1.0 - batch_distances  # Convert distance to similarity
            
            # Ontology alignment loss
            ont_loss = F.mse_loss(z_sim, ont_sim)
            
            # Total loss
            total_loss = loss + 0.1 * ont_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            n_batches += 1
        
        epoch_loss /= n_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    return model, type_to_idx


def evaluate_model(model, adata, splits, type_to_idx, seed=42):
    """
    Evaluate model.
    """
    device = get_device()
    model.eval()
    
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = torch.FloatTensor(X).to(device)
    
    y = adata.obs['cell_type'].values
    idx_to_type = {i: t for t, i in type_to_idx.items()}
    
    # Predict
    test_indices = splits['test']
    with torch.no_grad():
        _, logits = model(X[test_indices])
        y_pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
    
    y_pred = [idx_to_type[i] for i in y_pred_idx]
    y_test = y[test_indices]
    
    metrics = compute_classification_metrics(y_test, y_pred)
    
    return metrics


def evaluate_zero_shot(model, adata, splits, held_out_types, type_to_idx, seed=42):
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
    idx_to_type = {i: t for t, i in type_to_idx.items()}
    
    # Filter out held-out types from training
    is_held_out = np.isin(y, held_out_types)
    test_mask = np.isin(np.arange(len(y)), splits['test'])
    
    # Predict on test set
    test_indices = np.where(test_mask)[0]
    with torch.no_grad():
        _, logits = model(X[test_indices])
        y_pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
    
    y_pred = np.array([idx_to_type[i] for i in y_pred_idx])
    y_test = np.array(y[test_indices])
    
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
    print("Baseline 3: Ontology-Guided Method (OnClass-style)")
    print("=" * 60)
    
    data_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/data')
    output_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    adata = sc.read_h5ad(data_dir / 'pbmc3k_processed.h5ad')
    
    # Load zero-shot splits
    with open(data_dir / 'zero_shot_splits.json', 'r') as f:
        zero_shot_info = json.load(f)
    
    # Create ontology hierarchy
    cell_types = adata.obs['cell_type'].unique()
    ontology = OntologyHierarchy(cell_types)
    
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
        print("  Training ontology-guided model...")
        model, type_to_idx = train_ontology_model(adata, splits, ontology, seed=seed, epochs=50)
        
        # Save model
        model_path = output_dir / f'baseline_ontology_seed{seed}.pt'
        torch.save(model.state_dict(), model_path)
        
        # Evaluate
        print("  Evaluating...")
        test_metrics = evaluate_model(model, adata, splits, type_to_idx, seed=seed)
        zero_shot_results = evaluate_zero_shot(model, adata, splits, held_out_types, type_to_idx, seed=seed)
        
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
        'method': 'Ontology-Guided (OnClass-style)',
        'description': 'Ontology-guided classification with hierarchical structure',
        'seeds': seeds,
        'per_seed_results': all_results,
        'mean': mean_results,
        'config': {'epochs': 50, 'batch_size': 256, 'lr': 1e-3}
    }
    
    save_results(final_results, output_dir / 'baseline_ontology.json')
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Test Accuracy: {mean_results['test_accuracy']:.4f} ± {mean_results['test_accuracy_std']:.4f}")
    print(f"Zero-shot Accuracy: {mean_results['zero_shot_accuracy']:.4f} ± {mean_results['zero_shot_accuracy_std']:.4f}")
    print(f"Generalization Gap: {mean_results['generalization_gap']:.4f} ± {mean_results['generalization_gap_std']:.4f}")


if __name__ == '__main__':
    main()

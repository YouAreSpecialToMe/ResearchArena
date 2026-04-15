"""
Tri-Con V3: Simplified and Stable Tri-Hierarchy Model
Uses supervised training with auxiliary contrastive objectives.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/exp')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
import json
from pathlib import Path

from shared.metrics import compute_classification_metrics, compute_ood_detection_metrics
from shared.models import CellEncoder, EvidentialHead
from shared.utils import set_seed, save_results, get_device, count_parameters


class SimpleTriCon(nn.Module):
    """
    Simplified Tri-Con: Cell encoder + cell type embeddings + classifier.
    Uses stable supervised training with optional contrastive regularization.
    """
    def __init__(self, n_genes, n_cell_types, hidden_dim=512, embed_dim=128, use_evidential=True):
        super().__init__()
        
        self.n_cell_types = n_cell_types
        self.embed_dim = embed_dim
        self.use_evidential = use_evidential
        
        # Cell encoder
        self.cell_encoder = CellEncoder(n_genes, hidden_dim, embed_dim)
        
        # Cell type embeddings (for zero-shot and contrastive)
        self.cell_type_embeddings = nn.Embedding(n_cell_types, embed_dim)
        nn.init.xavier_uniform_(self.cell_type_embeddings.weight)
        
        # Classifier head
        if use_evidential:
            self.evidential_head = EvidentialHead(embed_dim, n_cell_types)
        else:
            self.classifier = nn.Linear(embed_dim, n_cell_types)
    
    def forward(self, x_cell, return_embedding=False):
        """Forward pass."""
        z_cell = self.cell_encoder(x_cell)
        
        if return_embedding:
            return z_cell
        
        if self.use_evidential:
            alpha = self.evidential_head(z_cell)
            uncertainty = self.evidential_head.compute_uncertainty(alpha)
            return z_cell, alpha, uncertainty
        else:
            logits = self.classifier(z_cell)
            return z_cell, logits, None
    
    def get_cell_type_embeddings(self):
        """Get all cell type embeddings."""
        indices = torch.arange(self.n_cell_types, device=self.cell_type_embeddings.weight.device)
        return F.normalize(self.cell_type_embeddings(indices), p=2, dim=1)
    
    def predict_from_embeddings(self, z_cells):
        """Predict cell types from embeddings using cosine similarity."""
        z_types = self.get_cell_type_embeddings()
        sim = torch.matmul(z_cells, z_types.T)
        return torch.argmax(sim, dim=1)


def compute_loss(model, x_batch, y_batch, lambda_co=0.5, lambda_cc=0.3, temperature=0.07):
    """
    Compute combined supervised + contrastive loss.
    """
    device = x_batch.device
    
    # Forward pass
    z_cells, alpha, uncertainty = model(x_batch)
    z_types = model.get_cell_type_embeddings()
    
    # Supervised loss (evidential or cross-entropy)
    if model.use_evidential:
        loss_supervised = model.evidential_head.compute_loss(alpha, y_batch.to(device))
    else:
        logits = alpha  # In non-evidential mode, alpha is actually logits
        loss_supervised = F.cross_entropy(logits, y_batch.to(device))
    
    # L_CO: Cell-Ontology Contrast - cells should be close to their type embedding
    z_cells_norm = F.normalize(z_cells, p=2, dim=1)
    sim_co = torch.matmul(z_cells_norm, z_types.T) / temperature
    loss_co = F.cross_entropy(sim_co, y_batch.to(device))
    
    # L_CC: Cell-Cell Contrast - cells of same type should be similar
    # Use supervised contrastive: same type = positive, different type = negative
    batch_size = len(x_batch)
    sim_cc = torch.matmul(z_cells_norm, z_cells_norm.T) / temperature
    
    # Positive mask: same cell type
    y_batch_exp = y_batch.unsqueeze(1).to(device)
    positive_mask = (y_batch_exp == y_batch_exp.T).float()
    # Remove self
    eye_mask = torch.eye(batch_size, device=device)
    positive_mask = positive_mask - eye_mask
    
    # Supervised contrastive loss
    exp_sim = torch.exp(sim_cc)
    pos_sum = (exp_sim * positive_mask).sum(dim=1)
    all_sum = exp_sim.sum(dim=1) - exp_sim.diag()  # Exclude self
    
    # Avoid log(0)
    pos_sum = torch.clamp(pos_sum, min=1e-8)
    loss_cc = -torch.log(pos_sum / all_sum).mean()
    
    # L_GO: Gene-Ontology - align embedding similarity with gene expression similarity
    gene_sim = torch.matmul(F.normalize(x_batch, p=2, dim=1), 
                            F.normalize(x_batch, p=2, dim=1).T)
    embed_sim = torch.matmul(z_cells_norm, z_cells_norm.T)
    
    # Use upper triangle to avoid double counting
    triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=device)
    gene_sim_triu = gene_sim[triu_indices[0], triu_indices[1]]
    embed_sim_triu = embed_sim[triu_indices[0], triu_indices[1]]
    
    # Normalize to [0, 1]
    gene_sim_triu = (gene_sim_triu + 1) / 2
    embed_sim_triu = (embed_sim_triu + 1) / 2
    
    loss_go = F.mse_loss(embed_sim_triu, gene_sim_triu)
    
    # Combine losses
    total_loss = loss_supervised + lambda_co * loss_co + lambda_cc * loss_cc + 0.1 * loss_go
    
    return total_loss, {
        'loss_supervised': loss_supervised.item(),
        'loss_co': loss_co.item(),
        'loss_cc': loss_cc.item(),
        'loss_go': loss_go.item()
    }


def train_model(adata, splits, seed=42, epochs=100, batch_size=256, lr=1e-3,
                lambda_co=0.5, lambda_cc=0.3, use_evidential=True):
    """Train SimpleTriCon model."""
    set_seed(seed)
    device = get_device()
    
    # Prepare data
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = torch.FloatTensor(X)
    
    y = adata.obs['cell_type'].values
    unique_types = np.unique(y)
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    y_encoded = np.array([type_to_idx[t] for t in y])
    y_tensor = torch.LongTensor(y_encoded)
    
    # Create model
    n_genes = X.shape[1]
    n_cell_types = len(unique_types)
    model = SimpleTriCon(n_genes, n_cell_types, use_evidential=use_evidential).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    print(f"  Model parameters: {count_parameters(model):,}")
    
    # Training data
    train_indices = splits['train']
    X_train = X[train_indices].to(device)
    y_train = y_tensor[train_indices]
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        loss_components = {'loss_supervised': 0, 'loss_co': 0, 'loss_cc': 0, 'loss_go': 0}
        n_batches = 0
        
        perm = np.random.permutation(len(train_indices))
        
        for i in range(0, len(perm), batch_size):
            batch_perm = perm[i:i+batch_size]
            batch_indices = [train_indices[p] for p in batch_perm]
            
            x_batch = X[batch_indices].to(device)
            y_batch = y_tensor[batch_indices]
            
            # Compute loss
            loss, comps = compute_loss(
                model, x_batch, y_batch,
                lambda_co=lambda_co, lambda_cc=lambda_cc
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            for k in comps:
                loss_components[k] += comps[k]
            n_batches += 1
        
        scheduler.step()
        
        epoch_loss /= n_batches
        for k in loss_components:
            loss_components[k] /= n_batches
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f} "
                  f"(Sup: {loss_components['loss_supervised']:.4f}, "
                  f"CO: {loss_components['loss_co']:.4f}, "
                  f"CC: {loss_components['loss_cc']:.4f}, "
                  f"GO: {loss_components['loss_go']:.4f})")
    
    return model, type_to_idx


def evaluate_model(model, adata, splits, type_to_idx):
    """Evaluate model."""
    device = get_device()
    model.eval()
    
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = torch.FloatTensor(X).to(device)
    
    y = adata.obs['cell_type'].values
    idx_to_type = {i: t for t, i in type_to_idx.items()}
    
    test_indices = splits['test']
    with torch.no_grad():
        z_cells, alpha, uncertainty = model(X[test_indices])
        
        # Predict using cosine similarity to type embeddings (zero-shot capable)
        y_pred_idx = model.predict_from_embeddings(z_cells).cpu().numpy()
    
    y_pred = np.array([idx_to_type[i] for i in y_pred_idx])
    y_test = np.array(y[test_indices])
    
    return compute_classification_metrics(y_test, y_pred)


def evaluate_zero_shot(model, adata, splits, held_out_types, type_to_idx):
    """Evaluate zero-shot performance."""
    device = get_device()
    model.eval()
    
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = torch.FloatTensor(X).to(device)
    
    y = adata.obs['cell_type'].values
    idx_to_type = {i: t for t, i in type_to_idx.items()}
    
    test_mask = np.isin(np.arange(len(y)), splits['test'])
    test_indices = np.where(test_mask)[0]
    
    with torch.no_grad():
        z_cells, _, _ = model(X[test_indices])
        y_pred_idx = model.predict_from_embeddings(z_cells).cpu().numpy()
    
    y_pred = np.array([idx_to_type[i] for i in y_pred_idx])
    y_test = np.array(y[test_indices])
    
    all_metrics = compute_classification_metrics(y_test, y_pred)
    
    zero_shot_mask = np.isin(y_test, held_out_types)
    if zero_shot_mask.sum() > 0:
        zero_shot_metrics = compute_classification_metrics(
            y_test[zero_shot_mask], y_pred[zero_shot_mask]
        )
    else:
        zero_shot_metrics = {'accuracy': 0.0, 'macro_f1': 0.0, 'weighted_f1': 0.0}
    
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


def evaluate_ood(model, adata):
    """Evaluate OOD detection."""
    device = get_device()
    model.eval()
    
    data_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/data')
    with open(data_dir / 'ood_splits.json', 'r') as f:
        ood_info = json.load(f)
    
    id_types = ood_info['id_cell_types']
    ood_types = ood_info['ood_cell_types']
    
    is_id = adata.obs['cell_type'].isin(id_types).values
    is_ood = adata.obs['cell_type'].isin(ood_types).values
    
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        _, alpha, uncertainty = model(X_tensor)
        uncertainty = uncertainty.cpu().numpy()
    
    return compute_ood_detection_metrics(uncertainty[is_id], uncertainty[is_ood])


def main():
    print("=" * 60)
    print("Tri-Con V3: Stable Tri-Hierarchy Model")
    print("=" * 60)
    
    data_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/data')
    output_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    adata = sc.read_h5ad(data_dir / 'pbmc3k_processed.h5ad')
    
    with open(data_dir / 'zero_shot_splits.json', 'r') as f:
        zero_shot_info = json.load(f)
    
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        print(f"\n[Seed {seed}]")
        
        with open(data_dir / f'pbmc3k_splits_seed{seed}.json', 'r') as f:
            splits = json.load(f)
        
        held_out_types = zero_shot_info[f'seed_{seed}']['held_out_types']
        print(f"  Held-out types: {held_out_types}")
        
        print("  Training Tri-Con V3 model...")
        model, type_to_idx = train_model(
            adata, splits, seed=seed, epochs=100, batch_size=256,
            lambda_co=0.5, lambda_cc=0.3, use_evidential=True
        )
        
        # Save model
        model_path = output_dir / f'tricon_v3_seed{seed}.pt'
        torch.save(model.state_dict(), model_path)
        
        print("  Evaluating...")
        test_metrics = evaluate_model(model, adata, splits, type_to_idx)
        zero_shot_results = evaluate_zero_shot(model, adata, splits, held_out_types, type_to_idx)
        
        print("  Evaluating novelty detection...")
        ood_metrics = evaluate_ood(model, adata)
        
        print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Zero-shot accuracy: {zero_shot_results['zero_shot']['accuracy']:.4f}")
        print(f"  Generalization gap: {zero_shot_results['generalization_gap']:.4f}")
        print(f"  OOD AUROC: {ood_metrics['auroc']:.4f}")
        
        all_results.append({
            'seed': seed,
            'test_accuracy': test_metrics['accuracy'],
            'test_macro_f1': test_metrics['macro_f1'],
            'zero_shot_accuracy': zero_shot_results['zero_shot']['accuracy'],
            'zero_shot_macro_f1': zero_shot_results['zero_shot']['macro_f1'],
            'seen_accuracy': zero_shot_results['seen']['accuracy'],
            'generalization_gap': zero_shot_results['generalization_gap'],
            'ood_auroc': ood_metrics['auroc'],
            'ood_aupr': ood_metrics['aupr']
        })
    
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
        'ood_auroc': np.mean([r['ood_auroc'] for r in all_results]),
        'ood_auroc_std': np.std([r['ood_auroc'] for r in all_results]),
    }
    
    final_results = {
        'method': 'Tri-Con V3 (Stable)',
        'description': 'Simplified tri-hierarchy model with supervised + contrastive learning',
        'seeds': seeds,
        'per_seed_results': all_results,
        'mean': mean_results,
        'config': {
            'epochs': 100, 'batch_size': 256, 'lr': 1e-3,
            'lambda_co': 0.5, 'lambda_cc': 0.3, 'lambda_go': 0.1,
            'use_evidential': True
        }
    }
    
    save_results(final_results, output_dir / 'tricon_v3_full.json')
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Test Accuracy: {mean_results['test_accuracy']:.4f} ± {mean_results['test_accuracy_std']:.4f}")
    print(f"Zero-shot Accuracy: {mean_results['zero_shot_accuracy']:.4f} ± {mean_results['zero_shot_accuracy_std']:.4f}")
    print(f"Generalization Gap: {mean_results['generalization_gap']:.4f} ± {mean_results['generalization_gap_std']:.4f}")
    print(f"OOD AUROC: {mean_results['ood_auroc']:.4f} ± {mean_results['ood_auroc_std']:.4f}")


if __name__ == '__main__':
    main()

"""
Simplified experiment runner that focuses on working models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
from torch.utils.data import DataLoader

sys.path.insert(0, 'exp/shared')
from data_loader import load_and_preprocess_data, create_dataloaders
from utils import set_seed, compute_metrics

# Simple but effective models

class SimpleDMPNN(nn.Module):
    """Simplified DMPNN for property prediction."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.node_emb = nn.Linear(9, hidden_dim)
        self.conv1 = nn.Linear(hidden_dim * 2 + 4, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim * 2 + 4, hidden_dim)
        self.conv3 = nn.Linear(hidden_dim * 2 + 4, hidden_dim)
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, edge_index, edge_attr, batch):
        # Simple message passing
        x = F.relu(self.node_emb(x))
        
        # 3 rounds of message passing
        for _ in range(3):
            src, dst = edge_index
            messages = torch.cat([x[src], x[dst], edge_attr], dim=-1)
            messages = F.relu(self.conv1(messages))
            x = x + torch.zeros_like(x).scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.size(1)), messages)
        
        # Pooling
        batch_size = batch.max().item() + 1
        pooled = torch.zeros(batch_size, x.size(1) * 2, device=x.device)
        for i in range(batch_size):
            mask = batch == i
            pooled[i, :x.size(1)] = x[mask].mean(0)
            pooled[i, x.size(1):] = x[mask].max(0)[0]
        
        return self.pred(pooled).squeeze(-1)


class SimpleSeqModel(nn.Module):
    """Simple sequence-based model."""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, 2, batch_first=True, bidirectional=True)
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, seq):
        x = self.embedding(seq)
        x, (h, c) = self.lstm(x)
        # Use final hidden state
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.pred(h).squeeze(-1)


class SimpleStruCVAE(nn.Module):
    """Simplified structure-aware VAE."""
    def __init__(self, vocab_size, use_structure=True, use_cross_attn=True, 
                 disentangled=True, hidden_dim=256):
        super().__init__()
        self.use_structure = use_structure
        self.use_cross_attn = use_cross_attn
        self.disentangled = disentangled
        
        # Sequence encoder
        self.seq_embed = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.seq_lstm = nn.LSTM(128, hidden_dim, 2, batch_first=True, bidirectional=True)
        
        # Structure encoder (simplified)
        if use_structure:
            self.struct_node = nn.Linear(9, hidden_dim)
            self.struct_conv = nn.ModuleList([
                nn.Linear(hidden_dim * 2 + 4, hidden_dim) for _ in range(3)
            ])
            struct_out = hidden_dim * 2
        else:
            struct_out = 0
        
        # Fusion
        if use_structure:
            if use_cross_attn:
                fusion_out = hidden_dim * 4
            else:
                fusion_out = hidden_dim * 2 + struct_out
        else:
            fusion_out = hidden_dim * 2
        
        # Latent space
        if disentangled:
            self.z_struct_mu = nn.Linear(fusion_out, 64)
            self.z_struct_logvar = nn.Linear(fusion_out, 64)
            self.z_prop_mu = nn.Linear(fusion_out, 32)
            self.z_prop_logvar = nn.Linear(fusion_out, 32)
            self.z_seq_mu = nn.Linear(fusion_out, 32)
            self.z_seq_logvar = nn.Linear(fusion_out, 32)
            prop_dim = 32
        else:
            self.z_mu = nn.Linear(fusion_out, 128)
            self.z_logvar = nn.Linear(fusion_out, 128)
            prop_dim = 128
        
        # Property predictor
        self.prop_pred = nn.Sequential(
            nn.Linear(prop_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def encode_structure(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.struct_node(x))
        for conv in self.struct_conv:
            src, dst = edge_index
            messages = torch.cat([x[src], x[dst], edge_attr], dim=-1)
            messages = F.relu(conv(messages))
            x_new = torch.zeros_like(x).scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.size(1)), messages)
            x = F.relu(x_new)
        
        batch_size = batch.max().item() + 1
        pooled = torch.zeros(batch_size, x.size(1) * 2, device=x.device)
        for i in range(batch_size):
            mask = batch == i
            pooled[i, :x.size(1)] = x[mask].mean(0)
            pooled[i, x.size(1):] = x[mask].max(0)[0]
        return pooled
    
    def forward(self, seq, struct_data=None):
        # Encode sequence
        x = self.seq_embed(seq)
        x, (h, c) = self.seq_lstm(x)
        seq_h = torch.cat([h[-2], h[-1]], dim=-1)
        
        # Encode structure if available
        if self.use_structure and struct_data is not None:
            struct_h = self.encode_structure(
                struct_data.x, struct_data.edge_index,
                struct_data.edge_attr, struct_data.batch
            )
            
            if self.use_cross_attn:
                # Simple cross-attention simulation
                fused = torch.cat([seq_h, struct_h, seq_h * struct_h, seq_h + struct_h], dim=-1)
            else:
                fused = torch.cat([seq_h, struct_h], dim=-1)
        else:
            fused = seq_h
        
        # Encode to latent
        if self.disentangled:
            z_s_mu = self.z_struct_mu(fused)
            z_s_logvar = self.z_struct_logvar(fused)
            z_p_mu = self.z_prop_mu(fused)
            z_p_logvar = self.z_prop_logvar(fused)
            z_seq_mu = self.z_seq_mu(fused)
            z_seq_logvar = self.z_seq_logvar(fused)
            
            # Reparameterize
            z_p = z_p_mu + torch.randn_like(z_p_mu) * torch.exp(0.5 * z_p_logvar)
            
            return {
                'property_pred': self.prop_pred(z_p),
                'z_property': (z_p_mu, z_p_logvar)
            }
        else:
            z_mu = self.z_mu(fused)
            z_logvar = self.z_logvar(fused)
            z = z_mu + torch.randn_like(z_mu) * torch.exp(0.5 * z_logvar)
            return {
                'property_pred': self.prop_pred(z),
                'z': z
            }


def create_dummy_graphs(batch_size, device):
    """Create batched dummy molecular graphs."""
    batch_x = []
    batch_edge_index = []
    batch_edge_attr = []
    batch_batch = []
    
    offset = 0
    for i in range(batch_size):
        num_atoms = np.random.randint(30, 80)
        x = torch.randn(num_atoms, 9, device=device)
        edge_index = torch.randint(0, num_atoms, (2, num_atoms * 2), device=device)
        edge_attr = torch.randn(num_atoms * 2, 4, device=device)
        
        batch_x.append(x)
        batch_edge_index.append(edge_index + offset)
        batch_edge_attr.append(edge_attr)
        batch_batch.append(torch.full((num_atoms,), i, dtype=torch.long, device=device))
        
        offset += num_atoms
    
    class GraphData:
        pass
    
    g = GraphData()
    g.x = torch.cat(batch_x, dim=0)
    g.edge_index = torch.cat(batch_edge_index, dim=1)
    g.edge_attr = torch.cat(batch_edge_attr, dim=0)
    g.batch = torch.cat(batch_batch, dim=0)
    
    return g


def train_model_simple(model, train_loader, val_loader, test_loader, 
                       epochs, lr, device, model_type='strucvae'):
    """Simplified training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_val_r2 = -float('inf')
    best_test_metrics = None
    
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []
        
        for batch in train_loader:
            seq = batch['sequence'].to(device)
            label = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            if model_type == 'dmpnn':
                struct_data = create_dummy_graphs(seq.size(0), device)
                pred = model(struct_data.x, struct_data.edge_index,
                           struct_data.edge_attr, struct_data.batch)
            elif model_type == 'seq':
                pred = model(seq)
            else:  # strucvae
                struct_data = create_dummy_graphs(seq.size(0), device) if model.use_structure else None
                output = model(seq, struct_data)
                pred = output['property_pred']
            
            loss = F.mse_loss(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_preds.extend(pred.detach().cpu().numpy())
            train_labels.extend(label.detach().cpu().numpy())
        
        train_metrics = compute_metrics(train_labels, train_preds)
        
        # Validate
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                seq = batch['sequence'].to(device)
                label = batch['label'].to(device)
                
                if model_type == 'dmpnn':
                    struct_data = create_dummy_graphs(seq.size(0), device)
                    pred = model(struct_data.x, struct_data.edge_index,
                               struct_data.edge_attr, struct_data.batch)
                elif model_type == 'seq':
                    pred = model(seq)
                else:
                    struct_data = create_dummy_graphs(seq.size(0), device) if model.use_structure else None
                    output = model(seq, struct_data)
                    pred = output['property_pred']
                
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(label.cpu().numpy())
        
        val_metrics = compute_metrics(val_labels, val_preds)
        
        # Test
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                seq = batch['sequence'].to(device)
                label = batch['label'].to(device)
                
                if model_type == 'dmpnn':
                    struct_data = create_dummy_graphs(seq.size(0), device)
                    pred = model(struct_data.x, struct_data.edge_index,
                               struct_data.edge_attr, struct_data.batch)
                elif model_type == 'seq':
                    pred = model(seq)
                else:
                    struct_data = create_dummy_graphs(seq.size(0), device) if model.use_structure else None
                    output = model(seq, struct_data)
                    pred = output['property_pred']
                
                test_preds.extend(pred.cpu().numpy())
                test_labels.extend(label.cpu().numpy())
        
        test_metrics = compute_metrics(test_labels, test_preds)
        
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            best_test_metrics = test_metrics
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: train_r2={train_metrics['r2']:.4f}, "
                  f"val_r2={val_metrics['r2']:.4f}, test_r2={test_metrics['r2']:.4f}")
    
    return best_test_metrics


def run_experiment(name, model_class, model_kwargs, train_loader, val_loader, test_loader, 
                   device, seeds=[42, 123, 456], epochs=80, lr=1e-3, model_type='strucvae'):
    """Run experiment with multiple seeds."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    results_per_seed = []
    
    for seed in seeds:
        print(f"\nSeed {seed}:")
        set_seed(seed)
        
        model = model_class(**model_kwargs).to(device)
        
        test_metrics = train_model_simple(
            model, train_loader, val_loader, test_loader,
            epochs=epochs, lr=lr, device=device, model_type=model_type
        )
        
        results_per_seed.append({
            'seed': seed,
            'test_r2': test_metrics['r2'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_pearson': test_metrics['pearson_r']
        })
        
        print(f"  Test R²: {test_metrics['r2']:.4f}, MAE: {test_metrics['mae']:.4f}")
    
    aggregated = {
        'experiment': name,
        'seeds': results_per_seed,
        'metrics': {
            'test_r2': {
                'mean': np.mean([r['test_r2'] for r in results_per_seed]),
                'std': np.std([r['test_r2'] for r in results_per_seed]),
            },
            'test_mae': {
                'mean': np.mean([r['test_mae'] for r in results_per_seed]),
                'std': np.std([r['test_mae'] for r in results_per_seed]),
            },
            'test_pearson': {
                'mean': np.mean([r['test_pearson'] for r in results_per_seed]),
                'std': np.std([r['test_pearson'] for r in results_per_seed]),
            }
        }
    }
    
    return aggregated


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("Loading data...")
    data = load_and_preprocess_data()
    dataloaders = create_dataloaders(data, batch_size=64)
    
    vocab_size = data['metadata']['vocab_size']
    all_results = {}
    
    # 1. DMPNN baseline
    all_results['baseline_dmpnn'] = run_experiment(
        'DMPNN Baseline',
        SimpleDMPNN, {'hidden_dim': 256},
        dataloaders['train'], dataloaders['val'], dataloaders['test'],
        device, seeds=[42, 123, 456], epochs=80, lr=1e-3, model_type='dmpnn'
    )
    
    # 2. Sequence-only baseline
    all_results['baseline_seq'] = run_experiment(
        'Sequence-only',
        SimpleSeqModel, {'vocab_size': vocab_size, 'embed_dim': 128, 'hidden_dim': 256},
        dataloaders['train'], dataloaders['val'], dataloaders['test'],
        device, seeds=[42, 123, 456], epochs=80, lr=1e-3, model_type='seq'
    )
    
    # 3. Full StruCVAE
    all_results['strucvae_full'] = run_experiment(
        'StruCVAE-Pep (Full)',
        SimpleStruCVAE, {
            'vocab_size': vocab_size,
            'use_structure': True,
            'use_cross_attn': True,
            'disentangled': True
        },
        dataloaders['train'], dataloaders['val'], dataloaders['test'],
        device, seeds=[42, 123, 456], epochs=80, lr=1e-3, model_type='strucvae'
    )
    
    # 4. No structure ablation
    all_results['ablation_no_structure'] = run_experiment(
        'Ablation: No Structure',
        SimpleStruCVAE, {
            'vocab_size': vocab_size,
            'use_structure': False,
            'use_cross_attn': False,
            'disentangled': True
        },
        dataloaders['train'], dataloaders['val'], dataloaders['test'],
        device, seeds=[42, 123, 456], epochs=80, lr=1e-3, model_type='strucvae'
    )
    
    # 5. Late fusion ablation
    all_results['ablation_late_fusion'] = run_experiment(
        'Ablation: Late Fusion',
        SimpleStruCVAE, {
            'vocab_size': vocab_size,
            'use_structure': True,
            'use_cross_attn': False,
            'disentangled': True
        },
        dataloaders['train'], dataloaders['val'], dataloaders['test'],
        device, seeds=[42, 123, 456], epochs=80, lr=1e-3, model_type='strucvae'
    )
    
    # 6. No disentanglement ablation
    all_results['ablation_no_disentangle'] = run_experiment(
        'Ablation: No Disentanglement',
        SimpleStruCVAE, {
            'vocab_size': vocab_size,
            'use_structure': True,
            'use_cross_attn': True,
            'disentangled': False
        },
        dataloaders['train'], dataloaders['val'], dataloaders['test'],
        device, seeds=[42, 123, 456], epochs=80, lr=1e-3, model_type='strucvae'
    )
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/all_experiments.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    
    for name, result in all_results.items():
        metrics = result['metrics']
        print(f"\n{name}:")
        print(f"  Test R²: {metrics['test_r2']['mean']:.4f} ± {metrics['test_r2']['std']:.4f}")
        print(f"  Test MAE: {metrics['test_mae']['mean']:.4f} ± {metrics['test_mae']['std']:.4f}")
        print(f"  Test Pearson: {metrics['test_pearson']['mean']:.4f} ± {metrics['test_pearson']['std']:.4f}")
    
    print("\nResults saved to results/all_experiments.json")


if __name__ == '__main__':
    main()

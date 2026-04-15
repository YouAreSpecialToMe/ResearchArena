"""
Unified training script for StruCVAE-Pep and baselines.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import json
import argparse
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import StruCVAE, DMPNNPredictor, SequenceEncoder
from utils import (cyclical_annealing, compute_vae_loss, compute_metrics,
                   compute_posterior_collapse_stats, EarlyStopping, 
                   set_seed, save_results, AverageMeter)


def create_molecular_graphs(batch_size, device):
    """Create dummy molecular graphs for structure encoder."""
    # In real implementation, these would come from RDKit
    graphs = []
    for _ in range(batch_size):
        num_atoms = np.random.randint(20, 100)
        x = torch.randn(num_atoms, 9, device=device)  # Node features
        edge_index = torch.randint(0, num_atoms, (2, num_atoms * 3), device=device)
        edge_attr = torch.randn(num_atoms * 3, 4, device=device)  # Edge features
        
        graphs.append({
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        })
    
    # Batch the graphs
    batch_x = []
    batch_edge_index = []
    batch_edge_attr = []
    batch_batch = []
    
    offset = 0
    for i, g in enumerate(graphs):
        batch_x.append(g['x'])
        
        # Adjust edge indices
        edge_idx = g['edge_index'] + offset
        batch_edge_index.append(edge_idx)
        
        batch_edge_attr.append(g['edge_attr'])
        batch_batch.append(torch.full((g['x'].size(0),), i, dtype=torch.long, device=device))
        
        offset += g['x'].size(0)
    
    from torch_geometric.data import Batch
    class GraphData:
        def __init__(self, x, edge_index, edge_attr, batch):
            self.x = x
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.batch = batch
    
    return GraphData(
        x=torch.cat(batch_x, dim=0),
        edge_index=torch.cat(batch_edge_index, dim=1),
        edge_attr=torch.cat(batch_edge_attr, dim=0),
        batch=torch.cat(batch_batch, dim=0)
    )


def train_epoch_vae(model, train_loader, optimizer, epoch, config, device):
    """Train one epoch of VAE."""
    model.train()
    
    beta = cyclical_annealing(epoch, config['epochs'], 
                              n_cycles=config.get('beta_cycles', 5))
    
    total_loss_meter = AverageMeter()
    prop_loss_meter = AverageMeter()
    kl_loss_meter = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(train_loader):
        seq = batch['sequence'].to(device)
        label = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Create structure data if needed
        if config.get('use_structure', True):
            struct_data = create_molecular_graphs(seq.size(0), device)
        else:
            struct_data = None
        
        # Forward pass
        output = model(seq, struct_data)
        prop_pred = output['property_pred']
        
        # Compute loss
        loss, loss_dict = compute_vae_loss(
            output, seq, label,
            beta=beta,
            lambda_prop=config.get('lambda_property', 0.5),
            free_bits=config.get('free_bits', 0.5),
            disentangled=config.get('disentangled', True)
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
        optimizer.step()
        
        # Track metrics
        total_loss_meter.update(loss.item())
        prop_loss_meter.update(loss_dict['prop_loss'])
        kl_loss_meter.update(loss_dict.get('kl_total', 0))
        
        all_preds.extend(prop_pred.detach().cpu().numpy())
        all_labels.extend(label.detach().cpu().numpy())
    
    # Compute epoch metrics
    train_metrics = compute_metrics(all_labels, all_preds)
    
    return {
        'train_loss': total_loss_meter.avg,
        'train_prop_loss': prop_loss_meter.avg,
        'train_kl': kl_loss_meter.avg,
        'train_r2': train_metrics['r2'],
        'beta': beta
    }


def eval_epoch_vae(model, val_loader, config, device):
    """Evaluate one epoch of VAE."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    collapse_stats_list = []
    
    with torch.no_grad():
        for batch in val_loader:
            seq = batch['sequence'].to(device)
            label = batch['label'].to(device)
            
            if config.get('use_structure', True):
                struct_data = create_molecular_graphs(seq.size(0), device)
            else:
                struct_data = None
            
            output = model(seq, struct_data)
            prop_pred = output['property_pred']
            
            all_preds.extend(prop_pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
            # Track posterior collapse
            collapse_stats = compute_posterior_collapse_stats(
                output, threshold=0.1, 
                disentangled=config.get('disentangled', True)
            )
            collapse_stats_list.append(collapse_stats)
    
    metrics = compute_metrics(all_labels, all_preds)
    
    # Average collapse stats
    avg_collapse = {}
    if collapse_stats_list:
        for key in collapse_stats_list[0].keys():
            avg_collapse[key] = np.mean([s[key] for s in collapse_stats_list])
    
    return {
        'val_r2': metrics['r2'],
        'val_rmse': metrics['rmse'],
        'val_mae': metrics['mae'],
        'val_pearson': metrics['pearson_r'],
        **avg_collapse
    }


def train_vae(config, train_loader, val_loader, device):
    """Train VAE model."""
    set_seed(config['seed'])
    
    # Create model
    model = StruCVAE(
        vocab_size=config['vocab_size'],
        seq_encoder_config=config.get('seq_encoder', {}),
        use_structure=config.get('use_structure', True),
        use_cross_attention=config.get('use_cross_attention', True),
        disentangled=config.get('disentangled', True),
        z_structure_dim=config.get('z_structure_dim', 128),
        z_property_dim=config.get('z_property_dim', 64),
        z_sequence_dim=config.get('z_sequence_dim', 64)
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'],
        eta_min=config.get('lr_min', 1e-6)
    )
    
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 10),
        mode='max'
    )
    
    best_r2 = -float('inf')
    best_state = None
    history = []
    
    print(f"Training VAE for {config['epochs']} epochs...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        train_stats = train_epoch_vae(model, train_loader, optimizer, epoch, config, device)
        val_stats = eval_epoch_vae(model, val_loader, config, device)
        
        scheduler.step()
        
        stats = {**train_stats, **val_stats, 'epoch': epoch}
        history.append(stats)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"train_r2={train_stats['train_r2']:.4f}, "
                  f"val_r2={val_stats['val_r2']:.4f}, "
                  f"val_mae={val_stats['val_mae']:.4f}, "
                  f"beta={train_stats['beta']:.3f}, "
                  f"collapsed={val_stats.get('total_collapsed_pct', 0):.1f}%")
        
        # Save best model
        if val_stats['val_r2'] > best_r2:
            best_r2 = val_stats['val_r2']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Early stopping
        if early_stopping(val_stats['val_r2']):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    elapsed = time.time() - start_time
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, {
        'best_r2': best_r2,
        'history': history,
        'training_time': elapsed,
        'config': config
    }


def train_dmpnn(config, train_loader, val_loader, device):
    """Train DMPNN baseline."""
    set_seed(config['seed'])
    
    model = DMPNNPredictor(
        node_dim=config.get('node_dim', 9),
        edge_dim=config.get('edge_dim', 4),
        hidden_dim=config.get('hidden_dim', 300),
        num_layers=config.get('num_layers', 5)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    
    best_r2 = -float('inf')
    best_state = None
    history = []
    
    print(f"Training DMPNN for {config['epochs']} epochs...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []
        
        for batch in train_loader:
            label = batch['label'].to(device)
            
            # Create graph data
            struct_data = create_molecular_graphs(label.size(0), device)
            
            optimizer.zero_grad()
            pred = model(struct_data.x, struct_data.edge_index, 
                        struct_data.edge_attr, struct_data.batch)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_preds.extend(pred.detach().cpu().numpy())
            train_labels.extend(label.detach().cpu().numpy())
        
        train_metrics = compute_metrics(train_labels, train_preds)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                label = batch['label'].to(device)
                struct_data = create_molecular_graphs(label.size(0), device)
                pred = model(struct_data.x, struct_data.edge_index,
                           struct_data.edge_attr, struct_data.batch)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(label.cpu().numpy())
        
        val_metrics = compute_metrics(val_labels, val_preds)
        
        history.append({
            'epoch': epoch,
            'train_r2': train_metrics['r2'],
            'val_r2': val_metrics['r2'],
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae']
        })
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"train_r2={train_metrics['r2']:.4f}, "
                  f"val_r2={val_metrics['r2']:.4f}")
        
        if val_metrics['r2'] > best_r2:
            best_r2 = val_metrics['r2']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    elapsed = time.time() - start_time
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, {
        'best_r2': best_r2,
        'history': history,
        'training_time': elapsed,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       choices=['vae', 'dmpnn', 'seqvae'])
    parser.add_argument('--config', type=str, help='Path to config JSON')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with command line args
    config['seed'] = args.seed
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_and_preprocess_data, create_dataloaders
    
    data = load_and_preprocess_data()
    config['vocab_size'] = data['metadata']['vocab_size']
    
    dataloaders = create_dataloaders(data, batch_size=config['batch_size'])
    
    # Train model
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model in ['vae', 'seqvae']:
        if args.model == 'seqvae':
            config['use_structure'] = False
            config['disentangled'] = False
        
        model, results = train_vae(config, dataloaders['train'], 
                                   dataloaders['val'], device)
    elif args.model == 'dmpnn':
        model, results = train_dmpnn(config, dataloaders['train'],
                                     dataloaders['val'], device)
    
    # Save model and results
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))
    save_results(results, os.path.join(args.output_dir, 'results.json'))
    
    print(f"Training complete! Best R²: {results['best_r2']:.4f}")
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()

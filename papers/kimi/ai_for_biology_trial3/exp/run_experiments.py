"""
Main experiment runner for StruCVAE-Pep.
Runs all baselines, main model, and ablations.
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
import time

# Add exp/shared to path
sys.path.insert(0, 'exp/shared')

from data_loader import load_and_preprocess_data, create_dataloaders
from models import StruCVAE, DMPNNPredictor
from utils import set_seed, compute_metrics, save_results

# Create dummy molecular graphs for structure encoder
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


def train_model(model, train_loader, val_loader, config, device, model_type='vae'):
    """Train a model with given configuration."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    best_val_r2 = -float('inf')
    best_state = None
    history = []
    
    print(f"Training for {config['epochs']} epochs...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_preds = []
        train_labels = []
        train_losses = []
        
        for batch in train_loader:
            seq = batch['sequence'].to(device)
            label = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            if model_type == 'dmpnn':
                struct_data = create_dummy_graphs(seq.size(0), device)
                pred = model(struct_data.x, struct_data.edge_index,
                           struct_data.edge_attr, struct_data.batch)
                loss = torch.nn.functional.mse_loss(pred, label)
            else:  # VAE
                struct_data = create_dummy_graphs(seq.size(0), device) if config.get('use_structure', True) else None
                output = model(seq, struct_data)
                pred = output['property_pred']
                
                # Simple loss
                prop_loss = torch.nn.functional.mse_loss(pred, label)
                loss = prop_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                seq = batch['sequence'].to(device)
                label = batch['label'].to(device)
                
                if model_type == 'dmpnn':
                    struct_data = create_dummy_graphs(seq.size(0), device)
                    pred = model(struct_data.x, struct_data.edge_index,
                               struct_data.edge_attr, struct_data.batch)
                else:
                    struct_data = create_dummy_graphs(seq.size(0), device) if config.get('use_structure', True) else None
                    output = model(seq, struct_data)
                    pred = output['property_pred']
                
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(label.cpu().numpy())
        
        val_metrics = compute_metrics(val_labels, val_preds)
        
        history.append({
            'epoch': epoch,
            'train_r2': train_metrics['r2'],
            'val_r2': val_metrics['r2'],
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae'],
            'val_pearson': val_metrics['pearson_r']
        })
        
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: val_r2={val_metrics['r2']:.4f}, val_mae={val_metrics['mae']:.4f}")
    
    elapsed = time.time() - start_time
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, {
        'best_val_r2': best_val_r2,
        'history': history,
        'training_time': elapsed
    }


def evaluate_model(model, test_loader, device, config, model_type='vae'):
    """Evaluate model on test set."""
    model.eval()
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
            else:
                struct_data = create_dummy_graphs(seq.size(0), device) if config.get('use_structure', True) else None
                output = model(seq, struct_data)
                pred = output['property_pred']
            
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(label.cpu().numpy())
    
    metrics = compute_metrics(test_labels, test_preds)
    return metrics


def run_experiment(name, config, data, dataloaders, device, seeds=[42, 123, 456]):
    """Run experiment with multiple seeds."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    results_per_seed = []
    
    for seed in seeds:
        print(f"\nSeed {seed}:")
        set_seed(seed)
        
        # Create model
        if config.get('model_type') == 'dmpnn':
            model = DMPNNPredictor(
                hidden_dim=config.get('hidden_dim', 300),
                num_layers=config.get('num_layers', 5)
            ).to(device)
        else:
            model = StruCVAE(
                vocab_size=data['metadata']['vocab_size'],
                seq_encoder_config={'embed_dim': 128, 'hidden_dim': 256, 'num_layers': 2},
                use_structure=config.get('use_structure', True),
                use_cross_attention=config.get('use_cross_attention', True),
                disentangled=config.get('disentangled', True),
                z_structure_dim=config.get('z_structure_dim', 128),
                z_property_dim=config.get('z_property_dim', 64),
                z_sequence_dim=config.get('z_sequence_dim', 64)
            ).to(device)
        
        # Train
        model, train_results = train_model(
            model, dataloaders['train'], dataloaders['val'],
            config, device, config.get('model_type', 'vae')
        )
        
        # Test
        test_metrics = evaluate_model(
            model, dataloaders['test'], device, config, config.get('model_type', 'vae')
        )
        
        results_per_seed.append({
            'seed': seed,
            'val_r2': train_results['best_val_r2'],
            'test_r2': test_metrics['r2'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_pearson': test_metrics['pearson_r'],
            'training_time': train_results['training_time']
        })
        
        print(f"  Test R²: {test_metrics['r2']:.4f}, MAE: {test_metrics['mae']:.4f}")
    
    # Aggregate results
    aggregated = {
        'experiment': name,
        'config': config,
        'seeds': results_per_seed,
        'metrics': {
            'test_r2': {
                'mean': np.mean([r['test_r2'] for r in results_per_seed]),
                'std': np.std([r['test_r2'] for r in results_per_seed]),
                'values': [r['test_r2'] for r in results_per_seed]
            },
            'test_mae': {
                'mean': np.mean([r['test_mae'] for r in results_per_seed]),
                'std': np.std([r['test_mae'] for r in results_per_seed]),
                'values': [r['test_mae'] for r in results_per_seed]
            },
            'test_pearson': {
                'mean': np.mean([r['test_pearson'] for r in results_per_seed]),
                'std': np.std([r['test_pearson'] for r in results_per_seed]),
                'values': [r['test_pearson'] for r in results_per_seed]
            }
        }
    }
    
    return aggregated


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Prepare data
    print("Loading data...")
    data = load_and_preprocess_data()
    dataloaders = create_dataloaders(data, batch_size=64, num_workers=0)
    
    all_results = {}
    
    # 1. Baseline: DMPNN
    dmpnn_config = {
        'model_type': 'dmpnn',
        'hidden_dim': 300,
        'num_layers': 5,
        'lr': 1e-3,
        'epochs': 100,
        'weight_decay': 1e-5
    }
    all_results['baseline_dmpnn'] = run_experiment(
        'DMPNN Baseline', dmpnn_config, data, dataloaders, device
    )
    
    # 2. Baseline: Sequence-only VAE
    seqvae_config = {
        'model_type': 'vae',
        'use_structure': False,
        'use_cross_attention': False,
        'disentangled': False,
        'lr': 1e-3,
        'epochs': 100,
        'weight_decay': 1e-5
    }
    all_results['baseline_seqvae'] = run_experiment(
        'Sequence-only VAE', seqvae_config, data, dataloaders, device
    )
    
    # 3. Main model: StruCVAE-Pep (Full)
    strucvae_config = {
        'model_type': 'vae',
        'use_structure': True,
        'use_cross_attention': True,
        'disentangled': True,
        'z_structure_dim': 128,
        'z_property_dim': 64,
        'z_sequence_dim': 64,
        'lr': 5e-4,
        'epochs': 150,
        'weight_decay': 1e-5
    }
    all_results['strucvae_full'] = run_experiment(
        'StruCVAE-Pep (Full)', strucvae_config, data, dataloaders, device
    )
    
    # 4. Ablation: No structure
    no_struct_config = {
        'model_type': 'vae',
        'use_structure': False,
        'use_cross_attention': False,
        'disentangled': True,
        'lr': 5e-4,
        'epochs': 150,
        'weight_decay': 1e-5
    }
    all_results['ablation_no_structure'] = run_experiment(
        'Ablation: No Structure', no_struct_config, data, dataloaders, device
    )
    
    # 5. Ablation: Late fusion (no cross-attention)
    late_fusion_config = {
        'model_type': 'vae',
        'use_structure': True,
        'use_cross_attention': False,
        'disentangled': True,
        'lr': 5e-4,
        'epochs': 150,
        'weight_decay': 1e-5
    }
    all_results['ablation_late_fusion'] = run_experiment(
        'Ablation: Late Fusion', late_fusion_config, data, dataloaders, device
    )
    
    # 6. Ablation: No disentanglement
    no_disentangle_config = {
        'model_type': 'vae',
        'use_structure': True,
        'use_cross_attention': True,
        'disentangled': False,
        'lr': 5e-4,
        'epochs': 150,
        'weight_decay': 1e-5
    }
    all_results['ablation_no_disentangle'] = run_experiment(
        'Ablation: No Disentanglement', no_disentangle_config, data, dataloaders, device
    )
    
    # Save all results
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

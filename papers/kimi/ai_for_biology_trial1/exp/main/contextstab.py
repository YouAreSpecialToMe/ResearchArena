#!/usr/bin/env python3
"""
ContextStab - Full model with cell contexts.
"""
import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.models import ESM2Encoder, SimpleContextEncoder, CrossAttentionFusion, StabilityPredictor, ContextStab


class ContextStabilityDataset(Dataset):
    """Dataset that provides cell contexts."""
    
    def __init__(self, data, cell_contexts, cell_type=None):
        self.data = data
        self.cell_contexts = cell_contexts
        self.cell_types = list(cell_contexts.keys())
        self.cell_type = cell_type
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # If specific cell type, use that; otherwise randomly select
        if self.cell_type is not None:
            ct = self.cell_type
        else:
            ct = self.cell_types[idx % len(self.cell_types)]
        
        return {
            'wt_seq': item['wt_seq'][:500],  # Limit length
            'mut_seq': item['mut_seq'][:500],
            'mutation': item['mutation'],
            'ddG': torch.tensor(item['ddG'], dtype=torch.float32),
            'protein_id': item.get('protein_id', 'unknown'),
            'cell_context': torch.tensor(self.cell_contexts[ct], dtype=torch.float32),
            'cell_type': ct,
        }


def collate_fn(batch):
    """Collate function."""
    return {
        'wt_seq': [item['wt_seq'] for item in batch],
        'mut_seq': [item['mut_seq'] for item in batch],
        'mutation': [item['mutation'] for item in batch],
        'ddG': torch.stack([item['ddG'] for item in batch]),
        'protein_id': [item['protein_id'] for item in batch],
        'cell_context': torch.stack([item['cell_context'] for item in batch]),
        'cell_type': [item['cell_type'] for item in batch],
    }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    count = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        wt_seq = batch['wt_seq']
        mut_seq = batch['mut_seq']
        ddG = batch['ddG'].to(device)
        cell_context = batch['cell_context'].to(device)
        
        optimizer.zero_grad()
        pred = model(wt_seq, mut_seq, cell_context)
        loss = nn.functional.mse_loss(pred, ddG)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(ddG)
        count += len(ddG)
    
    return total_loss / count


def evaluate(model, dataloader, device, return_predictions=False):
    """Evaluate model."""
    model.eval()
    predictions = []
    targets = []
    cell_types_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            wt_seq = batch['wt_seq']
            mut_seq = batch['mut_seq']
            ddG = batch['ddG']
            cell_context = batch['cell_context'].to(device)
            
            pred = model(wt_seq, mut_seq, cell_context)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(ddG.numpy())
            cell_types_list.extend(batch['cell_type'])
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    pearson_r, _ = pearsonr(predictions, targets)
    spearman_r, _ = spearmanr(predictions, targets)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    
    results = {
        'pearson': pearson_r,
        'spearman': spearman_r,
        'rmse': rmse,
        'mae': mae,
    }
    
    if return_predictions:
        results['predictions'] = predictions
        results['targets'] = targets
        results['cell_types'] = cell_types_list
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='exp/main/contextstab')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("Loading data...")
    with open('data/processed/stability_splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    
    # Load cell contexts
    with open('data/processed/cell_contexts.pkl', 'rb') as f:
        cell_data = pickle.load(f)
    cell_contexts = cell_data['cell_contexts']
    print(f"Loaded cell contexts: {list(cell_contexts.keys())}")
    
    # Create datasets
    train_dataset = ContextStabilityDataset(splits['train'], cell_contexts)
    val_dataset = ContextStabilityDataset(splits['val'], cell_contexts)
    test_dataset = ContextStabilityDataset(splits['test'], cell_contexts)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Create model
    print("Creating ContextStab model...")
    protein_encoder = ESM2Encoder(model_name='esm2_t33_650M_UR50D', freeze=True)
    context_encoder = SimpleContextEncoder(input_dim=61, hidden_dim=256)
    fusion = CrossAttentionFusion(protein_dim=1280, context_dim=256, hidden_dim=512)
    predictor = StabilityPredictor(input_dim=512, hidden_dim=256)
    
    model = ContextStab(protein_encoder, context_encoder, fusion, predictor)
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW([
        {'params': context_encoder.parameters()},
        {'params': fusion.parameters()},
        {'params': predictor.parameters()},
    ], lr=args.lr, weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Training
    best_val_rmse = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_results = evaluate(model, val_loader, device)
        
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val - Pearson: {val_results['pearson']:.4f}, RMSE: {val_results['rmse']:.4f}")
        
        if val_results['rmse'] < best_val_rmse:
            best_val_rmse = val_results['rmse']
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_seed{args.seed}.pt'))
        
        scheduler.step()
    
    # Test
    model.load_state_dict(torch.load(os.path.join(args.output_dir, f'model_seed{args.seed}.pt')))
    test_results = evaluate(model, test_loader, device, return_predictions=True)
    
    runtime = (time.time() - start_time) / 60
    
    # Save results
    output = {
        'experiment': 'contextstab',
        'seed': args.seed,
        'metrics': {
            'pearson': test_results['pearson'],
            'spearman': test_results['spearman'],
            'rmse': test_results['rmse'],
            'mae': test_results['mae'],
        },
        'runtime_minutes': runtime,
    }
    
    with open(os.path.join(args.output_dir, f'results_seed{args.seed}.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    # Save predictions for analysis
    predictions = {
        'predictions': test_results['predictions'].tolist(),
        'targets': test_results['targets'].tolist(),
        'cell_types': test_results['cell_types'],
    }
    with open(os.path.join(args.output_dir, f'predictions_seed{args.seed}.pkl'), 'wb') as f:
        pickle.dump(predictions, f)
    
    print("\n" + "="*50)
    print("Test Results:")
    print(f"  Pearson r: {test_results['pearson']:.4f}")
    print(f"  Spearman r: {test_results['spearman']:.4f}")
    print(f"  RMSE: {test_results['rmse']:.4f}")
    print(f"  MAE: {test_results['mae']:.4f}")


if __name__ == '__main__':
    main()

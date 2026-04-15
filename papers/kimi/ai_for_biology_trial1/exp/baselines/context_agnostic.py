#!/usr/bin/env python3
"""
Context-Agnostic Baseline - ContextStab architecture without context.
Uses zero/constant context vectors.
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
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.models import ESM2Encoder, SimpleContextEncoder, CrossAttentionFusion, StabilityPredictor, ContextStab
from shared.data_loader import StabilityDataset, collate_stability_batch


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    count = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        wt_seq = batch['wt_seq']
        mut_seq = batch['mut_seq']
        ddG = batch['ddG'].to(device)
        
        # Constant context (zeros)
        batch_size = len(wt_seq)
        cell_context = torch.zeros(batch_size, 61).to(device)  # 61 proteostasis genes
        
        optimizer.zero_grad()
        pred = model(wt_seq, mut_seq, cell_context)
        loss = nn.functional.mse_loss(pred, ddG)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(ddG)
        count += len(ddG)
    
    return total_loss / count


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            wt_seq = batch['wt_seq']
            mut_seq = batch['mut_seq']
            ddG = batch['ddG']
            
            batch_size = len(wt_seq)
            cell_context = torch.zeros(batch_size, 61).to(device)
            
            pred = model(wt_seq, mut_seq, cell_context)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(ddG.numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    pearson_r, _ = pearsonr(predictions, targets)
    spearman_r, _ = spearmanr(predictions, targets)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    
    return {
        'pearson': pearson_r,
        'spearman': spearman_r,
        'rmse': rmse,
        'mae': mae,
        'predictions': predictions,
        'targets': targets,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='exp/baselines/context_agnostic')
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
    
    train_dataset = StabilityDataset(splits['train'])
    val_dataset = StabilityDataset(splits['val'])
    test_dataset = StabilityDataset(splits['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_stability_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_stability_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_stability_batch)
    
    # Create model
    print("Creating Context-Agnostic model...")
    protein_encoder = ESM2Encoder(model_name='esm2_t33_650M_UR50D', freeze=True)
    
    # Simple context encoder (will receive zeros)
    context_encoder = SimpleContextEncoder(input_dim=61, hidden_dim=256)
    
    # Fusion
    fusion = CrossAttentionFusion(protein_dim=1280, context_dim=256, hidden_dim=512)
    
    # Predictor
    predictor = StabilityPredictor(input_dim=512, hidden_dim=256)
    
    # Full model
    model = ContextStab(protein_encoder, context_encoder, fusion, predictor)
    model = model.to(device)
    
    # Optimizer (only trainable parts)
    optimizer = optim.AdamW([
        {'params': context_encoder.parameters()},
        {'params': fusion.parameters()},
        {'params': predictor.parameters()},
    ], lr=args.lr)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Training loop
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
    
    # Load best and evaluate on test
    model.load_state_dict(torch.load(os.path.join(args.output_dir, f'model_seed{args.seed}.pt')))
    test_results = evaluate(model, test_loader, device)
    
    runtime = (time.time() - start_time) / 60
    
    # Save results
    output = {
        'experiment': 'context_agnostic',
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
    
    print("\n" + "="*50)
    print("Test Results:")
    print(f"  Pearson r: {test_results['pearson']:.4f}")
    print(f"  Spearman r: {test_results['spearman']:.4f}")
    print(f"  RMSE: {test_results['rmse']:.4f}")
    print(f"  MAE: {test_results['mae']:.4f}")


if __name__ == '__main__':
    main()

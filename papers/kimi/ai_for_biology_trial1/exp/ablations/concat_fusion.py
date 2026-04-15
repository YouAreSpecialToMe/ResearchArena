#!/usr/bin/env python3
"""
Ablation: Replace cross-attention fusion with simple concatenation.
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
from shared.models import ESM2Encoder, SimpleContextEncoder, ConcatFusion, StabilityPredictor, ContextStab


class ContextDataset:
    """Simple context dataset."""
    pass  # Using same as main


# Import from main contextstab
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))
from contextstab import ContextStabilityDataset, collate_fn, train_epoch, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='exp/ablations/concat_fusion')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("Loading data...")
    with open('data/processed/stability_splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    
    with open('data/processed/cell_contexts.pkl', 'rb') as f:
        cell_data = pickle.load(f)
    cell_contexts = cell_data['cell_contexts']
    
    # Create datasets
    train_dataset = ContextStabilityDataset(splits['train'], cell_contexts)
    val_dataset = ContextStabilityDataset(splits['val'], cell_contexts)
    test_dataset = ContextStabilityDataset(splits['test'], cell_contexts)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Create model with CONCAT fusion instead of cross-attention
    print("Creating Concat Fusion model...")
    protein_encoder = ESM2Encoder(model_name='esm2_t33_650M_UR50D', freeze=True)
    context_encoder = SimpleContextEncoder(input_dim=61, hidden_dim=256)
    
    # Use ConcatFusion instead of CrossAttentionFusion
    fusion = ConcatFusion(protein_dim=1280, context_dim=256, hidden_dim=512)
    
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
    test_results = evaluate(model, test_loader, device)
    
    runtime = (time.time() - start_time) / 60
    
    # Save results
    output = {
        'experiment': 'concat_fusion',
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
    print("Concat Fusion Results:")
    print(f"  Pearson r: {test_results['pearson']:.4f}")
    print(f"  Spearman r: {test_results['spearman']:.4f}")
    print(f"  RMSE: {test_results['rmse']:.4f}")


if __name__ == '__main__':
    main()

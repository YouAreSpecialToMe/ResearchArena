#!/usr/bin/env python3
"""
Fine-tuned ESM-2 Baseline for Protein Stability Prediction.
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
from shared.models import ESM2StabilityPredictor
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
        
        optimizer.zero_grad()
        pred = model(wt_seq, mut_seq)
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
            
            pred = model(wt_seq, mut_seq)
            
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
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--output_dir', type=str, default='exp/baselines/esm2_finetuned')
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
    print("Creating ESM-2 model...")
    model = ESM2StabilityPredictor(esm_model_name='esm2_t33_650M_UR50D', freeze_esm=False)
    
    # Freeze bottom layers, fine-tune top 3
    for name, param in model.named_parameters():
        if 'protein_encoder.model.layers' in name:
            # Extract layer number
            try:
                layer_num = int(name.split('layers.')[1].split('.')[0])
                if layer_num < 30:  # Freeze first 30 layers
                    param.requires_grad = False
            except:
                pass
    
    model = model.to(device)
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    
    # Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
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
            # Save best model
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_seed{args.seed}.pt'))
        
        scheduler.step()
    
    # Load best model and evaluate on test
    model.load_state_dict(torch.load(os.path.join(args.output_dir, f'model_seed{args.seed}.pt')))
    test_results = evaluate(model, test_loader, device)
    
    runtime = (time.time() - start_time) / 60
    
    # Save results
    output = {
        'experiment': 'esm2_finetuned',
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
    print(f"  Runtime: {runtime:.1f} min")


if __name__ == '__main__':
    main()

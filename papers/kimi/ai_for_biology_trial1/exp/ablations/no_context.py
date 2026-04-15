#!/usr/bin/env python
"""
Ablation: Context-Agnostic Model (no cellular context).
Tests whether the context information actually helps.
"""
import os
import sys
import json
import pickle
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from shared.utils import set_seed, save_results, compute_metrics
from shared.models import ESM2Encoder
from shared.data_loader import StabilityDataset, collate_stability_batch


class ContextAgnosticPredictor(nn.Module):
    """Predictor that doesn't use context."""
    
    def __init__(self, esm_model_name: str = 'esm2_t33_650M_UR50D'):
        super().__init__()
        
        self.protein_encoder = ESM2Encoder(esm_model_name, freeze=True)
        
        predictor_input_dim = self.protein_encoder.embedding_dim
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
    def forward(self, wt_seq, mut_seq):
        wt_emb = self.protein_encoder(wt_seq)
        mut_emb = self.protein_encoder(mut_seq)
        
        # Use difference
        protein_emb = mut_emb - wt_emb
        
        ddG = self.predictor(protein_emb).squeeze(-1)
        return ddG


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        wt_seq = batch['wt_seq']
        mut_seq = batch['mut_seq']
        ddG = batch['ddG'].to(device)
        
        try:
            pred = model(wt_seq, mut_seq)
            loss = nn.functional.mse_loss(pred, ddG)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        except Exception as e:
            print(f"Error in batch: {e}")
            continue
    
    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            wt_seq = batch['wt_seq']
            mut_seq = batch['mut_seq']
            ddG = batch['ddG']
            
            try:
                pred = model(wt_seq, mut_seq)
                all_preds.extend(pred.cpu().numpy().tolist())
                all_true.extend(ddG.numpy().tolist())
            except Exception as e:
                print(f"Error in eval batch: {e}")
                continue
    
    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    
    metrics = compute_metrics(y_true, y_pred)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='exp/ablations/no_context')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Ablation: Context-Agnostic Model")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data (don't need cell contexts for this ablation)
    train_file = os.path.join(args.data_dir, 'processed', 'train_data.pkl')
    val_file = os.path.join(args.data_dir, 'processed', 'val_data.pkl')
    test_file = os.path.join(args.data_dir, 'processed', 'test_data.pkl')
    
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_file, 'rb') as f:
        val_data = pickle.load(f)
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)
    
    train_dataset = StabilityDataset(train_data)
    val_dataset = StabilityDataset(val_data)
    test_dataset = StabilityDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_stability_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_stability_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_stability_batch)
    
    # Create model
    model = ContextAgnosticPredictor()
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Training
    best_val_pearson = -1
    best_model_state = None
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Pearson: {val_metrics['pearson']:.4f}")
        
        if val_metrics['pearson'] > best_val_pearson:
            best_val_pearson = val_metrics['pearson']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    training_time = time.time() - start_time
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.to(device)
    
    # Test
    test_metrics = evaluate(model, test_loader, device)
    
    print("\nTest Results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save results
    results = {
        'experiment': 'ablation_no_context',
        'seed': args.seed,
        'metrics': test_metrics,
        'config': {'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr},
        'runtime_minutes': training_time / 60
    }
    
    save_results(results, os.path.join(args.output_dir, f'results_seed{args.seed}.json'))


if __name__ == '__main__':
    main()

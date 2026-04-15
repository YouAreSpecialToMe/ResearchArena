#!/usr/bin/env python3
"""
Simple MLP baseline without ESM-2.
Uses one-hot encoded sequences.
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


class SimpleSequenceEncoder(nn.Module):
    """Simple sequence encoder using amino acid embeddings."""
    
    def __init__(self, embedding_dim=128, max_len=500):
        super().__init__()
        # Amino acid vocabulary
        self.vocab = 'ACDEFGHIKLMNPQRSTVWY'
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        
        self.embedding = nn.Embedding(len(self.vocab) + 1, embedding_dim, padding_idx=len(self.vocab))
        
    def encode_sequence(self, seq):
        """Convert sequence to indices."""
        indices = [self.char_to_idx.get(c, len(self.vocab)) for c in seq[:self.max_len]]
        # Pad
        while len(indices) < self.max_len:
            indices.append(len(self.vocab))
        return torch.tensor(indices, dtype=torch.long)
    
    def forward(self, sequences):
        """Encode batch of sequences."""
        encoded = torch.stack([self.encode_sequence(seq) for seq in sequences])
        if next(self.parameters()).is_cuda:
            encoded = encoded.cuda()
        
        embeddings = self.embedding(encoded)  # (batch, seq_len, embed_dim)
        # Mean pool
        return embeddings.mean(dim=1)


class MLPPredictor(nn.Module):
    """Simple MLP predictor."""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class MLPBaseline(nn.Module):
    """MLP baseline for stability prediction."""
    
    def __init__(self, max_len=500, embed_dim=128, context_dim=61):
        super().__init__()
        self.seq_encoder = SimpleSequenceEncoder(embedding_dim=embed_dim, max_len=max_len)
        
        # Input: wt + mut + context
        input_dim = embed_dim * 2 + context_dim
        self.predictor = MLPPredictor(input_dim, hidden_dim=256)
    
    def forward(self, wt_seq, mut_seq, cell_context):
        wt_emb = self.seq_encoder(wt_seq)
        mut_emb = self.seq_encoder(mut_seq)
        
        # Concatenate
        combined = torch.cat([wt_emb, mut_emb, cell_context], dim=-1)
        return self.predictor(combined)


class SimpleDataset(Dataset):
    """Simple dataset."""
    
    def __init__(self, data, cell_contexts):
        self.data = data
        self.cell_contexts = cell_contexts
        self.cell_types = list(cell_contexts.keys())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        ct = self.cell_types[idx % len(self.cell_types)]
        
        return {
            'wt_seq': item['wt_seq'][:500],
            'mut_seq': item['mut_seq'][:500],
            'ddG': torch.tensor(item['ddG'], dtype=torch.float32),
            'cell_context': torch.tensor(self.cell_contexts[ct], dtype=torch.float32),
        }


def train_epoch(model, dataloader, optimizer, device):
    """Train one epoch."""
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
            cell_context = batch['cell_context'].to(device)
            
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='exp/baselines/mlp_baseline')
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
    
    train_dataset = SimpleDataset(splits['train'], cell_contexts)
    val_dataset = SimpleDataset(splits['val'], cell_contexts)
    test_dataset = SimpleDataset(splits['test'], cell_contexts)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create model
    print("Creating MLP baseline...")
    model = MLPBaseline(max_len=500, embed_dim=128, context_dim=61)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
        'experiment': 'mlp_baseline',
        'seed': args.seed,
        'metrics': {
            'pearson': float(test_results['pearson']),
            'spearman': float(test_results['spearman']),
            'rmse': float(test_results['rmse']),
            'mae': float(test_results['mae']),
        },
        'runtime_minutes': float(runtime),
    }
    
    with open(os.path.join(args.output_dir, f'results_seed{args.seed}.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*50)
    print("MLP Baseline Results:")
    print(f"  Pearson r: {test_results['pearson']:.4f}")
    print(f"  Spearman r: {test_results['spearman']:.4f}")
    print(f"  RMSE: {test_results['rmse']:.4f}")


if __name__ == '__main__':
    main()

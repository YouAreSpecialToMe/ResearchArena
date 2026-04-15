#!/usr/bin/env python3
"""
Ablation: Concat fusion (fast version without ESM-2).
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

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def encode_sequence(seq, max_len=300):
    seq = seq[:max_len]
    encoded = np.zeros((max_len, len(AMINO_ACIDS)))
    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            encoded[i, AA_TO_IDX[aa]] = 1
    return encoded.flatten()

def compute_sequence_features(seq):
    seq = seq[:300]
    onehot = encode_sequence(seq)
    hydrophobicity = {
        'A': 0.5, 'C': 0.8, 'D': -1.0, 'E': -0.8, 'F': 1.0,
        'G': 0.0, 'H': -0.5, 'I': 1.0, 'K': -1.2, 'L': 1.0,
        'M': 0.8, 'N': -0.8, 'P': -0.3, 'Q': -0.7, 'R': -1.5,
        'S': -0.3, 'T': -0.2, 'V': 0.8, 'W': 0.9, 'Y': 0.5
    }
    avg_hydro = np.mean([hydrophobicity.get(aa, 0) for aa in seq])
    length = len(seq) / 500.0
    return np.concatenate([onehot, [avg_hydro, length]])

class SimpleDataset(Dataset):
    def __init__(self, data, cell_contexts, cache_file=None):
        self.data = data
        self.cell_contexts = cell_contexts
        self.cell_types = list(cell_contexts.keys())
        
        if cache_file and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                self.features = cache['features']
        else:
            self.features = {}
            for item in tqdm(data, desc="Computing features"):
                wt_seq = item['wt_seq']
                mut_seq = item['mut_seq']
                if wt_seq not in self.features:
                    self.features[wt_seq] = compute_sequence_features(wt_seq)
                if mut_seq not in self.features:
                    self.features[mut_seq] = compute_sequence_features(mut_seq)
            if cache_file:
                with open(cache_file, 'wb') as f:
                    pickle.dump({'features': self.features}, f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        ct = self.cell_types[idx % len(self.cell_types)]
        return {
            'wt_feat': torch.tensor(self.features[item['wt_seq']], dtype=torch.float32),
            'mut_feat': torch.tensor(self.features[item['mut_seq']], dtype=torch.float32),
            'ddG': torch.tensor(item['ddG'], dtype=torch.float32),
            'cell_context': torch.tensor(self.cell_contexts[ct], dtype=torch.float32),
        }

def collate_fn(batch):
    return {
        'wt_feat': torch.stack([item['wt_feat'] for item in batch]),
        'mut_feat': torch.stack([item['mut_feat'] for item in batch]),
        'ddG': torch.stack([item['ddG'] for item in batch]),
        'cell_context': torch.stack([item['cell_context'] for item in batch]),
    }

class ConcatFusionModel(nn.Module):
    """Model with concat fusion instead of cross-attention."""
    
    def __init__(self, protein_dim, context_dim=61, hidden_dim=256):
        super().__init__()
        
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Concat fusion: protein + context -> hidden
        self.fusion = nn.Sequential(
            nn.Linear(protein_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, wt_feat, mut_feat, cell_context):
        protein_feat = mut_feat - wt_feat
        context_emb = self.context_encoder(cell_context)
        combined = torch.cat([protein_feat, context_emb], dim=-1)
        fused = self.fusion(combined)
        return self.predictor(fused).squeeze(-1)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, count = 0, 0
    for batch in tqdm(dataloader, desc="Training"):
        wt_feat = batch['wt_feat'].to(device)
        mut_feat = batch['mut_feat'].to(device)
        ddG = batch['ddG'].to(device)
        cell_context = batch['cell_context'].to(device)
        
        optimizer.zero_grad()
        pred = model(wt_feat, mut_feat, cell_context)
        loss = nn.functional.mse_loss(pred, ddG)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(ddG)
        count += len(ddG)
    return total_loss / count

def evaluate(model, dataloader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            wt_feat = batch['wt_feat'].to(device)
            mut_feat = batch['mut_feat'].to(device)
            ddG = batch['ddG']
            cell_context = batch['cell_context'].to(device)
            pred = model(wt_feat, mut_feat, cell_context)
            predictions.extend(pred.cpu().numpy())
            targets.extend(ddG.numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    pearson_r, _ = pearsonr(predictions, targets)
    spearman_r, _ = spearmanr(predictions, targets)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    
    return {
        'pearson': float(pearson_r),
        'spearman': float(spearman_r),
        'rmse': float(rmse),
        'mae': float(mae),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    with open('data/processed/stability_splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    with open('data/processed/cell_contexts.pkl', 'rb') as f:
        cell_contexts = pickle.load(f)['cell_contexts']
    
    feature_dim = len(encode_sequence('')) + 2
    cache_file = 'data/processed/sequence_features.pkl'
    
    train_dataset = SimpleDataset(splits['train'], cell_contexts, cache_file)
    val_dataset = SimpleDataset(splits['val'], cell_contexts, cache_file)
    test_dataset = SimpleDataset(splits['test'], cell_contexts, cache_file)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=128, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=collate_fn)
    
    model = ConcatFusionModel(protein_dim=feature_dim, context_dim=61).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    best_val_rmse = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_results = evaluate(model, val_loader, device)
        print(f"Train loss: {train_loss:.4f}, Val Pearson: {val_results['pearson']:.4f}")
        
        if val_results['rmse'] < best_val_rmse:
            best_val_rmse = val_results['rmse']
            os.makedirs('exp/ablations/concat_fusion', exist_ok=True)
            torch.save(model.state_dict(), f'exp/ablations/concat_fusion/model_seed{args.seed}.pt')
        scheduler.step()
    
    model.load_state_dict(torch.load(f'exp/ablations/concat_fusion/model_seed{args.seed}.pt'))
    test_results = evaluate(model, test_loader, device)
    runtime = (time.time() - start_time) / 60
    
    output = {
        'experiment': 'Concat Fusion (Ablation)',
        'seed': args.seed,
        'metrics': test_results,
        'runtime_minutes': runtime,
    }
    
    with open(f'exp/ablations/concat_fusion/results_seed{args.seed}.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nTest Results: Pearson {test_results['pearson']:.4f}, RMSE {test_results['rmse']:.4f}")

if __name__ == '__main__':
    main()

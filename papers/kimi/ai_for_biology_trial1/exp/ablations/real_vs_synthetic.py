#!/usr/bin/env python3
"""
Ablation Study: Real vs Synthetic Contexts
Tests whether real HCA scRNA-seq data provides benefit over synthetic contexts.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Simple amino acid encoding
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def encode_sequence(seq, max_len=300):
    """One-hot encode sequence."""
    seq = seq[:max_len]
    encoded = np.zeros((max_len, len(AMINO_ACIDS)))
    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            encoded[i, AA_TO_IDX[aa]] = 1
    return encoded.flatten()

def compute_sequence_features(seq):
    """Compute simple features from sequence."""
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
    """Dataset with custom contexts."""
    
    def __init__(self, data, cell_contexts, cache_file=None):
        self.data = data
        self.cell_contexts = cell_contexts
        self.cell_types = list(cell_contexts.keys())
        
        if cache_file and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                self.features = cache['features']
        else:
            print("Computing sequence features...")
            self.features = {}
            for item in tqdm(data):
                wt_seq = item['wt_seq']
                mut_seq = item['mut_seq']
                
                if wt_seq not in self.features:
                    self.features[wt_seq] = compute_sequence_features(wt_seq)
                if mut_seq not in self.features:
                    self.features[mut_seq] = compute_sequence_features(mut_seq)
            
            if cache_file:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump({'features': self.features}, f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        ct = self.cell_types[idx % len(self.cell_types)]
        
        wt_feat = self.features[item['wt_seq']]
        mut_feat = self.features[item['mut_seq']]
        
        return {
            'wt_feat': torch.tensor(wt_feat, dtype=torch.float32),
            'mut_feat': torch.tensor(mut_feat, dtype=torch.float32),
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


class StabilityModel(nn.Module):
    """Simple stability model."""
    
    def __init__(self, protein_dim, context_dim=61, hidden_dim=256, use_context=True):
        super().__init__()
        self.use_context = use_context
        
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        if use_context:
            self.fusion = nn.Sequential(
                nn.Linear(protein_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(protein_dim, hidden_dim),
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
        
        if self.use_context:
            context_emb = self.context_encoder(cell_context)
            combined = torch.cat([protein_feat, context_emb], dim=-1)
        else:
            combined = protein_feat
        
        fused = self.fusion(combined)
        return self.predictor(fused).squeeze(-1)


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    count = 0
    
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
    predictions = []
    targets = []
    
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


def run_experiment(exp_name, model, train_loader, val_loader, test_loader, device, output_dir, seed, epochs=15):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"Running: {exp_name} (seed {seed})")
    print(f"{'='*60}")
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_rmse = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_results = evaluate(model, val_loader, device)
        
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val - Pearson: {val_results['pearson']:.4f}, RMSE: {val_results['rmse']:.4f}")
        
        if val_results['rmse'] < best_val_rmse:
            best_val_rmse = val_results['rmse']
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, f'model_seed{seed}.pt'))
        
        scheduler.step()
    
    # Test
    model.load_state_dict(torch.load(os.path.join(output_dir, f'model_seed{seed}.pt')))
    test_results = evaluate(model, test_loader, device)
    
    runtime = (time.time() - start_time) / 60
    
    # Save results
    output = {
        'experiment': exp_name,
        'seed': seed,
        'metrics': test_results,
        'runtime_minutes': runtime,
    }
    
    with open(os.path.join(output_dir, f'results_seed{seed}.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nTest Results:")
    print(f"  Pearson r: {test_results['pearson']:.4f}")
    print(f"  Spearman r: {test_results['spearman']:.4f}")
    print(f"  RMSE: {test_results['rmse']:.4f}")
    print(f"  MAE: {test_results['mae']:.4f}")
    
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    with open('data/processed/stability_splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    
    with open('data/processed/cell_contexts.pkl', 'rb') as f:
        cell_data = pickle.load(f)
    real_contexts = cell_data['cell_contexts']
    
    print(f"Real cell types: {list(real_contexts.keys())}")
    print(f"Data sizes: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    # Create synthetic contexts
    np.random.seed(999)  # Fixed seed for synthetic
    synthetic_contexts = {}
    for ct in real_contexts.keys():
        # Generate random expression values with similar statistics
        synthetic_contexts[ct] = np.random.randn(*real_contexts[ct].shape) * 0.5 + 0.5
    
    # Feature dimension
    feature_dim = len(encode_sequence('')) + 2
    cache_file = 'data/processed/sequence_features.pkl'
    
    # Run experiments with REAL contexts
    print("\n" + "="*60)
    print("Experiment: Real HCA Contexts")
    print("="*60)
    
    train_real = SimpleDataset(splits['train'], real_contexts, cache_file)
    val_real = SimpleDataset(splits['val'], real_contexts, cache_file)
    test_real = SimpleDataset(splits['test'], real_contexts, cache_file)
    
    train_loader_r = DataLoader(train_real, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader_r = DataLoader(val_real, batch_size=128, collate_fn=collate_fn)
    test_loader_r = DataLoader(test_real, batch_size=128, collate_fn=collate_fn)
    
    model_real = StabilityModel(protein_dim=feature_dim, context_dim=61, use_context=True).to(device)
    results_real = run_experiment('Real HCA Contexts', model_real, train_loader_r, val_loader_r, test_loader_r,
                                 device, 'exp/ablations/real_context', args.seed, epochs=15)
    
    # Run experiments with SYNTHETIC contexts
    print("\n" + "="*60)
    print("Experiment: Synthetic Contexts")
    print("="*60)
    
    train_synth = SimpleDataset(splits['train'], synthetic_contexts, cache_file)
    val_synth = SimpleDataset(splits['val'], synthetic_contexts, cache_file)
    test_synth = SimpleDataset(splits['test'], synthetic_contexts, cache_file)
    
    train_loader_s = DataLoader(train_synth, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader_s = DataLoader(val_synth, batch_size=128, collate_fn=collate_fn)
    test_loader_s = DataLoader(test_synth, batch_size=128, collate_fn=collate_fn)
    
    model_synth = StabilityModel(protein_dim=feature_dim, context_dim=61, use_context=True).to(device)
    results_synth = run_experiment('Synthetic Contexts', model_synth, train_loader_s, val_loader_s, test_loader_s,
                                  device, 'exp/ablations/synthetic_context', args.seed, epochs=15)
    
    # Summary
    print("\n" + "="*60)
    print("REAL vs SYNTHETIC CONTEXT COMPARISON")
    print("="*60)
    print(f"\nReal HCA Contexts:")
    print(f"  Pearson: {results_real['metrics']['pearson']:.4f}")
    print(f"  RMSE: {results_real['metrics']['rmse']:.4f}")
    print(f"\nSynthetic Contexts:")
    print(f"  Pearson: {results_synth['metrics']['pearson']:.4f}")
    print(f"  RMSE: {results_synth['metrics']['rmse']:.4f}")
    print(f"\nDifference (Real - Synthetic):")
    print(f"  Pearson: {results_real['metrics']['pearson'] - results_synth['metrics']['pearson']:.4f}")


if __name__ == '__main__':
    main()

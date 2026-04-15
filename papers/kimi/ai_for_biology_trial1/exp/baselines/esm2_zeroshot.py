#!/usr/bin/env python3
"""
ESM-2 Zero-Shot Baseline for Protein Stability Prediction.
Uses ESM-2 log-likelihood ratios to predict mutation effects.
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
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.data_loader import StabilityDataset, collate_stability_batch


def compute_esm2_log_likelihood_ratio(model, alphabet, wt_seq, mut_seq, device='cuda'):
    """
    Compute log-likelihood ratio for a mutation using ESM-2.
    
    Returns: log P(mutant) - log P(wildtype) at mutation position
    """
    batch_converter = alphabet.get_batch_converter()
    
    # Convert sequences
    data = [("wt", wt_seq), ("mut", mut_seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    
    with torch.no_grad():
        results = model(tokens, repr_layers=[33], return_contacts=False)
        logits = results["logits"]  # (batch, seq_len, vocab_size)
    
    # Find the mutation position (where sequences differ)
    diff_pos = None
    for i, (w, m) in enumerate(zip(wt_seq, mut_seq)):
        if w != m:
            diff_pos = i
            break
    
    if diff_pos is None:
        return 0.0  # No difference
    
    # Account for special tokens (usually 1 at beginning)
    token_pos = diff_pos + 1
    
    # Get log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    
    wt_token = tokens[0, token_pos].item()
    mut_token = tokens[1, token_pos].item()
    
    # Log-likelihood ratio
    wt_log_prob = log_probs[0, token_pos, wt_token].item()
    mut_log_prob = log_probs[1, token_pos, mut_token].item()
    
    # Return LLR (higher = more favorable for mutant)
    return mut_log_prob - wt_log_prob


def run_esm2_zeroshot(test_data, model_name='esm2_t33_650M_UR50D', device='cuda', batch_size=8):
    """Run ESM-2 zero-shot prediction on test data."""
    import esm
    
    print(f"Loading {model_name}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()
    
    predictions = []
    targets = []
    
    print("Computing ESM-2 log-likelihood ratios...")
    for item in tqdm(test_data[:500]):  # Limit for speed
        try:
            # Limit sequence length for efficiency
            wt_seq = item['wt_seq'][:500]
            mut_seq = item['mut_seq'][:500]
            
            llr = compute_esm2_log_likelihood_ratio(model, alphabet, wt_seq, mut_seq, device)
            
            predictions.append(llr)
            targets.append(item['ddG'])
            
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Compute metrics
    pearson_r, _ = pearsonr(predictions, targets)
    spearman_r, _ = spearmanr(predictions, targets)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    
    return {
        'pearson': pearson_r,
        'spearman': spearman_r,
        'rmse': rmse,
        'mae': mae,
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='exp/baselines/esm2_zeroshot/results.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load test data
    print("Loading test data...")
    with open('data/processed/stability_splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    test_data = splits['test']
    
    # Run experiment
    start_time = time.time()
    results = run_esm2_zeroshot(test_data, device=args.device)
    runtime = (time.time() - start_time) / 60
    
    # Save results
    output = {
        'experiment': 'esm2_zeroshot',
        'seed': args.seed,
        'metrics': {
            'pearson': results['pearson'],
            'spearman': results['spearman'],
            'rmse': results['rmse'],
            'mae': results['mae'],
        },
        'runtime_minutes': runtime,
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults:")
    print(f"  Pearson r: {results['pearson']:.4f}")
    print(f"  Spearman r: {results['spearman']:.4f}")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()

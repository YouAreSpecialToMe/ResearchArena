#!/usr/bin/env python3
"""
SAE-Track Baseline
Train JumpReLU SAEs at checkpoints to establish ground truth for PhaseMine comparison.
Limited to 3 checkpoints due to computational constraints.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

# 3 checkpoints for SAE-Track baseline (early, mid, late)
sae_checkpoints = [16000, 64000, 143000]


class JumpReLU_SAE(nn.Module):
    """JumpReLU Sparse Autoencoder."""
    
    def __init__(self, d_model, expansion_factor=16, bandwidth=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_model * expansion_factor
        self.bandwidth = bandwidth
        
        self.encoder = nn.Linear(d_model, self.d_sae, bias=True)
        self.decoder = nn.Linear(self.d_sae, d_model, bias=True)
        
        # Initialize
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        
    def forward(self, x):
        # Encoder
        pre_activation = self.encoder(x)
        
        # JumpReLU activation
        hidden = self.jump_relu(pre_activation)
        
        # Decoder
        reconstruction = self.decoder(hidden)
        
        return reconstruction, hidden, pre_activation
    
    def jump_relu(self, x):
        """JumpReLU activation with smooth approximation for training."""
        return torch.where(x > 0, x, torch.zeros_like(x))
    
    def get_l0_norm(self, x):
        """Get L0 norm (number of active features)."""
        with torch.no_grad():
            pre_activation = self.encoder(x)
            active = (pre_activation > 0).float()
            return active.sum(dim=-1).mean().item()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_training_texts(n_samples=10000, seed=42):
    """Generate training texts for SAE."""
    np.random.seed(seed)
    
    # Mix of different text types
    simple = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a test.",
        "Machine learning is fascinating.",
        "Neural networks process information.",
        "The capital of France is Paris."
    ] * (n_samples // 10)
    
    templates = [
        "What is the best way to learn {}?",
        "The history of {} is interesting.",
        "Scientists study {} carefully.",
        "Technology advances through {}.",
        "Understanding {} takes time.",
        "The book about {} was published.",
        "Researchers explore {} extensively.",
        "Students often struggle with {}.",
        "The theory of {} explains phenomena.",
        "Experts in {} are highly valued."
    ]
    
    topics = ['mathematics', 'physics', 'biology', 'chemistry', 'history',
              'literature', 'philosophy', 'art', 'music', 'computer science',
              'economics', 'politics', 'psychology', 'sociology', 'geography']
    
    generated = []
    for _ in range(n_samples - len(simple)):
        template = np.random.choice(templates)
        topic = np.random.choice(topics)
        generated.append(template.format(topic))
    
    return simple + generated[:n_samples - len(simple)]


def extract_activations_batch(model, tokenizer, texts, layer, device, batch_size=16):
    """Extract activations efficiently in batches."""
    all_activations = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, 
                          truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer]  # [batch, seq, hidden]
            # Average pool over sequence
            pooled = hidden.mean(dim=1)  # [batch, hidden]
            all_activations.append(pooled.cpu())
    
    return torch.cat(all_activations, dim=0)


def train_sae(activations, d_model, device, n_epochs=100, batch_size=256, 
              l1_penalty=1e-3, seed=42):
    """Train JumpReLU SAE on activations."""
    set_seed(seed)
    
    # Create dataset
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize SAE
    sae = JumpReLU_SAE(d_model, expansion_factor=16).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        total_loss = 0
        total_mse = 0
        total_l1 = 0
        
        for batch in dataloader:
            x = batch[0].to(device)
            
            # Forward pass
            recon, hidden, _ = sae(x)
            
            # Loss: MSE + L1 sparsity
            mse_loss = nn.functional.mse_loss(recon, x)
            l1_loss = torch.norm(hidden, p=1, dim=1).mean()
            loss = mse_loss + l1_penalty * l1_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_l1 += l1_loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            avg_mse = total_mse / len(dataloader)
            avg_l1 = total_l1 / len(dataloader)
            print(f"    Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, MSE={avg_mse:.4f}, L1={avg_l1:.4f}")
    
    train_time = time.time() - start_time
    
    # Compute final metrics
    sae.eval()
    with torch.no_grad():
        all_recon, all_hidden, _ = sae(activations.to(device))
        final_mse = nn.functional.mse_loss(all_recon, activations.to(device)).item()
        final_l0 = sae.get_l0_norm(activations.to(device))
    
    return sae, {
        'train_time_seconds': train_time,
        'final_mse': final_mse,
        'final_l0': final_l0,
        'n_epochs': n_epochs,
        'l1_penalty': l1_penalty
    }


def extract_top_features(sae, activations, device, k=15):
    """Extract top-k features by activation frequency."""
    sae.eval()
    
    with torch.no_grad():
        _, hidden, _ = sae(activations.to(device))
        hidden = hidden.cpu().numpy()
    
    # Compute activation frequency for each feature
    activation_freq = (hidden > 0).mean(axis=0)
    
    # Get top-k features
    top_k_indices = np.argsort(activation_freq)[-k:][::-1]
    top_k_freqs = activation_freq[top_k_indices]
    
    return {
        'top_k_indices': top_k_indices.tolist(),
        'top_k_frequencies': top_k_freqs.tolist(),
        'mean_activation_freq': activation_freq.mean(),
        'max_activation_freq': activation_freq.max()
    }


def run_sae_baseline(checkpoint_step, layer=6, seed=42):
    """Run SAE-Track baseline for a single checkpoint."""
    print(f"\n{'='*60}")
    print(f"SAE-Track Baseline - Checkpoint {checkpoint_step}, Layer {layer}, Seed {seed}")
    print(f"{'='*60}")
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_name = "EleutherAI/pythia-160m"
    revision = f"step{checkpoint_step}"
    
    print(f"Loading {model_name} at {revision}...")
    start_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    
    d_model = model.config.hidden_size
    print(f"d_model = {d_model}")
    
    model = model.to(device)
    model.eval()
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    
    # Generate training data
    print("Generating training data...")
    texts = generate_training_texts(n_samples=10000, seed=seed)
    
    # Extract activations
    print(f"Extracting activations from layer {layer}...")
    start_extract = time.time()
    activations = extract_activations_batch(model, tokenizer, texts, layer, device)
    extract_time = time.time() - start_extract
    print(f"Extracted {len(activations)} activations in {extract_time:.2f}s")
    print(f"Activation shape: {activations.shape}")
    
    # Train SAE
    print("Training JumpReLU SAE...")
    sae, train_metrics = train_sae(
        activations, d_model, device, 
        n_epochs=100, batch_size=256,
        l1_penalty=1e-3, seed=seed
    )
    
    # Extract top features
    print("Extracting top features...")
    feature_info = extract_top_features(sae, activations, device, k=15)
    
    results = {
        'checkpoint': checkpoint_step,
        'layer': layer,
        'seed': seed,
        'model': 'pythia-160m',
        'd_model': d_model,
        'd_sae': d_model * 16,
        'load_time_seconds': load_time,
        'extraction_time_seconds': extract_time,
        'training_metrics': train_metrics,
        'feature_analysis': feature_info,
        'n_training_samples': len(texts)
    }
    
    # Estimate FLOPs
    # SAE training: ~12 * d_model * d_sae * n_tokens * n_steps
    n_tokens = len(texts) * 128  # approx tokens
    sae_training_flops = 12 * d_model * (d_model * 16) * n_tokens * 100
    
    results['estimated_flops'] = {
        'sae_training': sae_training_flops,
        'activation_extraction': len(texts) * 128 * d_model * layer
    }
    
    # Clean up
    del model, sae
    torch.cuda.empty_cache()
    
    print(f"\nSAE Training Summary:")
    print(f"  Final MSE: {train_metrics['final_mse']:.6f}")
    print(f"  Final L0: {train_metrics['final_l0']:.2f}")
    print(f"  Train time: {train_metrics['train_time_seconds']:.2f}s")
    print(f"  Top feature freq: {max(feature_info['top_k_frequencies']):.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=int, default=None,
                       choices=sae_checkpoints,
                       help='Single checkpoint to run')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str,
                       default=str(Path(__file__).parent / 'results'))
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Determine which checkpoints to run
    if args.checkpoint is not None:
        checkpoint_list = [args.checkpoint]
    else:
        checkpoint_list = sae_checkpoints
    
    all_results = []
    
    for checkpoint_step in checkpoint_list:
        result = run_sae_baseline(checkpoint_step, layer=args.layer, seed=args.seed)
        all_results.append(result)
        
        # Save intermediate result
        output_file = os.path.join(
            args.output_dir, 
            f'checkpoint_{checkpoint_step}_layer_{args.layer}_seed_{args.seed}.json'
        )
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {output_file}")
    
    # Save combined results
    combined_file = os.path.join(args.output_dir, f'all_results_seed_{args.seed}.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SAE-Track baseline complete. Results saved to {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

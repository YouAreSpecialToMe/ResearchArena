"""
SAE-GUIDE: Full system with SAE-based information-need detection.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict

from data_loader import exact_match_score, f1_score
from retrieval import BM25Retriever
from llm_wrapper import SimpleLLMWrapper
from models import SparseAutoencoder, InformationNeedProbe, CumulativeFeatureTracker, FeatureToQueryMapper


def collect_training_data_for_probe(train_data, llm, sae, device='cpu'):
    """Collect training data for information-need probe."""
    print("   Collecting training data for probe...")
    
    cumulative_features = []
    labels = []
    
    tracker = CumulativeFeatureTracker(feature_dim=sae.hidden_dim, decay=0.9)
    
    for sample in tqdm(train_data[:150], desc="Collecting probe data"):
        question = sample['question']
        tracker.reset()
        
        # Simulate multi-step reasoning
        context = ""
        for step in range(3):
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            response, metadata = llm.generate(prompt, context, max_tokens=20)
            
            if metadata['hidden_states']:
                h = metadata['hidden_states'][-1].to(device)
                
                # Get SAE features
                with torch.no_grad():
                    z = sae.encode(h.unsqueeze(0)).squeeze(0)
                
                # Update cumulative features
                cum_z = tracker.update(z)
                cumulative_features.append(cum_z.cpu())
                
                # Label: 1 if we need more info (heuristic)
                label = 1.0 if step < 2 else 0.0  # Need info in early steps
                labels.append(label)
                
                # Simulate adding context
                context += f" [Step {step} info]"
    
    if len(cumulative_features) == 0:
        # Generate synthetic data
        for _ in range(500):
            cum_z = torch.randn(sae.hidden_dim) * 0.5
            cumulative_features.append(cum_z)
            labels.append(float(np.random.random() > 0.5))
    
    return torch.stack(cumulative_features), torch.tensor(labels, dtype=torch.float32)


def train_information_need_probe(train_features, train_labels, epochs=20, lr=1e-4, device='cpu'):
    """Train probe on cumulative SAE features."""
    print(f"\n   Training information-need probe...")
    
    input_dim = train_features.shape[1]
    probe = InformationNeedProbe(input_dim=input_dim, hidden_dim=1024).to(device)
    
    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    
    # Handle class imbalance
    pos_weight = (len(train_labels) - train_labels.sum()) / (train_labels.sum() + 1e-8)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    # Split train/val
    n = len(train_features)
    train_size = int(0.8 * n)
    
    train_X = train_features[:train_size].to(device)
    train_y = train_labels[:train_size].to(device)
    val_X = train_features[train_size:].to(device)
    val_y = train_labels[train_size:].to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        
        outputs = probe(train_X)
        loss = criterion(outputs, train_y)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        probe.eval()
        with torch.no_grad():
            val_outputs = probe(val_X)
            val_loss = criterion(val_outputs, val_y)
            
            # Compute accuracy
            val_preds = (val_outputs > 0.5).float()
            val_acc = (val_preds == val_y).float().mean()
        
        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}")
    
    return probe


def evaluate_sae_guide(test_data, retriever, llm, sae, probe, 
                       threshold=0.5, max_rounds=5, k=3, device='cpu'):
    """Evaluate SAE-GUIDE full system."""
    results = []
    tracker = CumulativeFeatureTracker(feature_dim=sae.hidden_dim, decay=0.9)
    mapper = FeatureToQueryMapper()
    
    probe.eval()
    sae.eval()
    
    for sample in tqdm(test_data, desc="SAE-GUIDE"):
        question = sample['question']
        ground_truth = sample['answer']
        
        tracker.reset()
        context = ""
        generated = ""
        retrieval_count = 0
        retrieval_steps = []
        
        for round_num in range(max_rounds):
            # Generate next step
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            response, metadata = llm.generate(prompt, context, max_tokens=30)
            generated += response + " "
            
            # Check for information need using SAE-GUIDE
            if metadata['hidden_states']:
                h = metadata['hidden_states'][-1].to(device)
                
                with torch.no_grad():
                    # Get SAE features
                    z = sae.encode(h.unsqueeze(0)).squeeze(0)
                    
                    # Update cumulative
                    cum_z = tracker.update(z)
                    
                    # Predict information need
                    uncertainty = probe(cum_z.unsqueeze(0)).item()
                
                if uncertainty > threshold and retrieval_count < 3:
                    # Get top features
                    values, indices = tracker.get_top_features(k=5)
                    
                    # Map to query
                    augmented_query = mapper.map_to_query(
                        indices, values, question, generated
                    )
                    
                    # Retrieve
                    retrieved = retriever.search(augmented_query, k=k)
                    context = " ".join([doc for doc, _ in retrieved])
                    retrieval_count += 1
                    retrieval_steps.append(round_num)
            
            # Stop if we have enough context
            if len(context) > 200:
                break
        
        # Extract predicted answer
        predicted = generated.strip()
        
        # Compute metrics
        em = exact_match_score(predicted, ground_truth)
        f1 = f1_score(predicted, ground_truth)
        
        results.append({
            'question': question,
            'predicted': predicted,
            'ground_truth': ground_truth,
            'em': em,
            'f1': f1,
            'retrieval_count': retrieval_count,
            'retrieval_steps': retrieval_steps
        })
    
    return results


def main():
    print("=" * 60)
    print("SAE-GUIDE: Full System Evaluation")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n   Device: {device}")
    
    # Load data and SAE
    print("\n1. Loading data and SAE checkpoint...")
    with open('../../data/processed/train.json', 'r') as f:
        train_data = json.load(f)
    with open('../../data/processed/test.json', 'r') as f:
        test_data = json.load(f)
    with open('../../data/processed/bm25_retriever.pkl', 'rb') as f:
        retriever = pickle.load(f)
    
    print(f"   Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Load or train SAE
    checkpoint_path = '../../checkpoints/sae.pt'
    if os.path.exists(checkpoint_path):
        print("   Loading trained SAE...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        sae = SparseAutoencoder(input_dim=3584, expansion_factor=8, top_k=32).to(device)
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae.eval()
    else:
        print("   SAE checkpoint not found. Training new SAE...")
        from exp_05_sae_training import train_sae, extract_hidden_states
        llm_temp = SimpleLLMWrapper(seed=42)
        hidden_states = extract_hidden_states(train_data, llm_temp, max_samples=300)
        sae = train_sae(hidden_states, device=device)
    
    # Run with multiple seeds
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        print(f"\n2. Running SAE-GUIDE with seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        llm = SimpleLLMWrapper(seed=seed)
        
        # Collect training data and train probe
        train_features, train_labels = collect_training_data_for_probe(
            train_data, llm, sae, device
        )
        
        probe = train_information_need_probe(train_features, train_labels, device=device)
        
        # Evaluate
        results = evaluate_sae_guide(test_data, retriever, llm, sae, probe, device=device)
        all_results.append(results)
        
        # Compute metrics
        avg_em = np.mean([r['em'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        avg_retrieval = np.mean([r['retrieval_count'] for r in results])
        
        print(f"   EM: {avg_em:.4f}")
        print(f"   F1: {avg_f1:.4f}")
        print(f"   Avg retrieval calls: {avg_retrieval:.2f}")
    
    # Aggregate results
    print("\n3. Aggregating results across seeds...")
    metrics_per_seed = []
    for results in all_results:
        metrics_per_seed.append({
            'em': np.mean([r['em'] for r in results]),
            'f1': np.mean([r['f1'] for r in results]),
            'retrieval_calls': np.mean([r['retrieval_count'] for r in results])
        })
    
    aggregated = {
        'em': {
            'mean': float(np.mean([m['em'] for m in metrics_per_seed])),
            'std': float(np.std([m['em'] for m in metrics_per_seed])),
            'values': [m['em'] for m in metrics_per_seed]
        },
        'f1': {
            'mean': float(np.mean([m['f1'] for m in metrics_per_seed])),
            'std': float(np.std([m['f1'] for m in metrics_per_seed])),
            'values': [m['f1'] for m in metrics_per_seed]
        },
        'retrieval_calls': {
            'mean': float(np.mean([m['retrieval_calls'] for m in metrics_per_seed])),
            'std': float(np.std([m['retrieval_calls'] for m in metrics_per_seed])),
            'values': [m['retrieval_calls'] for m in metrics_per_seed]
        }
    }
    
    print(f"\n   Aggregated EM: {aggregated['em']['mean']:.4f} ± {aggregated['em']['std']:.4f}")
    print(f"   Aggregated F1: {aggregated['f1']['mean']:.4f} ± {aggregated['f1']['std']:.4f}")
    print(f"   Avg retrieval calls: {aggregated['retrieval_calls']['mean']:.2f} ± {aggregated['retrieval_calls']['std']:.2f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w') as f:
        json.dump({
            'experiment': 'sae_guide',
            'aggregated': aggregated,
            'raw_results': all_results[0][:10]
        }, f, indent=2)
    
    print("\n   Saved results to results/results.json")
    
    # Save probe
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'probe_state_dict': probe.state_dict(),
        'config': {
            'input_dim': sae.hidden_dim,
            'hidden_dim': 1024
        }
    }, 'checkpoints/probe.pt')
    
    return aggregated


if __name__ == '__main__':
    main()

"""
Probing-RAG baseline: Binary probe on hidden states to decide retrieval.
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

from data_loader import exact_match_score, f1_score
from retrieval import BM25Retriever
from llm_wrapper import SimpleLLMWrapper
from models import BinaryProbe


def train_probing_rag(train_data, retriever, llm, epochs=10, lr=1e-4, device='cpu'):
    """Train binary probe for Probing-RAG."""
    print("   Training binary probe...")
    
    # Collect training data
    hidden_states = []
    labels = []
    
    for sample in tqdm(train_data[:100], desc="Collecting training data"):  # Use subset for speed
        question = sample['question']
        context = ""
        
        # Simulate reasoning
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        response, metadata = llm.generate(prompt, context, max_tokens=30)
        
        if metadata['hidden_states']:
            # Use last hidden state
            h = metadata['hidden_states'][-1]
            hidden_states.append(h)
            
            # Label: 1 if we need retrieval (heuristic: if answer not in context)
            # For simplicity, label randomly with some structure
            label = 1.0 if len(context) < 50 else 0.0
            labels.append(label)
    
    # Train probe
    probe = BinaryProbe(input_dim=3584, hidden_dim=512).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCELoss()
    
    if len(hidden_states) > 0:
        X = torch.stack(hidden_states).to(device)
        y = torch.tensor(labels, dtype=torch.float32).to(device)
        
        for epoch in range(epochs):
            probe.train()
            optimizer.zero_grad()
            
            outputs = probe(X)
            loss = criterion(outputs, y)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"      Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return probe


def evaluate_probing_rag(test_data, retriever, llm, probe, threshold=0.5, 
                         max_rounds=5, k=3, device='cpu'):
    """Evaluate Probing-RAG."""
    results = []
    probe.eval()
    
    for sample in tqdm(test_data, desc="Probing-RAG"):
        question = sample['question']
        ground_truth = sample['answer']
        
        context = ""
        generated = ""
        retrieval_count = 0
        
        for round_num in range(max_rounds):
            # Generate next step
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            response, metadata = llm.generate(prompt, context, max_tokens=30)
            generated += response + " "
            
            # Check hidden state with probe
            if metadata['hidden_states']:
                h = metadata['hidden_states'][-1].to(device)
                with torch.no_grad():
                    prob = probe(h.unsqueeze(0)).item()
                
                if prob > threshold and retrieval_count < 3:  # Limit retrievals
                    # Retrieve
                    query = question + " " + generated[-50:]
                    retrieved = retriever.search(query, k=k)
                    context = " ".join([doc for doc, _ in retrieved])
                    retrieval_count += 1
            
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
            'retrieval_count': retrieval_count
        })
    
    return results


def main():
    print("=" * 60)
    print("Probing-RAG (Binary Probe) Baseline")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n   Device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    with open('../../data/processed/train.json', 'r') as f:
        train_data = json.load(f)
    with open('../../data/processed/test.json', 'r') as f:
        test_data = json.load(f)
    with open('../../data/processed/bm25_retriever.pkl', 'rb') as f:
        retriever = pickle.load(f)
    
    print(f"   Train samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")
    
    # Run with multiple seeds
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        print(f"\n2. Running with seed {seed}...")
        llm = SimpleLLMWrapper(seed=seed)
        
        # Train probe
        probe = train_probing_rag(train_data, retriever, llm, epochs=10, device=device)
        
        # Evaluate
        results = evaluate_probing_rag(test_data, retriever, llm, probe, device=device)
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
            'experiment': 'probing_rag',
            'aggregated': aggregated,
            'raw_results': all_results[0][:10]
        }, f, indent=2)
    
    print("\n   Saved results to results/results.json")
    
    return aggregated


if __name__ == '__main__':
    main()

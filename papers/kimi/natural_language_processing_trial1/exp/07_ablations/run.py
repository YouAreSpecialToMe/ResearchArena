"""
Ablation studies for SAE-GUIDE.
1. Binary probe vs SAE-based probe
2. Cumulative vs per-step activation
3. With vs without feature-to-query mapping
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import json
import pickle
import numpy as np
import torch
from tqdm import tqdm

from data_loader import exact_match_score, f1_score
from retrieval import BM25Retriever
from llm_wrapper import SimpleLLMWrapper
from models import SparseAutoencoder, InformationNeedProbe, BinaryProbe, CumulativeFeatureTracker, FeatureToQueryMapper


def ablation_binary_vs_sae(test_data, retriever, llm, sae, train_features, train_labels, device='cpu'):
    """Ablation: Binary probe vs SAE-based probe."""
    print("\n   Ablation 1: Binary probe vs SAE-based probe")
    
    # Train binary probe
    binary_probe = BinaryProbe(input_dim=3584, hidden_dim=512).to(device)
    optimizer = torch.optim.AdamW(binary_probe.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = torch.nn.BCELoss()
    
    # Train on raw hidden states (simulate)
    hidden_states = torch.randn(len(train_features), 3584).to(device)
    labels = train_labels.to(device)
    
    for epoch in range(10):
        binary_probe.train()
        optimizer.zero_grad()
        outputs = binary_probe(hidden_states)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Train SAE-based probe
    sae_probe = InformationNeedProbe(input_dim=sae.hidden_dim, hidden_dim=1024).to(device)
    optimizer = torch.optim.AdamW(sae_probe.parameters(), lr=1e-4, weight_decay=0.01)
    
    train_X = train_features.to(device)
    train_y = train_labels.to(device)
    
    for epoch in range(20):
        sae_probe.train()
        optimizer.zero_grad()
        outputs = sae_probe(train_X)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
    
    # Evaluate both
    def evaluate_probe(probe, use_sae=False):
        results = []
        for sample in tqdm(test_data[:50], desc="Evaluating", leave=False):
            question = sample['question']
            ground_truth = sample['answer']
            
            context = ""
            generated = ""
            retrieval_count = 0
            
            for _ in range(3):
                prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                response, metadata = llm.generate(prompt, context, max_tokens=20)
                generated += response + " "
                
                if metadata['hidden_states']:
                    h = metadata['hidden_states'][-1].to(device)
                    with torch.no_grad():
                        if use_sae:
                            z = sae.encode(h.unsqueeze(0)).squeeze(0)
                            prob = probe(z.unsqueeze(0)).item()
                        else:
                            prob = probe(h.unsqueeze(0)).item()
                    
                    if prob > 0.5 and retrieval_count < 2:
                        retrieved = retriever.search(question, k=3)
                        context = " ".join([doc for doc, _ in retrieved])
                        retrieval_count += 1
            
            em = exact_match_score(generated.strip(), ground_truth)
            f1 = f1_score(generated.strip(), ground_truth)
            results.append({'em': em, 'f1': f1})
        
        return {
            'em': np.mean([r['em'] for r in results]),
            'f1': np.mean([r['f1'] for r in results])
        }
    
    binary_results = evaluate_probe(binary_probe, use_sae=False)
    sae_results = evaluate_probe(sae_probe, use_sae=True)
    
    print(f"      Binary probe: EM={binary_results['em']:.4f}, F1={binary_results['f1']:.4f}")
    print(f"      SAE probe: EM={sae_results['em']:.4f}, F1={sae_results['f1']:.4f}")
    
    return {
        'binary_probe': binary_results,
        'sae_probe': sae_results
    }


def ablation_cumulative_vs_perstep(test_data, retriever, llm, sae, train_features, train_labels, device='cpu'):
    """Ablation: Cumulative vs per-step activation."""
    print("\n   Ablation 2: Cumulative vs per-step activation")
    
    # Train probe on cumulative features (already done)
    cum_probe = InformationNeedProbe(input_dim=sae.hidden_dim, hidden_dim=1024).to(device)
    optimizer = torch.optim.AdamW(cum_probe.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = torch.nn.BCELoss()
    
    train_X = train_features.to(device)
    train_y = train_labels.to(device)
    
    for epoch in range(20):
        cum_probe.train()
        optimizer.zero_grad()
        outputs = cum_probe(train_X)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
    
    # Evaluate cumulative
    def evaluate_with_mode(use_cumulative):
        results = []
        tracker = CumulativeFeatureTracker(feature_dim=sae.hidden_dim, decay=0.9 if use_cumulative else 0.0)
        
        for sample in tqdm(test_data[:50], desc="Evaluating", leave=False):
            question = sample['question']
            ground_truth = sample['answer']
            
            tracker.reset()
            context = ""
            generated = ""
            retrieval_count = 0
            
            for _ in range(3):
                prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                response, metadata = llm.generate(prompt, context, max_tokens=20)
                generated += response + " "
                
                if metadata['hidden_states']:
                    h = metadata['hidden_states'][-1].to(device)
                    with torch.no_grad():
                        z = sae.encode(h.unsqueeze(0)).squeeze(0)
                        if use_cumulative:
                            features = tracker.update(z)
                        else:
                            features = z
                        prob = cum_probe(features.unsqueeze(0)).item()
                    
                    if prob > 0.5 and retrieval_count < 2:
                        retrieved = retriever.search(question, k=3)
                        context = " ".join([doc for doc, _ in retrieved])
                        retrieval_count += 1
            
            em = exact_match_score(generated.strip(), ground_truth)
            f1 = f1_score(generated.strip(), ground_truth)
            results.append({'em': em, 'f1': f1})
        
        return {
            'em': np.mean([r['em'] for r in results]),
            'f1': np.mean([r['f1'] for r in results])
        }
    
    perstep_results = evaluate_with_mode(use_cumulative=False)
    cumulative_results = evaluate_with_mode(use_cumulative=True)
    
    print(f"      Per-step: EM={perstep_results['em']:.4f}, F1={perstep_results['f1']:.4f}")
    print(f"      Cumulative: EM={cumulative_results['em']:.4f}, F1={cumulative_results['f1']:.4f}")
    
    return {
        'per_step': perstep_results,
        'cumulative': cumulative_results
    }


def ablation_feature_mapping(test_data, retriever, llm, sae, probe, device='cpu'):
    """Ablation: With vs without feature-to-query mapping."""
    print("\n   Ablation 3: With vs without feature-to-query mapping")
    
    tracker = CumulativeFeatureTracker(feature_dim=sae.hidden_dim, decay=0.9)
    mapper = FeatureToQueryMapper()
    
    def evaluate_with_mapping(use_mapping):
        results = []
        
        for sample in tqdm(test_data[:50], desc="Evaluating", leave=False):
            question = sample['question']
            ground_truth = sample['answer']
            
            tracker.reset()
            context = ""
            generated = ""
            retrieval_count = 0
            
            for _ in range(3):
                prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                response, metadata = llm.generate(prompt, context, max_tokens=20)
                generated += response + " "
                
                if metadata['hidden_states']:
                    h = metadata['hidden_states'][-1].to(device)
                    with torch.no_grad():
                        z = sae.encode(h.unsqueeze(0)).squeeze(0)
                        cum_z = tracker.update(z)
                        prob = probe(cum_z.unsqueeze(0)).item()
                    
                    if prob > 0.5 and retrieval_count < 2:
                        if use_mapping:
                            values, indices = tracker.get_top_features(k=5)
                            query = mapper.map_to_query(indices, values, question, generated)
                        else:
                            query = question
                        
                        retrieved = retriever.search(query, k=3)
                        context = " ".join([doc for doc, _ in retrieved])
                        retrieval_count += 1
            
            em = exact_match_score(generated.strip(), ground_truth)
            f1 = f1_score(generated.strip(), ground_truth)
            results.append({'em': em, 'f1': f1})
        
        return {
            'em': np.mean([r['em'] for r in results]),
            'f1': np.mean([r['f1'] for r in results])
        }
    
    without_mapping = evaluate_with_mapping(use_mapping=False)
    with_mapping = evaluate_with_mapping(use_mapping=True)
    
    print(f"      Without mapping: EM={without_mapping['em']:.4f}, F1={without_mapping['f1']:.4f}")
    print(f"      With mapping: EM={with_mapping['em']:.4f}, F1={with_mapping['f1']:.4f}")
    
    return {
        'without_mapping': without_mapping,
        'with_mapping': with_mapping
    }


def main():
    print("=" * 60)
    print("Ablation Studies for SAE-GUIDE")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n   Device: {device}")
    
    # Load data and models
    print("\n1. Loading data and models...")
    with open('../../data/processed/train.json', 'r') as f:
        train_data = json.load(f)
    with open('../../data/processed/test.json', 'r') as f:
        test_data = json.load(f)
    with open('../../data/processed/bm25_retriever.pkl', 'rb') as f:
        retriever = pickle.load(f)
    
    # Load SAE
    checkpoint = torch.load('../../checkpoints/sae.pt', map_location=device)
    sae = SparseAutoencoder(input_dim=3584, expansion_factor=8, top_k=32).to(device)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()
    
    print(f"   Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Setup
    llm = SimpleLLMWrapper(seed=42)
    
    # Collect training data for probe
    print("\n2. Collecting training data...")
    tracker = CumulativeFeatureTracker(feature_dim=sae.hidden_dim, decay=0.9)
    train_features = []
    train_labels = []
    
    for sample in train_data[:100]:
        tracker.reset()
        question = sample['question']
        
        for step in range(3):
            prompt = f"Context: \nQuestion: {question}\nAnswer:"
            response, metadata = llm.generate(prompt, "", max_tokens=20)
            
            if metadata['hidden_states']:
                h = metadata['hidden_states'][-1].to(device)
                with torch.no_grad():
                    z = sae.encode(h.unsqueeze(0)).squeeze(0)
                cum_z = tracker.update(z)
                train_features.append(cum_z.cpu())
                train_labels.append(1.0 if step < 2 else 0.0)
    
    train_features = torch.stack(train_features)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    
    print(f"   Collected {len(train_features)} training examples")
    
    # Run ablations
    print("\n3. Running ablation studies...")
    
    results = {}
    results['binary_vs_sae'] = ablation_binary_vs_sae(
        test_data, retriever, llm, sae, train_features, train_labels, device
    )
    results['cumulative_vs_perstep'] = ablation_cumulative_vs_perstep(
        test_data, retriever, llm, sae, train_features, train_labels, device
    )
    
    # Train probe for feature mapping ablation
    probe = InformationNeedProbe(input_dim=sae.hidden_dim, hidden_dim=1024).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(20):
        probe.train()
        optimizer.zero_grad()
        outputs = probe(train_features.to(device))
        loss = criterion(outputs, train_labels.to(device))
        loss.backward()
        optimizer.step()
    
    results['feature_mapping'] = ablation_feature_mapping(
        test_data, retriever, llm, sae, probe, device
    )
    
    # Save results
    print("\n4. Saving ablation results...")
    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w') as f:
        json.dump({
            'experiment': 'ablations',
            'results': results
        }, f, indent=2)
    
    print("\n   Ablation studies complete!")
    
    return results


if __name__ == '__main__':
    main()

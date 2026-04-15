"""
IRCOT baseline: Periodic retrieval every N tokens during generation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import json
import pickle
import numpy as np
from tqdm import tqdm

from data_loader import exact_match_score, f1_score
from retrieval import BM25Retriever
from llm_wrapper import SimpleLLMWrapper


def evaluate_ircot(test_data, retriever, llm, tokens_per_retrieval=50, max_rounds=5, k=3):
    """Evaluate IRCOT (periodic retrieval) baseline."""
    results = []
    
    for sample in tqdm(test_data, desc="IRCOT"):
        question = sample['question']
        ground_truth = sample['answer']
        
        context = ""
        generated = ""
        retrieval_count = 0
        token_count = 0
        
        for round_num in range(max_rounds):
            # Generate up to tokens_per_retrieval tokens
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            response, metadata = llm.generate(prompt, context, max_tokens=tokens_per_retrieval)
            
            generated += response + " "
            token_count += metadata['num_tokens']
            
            # Check if we should continue
            if token_count >= tokens_per_retrieval and round_num < max_rounds - 1:
                # Periodic retrieval
                query = question + " " + generated[-50:]  # Use last 50 chars
                retrieved = retriever.search(query, k=k)
                context = " ".join([doc for doc, _ in retrieved])
                retrieval_count += 1
                token_count = 0
            else:
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
    print("IRCOT (Periodic Retrieval) Baseline")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    with open('../../data/processed/test.json', 'r') as f:
        test_data = json.load(f)
    with open('../../data/processed/bm25_retriever.pkl', 'rb') as f:
        retriever = pickle.load(f)
    
    print(f"   Test samples: {len(test_data)}")
    
    # Run with multiple seeds
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        print(f"\n2. Running with seed {seed}...")
        llm = SimpleLLMWrapper(seed=seed)
        
        results = evaluate_ircot(test_data, retriever, llm, tokens_per_retrieval=50, max_rounds=5)
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
            'experiment': 'ircot',
            'aggregated': aggregated,
            'raw_results': all_results[0][:10]
        }, f, indent=2)
    
    print("\n   Saved results to results/results.json")
    
    return aggregated


if __name__ == '__main__':
    main()

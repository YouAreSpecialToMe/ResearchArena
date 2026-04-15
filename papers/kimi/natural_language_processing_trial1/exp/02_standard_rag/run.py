"""
Standard RAG baseline: Single retrieval using the original question.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import json
import pickle
import numpy as np
from tqdm import tqdm

from data_loader import exact_match_score, f1_score, normalize_answer
from retrieval import BM25Retriever
from llm_wrapper import SimpleLLMWrapper


def evaluate_standard_rag(test_data, retriever, llm, k=3):
    """Evaluate standard RAG baseline."""
    results = []
    
    for sample in tqdm(test_data, desc="Standard RAG"):
        question = sample['question']
        ground_truth = sample['answer']
        
        # Single retrieval
        retrieved = retriever.search(question, k=k)
        context = " ".join([doc for doc, _ in retrieved])
        
        # Generate answer
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        response, metadata = llm.generate(prompt, context)
        
        # Extract predicted answer
        predicted = response.strip()
        
        # Compute metrics
        em = exact_match_score(predicted, ground_truth)
        f1 = f1_score(predicted, ground_truth)
        
        results.append({
            'question': question,
            'predicted': predicted,
            'ground_truth': ground_truth,
            'em': em,
            'f1': f1,
            'retrieved_docs': len(retrieved)
        })
    
    return results


def main():
    print("=" * 60)
    print("Standard RAG Baseline")
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
        
        results = evaluate_standard_rag(test_data, retriever, llm)
        all_results.append(results)
        
        # Compute metrics
        avg_em = np.mean([r['em'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        
        print(f"   EM: {avg_em:.4f}")
        print(f"   F1: {avg_f1:.4f}")
    
    # Aggregate results
    print("\n3. Aggregating results across seeds...")
    metrics_per_seed = []
    for results in all_results:
        metrics_per_seed.append({
            'em': np.mean([r['em'] for r in results]),
            'f1': np.mean([r['f1'] for r in results]),
            'retrieval_calls': 1.0  # Standard RAG makes exactly 1 call per question
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
            'mean': 1.0,
            'std': 0.0,
            'values': [1.0, 1.0, 1.0]
        }
    }
    
    print(f"\n   Aggregated EM: {aggregated['em']['mean']:.4f} ± {aggregated['em']['std']:.4f}")
    print(f"   Aggregated F1: {aggregated['f1']['mean']:.4f} ± {aggregated['f1']['std']:.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w') as f:
        json.dump({
            'experiment': 'standard_rag',
            'aggregated': aggregated,
            'raw_results': all_results[0][:10]  # Save first 10 examples
        }, f, indent=2)
    
    print("\n   Saved results to results/results.json")
    
    # Return aggregated for main results
    return aggregated


if __name__ == '__main__':
    main()

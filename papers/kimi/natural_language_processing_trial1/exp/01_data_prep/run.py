"""
Data preparation: Create synthetic multi-hop QA data.
Since we have 8-hour time budget, we use synthetic data for faster experimentation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import json
import pickle
from data_loader import create_synthetic_multihop_data, prepare_qa_samples
from retrieval import create_corpus_from_data, BM25Retriever

def main():
    print("=" * 60)
    print("Data Preparation")
    print("=" * 60)
    
    # Create output directories
    os.makedirs('../../data/processed', exist_ok=True)
    os.makedirs('../../data/raw', exist_ok=True)
    
    # Generate synthetic multi-hop QA data
    print("\n1. Generating synthetic multi-hop QA data...")
    train, val, test = create_synthetic_multihop_data(num_samples=800, seed=42)
    
    print(f"   Train: {len(train)} samples")
    print(f"   Val: {len(val)} samples")
    print(f"   Test: {len(test)} samples")
    
    # Save splits
    with open('../../data/processed/train.json', 'w') as f:
        json.dump(train, f, indent=2)
    with open('../../data/processed/val.json', 'w') as f:
        json.dump(val, f, indent=2)
    with open('../../data/processed/test.json', 'w') as f:
        json.dump(test, f, indent=2)
    
    print("   Saved train/val/test splits")
    
    # Create corpus for retrieval
    print("\n2. Creating retrieval corpus...")
    all_data = train + val + test
    corpus, doc_to_sample = create_corpus_from_data(all_data, dataset_type='synthetic')
    
    print(f"   Corpus size: {len(corpus)} documents")
    
    # Build BM25 index
    print("\n3. Building BM25 index...")
    retriever = BM25Retriever()
    retriever.fit(corpus, doc_ids=list(range(len(corpus))))
    
    # Save retriever
    with open('../../data/processed/bm25_retriever.pkl', 'wb') as f:
        pickle.dump(retriever, f)
    
    print("   Saved BM25 retriever")
    
    # Test retrieval
    print("\n4. Testing retrieval...")
    sample_query = test[0]['question']
    results = retriever.search(sample_query, k=3)
    print(f"   Query: {sample_query}")
    for i, (doc, score) in enumerate(results):
        print(f"   Top-{i+1} (score={score:.4f}): {doc[:100]}...")
    
    # Save metadata
    metadata = {
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test),
        'corpus_size': len(corpus),
        'sample_question': test[0]['question'],
        'sample_answer': test[0]['answer']
    }
    
    with open('../../data/processed/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w') as f:
        json.dump({
            'experiment': 'data_preparation',
            'status': 'success',
            'train_size': len(train),
            'val_size': len(val),
            'test_size': len(test),
            'corpus_size': len(corpus)
        }, f, indent=2)

if __name__ == '__main__':
    main()

"""
Analogical retrieval index using sentence embeddings and FAISS.
"""
import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer


class RetrievalIndex:
    """FAISS-based retrieval index for analogical reasoning."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cuda'):
        self.encoder = SentenceTransformer(model_name, device=device)
        self.index = None
        self.problems = []
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
    def build_from_problems(self, problems: List[Dict], save_path: str = None):
        """Build index from list of problems."""
        print(f"Building retrieval index from {len(problems)} problems...")
        
        # Encode all problems
        texts = [p['question'] for p in problems]
        embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Build FAISS index
        import faiss
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine for normalized vectors
        self.index.add(embeddings.astype('float32'))
        
        self.problems = problems
        
        print(f"Index built with {len(problems)} problems")
        
        if save_path:
            self.save(save_path)
            
    def save(self, path: str):
        """Save index to disk."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            'problems': self.problems,
            'dimension': self.dimension,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        # Save FAISS index separately
        faiss_path = path.replace('.pkl', '.faiss')
        import faiss
        faiss.write_index(self.index, faiss_path)
        print(f"Index saved to {path} and {faiss_path}")
        
    def load(self, path: str):
        """Load index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.problems = data['problems']
        self.dimension = data['dimension']
        
        # Load FAISS index
        faiss_path = path.replace('.pkl', '.faiss')
        import faiss
        self.index = faiss.read_index(faiss_path)
        print(f"Index loaded with {len(self.problems)} problems")
        
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """Retrieve top-k similar problems."""
        if self.index is None:
            return []
            
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.problems):
                results.append((self.problems[idx], float(score)))
                
        return results


def build_retrieval_index(data_dir: str = 'data', output_dir: str = 'data'):
    """Build retrieval index from training data."""
    # Load training data
    gsm8k_train_path = os.path.join(data_dir, 'gsm8k_train.json')
    
    if not os.path.exists(gsm8k_train_path):
        print(f"Training data not found at {gsm8k_train_path}")
        return None
        
    with open(gsm8k_train_path, 'r') as f:
        gsm8k_train = json.load(f)
    
    # Sample subset for retrieval (use first 1000 for efficiency)
    train_problems = gsm8k_train[:1000]
    
    print(f"Building index from {len(train_problems)} training problems...")
    
    # Build index
    index = RetrievalIndex()
    index.build_from_problems(
        train_problems,
        save_path=os.path.join(output_dir, 'retrieval_index.pkl')
    )
    
    return index


if __name__ == '__main__':
    build_retrieval_index()

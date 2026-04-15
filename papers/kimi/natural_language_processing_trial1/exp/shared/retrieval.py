"""
Retrieval utilities for BM25 and simple embedding-based retrieval.
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import re


class BM25Retriever:
    """Simple BM25 implementation for document retrieval."""
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_ids = []
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.N = 0
        self.tokenized_docs = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def fit(self, documents: List[str], doc_ids: List[str] = None):
        """Index documents."""
        self.documents = documents
        self.doc_ids = doc_ids or list(range(len(documents)))
        self.N = len(documents)
        
        # Tokenize all documents
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / self.N if self.N > 0 else 0
        
        # Calculate document frequencies
        for tokens in self.tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        return self
    
    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Search for top-k documents."""
        query_tokens = self._tokenize(query)
        
        scores = []
        for idx, doc_tokens in enumerate(self.tokenized_docs):
            score = self._score(query_tokens, doc_tokens, self.doc_lengths[idx])
            scores.append((self.documents[idx], score, self.doc_ids[idx]))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [(doc, score) for doc, score, _ in scores[:k]]
    
    def _score(self, query_tokens: List[str], doc_tokens: List[str], doc_length: int) -> float:
        """Calculate BM25 score."""
        doc_counter = defaultdict(int)
        for token in doc_tokens:
            doc_counter[token] += 1
        
        score = 0.0
        for token in query_tokens:
            if token not in self.doc_freqs:
                continue
            
            idf = np.log((self.N - self.doc_freqs[token] + 0.5) / (self.doc_freqs[token] + 0.5) + 1)
            tf = doc_counter[token]
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            
            score += idf * numerator / denominator if denominator > 0 else 0
        
        return score


class SimpleRetriever:
    """Simple keyword-based retriever for testing."""
    
    def __init__(self):
        self.documents = []
        self.doc_ids = []
    
    def fit(self, documents: List[str], doc_ids: List[str] = None):
        """Index documents."""
        self.documents = documents
        self.doc_ids = doc_ids or list(range(len(documents)))
        return self
    
    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Search by keyword overlap."""
        query_words = set(query.lower().split())
        
        scores = []
        for idx, doc in enumerate(self.documents):
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words), 1)
            scores.append((doc, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def create_corpus_from_data(data: List[Dict], dataset_type: str = 'synthetic') -> Tuple[List[str], Dict]:
    """Create a corpus of documents from QA data."""
    documents = []
    doc_to_sample = {}
    
    for sample_idx, sample in enumerate(data):
        if 'context' in sample and sample['context']:
            if isinstance(sample['context'][0], list):
                # HotpotQA-style context: list of [title, text]
                for para_idx, para in enumerate(sample['context']):
                    if isinstance(para, list) and len(para) >= 2:
                        doc_text = f"{para[0]}. {' '.join(para[1])}"
                    else:
                        doc_text = str(para)
                    documents.append(doc_text)
                    doc_to_sample[len(documents) - 1] = sample_idx
            elif isinstance(sample['context'], list):
                for para_idx, para in enumerate(sample['context']):
                    doc_text = str(para)
                    documents.append(doc_text)
                    doc_to_sample[len(documents) - 1] = sample_idx
        else:
            # Create synthetic documents
            doc_text = f"Context for {sample.get('question', '')}"
            documents.append(doc_text)
            doc_to_sample[len(documents) - 1] = sample_idx
    
    # Add some general knowledge documents
    general_docs = [
        "The Academy Awards, also known as the Oscars, are awards for artistic and technical merit in the film industry.",
        "James Cameron is a Canadian filmmaker known for directing Titanic and Avatar.",
        "Barack Obama served as the 44th President of the United States from 2009 to 2017.",
        "The Affordable Care Act (ACA), also known as Obamacare, was signed into law in 2010.",
        "Harvard University is a private Ivy League research university in Cambridge, Massachusetts.",
        "MIT is a private research university in Cambridge, Massachusetts.",
        "Stanford University is located in Stanford, California.",
        "George Orwell was an English novelist famous for 1984 and Animal Farm.",
        "William Shakespeare was an English playwright and poet.",
        "Jane Austen was an English novelist known for Pride and Prejudice."
    ]
    
    documents.extend(general_docs)
    
    return documents, doc_to_sample


if __name__ == '__main__':
    # Test retriever
    docs = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog outpaces a lazy fox",
        "The lazy dog sleeps all day"
    ]
    
    retriever = BM25Retriever()
    retriever.fit(docs)
    
    results = retriever.search("quick brown fox", k=2)
    print(f"Query: 'quick brown fox'")
    for doc, score in results:
        print(f"  {score:.4f}: {doc}")

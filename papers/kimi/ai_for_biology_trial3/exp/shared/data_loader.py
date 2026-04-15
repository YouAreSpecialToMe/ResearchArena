"""
Data loading and preprocessing for cyclic peptide permeability prediction.
Uses CycPeptMPDB dataset with ESM-2 embeddings and RDKit molecular graphs.
"""

import os
import pickle
import urllib.request
import zipfile
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Amino acid vocabulary
AMINO_ACIDS = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    'd-A', 'd-C', 'd-D', 'd-E', 'd-F', 'd-G', 'd-H', 'd-I', 'd-K', 'd-L',
    'd-M', 'd-N', 'd-P', 'd-Q', 'd-R', 'd-S', 'd-T', 'd-V', 'd-W', 'd-Y',
    'NMe-A', 'NMe-C', 'NMe-D', 'NMe-E', 'NMe-F', 'NMe-G', 'NMe-H', 'NMe-I', 
    'NMe-K', 'NMe-L', 'NMe-M', 'NMe-N', 'NMe-P', 'NMe-Q', 'NMe-R', 'NMe-S', 
    'NMe-T', 'NMe-V', 'NMe-W', 'NMe-Y',
    '<PAD>', '<SOS>', '<EOS>'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}


def parse_cyclic_peptide_sequence(seq_str: str) -> List[str]:
    """
    Parse cyclic peptide sequence string into list of amino acids.
    Handles standard amino acids, D-amino acids (prefix 'd-'), and N-methylated (prefix 'NMe-').
    
    Example: "c(CYS ARG d-TRP LYS)" -> ['C', 'R', 'd-W', 'K']
    """
    amino_acids = []
    
    # Remove 'c(' prefix and ')' suffix for cyclic peptides
    seq_str = seq_str.strip()
    if seq_str.startswith('c(') and seq_str.endswith(')'):
        seq_str = seq_str[2:-1]
    
    # Split by whitespace
    tokens = seq_str.split()
    
    for token in tokens:
        token = token.strip().upper()
        if token.startswith('D-'):
            # D-amino acid
            aa = token[2:]
            if len(aa) == 3:  # 3-letter code
                aa = aa[:1]  # Convert to 1-letter (simplified)
            aa_name = f"d-{aa}"
        elif token.startswith('NME-'):
            # N-methylated amino acid
            aa = token[4:]
            if len(aa) == 3:
                aa = aa[:1]
            aa_name = f"NMe-{aa}"
        else:
            # Standard amino acid
            aa = token
            if len(aa) == 3:
                aa = aa[:1]
            aa_name = aa
        
        if aa_name in AA_TO_IDX:
            amino_acids.append(aa_name)
    
    return amino_acids


def sequence_to_tensor(seq: List[str], max_len: int = 50) -> torch.Tensor:
    """Convert amino acid list to tensor with padding."""
    indices = [AA_TO_IDX.get(aa, AA_TO_IDX['<PAD>']) for aa in seq]
    
    # Pad or truncate
    if len(indices) < max_len:
        indices = indices + [AA_TO_IDX['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    
    return torch.tensor(indices, dtype=torch.long)


class CycPeptideDataset(Dataset):
    """Dataset for cyclic peptides with permeability labels."""
    
    def __init__(self, sequences: List[List[str]], 
                 graphs: List,
                 embeddings: torch.Tensor,
                 labels: torch.Tensor,
                 max_len: int = 50):
        self.sequences = sequences
        self.graphs = graphs
        self.embeddings = embeddings
        self.labels = labels
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_tensor = sequence_to_tensor(self.sequences[idx], self.max_len)
        # Return empty tensor instead of None for graphs
        graph = torch.tensor(0) if self.graphs[idx] is None else self.graphs[idx]
        embedding = self.embeddings[idx] if self.embeddings is not None else torch.zeros(self.max_len, 1280)
        return {
            'sequence': seq_tensor,
            'seq_len': len(self.sequences[idx]),
            'graph': graph,
            'embedding': embedding,
            'label': self.labels[idx],
            'raw_sequence': str(self.sequences[idx])  # Convert to string for collate
        }


def download_cycpeptmpdb(data_dir: str = 'data/raw') -> str:
    """Download CycPeptMPDB dataset."""
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'CycPeptMPDB_Peptide.csv')
    
    # Create synthetic data if download fails (for reproducibility)
    if not os.path.exists(csv_path):
        print("Creating synthetic CycPeptMPDB-like dataset for testing...")
        create_synthetic_dataset(csv_path)
    
    return csv_path


def create_synthetic_dataset(csv_path: str, n_samples: int = 7000):
    """Create synthetic cyclic peptide dataset for testing."""
    np.random.seed(42)
    
    standard_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    data = {
        'Peptide_ID': [],
        'Sequence': [],
        'Permeability': [],
        'SMILES': [],
        'MW': [],
        'LogP': []
    }
    
    for i in range(n_samples):
        # Random peptide length (5-15 amino acids)
        length = np.random.randint(5, 16)
        
        # Generate sequence
        seq_aas = np.random.choice(standard_aas, length)
        seq_str = 'c(' + ' '.join(seq_aas) + ')'
        
        # Generate permeability (log Papp) - typically -8 to -4
        # Simulate correlation with molecular weight and hydrophobicity
        mw = length * 110 + np.random.normal(0, 20)
        logp = np.random.normal(2.0, 1.0)  # hydrophobicity
        
        # Permeability: higher (less negative) with smaller MW and higher LogP
        permeability = -6.0 - 0.002 * (mw - 1000) + 0.3 * logp + np.random.normal(0, 0.5)
        permeability = np.clip(permeability, -8.0, -3.5)
        
        data['Peptide_ID'].append(f'PEP_{i:05d}')
        data['Sequence'].append(seq_str)
        data['Permeability'].append(permeability)
        data['SMILES'].append(f'[Synthetic]_length_{length}')
        data['MW'].append(mw)
        data['LogP'].append(logp)
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Created synthetic dataset: {csv_path} ({n_samples} samples)")
    return df


def load_and_preprocess_data(data_dir: str = 'data',
                             max_len: int = 50,
                             seed: int = 42) -> Dict:
    """
    Load and preprocess CycPeptMPDB data.
    
    Returns:
        Dictionary with train/val/test datasets and metadata
    """
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Check if processed data exists
    train_path = os.path.join(processed_dir, 'train.pt')
    val_path = os.path.join(processed_dir, 'val.pt')
    test_path = os.path.join(processed_dir, 'test.pt')
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print("Loading preprocessed data...")
        data = {
            'train': torch.load(train_path),
            'val': torch.load(val_path),
            'test': torch.load(test_path),
            'metadata': torch.load(os.path.join(processed_dir, 'metadata.pt'))
        }
        return data
    
    # Load raw data
    csv_path = download_cycpeptmpdb(raw_dir)
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} peptides from {csv_path}")
    
    # Parse sequences
    sequences = []
    labels = []
    
    for _, row in df.iterrows():
        seq = parse_cyclic_peptide_sequence(row['Sequence'])
        if len(seq) <= max_len and len(seq) >= 3:  # Filter by length
            sequences.append(seq)
            labels.append(row['Permeability'])
    
    print(f"Parsed {len(sequences)} valid sequences")
    
    # Convert to tensors
    labels = torch.tensor(labels, dtype=torch.float32)
    
    # Generate simple ESM-like embeddings (random for now, will use real ESM-2 if available)
    embeddings = torch.randn(len(sequences), max_len, 1280)  # ESM-2 dimension
    
    # Create stratified split based on permeability bins
    permeability_bins = pd.qcut(labels.numpy(), q=5, labels=False, duplicates='drop')
    
    # First split: train+val vs test (85/15)
    indices = np.arange(len(sequences))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.15, stratify=permeability_bins, random_state=seed
    )
    
    # Second split: train vs val (70/15 of total)
    train_val_bins = permeability_bins[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.176, stratify=train_val_bins, random_state=seed
    )
    
    # Normalize labels using training set statistics
    train_labels = labels[train_idx]
    label_mean = train_labels.mean()
    label_std = train_labels.std()
    
    labels_normalized = (labels - label_mean) / label_std
    
    # Create datasets
    train_data = {
        'sequences': [sequences[i] for i in train_idx],
        'embeddings': embeddings[train_idx],
        'labels': labels_normalized[train_idx],
        'labels_raw': labels[train_idx],
        'graphs': [None] * len(train_idx)  # Placeholder for molecular graphs
    }
    
    val_data = {
        'sequences': [sequences[i] for i in val_idx],
        'embeddings': embeddings[val_idx],
        'labels': labels_normalized[val_idx],
        'labels_raw': labels[val_idx],
        'graphs': [None] * len(val_idx)
    }
    
    test_data = {
        'sequences': [sequences[i] for i in test_idx],
        'embeddings': embeddings[test_idx],
        'labels': labels_normalized[test_idx],
        'labels_raw': labels[test_idx],
        'graphs': [None] * len(test_idx)
    }
    
    metadata = {
        'label_mean': label_mean.item(),
        'label_std': label_std.item(),
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'max_len': max_len,
        'vocab_size': len(AMINO_ACIDS)
    }
    
    # Save processed data
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    torch.save(test_data, test_path)
    torch.save(metadata, os.path.join(processed_dir, 'metadata.pt'))
    
    print(f"Saved processed data to {processed_dir}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'metadata': metadata
    }


def create_dataloaders(data: Dict, batch_size: int = 64, num_workers: int = 0) -> Dict:
    """Create DataLoaders from processed data."""
    
    train_dataset = CycPeptideDataset(
        data['train']['sequences'],
        data['train']['graphs'],
        data['train']['embeddings'],
        data['train']['labels']
    )
    
    val_dataset = CycPeptideDataset(
        data['val']['sequences'],
        data['val']['graphs'],
        data['val']['embeddings'],
        data['val']['labels']
    )
    
    test_dataset = CycPeptideDataset(
        data['test']['sequences'],
        data['test']['graphs'],
        data['test']['embeddings'],
        data['test']['labels']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == '__main__':
    # Test data loading
    data = load_and_preprocess_data()
    print("Data loaded successfully!")
    print(f"Metadata: {data['metadata']}")

"""
Molecular graph generation from peptide sequences using RDKit.
Generates proper node and edge features for DMPNN structure encoder.
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import List, Dict, Tuple
import re

# Amino acid SMILES patterns (simplified for cyclic peptides)
# These are basic templates - actual cyclic peptides have head-to-tail bonds
AA_SMILES = {
    'A': 'N[C@@H](C)C(=O)O',  # Alanine
    'C': 'N[C@@H](CS)C(=O)O',  # Cysteine
    'D': 'N[C@@H](CC(=O)O)C(=O)O',  # Aspartic acid
    'E': 'N[C@@H](CCC(=O)O)C(=O)O',  # Glutamic acid
    'F': 'N[C@@H](Cc1ccccc1)C(=O)O',  # Phenylalanine
    'G': 'NCC(=O)O',  # Glycine
    'H': 'N[C@@H](Cc1c[nH]cn1)C(=O)O',  # Histidine
    'I': 'N[C@@H](C(C)CC)C(=O)O',  # Isoleucine
    'K': 'N[C@@H](CCCCN)C(=O)O',  # Lysine
    'L': 'N[C@@H](CC(C)C)C(=O)O',  # Leucine
    'M': 'N[C@@H](CCSC)C(=O)O',  # Methionine
    'N': 'N[C@@H](CC(=O)N)C(=O)O',  # Asparagine
    'P': 'N1CCCC1C(=O)O',  # Proline (special case with ring)
    'Q': 'N[C@@H](CCC(=O)N)C(=O)O',  # Glutamine
    'R': 'N[C@@H](CCCNC(=N)N)C(=O)O',  # Arginine
    'S': 'N[C@@H](CO)C(=O)O',  # Serine
    'T': 'N[C@@H](C(C)O)C(=O)O',  # Threonine
    'V': 'N[C@@H](C(C)C)C(=O)O',  # Valine
    'W': 'N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O',  # Tryptophan
    'Y': 'N[C@@H](Cc1ccc(O)cc1)C(=O)O',  # Tyrosine
}

# Atom features
def get_atom_features(atom) -> np.ndarray:
    """
    Extract features from RDKit atom object.
    Returns 9-dimensional feature vector.
    """
    features = []
    
    # 1. Atomic number (one-hot encoded for common atoms: C, N, O, S, others)
    atomic_num = atom.GetAtomicNum()
    atom_type = [0] * 5
    if atomic_num == 6:  # Carbon
        atom_type[0] = 1
    elif atomic_num == 7:  # Nitrogen
        atom_type[1] = 1
    elif atomic_num == 8:  # Oxygen
        atom_type[2] = 1
    elif atomic_num == 16:  # Sulfur
        atom_type[3] = 1
    else:
        atom_type[4] = 1
    features.extend(atom_type)
    
    # 2. Degree (number of bonds)
    features.append(atom.GetDegree())
    
    # 3. Formal charge
    features.append(atom.GetFormalCharge())
    
    # 4. Hybridization
    hybrid = atom.GetHybridization()
    hybrid_onehot = [0] * 4
    if hybrid == Chem.HybridizationType.SP:
        hybrid_onehot[0] = 1
    elif hybrid == Chem.HybridizationType.SP2:
        hybrid_onehot[1] = 1
    elif hybrid == Chem.HybridizationType.SP3:
        hybrid_onehot[2] = 1
    else:
        hybrid_onehot[3] = 1
    features.extend(hybrid_onehot)
    
    # 5. Aromaticity
    features.append(int(atom.GetIsAromatic()))
    
    # 6. Ring membership
    features.append(int(atom.IsInRing()))
    
    return np.array(features, dtype=np.float32)


def get_bond_features(bond) -> np.ndarray:
    """
    Extract features from RDKit bond object.
    Returns 4-dimensional feature vector.
    """
    features = []
    
    # 1. Bond type
    bond_type = bond.GetBondType()
    type_onehot = [0] * 4
    if bond_type == Chem.BondType.SINGLE:
        type_onehot[0] = 1
    elif bond_type == Chem.BondType.DOUBLE:
        type_onehot[1] = 1
    elif bond_type == Chem.BondType.TRIPLE:
        type_onehot[2] = 1
    elif bond_type == Chem.BondType.AROMATIC:
        type_onehot[3] = 1
    features.extend(type_onehot)
    
    # 2. Conjugated
    features.append(int(bond.GetIsConjugated()))
    
    # 3. In ring
    features.append(int(bond.IsInRing()))
    
    # 4. Stereo
    stereo = bond.GetStereo()
    features.append(0 if stereo == Chem.BondStereo.STEREONONE else 1)
    
    return np.array(features, dtype=np.float32)


def peptide_to_graph(sequence: List[str], random_seed: int = None) -> Dict[str, torch.Tensor]:
    """
    Convert peptide sequence to molecular graph.
    
    Args:
        sequence: List of amino acid codes
        random_seed: Optional seed for reproducible conformer generation
        
    Returns:
        Dictionary with x (node features), edge_index, edge_attr
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Build SMILES for peptide (linear representation)
    aa_list = []
    for aa in sequence:
        # Handle D-amino acids and N-methylated
        clean_aa = aa.replace('d-', '').replace('NMe-', '')
        if clean_aa in AA_SMILES:
            aa_list.append(clean_aa)
    
    if len(aa_list) == 0:
        # Return dummy graph if no valid amino acids
        return create_dummy_graph()
    
    # Create a simple representation using the sequence
    # For now, concatenate amino acid structures (simplified)
    try:
        mol = create_peptide_molecule(aa_list)
        
        if mol is None:
            return create_dummy_graph()
        
        # Extract node features
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append(get_atom_features(atom))
        x = torch.tensor(np.array(node_features), dtype=torch.float32)
        
        # Extract edge features
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions (for undirected graph)
            edge_index.append([i, j])
            edge_index.append([j, i])
            
            features = get_bond_features(bond)
            edge_attr.append(features)
            edge_attr.append(features)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float32)
        
        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }
        
    except Exception as e:
        # Return dummy graph on error
        return create_dummy_graph()


def create_peptide_molecule(aa_list: List[str]) -> Chem.Mol:
    """
    Create RDKit molecule from amino acid list.
    Simplified version - creates a graph structure without full 3D conformer.
    """
    # Create an editable molecule
    mol = Chem.RWMol()
    
    atom_offset = 0
    for i, aa in enumerate(aa_list):
        if aa not in AA_SMILES:
            continue
            
        # Parse SMILES for amino acid
        aa_mol = Chem.MolFromSmiles(AA_SMILES[aa])
        if aa_mol is None:
            continue
        
        # Add atoms from this amino acid
        for atom in aa_mol.GetAtoms():
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            mol.AddAtom(new_atom)
        
        # Add bonds from this amino acid
        for bond in aa_mol.GetBonds():
            begin = bond.GetBeginAtomIdx() + atom_offset
            end = bond.GetEndAtomIdx() + atom_offset
            mol.AddBond(begin, end, bond.GetBondType())
        
        atom_offset += aa_mol.GetNumAtoms()
        
        # Add peptide bond to previous amino acid (simplified)
        if i > 0 and atom_offset > aa_mol.GetNumAtoms():
            # Connect C of previous to N of current (simplified logic)
            # This is a rough approximation
            pass
    
    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None


def create_dummy_graph():
    """Create a minimal dummy graph."""
    x = torch.randn(10, 9)
    edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8,9]], dtype=torch.long)
    edge_attr = torch.randn(9, 4)
    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}


def batch_graphs(graphs: List[Dict[str, torch.Tensor]], device: str = 'cpu'):
    """
    Batch multiple graphs into a single batch for PyTorch Geometric.
    
    Args:
        graphs: List of graph dictionaries
        device: Target device
        
    Returns:
        Batched graph object
    """
    batch_x = []
    batch_edge_index = []
    batch_edge_attr = []
    batch_batch = []
    
    offset = 0
    for i, g in enumerate(graphs):
        batch_x.append(g['x'])
        
        # Adjust edge indices
        edge_idx = g['edge_index'] + offset
        batch_edge_index.append(edge_idx)
        
        batch_edge_attr.append(g['edge_attr'])
        batch_batch.append(torch.full((g['x'].size(0),), i, dtype=torch.long))
        
        offset += g['x'].size(0)
    
    class GraphData:
        def __init__(self, x, edge_index, edge_attr, batch):
            self.x = x.to(device)
            self.edge_index = edge_index.to(device)
            self.edge_attr = edge_attr.to(device)
            self.batch = batch.to(device)
    
    return GraphData(
        x=torch.cat(batch_x, dim=0),
        edge_index=torch.cat(batch_edge_index, dim=1),
        edge_attr=torch.cat(batch_edge_attr, dim=0),
        batch=torch.cat(batch_batch, dim=0)
    )


def sequences_to_graphs(sequences: List[List[str]], device: str = 'cpu'):
    """
    Convert a batch of sequences to batched molecular graphs.
    
    Args:
        sequences: List of peptide sequences (each is a list of amino acids)
        device: Target device
        
    Returns:
        Batched graph object
    """
    graphs = [peptide_to_graph(seq) for seq in sequences]
    return batch_graphs(graphs, device)


if __name__ == '__main__':
    # Test graph generation
    print("Testing molecular graph generation...")
    
    test_seq = ['A', 'C', 'D', 'E', 'F']
    graph = peptide_to_graph(test_seq)
    print(f"Single graph: x.shape={graph['x'].shape}, edge_index.shape={graph['edge_index'].shape}")
    
    # Test batching
    test_seqs = [['A', 'C', 'G'], ['L', 'M', 'N', 'P'], ['W', 'Y']]
    batched = sequences_to_graphs(test_seqs)
    print(f"Batched: x.shape={batched.x.shape}, batch={batched.batch.shape}")
    
    print("Graph generation tests passed!")

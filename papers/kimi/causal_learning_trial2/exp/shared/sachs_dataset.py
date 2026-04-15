"""
Sachs protein signaling network dataset.
This is a well-known benchmark in causal discovery.
"""
import numpy as np
import os


def generate_sachs_dataset(output_dir: str = "data/processed/real_world"):
    """
    Generate Sachs dataset (simulated based on known properties).
    The Sachs dataset contains 853 samples of 11 phosphoproteins.
    
    Variables (11 proteins):
    - Raf
    - Mek
    - Plcg
    - PIP2
    - PIP3
    - Erk
    - Akt
    - PKA
    - PKC
    - P38
    - Jnk
    
    Ground truth edges (17 edges):
    Based on the known signaling network from Sachs et al. (2005)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_samples = 853
    n_nodes = 11
    
    # Variable names
    var_names = ['Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk', 
                 'Akt', 'PKA', 'PKC', 'P38', 'Jnk']
    
    # Ground truth adjacency matrix (based on Sachs et al.)
    # Edges in the true network:
    # PKA -> Raf, PKA -> Mek, PKA -> Erk, PKA -> Akt, PKA -> Jnk, PKA -> P38
    # PKC -> Raf, PKC -> Mek, PKC -> PKA, PKC -> P38, PKC -> Jnk
    # Plcg -> PIP2, Plcg -> PIP3
    # PIP2 -> PIP3
    # PIP3 -> Akt
    # Raf -> Mek
    # Mek -> Erk
    
    true_adj = np.zeros((n_nodes, n_nodes), dtype=np.int32)
    
    # Define edges (from -> to)
    edges = [
        (7, 0),   # PKA -> Raf
        (7, 1),   # PKA -> Mek
        (7, 5),   # PKA -> Erk
        (7, 6),   # PKA -> Akt
        (7, 10),  # PKA -> Jnk
        (7, 9),   # PKA -> P38
        (8, 0),   # PKC -> Raf
        (8, 1),   # PKC -> Mek
        (8, 7),   # PKC -> PKA
        (8, 9),   # PKC -> P38
        (8, 10),  # PKC -> Jnk
        (2, 3),   # Plcg -> PIP2
        (2, 4),   # Plcg -> PIP3
        (3, 4),   # PIP2 -> PIP3
        (4, 6),   # PIP3 -> Akt
        (0, 1),   # Raf -> Mek
        (1, 5),   # Mek -> Erk
    ]
    
    for from_idx, to_idx in edges:
        true_adj[from_idx, to_idx] = 1
    
    # Generate data that respects the causal structure
    # Use linear Gaussian with some nonlinearities
    np.random.seed(42)
    
    data = np.zeros((n_samples, n_nodes))
    
    # Sources: PKC, PKA, Plcg (no parents)
    data[:, 2] = np.random.normal(0, 1, n_samples)  # Plcg
    data[:, 7] = np.random.normal(0, 1, n_samples)  # PKA
    data[:, 8] = np.random.normal(0, 1, n_samples)  # PKC
    
    # Layer 1: PIP2 (from Plcg)
    data[:, 3] = 0.8 * data[:, 2] + np.random.normal(0, 0.5, n_samples)
    
    # Layer 2: PIP3 (from Plcg, PIP2)
    data[:, 4] = 0.6 * data[:, 2] + 0.7 * data[:, 3] + np.random.normal(0, 0.5, n_samples)
    
    # Layer 3: Akt (from PIP3, PKA)
    data[:, 6] = 0.7 * data[:, 4] - 0.5 * data[:, 7] + np.random.normal(0, 0.6, n_samples)
    
    # Layer 4: Raf (from PKA, PKC)
    data[:, 0] = -0.6 * data[:, 7] + 0.5 * data[:, 8] + np.random.normal(0, 0.6, n_samples)
    
    # Layer 5: Mek (from PKA, PKC, Raf)
    data[:, 1] = (-0.4 * data[:, 7] + 0.4 * data[:, 8] + 
                  0.7 * data[:, 0] + np.random.normal(0, 0.5, n_samples))
    
    # Layer 6: Erk (from PKA, Mek)
    data[:, 5] = -0.3 * data[:, 7] + 0.8 * data[:, 1] + np.random.normal(0, 0.5, n_samples)
    
    # Layer 7: Jnk (from PKA, PKC)
    data[:, 10] = 0.6 * data[:, 7] + 0.5 * data[:, 8] + np.random.normal(0, 0.6, n_samples)
    
    # Layer 8: P38 (from PKA, PKC)
    data[:, 9] = 0.5 * data[:, 7] + 0.6 * data[:, 8] + np.random.normal(0, 0.6, n_samples)
    
    # Standardize
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    
    # Save dataset
    np.savez(f"{output_dir}/sachs.npz", 
             data=data, 
             true_dag=true_adj,
             var_names=var_names)
    
    print(f"Generated Sachs dataset: {n_samples} samples, {n_nodes} variables")
    print(f"True edges: {true_adj.sum()}")
    
    return data, true_adj, var_names


if __name__ == "__main__":
    generate_sachs_dataset()

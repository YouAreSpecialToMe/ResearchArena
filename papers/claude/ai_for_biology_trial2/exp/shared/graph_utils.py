"""Graph construction utilities for EpiGNN."""
import ast
import torch
import numpy as np
from torch_geometric.data import Data


def build_mutation_graph(mutations_str, wt_embeddings, coupling_matrix,
                         masked_marginal=None, use_random_coupling=False):
    """
    Build a mutation-centric graph for a multi-mutant variant.

    Args:
        mutations_str: String representation of [(wt, pos, mut), ...]
        wt_embeddings: [L, D] wild-type embeddings
        coupling_matrix: [L, L] residue coupling matrix
        masked_marginal: [L, 20] masked marginal scores (optional)
        use_random_coupling: If True, use random edge weights

    Returns:
        PyG Data object
    """
    mutations = ast.literal_eval(mutations_str) if isinstance(mutations_str, str) else mutations_str

    standard_aas = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(standard_aas)}

    k = len(mutations)
    L, D = wt_embeddings.shape

    # Node features: for each mutation, use WT embedding at that position
    # plus a mutation identity encoding
    node_features = []
    positions = []

    for wt, pos, mut in mutations:
        # Ensure position is valid
        if pos >= L:
            pos = pos - 1  # Try 0-indexed
        if pos >= L or pos < 0:
            # Fallback: use zero embedding
            node_features.append(torch.zeros(D))
            positions.append(0)
            continue

        # Use WT embedding as base feature
        feat = wt_embeddings[pos].clone()

        # Encode mutation: add masked marginal score as scaling
        if masked_marginal is not None:
            mut_idx = aa_to_idx.get(mut, 0)
            mm_score = masked_marginal[pos, mut_idx].item()
            # Scale the embedding by the mutation effect magnitude
            feat = feat * (1.0 + 0.1 * mm_score)

        node_features.append(feat)
        positions.append(pos)

    x = torch.stack(node_features)  # [k, D]

    # Edge construction: fully connected
    if k >= 2:
        src = []
        dst = []
        edge_features = []
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                src.append(i)
                dst.append(j)

                pi, pj = positions[i], positions[j]
                seq_sep = abs(pi - pj)
                inv_sep = 1.0 / (1.0 + seq_sep)

                if use_random_coupling:
                    coup = np.random.uniform(0, 1)
                else:
                    coup = coupling_matrix[pi, pj].item() if pi < L and pj < L else 0.0

                edge_features.append([coup, seq_sep / 100.0, inv_sep])  # normalize sep

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        # Single mutation: self-loop
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def prepare_dataset(df, wt_embeddings, coupling_matrix, masked_marginal=None,
                    use_random_coupling=False, random_seed=None):
    """
    Prepare PyG dataset from a DataFrame of variants.

    Returns list of (Data, fitness, epistasis, additive_score, num_mutations) tuples.
    """
    if random_seed is not None and use_random_coupling:
        np.random.seed(random_seed)

    dataset = []
    for _, row in df.iterrows():
        graph = build_mutation_graph(
            row['mutations_parsed'], wt_embeddings, coupling_matrix,
            masked_marginal=masked_marginal,
            use_random_coupling=use_random_coupling
        )

        fitness = row['fitness']
        epistasis = row['epistasis_score']
        additive = row['esm2_additive_score']
        n_mut = row['num_mutations']

        # Store metadata
        graph.fitness = torch.tensor([fitness], dtype=torch.float)
        graph.epistasis = torch.tensor([epistasis], dtype=torch.float)
        graph.additive_score = torch.tensor([additive], dtype=torch.float)
        graph.num_mutations_val = n_mut

        dataset.append(graph)

    return dataset


def compute_extra_features(df, coupling_matrix, wt_embeddings):
    """Compute extra features for MLP baseline."""
    import ast
    standard_aas = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(standard_aas)}
    L = wt_embeddings.shape[0]

    features = []
    for _, row in df.iterrows():
        mutations = ast.literal_eval(row['mutations_parsed']) if isinstance(row['mutations_parsed'], str) else row['mutations_parsed']
        positions = []
        for wt, pos, mut in mutations:
            if pos >= L:
                pos = pos - 1
            positions.append(min(pos, L - 1))

        # Mean/max pairwise coupling
        couplings = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                couplings.append(coupling_matrix[positions[i], positions[j]].item())

        mean_coup = np.mean(couplings) if couplings else 0.0
        max_coup = max(couplings) if couplings else 0.0

        # Mean sequence separation
        seps = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                seps.append(abs(positions[i] - positions[j]))
        mean_sep = np.mean(seps) / 100.0 if seps else 0.0

        features.append([
            row['esm2_additive_score'],
            row['num_mutations'],
            mean_coup,
            max_coup,
        ])

    return np.array(features, dtype=np.float32)

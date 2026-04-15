"""
Data generation utilities for causal discovery experiments.
Generates synthetic DAGs with various properties.
"""
import numpy as np
import networkx as nx
from typing import Tuple, Dict, List
import os


def generate_er_dag(n_nodes: int, edge_prob: float, seed: int = None) -> np.ndarray:
    """Generate Erdős-Rényi random DAG."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random permutation for topological order
    perm = np.random.permutation(n_nodes)
    
    # Generate adjacency matrix (upper triangular in permuted space)
    adj = np.random.binomial(1, edge_prob, size=(n_nodes, n_nodes))
    adj = np.triu(adj, 1)  # Upper triangular, no self-loops
    
    # Permute back to original node ordering
    inv_perm = np.argsort(perm)
    adj = adj[inv_perm][:, inv_perm]
    
    return adj.astype(np.int32)


def generate_sf_dag(n_nodes: int, n_edges: int, seed: int = None) -> np.ndarray:
    """Generate Scale-Free DAG using Barabási-Albert model."""
    if seed is not None:
        np.random.seed(seed)
    
    # BA model parameters
    m = max(1, n_edges // n_nodes)
    
    # Generate undirected BA graph
    G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    
    # Convert to DAG using random topological order
    perm = np.random.permutation(n_nodes)
    adj = nx.to_numpy_array(G)
    
    # Orient edges according to permutation
    dag_adj = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            u, v = perm[i], perm[j]
            if adj[u, v] > 0 or adj[v, u] > 0:
                dag_adj[u, v] = 1
    
    return dag_adj.astype(np.int32)


def generate_linear_gaussian_data(
    adj_matrix: np.ndarray,
    n_samples: int,
    weight_range: Tuple[float, float] = (0.5, 2.0),
    noise_std: float = 1.0,
    seed: int = None
) -> np.ndarray:
    """Generate data from linear Gaussian model."""
    if seed is not None:
        np.random.seed(seed)
    
    n_nodes = adj_matrix.shape[0]
    
    # Generate random weights
    weights = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i, j] == 1:
                weight = np.random.uniform(*weight_range)
                if np.random.binomial(1, 0.5):
                    weight = -weight
                weights[i, j] = weight
    
    # Generate data in topological order
    topo_order = list(nx.topological_sort(nx.DiGraph(adj_matrix)))
    
    data = np.zeros((n_samples, n_nodes))
    for node in topo_order:
        parents = np.where(adj_matrix[:, node] == 1)[0]
        if len(parents) > 0:
            data[:, node] = data[:, parents] @ weights[parents, node]
        data[:, node] += np.random.normal(0, noise_std, n_samples)
    
    return data.astype(np.float32)


def generate_linear_nongaussian_data(
    adj_matrix: np.ndarray,
    n_samples: int,
    weight_range: Tuple[float, float] = (0.5, 2.0),
    seed: int = None
) -> np.ndarray:
    """Generate data from linear non-Gaussian model (using Gumbel noise)."""
    if seed is not None:
        np.random.seed(seed)
    
    n_nodes = adj_matrix.shape[0]
    
    # Generate random weights
    weights = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i, j] == 1:
                weight = np.random.uniform(*weight_range)
                if np.random.binomial(1, 0.5):
                    weight = -weight
                weights[i, j] = weight
    
    # Generate data in topological order
    topo_order = list(nx.topological_sort(nx.DiGraph(adj_matrix)))
    
    data = np.zeros((n_samples, n_nodes))
    for node in topo_order:
        parents = np.where(adj_matrix[:, node] == 1)[0]
        if len(parents) > 0:
            data[:, node] = data[:, parents] @ weights[parents, node]
        # Gumbel noise
        data[:, node] += np.random.gumbel(0, 1, n_samples)
    
    return data.astype(np.float32)


def generate_nonlinear_data(
    adj_matrix: np.ndarray,
    n_samples: int,
    seed: int = None
) -> np.ndarray:
    """Generate data from nonlinear model (sine and quadratic functions)."""
    if seed is not None:
        np.random.seed(seed)
    
    n_nodes = adj_matrix.shape[0]
    topo_order = list(nx.topological_sort(nx.DiGraph(adj_matrix)))
    
    data = np.zeros((n_samples, n_nodes))
    for node in topo_order:
        parents = np.where(adj_matrix[:, node] == 1)[0]
        if len(parents) > 0:
            # Mix of sine and quadratic functions
            for i, parent in enumerate(parents):
                if i % 2 == 0:
                    data[:, node] += np.sin(data[:, parent])
                else:
                    data[:, node] += 0.5 * data[:, parent] ** 2
        data[:, node] += np.random.normal(0, 1, n_samples)
    
    return data.astype(np.float32)


def generate_anm_data(
    adj_matrix: np.ndarray,
    n_samples: int,
    seed: int = None
) -> np.ndarray:
    """Generate data from Additive Noise Model with nonlinear functions."""
    if seed is not None:
        np.random.seed(seed)
    
    n_nodes = adj_matrix.shape[0]
    topo_order = list(nx.topological_sort(nx.DiGraph(adj_matrix)))
    
    data = np.zeros((n_samples, n_nodes))
    for node in topo_order:
        parents = np.where(adj_matrix[:, node] == 1)[0]
        if len(parents) > 0:
            # Nonlinear function of parents
            parent_sum = np.sum(data[:, parents], axis=1)
            data[:, node] = np.tanh(parent_sum) + 0.5 * parent_sum
        data[:, node] += np.random.normal(0, 1, n_samples)
    
    return data.astype(np.float32)


def generate_all_datasets(output_dir: str = "data/processed"):
    """Generate all synthetic datasets according to plan."""
    
    # Parameters from plan.json
    n_nodes_list = [10, 20, 30]
    edge_densities = [1.0, 2.0]  # expected edges per node
    graph_types = ['ER', 'SF']
    sample_sizes = [50, 100, 200, 500, 1000]
    mechanisms = ['linear_gaussian', 'linear_nongaussian', 'nonlinear', 'anm']
    n_graphs_per_config = 10  # 10 different graphs per configuration
    n_seeds = 3  # 3 random seeds per graph for data generation
    
    os.makedirs(f"{output_dir}/ground_truth", exist_ok=True)
    os.makedirs(f"{output_dir}/datasets", exist_ok=True)
    
    graph_id = 0
    
    for n_nodes in n_nodes_list:
        for graph_type in graph_types:
            for edge_density in edge_densities:
                n_edges_target = int(n_nodes * edge_density)
                
                for graph_seed in range(1, n_graphs_per_config + 1):
                    graph_id += 1
                    
                    # Generate graph structure
                    if graph_type == 'ER':
                        edge_prob = edge_density / (n_nodes - 1)
                        adj = generate_er_dag(n_nodes, edge_prob, seed=graph_seed * 1000 + graph_id)
                    else:  # SF
                        adj = generate_sf_dag(n_nodes, n_edges_target, seed=graph_seed * 1000 + graph_id)
                    
                    # Save ground truth
                    np.save(f"{output_dir}/ground_truth/graph_{graph_id:04d}.npy", adj)
                    
                    # Generate data for each mechanism and sample size
                    for mechanism in mechanisms:
                        for n_samples in sample_sizes:
                            for data_seed in range(1, n_seeds + 1):
                                seed = graph_seed * 1000 + data_seed * 100 + n_samples
                                
                                if mechanism == 'linear_gaussian':
                                    data = generate_linear_gaussian_data(adj, n_samples, seed=seed)
                                elif mechanism == 'linear_nongaussian':
                                    data = generate_linear_nongaussian_data(adj, n_samples, seed=seed)
                                elif mechanism == 'nonlinear':
                                    data = generate_nonlinear_data(adj, n_samples, seed=seed)
                                else:  # anm
                                    data = generate_anm_data(adj, n_samples, seed=seed)
                                
                                # Save dataset
                                filename = f"{output_dir}/datasets/graph_{graph_id:04d}_{mechanism}_N{n_samples}_seed{data_seed}.npz"
                                np.savez(filename, data=data, adj=adj, 
                                        graph_id=graph_id, mechanism=mechanism,
                                        n_samples=n_samples, seed=data_seed)
                    
                    print(f"Generated graph {graph_id}: {graph_type}, n={n_nodes}, density={edge_density}")
    
    print(f"Total graphs generated: {graph_id}")
    return graph_id


if __name__ == "__main__":
    generate_all_datasets()

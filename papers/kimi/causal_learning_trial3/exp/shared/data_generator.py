"""
Synthetic data generation for causal discovery experiments.
Generates Erdős-Rényi and Barabási-Albert DAGs with linear Gaussian SEM.
"""
import numpy as np
import networkx as nx
from typing import Tuple, Optional
import pickle
import os


def generate_erdos_renyi_dag(n_nodes: int, edge_prob: float, seed: int = 42) -> nx.DiGraph:
    """
    Generate an Erdős-Rényi random DAG.
    
    Args:
        n_nodes: Number of nodes
        edge_prob: Probability of edge between any pair of nodes
        seed: Random seed
        
    Returns:
        A directed acyclic graph
    """
    rng = np.random.RandomState(seed)
    
    # Generate random topological order
    nodes = list(range(n_nodes))
    rng.shuffle(nodes)
    
    # Create DAG by only allowing edges from earlier to later in topological order
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                # Add edge from node[i] to node[j] (respects topological order)
                u, v = nodes[i], nodes[j]
                G.add_edge(u, v)
    
    return G


def generate_barabasi_albert_dag(n_nodes: int, m: int, seed: int = 42) -> nx.DiGraph:
    """
    Generate a scale-free DAG using Barabási-Albert model.
    
    Args:
        n_nodes: Number of nodes
        m: Number of edges to attach from a new node to existing nodes
        seed: Random seed
        
    Returns:
        A directed acyclic graph
    """
    rng = np.random.RandomState(seed)
    
    # Generate undirected BA graph first
    G_undirected = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    
    # Convert to DAG by orienting edges according to random topological order
    nodes = list(range(n_nodes))
    rng.shuffle(nodes)
    order_map = {node: i for i, node in enumerate(nodes)}
    
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    
    for u, v in G_undirected.edges():
        # Orient edge based on topological order
        if order_map[u] < order_map[v]:
            G.add_edge(u, v)
        else:
            G.add_edge(v, u)
    
    return G


def generate_linear_gaussian_data(
    G: nx.DiGraph,
    n_samples: int,
    seed: int = 42,
    weight_range: Tuple[float, float] = (0.3, 1.0),
    noise_std: float = 1.0
) -> np.ndarray:
    """
    Generate data from a linear Gaussian structural equation model.
    
    Args:
        G: DAG structure
        n_samples: Number of samples to generate
        seed: Random seed
        weight_range: Range for edge weights (uniform sampling)
        noise_std: Standard deviation of noise
        
    Returns:
        Data matrix of shape (n_samples, n_nodes)
    """
    rng = np.random.RandomState(seed)
    n_nodes = G.number_of_nodes()
    
    # Get topological order for generation
    topo_order = list(nx.topological_sort(G))
    
    # Generate edge weights
    weights = {}
    for u, v in G.edges():
        weights[(u, v)] = rng.uniform(weight_range[0], weight_range[1]) * (1 if rng.random() > 0.5 else -1)
    
    # Generate data
    data = np.zeros((n_samples, n_nodes))
    noise = rng.normal(0, noise_std, (n_samples, n_nodes))
    
    for node in topo_order:
        parents = list(G.predecessors(node))
        if len(parents) == 0:
            data[:, node] = noise[:, node]
        else:
            parent_contrib = np.zeros(n_samples)
            for parent in parents:
                parent_contrib += weights[(parent, node)] * data[:, parent]
            data[:, node] = parent_contrib + noise[:, node]
    
    return data


def generate_dataset(
    graph_type: str,
    n_nodes: int,
    n_samples: int,
    graph_param: float,  # edge_prob for ER, m for BA
    seed: int = 42,
    return_graph: bool = True
) -> Tuple[np.ndarray, Optional[nx.DiGraph]]:
    """
    Generate a complete dataset with ground truth graph.
    
    Args:
        graph_type: 'er' for Erdős-Rényi or 'ba' for Barabási-Albert
        n_nodes: Number of nodes
        n_samples: Number of samples
        graph_param: Edge probability (ER) or m parameter (BA)
        seed: Random seed
        return_graph: Whether to return the ground truth graph
        
    Returns:
        Tuple of (data, ground_truth_graph)
    """
    if graph_type == 'er':
        G = generate_erdos_renyi_dag(n_nodes, graph_param, seed)
    elif graph_type == 'ba':
        G = generate_barabasi_albert_dag(n_nodes, int(graph_param), seed)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    data = generate_linear_gaussian_data(G, n_samples, seed)
    
    if return_graph:
        return data, G
    return data


def adjacency_to_numpy(G: nx.DiGraph) -> np.ndarray:
    """Convert directed graph to adjacency matrix."""
    n = G.number_of_nodes()
    adj = np.zeros((n, n))
    for u, v in G.edges():
        adj[u, v] = 1
    return adj


def get_graph_stats(G: nx.DiGraph) -> dict:
    """Get statistics about a graph."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    # Average degree
    avg_degree = 2 * m / n if n > 0 else 0
    
    # Max in-degree and out-degree
    if n > 0:
        max_in_degree = max(G.in_degree(node) for node in G.nodes())
        max_out_degree = max(G.out_degree(node) for node in G.nodes())
    else:
        max_in_degree = 0
        max_out_degree = 0
    
    return {
        'n_nodes': n,
        'n_edges': m,
        'avg_degree': avg_degree,
        'max_in_degree': max_in_degree,
        'max_out_degree': max_out_degree,
        'density': nx.density(G)
    }


if __name__ == "__main__":
    # Test data generation
    print("Testing data generation...")
    
    # Test ER graph
    G_er = generate_erdos_renyi_dag(10, 0.3, seed=42)
    print(f"ER graph: {G_er.number_of_nodes()} nodes, {G_er.number_of_edges()} edges")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(G_er)}")
    
    # Test BA graph
    G_ba = generate_barabasi_albert_dag(10, 2, seed=42)
    print(f"BA graph: {G_ba.number_of_nodes()} nodes, {G_ba.number_of_edges()} edges")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(G_ba)}")
    
    # Test data generation
    data, G = generate_dataset('er', 20, 1000, 0.2, seed=42)
    print(f"Data shape: {data.shape}")
    print(f"Graph edges: {G.number_of_edges()}")
    print(f"Graph stats: {get_graph_stats(G)}")

"""
Phase 2: Structural Constraint Extraction
Extracts structural parameters (edge-connectivity, feedback vertex set) for FPT guarantees.
"""
import numpy as np
import networkx as nx


def compute_edge_connectivity(skeleton: np.ndarray) -> int:
    """
    Compute maximum edge-connectivity of the skeleton graph.
    
    Edge-connectivity is the minimum number of edges to remove to disconnect
    the graph. Lower connectivity implies simpler structure.
    
    Args:
        skeleton: Undirected adjacency matrix
        
    Returns:
        Maximum edge-connectivity
    """
    G = nx.Graph(skeleton)
    
    if not nx.is_connected(G):
        # Return connectivity of largest component
        components = list(nx.connected_components(G))
        max_conn = 0
        for comp in components:
            if len(comp) > 1:
                subgraph = G.subgraph(comp)
                try:
                    conn = nx.edge_connectivity(subgraph)
                    max_conn = max(max_conn, conn)
                except:
                    pass
        return max_conn
    
    try:
        return nx.edge_connectivity(G)
    except:
        return 1


def compute_feedback_vertex_set(skeleton: np.ndarray, max_iterations: int = 100) -> tuple:
    """
    Approximate minimum feedback vertex set (FVS) using greedy algorithm.
    
    FVS is a set of nodes whose removal makes the graph acyclic.
    Smaller FVS implies more tree-like structure.
    
    Args:
        skeleton: Undirected adjacency matrix
        max_iterations: Maximum iterations for greedy algorithm
        
    Returns:
        (fvs_nodes, fvs_size)
    """
    G = nx.Graph(skeleton)
    
    fvs = set()
    remaining = set(G.nodes())
    
    iterations = 0
    while remaining and iterations < max_iterations:
        iterations += 1
        
        # Find cycles in remaining subgraph
        subgraph = G.subgraph(remaining)
        
        try:
            cycles = list(nx.cycle_basis(subgraph))
        except:
            break
        
        if not cycles:
            break
        
        # Greedy: remove node that breaks most cycles
        cycle_count = {node: 0 for node in remaining}
        for cycle in cycles:
            for node in cycle:
                if node in cycle_count:
                    cycle_count[node] += 1
        
        if not cycle_count:
            break
        
        # Remove node with highest cycle participation
        node_to_remove = max(cycle_count, key=cycle_count.get)
        fvs.add(node_to_remove)
        remaining.remove(node_to_remove)
    
    return list(fvs), len(fvs)


def extract_structural_constraints(skeleton: np.ndarray) -> dict:
    """
    Extract structural constraints from skeleton.
    
    Returns a dictionary with:
    - edge_connectivity: Maximum edge-connectivity
    - fvs_nodes: Feedback vertex set nodes
    - fvs_size: Size of FVS
    - modular_components: 2-connected components
    - topological_hints: Partial ordering from FVS
    
    Args:
        skeleton: Undirected adjacency matrix
        
    Returns:
        Dictionary of structural constraints
    """
    G = nx.Graph(skeleton)
    n_nodes = skeleton.shape[0]
    
    # Compute edge-connectivity
    connectivity = compute_edge_connectivity(skeleton)
    
    # Compute feedback vertex set
    fvs_nodes, fvs_size = compute_feedback_vertex_set(skeleton)
    
    # Find 2-connected components (biconnected components)
    try:
        bicomponents = list(nx.biconnected_components(G))
    except:
        bicomponents = []
    
    # Create topological hints from FVS
    # Nodes in FVS are likely to be "earlier" in causal ordering
    topological_hints = {
        'early_nodes': set(fvs_nodes),  # Likely ancestors
        'late_nodes': set(range(n_nodes)) - set(fvs_nodes)  # Likely descendants
    }
    
    constraints = {
        'edge_connectivity': connectivity,
        'fvs_nodes': fvs_nodes,
        'fvs_size': fvs_size,
        'fvs_ratio': fvs_size / n_nodes if n_nodes > 0 else 0,
        'num_bicomponents': len(bicomponents),
        'bicomponent_sizes': [len(c) for c in bicomponents],
        'topological_hints': topological_hints
    }
    
    return constraints


def create_constraint_penalty_matrix(constraints: dict, n_nodes: int) -> np.ndarray:
    """
    Create penalty matrix for structural constraint term in optimization.
    
    Penalizes edges that violate topological hints from FVS analysis.
    
    Args:
        constraints: Structural constraints dictionary
        n_nodes: Number of nodes
        
    Returns:
        Penalty matrix C where C[i,j] is penalty for edge i->j
    """
    C = np.zeros((n_nodes, n_nodes))
    
    topological_hints = constraints.get('topological_hints', {})
    early_nodes = topological_hints.get('early_nodes', set())
    late_nodes = topological_hints.get('late_nodes', set())
    
    # Penalize edges from late nodes to early nodes
    # (These are less likely in a causal ordering)
    for i in late_nodes:
        for j in early_nodes:
            if i != j:
                C[i, j] = 1.0
    
    # Scale by FVS ratio (smaller FVS = stronger constraints)
    fvs_ratio = constraints.get('fvs_ratio', 1.0)
    if fvs_ratio < 0.3:  # Strong structural constraints
        C *= 2.0
    elif fvs_ratio < 0.5:  # Moderate constraints
        C *= 1.0
    else:  # Weak constraints
        C *= 0.5
    
    return C

"""
Generate n=50 datasets for scalability testing.
Addresses self-review feedback about missing n=50 experiments.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'exp'))

import numpy as np
from shared.data_generator import (
    generate_er_dag, generate_sf_dag,
    generate_linear_gaussian_data, generate_linear_nongaussian_data,
    generate_nonlinear_data, generate_anm_data
)


def generate_n50_datasets():
    """Generate n=50 datasets for actual scalability testing."""
    
    output_dir = os.path.join(PROJECT_ROOT, "data/processed")
    os.makedirs(f"{output_dir}/ground_truth", exist_ok=True)
    os.makedirs(f"{output_dir}/datasets", exist_ok=True)
    
    n_nodes = 50
    edge_densities = [1.0, 2.0]
    graph_types = ['ER', 'SF']
    sample_sizes = [50, 100, 200, 500, 1000]
    mechanisms = ['linear_gaussian', 'linear_nongaussian', 'nonlinear', 'anm']
    n_graphs_per_config = 10  # 10 graphs per configuration
    n_seeds = 3
    
    # Start IDs from 101 to avoid overlap with existing graphs (1-60 for n=10,20,30)
    start_id = 101
    graph_id = start_id - 1
    
    for graph_type in graph_types:
        for edge_density in edge_densities:
            n_edges_target = int(n_nodes * edge_density)
            
            for graph_seed in range(1, n_graphs_per_config + 1):
                graph_id += 1
                
                # Generate graph structure
                if graph_type == 'ER':
                    edge_prob = edge_density / (n_nodes - 1)
                    adj = generate_er_dag(n_nodes, edge_prob, seed=graph_seed * 10000 + graph_id)
                else:  # SF
                    adj = generate_sf_dag(n_nodes, n_edges_target, seed=graph_seed * 10000 + graph_id)
                
                # Save ground truth
                np.save(f"{output_dir}/ground_truth/graph_{graph_id:04d}.npy", adj)
                
                # Generate data for each mechanism and sample size
                for mechanism in mechanisms:
                    for n_samples in sample_sizes:
                        for data_seed in range(1, n_seeds + 1):
                            seed = graph_seed * 10000 + data_seed * 1000 + n_samples
                            
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
                
                print(f"Generated graph {graph_id}: {graph_type}, n={n_nodes}, density={edge_density}, edges={int(adj.sum())}")
    
    print(f"\nTotal n=50 graphs generated: {graph_id - start_id + 1}")
    return graph_id - start_id + 1


if __name__ == "__main__":
    print("Generating n=50 datasets for scalability testing...")
    count = generate_n50_datasets()
    print(f"Done! Generated {count} n=50 graphs.")

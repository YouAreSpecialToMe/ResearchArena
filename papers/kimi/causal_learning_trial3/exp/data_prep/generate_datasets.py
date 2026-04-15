"""
Generate all synthetic benchmark datasets for experiments.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import pickle
from tqdm import tqdm
from shared.data_generator import generate_dataset, get_graph_stats, adjacency_to_numpy


def generate_all_datasets(output_dir: str = "data/synthetic"):
    """Generate all synthetic datasets according to plan."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration from plan.json
    graph_types = ['er', 'ba']
    n_nodes_list = [20, 50, 100]
    edge_probs = {
        'er': [0.1, 0.2, 0.3],
        'ba': [1, 2, 3]  # m parameter for BA
    }
    n_samples_list = [500, 1000, 2000]
    seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
    
    manifest = {
        'datasets': [],
        'configurations': {
            'graph_types': graph_types,
            'n_nodes': n_nodes_list,
            'edge_probs': edge_probs,
            'n_samples': n_samples_list,
            'n_seeds': len(seeds)
        }
    }
    
    total_datasets = len(graph_types) * len(n_nodes_list) * (
        len(edge_probs['er']) + len(edge_probs['ba'])
    ) * len(n_samples_list) * len(seeds)
    
    print(f"Generating {total_datasets} datasets...")
    
    pbar = tqdm(total=total_datasets)
    
    for graph_type in graph_types:
        for n_nodes in n_nodes_list:
            for edge_param in edge_probs[graph_type]:
                for n_samples in n_samples_list:
                    for seed in seeds:
                        # Generate dataset
                        data, G = generate_dataset(
                            graph_type=graph_type,
                            n_nodes=n_nodes,
                            n_samples=n_samples,
                            graph_param=edge_param,
                            seed=seed
                        )
                        
                        # Get stats
                        stats = get_graph_stats(G)
                        
                        # Save dataset
                        dataset_name = f"{graph_type}_p{n_nodes}_e{edge_param}_n{n_samples}_s{seed}"
                        dataset_path = os.path.join(output_dir, f"{dataset_name}.pkl")
                        
                        with open(dataset_path, 'wb') as f:
                            pickle.dump({
                                'data': data,
                                'graph': G,
                                'adjacency': adjacency_to_numpy(G),
                                'stats': stats,
                                'config': {
                                    'graph_type': graph_type,
                                    'n_nodes': n_nodes,
                                    'edge_param': edge_param,
                                    'n_samples': n_samples,
                                    'seed': seed
                                }
                            }, f)
                        
                        # Add to manifest
                        manifest['datasets'].append({
                            'name': dataset_name,
                            'path': dataset_path,
                            'config': {
                                'graph_type': graph_type,
                                'n_nodes': n_nodes,
                                'edge_param': edge_param,
                                'n_samples': n_samples,
                                'seed': seed
                            },
                            'stats': stats
                        })
                        
                        pbar.update(1)
    
    pbar.close()
    
    # Save manifest
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Generated {len(manifest['datasets'])} datasets")
    print(f"Manifest saved to {manifest_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    for n_nodes in n_nodes_list:
        count = sum(1 for d in manifest['datasets'] if d['config']['n_nodes'] == n_nodes)
        print(f"  {n_nodes} nodes: {count} datasets")


if __name__ == "__main__":
    generate_all_datasets()

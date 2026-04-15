"""
Data loading and generation utilities for AIT-LCD experiments.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling


NETWORKS = ['asia', 'child', 'insurance', 'alarm', 'hailfinder']
SAMPLE_SIZES = [100, 200, 500, 1000]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def get_network_info(network_name):
    """Load a Bayesian network and extract its structure."""
    model = get_example_model(network_name)
    
    nodes = list(model.nodes())
    edges = list(model.edges())
    
    # For each node, compute its PC set (parents + children)
    pc_sets = {}
    mb_sets = {}
    parents = {}
    children = {}
    
    for node in nodes:
        # Parents
        node_parents = list(model.get_parents(node))
        parents[node] = node_parents
        
        # Children
        node_children = list(model.get_children(node))
        children[node] = node_children
        
        # PC set = parents + children
        pc_set = node_parents + node_children
        pc_sets[node] = pc_set
        
        # MB set = PC + spouses (parents of children, excluding self)
        spouses = []
        for child in node_children:
            spouses.extend(model.get_parents(child))
        spouses = list(set(spouses) - {node})
        
        mb_set = pc_set + spouses
        mb_sets[node] = mb_set
    
    return {
        'name': network_name,
        'nodes': nodes,
        'edges': edges,
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'pc_sets': pc_sets,
        'mb_sets': mb_sets,
        'parents': parents,
        'children': children
    }


def generate_dataset(network_name, n_samples, seed, data_dir='data'):
    """Generate a synthetic dataset from a Bayesian network."""
    np.random.seed(seed)
    
    model = get_example_model(network_name)
    sampler = BayesianModelSampling(model)
    
    # Generate samples
    data = sampler.forward_sample(size=n_samples, show_progress=False)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Convert all columns to integer codes (for discrete/categorical data)
    for col in df.columns:
        # Convert to categorical then to codes
        df[col] = pd.Categorical(df[col]).codes
    
    return df


def save_dataset(df, network_name, n_samples, seed, data_dir='data'):
    """Save dataset to CSV file."""
    path = Path(data_dir) / network_name / f'n{n_samples}'
    path.mkdir(parents=True, exist_ok=True)
    
    filepath = path / f'seed{seed}.csv'
    df.to_csv(filepath, index=False)
    return filepath


def _get_project_root():
    """Get the project root directory."""
    # Try to find the project root by looking for data/ directory
    current = Path.cwd()
    
    # Check if we're in the exp/ subdirectory
    if current.name == 'exp':
        return current.parent
    
    # Check if current directory has data/
    if (current / 'data').exists():
        return current
    
    # Try parent
    if (current.parent / 'data').exists():
        return current.parent
    
    return current


def load_dataset(network_name, n_samples, seed, data_dir=None):
    """Load a dataset - generate on-the-fly if not cached."""
    if data_dir is None:
        data_dir = _get_project_root() / 'data'
    else:
        data_dir = Path(data_dir)
    
    filepath = data_dir / network_name / f'n{n_samples}' / f'seed{seed}.csv'
    
    # Check if cached
    if filepath.exists():
        return pd.read_csv(filepath)
    
    # Generate on-the-fly
    df = generate_dataset(network_name, n_samples, seed, data_dir)
    
    # Cache for future use
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    
    return df


def save_ground_truth(network_info, data_dir=None):
    """Save ground truth structure to JSON."""
    if data_dir is None:
        data_dir = _get_project_root() / 'data'
    else:
        data_dir = Path(data_dir)
    
    path = data_dir / network_info['name']
    path.mkdir(parents=True, exist_ok=True)
    
    filepath = path / 'ground_truth.json'
    with open(filepath, 'w') as f:
        json.dump(network_info, f, indent=2)
    return filepath


def load_ground_truth(network_name, data_dir=None):
    """Load ground truth structure from JSON."""
    if data_dir is None:
        data_dir = _get_project_root() / 'data'
    else:
        data_dir = Path(data_dir)
    
    filepath = data_dir / network_name / 'ground_truth.json'
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_all_datasets(networks=None, sample_sizes=None, seeds=None, data_dir='data'):
    """Generate all datasets for the experiments."""
    networks = networks or NETWORKS
    sample_sizes = sample_sizes or SAMPLE_SIZES
    seeds = seeds or SEEDS
    
    results = []
    
    for network_name in networks:
        print(f"Processing network: {network_name}")
        
        # Get network info and save ground truth
        info = get_network_info(network_name)
        save_ground_truth(info, data_dir)
        
        for n_samples in sample_sizes:
            for seed in seeds:
                df = generate_dataset(network_name, n_samples, seed, data_dir)
                filepath = save_dataset(df, network_name, n_samples, seed, data_dir)
                results.append({
                    'network': network_name,
                    'n_samples': n_samples,
                    'seed': seed,
                    'filepath': str(filepath),
                    'shape': df.shape
                })
                print(f"  Generated: {filepath}")
    
    return results


if __name__ == '__main__':
    # Generate all datasets
    results = generate_all_datasets()
    print(f"\nGenerated {len(results)} datasets")

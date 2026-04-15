#!/usr/bin/env python3
"""
Scalability experiments on synthetic datasets.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import json
import time
from shared.cleanbp import CleanBP


def run_scalability(size, violation_rate, seed=42):
    """Run scalability experiment."""
    
    dataset_path = f'data/synthetic/syn_{size}_{int(violation_rate*100)}pct.csv'
    
    # Load data
    df = pd.read_csv(dataset_path)
    
    # Load stats
    with open(f'data/synthetic/syn_{size}_{int(violation_rate*100)}pct_stats.json') as f:
        stats = json.load(f)
    
    print(f"Dataset: {size} tuples, {violation_rate*100}% violation rate")
    print(f"  Total violations: {stats['total_violations']}")
    
    # Define FDs for synthetic data
    fds = [
        (['A'], ['B']),
        (['C'], ['D'])
    ]
    
    # Run CleanBP
    cleanbp = CleanBP(max_iterations=30, convergence_threshold=1e-6, damping=0.5)
    
    start_time = time.time()
    _, info = cleanbp.repair(
        df, fds,
        violation_only=True,
        separate_attributes=True,
        priority_scheduling=True
    )
    elapsed = time.time() - start_time
    
    print(f"  Time: {elapsed:.2f}s, Iterations: {info.get('iterations', 0)}")
    print(f"  Graph: {info['graph_stats']['n_cells']} cells, {info['graph_stats']['n_violations']} violations")
    
    # Save results
    result = {
        'size': size,
        'violation_rate': violation_rate,
        'n_tuples': len(df),
        'n_violations': stats['total_violations'],
        'runtime_seconds': elapsed,
        'iterations': info.get('iterations', 0),
        'graph_stats': info['graph_stats']
    }
    
    output_path = f'results/scalability_{size}_{int(violation_rate*100)}pct.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  Results saved to {output_path}")
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--violation-rate', type=float, required=True)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Running scalability experiment")
    print(f"{'='*60}")
    
    run_scalability(args.size, args.violation_rate)

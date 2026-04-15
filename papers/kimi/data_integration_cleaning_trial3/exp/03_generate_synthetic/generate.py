#!/usr/bin/env python3
"""
Generate synthetic datasets for scalability testing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
import argparse
from shared.data_loader import compute_violation_stats


def generate_synthetic_dataset(n_tuples: int, violation_rate: float, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset with FDs.
    
    Schema: A, B, C, D, E where A -> B, C -> D
    """
    np.random.seed(seed)
    
    # Domain sizes
    n_a_values = max(10, n_tuples // 100)
    n_c_values = max(10, n_tuples // 100)
    
    # Generate A (LHS for first FD)
    A = np.random.choice([f'a{i}' for i in range(n_a_values)], n_tuples)
    
    # Generate C (LHS for second FD)
    C = np.random.choice([f'c{i}' for i in range(n_c_values)], n_tuples)
    
    # Generate B based on A (A -> B)
    b_map = {f'a{i}': f'b{np.random.randint(100)}' for i in range(n_a_values)}
    B = [b_map[a] for a in A]
    
    # Generate D based on C (C -> D)
    d_map = {f'c{i}': f'd{np.random.randint(100)}' for i in range(n_c_values)}
    D = [d_map[c] for c in C]
    
    # Generate E (independent)
    E = np.random.choice([f'e{i}' for i in range(50)], n_tuples)
    
    df = pd.DataFrame({
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'E': E
    })
    
    # Now inject violations to achieve target violation rate
    fds = [
        (['A'], ['B']),
        (['C'], ['D'])
    ]
    
    # Count current violations
    current_violations = 0
    for lhs, rhs in fds:
        groups = df.groupby(lhs, sort=False)
        for group_key, group_df in groups:
            if len(group_df) <= 1:
                continue
            for r_attr in rhs:
                unique_vals = group_df[r_attr].nunique()
                if unique_vals > 1:
                    current_violations += (len(group_df) * (unique_vals - 1))
    
    # Target violations
    target_violations = int(n_tuples * violation_rate)
    
    # Inject more violations if needed
    if current_violations < target_violations:
        # Corrupt some tuples
        n_to_corrupt = min(target_violations - current_violations, n_tuples // 2)
        indices_to_corrupt = np.random.choice(df.index, n_to_corrupt, replace=False)
        
        for idx in indices_to_corrupt:
            # Randomly choose which FD to violate
            if np.random.rand() < 0.5:
                # Violate A -> B
                current_b = df.at[idx, 'B']
                df.at[idx, 'B'] = f'b{np.random.randint(100, 200)}'
            else:
                # Violate C -> D
                current_d = df.at[idx, 'D']
                df.at[idx, 'D'] = f'd{np.random.randint(100, 200)}'
    
    return df


def main():
    # Generate datasets of different sizes
    sizes = [1000, 5000, 10000, 50000, 100000]
    violation_rates = [0.01, 0.05, 0.10]
    
    for size in sizes:
        for rate in violation_rates:
            print(f"Generating synthetic dataset: {size} tuples, {rate*100}% violation rate...")
            
            df = generate_synthetic_dataset(size, rate, seed=42)
            
            # Compute stats
            fds = [(['A'], ['B']), (['C'], ['D'])]
            stats = compute_violation_stats(df, fds)
            
            print(f"  Total violations: {stats['total_violations']}")
            
            # Save
            filename = f'data/synthetic/syn_{size}_{int(rate*100)}pct.csv'
            df.to_csv(filename, index=False)
            
            # Save stats
            with open(f'data/synthetic/syn_{size}_{int(rate*100)}pct_stats.json', 'w') as f:
                json.dump(stats, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else x)
    
    print("\nDone! Synthetic datasets generated.")


if __name__ == '__main__':
    main()

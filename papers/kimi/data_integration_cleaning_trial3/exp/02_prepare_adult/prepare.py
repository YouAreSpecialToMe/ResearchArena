#!/usr/bin/env python3
"""
Prepare Adult Census dataset for experiments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
from shared.data_loader import load_adult_dataset, create_dirty_dataset, compute_violation_stats

# FDs for Adult dataset (approximate)
ADULT_FDS = [
    (['education'], ['education-num']),
    (['relationship', 'sex'], ['marital-status']),
]

def main():
    print("Loading Adult Census dataset...")
    df = load_adult_dataset('data/adult/adult.data')
    print(f"Loaded {len(df)} records")
    
    # Take a subset for faster experiments
    df = df.sample(n=min(10000, len(df)), random_state=42).reset_index(drop=True)
    print(f"Using subset of {len(df)} records")
    
    # Clean version: resolve FD violations
    print("Creating clean version...")
    clean_df = df.copy()
    
    for lhs, rhs in ADULT_FDS:
        groups = clean_df.groupby(lhs, sort=False)
        for group_key, group_df in groups:
            if len(group_df) <= 1:
                continue
            for r_attr in rhs:
                mode_val = group_df[r_attr].mode()
                if len(mode_val) > 0:
                    clean_df.loc[group_df.index, r_attr] = mode_val[0]
    
    # Compute stats on clean data
    print("Computing violation statistics on clean data...")
    clean_stats = compute_violation_stats(clean_df, ADULT_FDS)
    print(f"Clean data violations: {clean_stats['total_violations']}")
    
    # Create dirty version with 10% error rate
    print("\nCreating dirty version with 10% error rate...")
    dirty_df, error_cells = create_dirty_dataset(clean_df, ADULT_FDS, 0.10, seed=42)
    
    # Compute stats
    dirty_stats = compute_violation_stats(dirty_df, ADULT_FDS)
    
    print(f"  Total errors injected: {len(error_cells)}")
    print(f"  Total violations: {dirty_stats['total_violations']}")
    
    # Save
    clean_df.to_csv('data/adult/adult_clean.csv', index=False)
    dirty_df.to_csv('data/adult/adult_dirty.csv', index=False)
    
    # Save error cells
    with open('data/adult/adult_errors.json', 'w') as f:
        json.dump({
            'error_cells': [[int(x), str(y)] for x, y in error_cells],
            'n_errors': len(error_cells),
            'violation_stats': dirty_stats
        }, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else x)
    
    # Save FDs
    with open('data/adult/fds.json', 'w') as f:
        json.dump([{'lhs': lhs, 'rhs': rhs} for lhs, rhs in ADULT_FDS], f, indent=2)
    
    print("\nDone! Files created:")
    print("  - data/adult/adult_clean.csv")
    print("  - data/adult/adult_dirty.csv")
    print("  - data/adult/fds.json")

if __name__ == '__main__':
    main()

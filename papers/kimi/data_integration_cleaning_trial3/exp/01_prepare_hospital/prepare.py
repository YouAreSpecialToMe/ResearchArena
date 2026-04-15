#!/usr/bin/env python3
"""
Prepare Hospital dataset for experiments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
from shared.data_loader import load_hospital_dataset, create_dirty_dataset, compute_violation_stats

# FDs for Hospital dataset
HOSPITAL_FDS = [
    (['ProviderNumber'], ['HospitalName']),
    (['ProviderNumber'], ['Address1']),
    (['ProviderNumber'], ['City']),
    (['ProviderNumber'], ['State']),
    (['ProviderNumber'], ['ZipCode']),
    (['ZipCode'], ['City']),
    (['ZipCode'], ['State']),
    (['HospitalName'], ['PhoneNumber']),
    (['HospitalName', 'MeasureCode'], ['Score']),
]

def main():
    print("Loading Hospital dataset...")
    df = load_hospital_dataset('data/hospital/hospital.csv')
    print(f"Loaded {len(df)} records")
    
    # Clean version: take most frequent value for each LHS group
    print("Creating clean version...")
    clean_df = df.copy()
    
    for lhs, rhs in HOSPITAL_FDS:
        groups = clean_df.groupby(lhs, sort=False)
        for group_key, group_df in groups:
            if len(group_df) <= 1:
                continue
            # Take most frequent value for each RHS attribute
            for r_attr in rhs:
                mode_val = group_df[r_attr].mode()
                if len(mode_val) > 0:
                    clean_df.loc[group_df.index, r_attr] = mode_val[0]
    
    # Compute stats on clean data
    print("Computing violation statistics on clean data...")
    clean_stats = compute_violation_stats(clean_df, HOSPITAL_FDS)
    print(f"Clean data violations: {clean_stats['total_violations']}")
    
    # Create dirty versions with different error rates
    for error_rate in [0.05, 0.10, 0.15]:
        print(f"\nCreating dirty version with {error_rate*100}% error rate...")
        dirty_df, error_cells = create_dirty_dataset(clean_df, HOSPITAL_FDS, error_rate, seed=42)
        
        # Compute stats
        dirty_stats = compute_violation_stats(dirty_df, HOSPITAL_FDS)
        
        print(f"  Total errors injected: {len(error_cells)}")
        print(f"  Total violations: {dirty_stats['total_violations']}")
        
        # Save
        dirty_df.to_csv(f'data/hospital/hospital_dirty_{int(error_rate*100)}pct.csv', index=False)
        
        # Save error cells
        with open(f'data/hospital/hospital_errors_{int(error_rate*100)}pct.json', 'w') as f:
            json.dump({
                'error_cells': [[int(x), str(y)] for x, y in error_cells],
                'n_errors': len(error_cells),
                'violation_stats': dirty_stats
            }, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else x)
    
    # Save clean version
    clean_df.to_csv('data/hospital/hospital_clean.csv', index=False)
    
    # Save FDs
    with open('data/hospital/fds.json', 'w') as f:
        json.dump([{'lhs': lhs, 'rhs': rhs} for lhs, rhs in HOSPITAL_FDS], f, indent=2)
    
    print("\nDone! Files created:")
    print("  - data/hospital/hospital_clean.csv")
    print("  - data/hospital/hospital_dirty_5pct.csv")
    print("  - data/hospital/hospital_dirty_10pct.csv")
    print("  - data/hospital/hospital_dirty_15pct.csv")
    print("  - data/hospital/fds.json")

if __name__ == '__main__':
    main()

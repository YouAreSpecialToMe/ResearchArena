#!/usr/bin/env python3
"""
Data preparation script for CDHR experiments.
Downloads and preprocesses all datasets.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import sys
from shared.data_loader import prepare_datasets


def main():
    print("=" * 60)
    print("CDHR Data Preparation")
    print("=" * 60)
    
    # Prepare all datasets
    datasets = prepare_datasets(data_dir="data")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    for name, data in datasets.items():
        print(f"{name}: {len(data)} problems")
    
    print("\nData preparation complete!")
    
    # Create a manifest file
    manifest = {
        "datasets": {
            name: {
                "path": f"data/{name}.json",
                "num_problems": len(data)
            }
            for name, data in datasets.items()
        }
    }
    
    with open("data/manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("Manifest saved to data/manifest.json")


if __name__ == "__main__":
    main()

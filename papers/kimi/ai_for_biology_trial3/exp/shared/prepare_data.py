"""
Prepare CycPeptMPDB dataset for training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_and_preprocess_data

if __name__ == '__main__':
    print("Preparing CycPeptMPDB dataset...")
    data = load_and_preprocess_data()
    print("Data preparation complete!")
    print(f"Metadata: {data['metadata']}")

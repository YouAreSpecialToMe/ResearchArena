"""Download and preprocess CLEAN benchmark data for enzyme function prediction."""
import os
import json
import csv
import subprocess
import sys
from collections import Counter, defaultdict

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def download_clean_data():
    """Download CLEAN dataset from official repository."""
    repo_dir = os.path.join(DATA_DIR, "CLEAN_repo")
    if not os.path.exists(repo_dir):
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/tttianhao/CLEAN.git", repo_dir],
            check=True
        )
    return repo_dir

def parse_fasta(filepath):
    """Parse FASTA file, return list of (id, sequence) tuples."""
    sequences = []
    current_id = None
    current_seq = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        sequences.append((current_id, "".join(current_seq)))
    return sequences

def parse_ec_levels(ec_full):
    """Parse a full EC number into 4 hierarchical levels."""
    parts = ec_full.split(".")
    if len(parts) != 4:
        return None
    return {
        "ec_full": ec_full,
        "ec_l1": parts[0],
        "ec_l2": f"{parts[0]}.{parts[1]}",
        "ec_l3": f"{parts[0]}.{parts[1]}.{parts[2]}",
        "ec_l4": ec_full
    }

def load_clean_splits(repo_dir):
    """Load train, New-392, and Price-149 splits from CLEAN data."""
    # The CLEAN repo has data in app/data/ directory
    data_base = os.path.join(repo_dir, "app", "data")

    # Check what files exist
    if not os.path.exists(data_base):
        # Try alternative paths
        for candidate in ["data", "CLEAN/data", "CLEAN/app/data"]:
            alt = os.path.join(repo_dir, candidate)
            if os.path.exists(alt):
                data_base = alt
                break

    print(f"Data base directory: {data_base}")
    print(f"Contents: {os.listdir(data_base) if os.path.exists(data_base) else 'NOT FOUND'}")

    # Look for the split files
    # CLEAN uses .csv files with columns: Entry, EC number, Sequence
    # and separate test sets
    train_data = []
    test_new392 = []
    test_price149 = []

    # Find all relevant files
    for root, dirs, files in os.walk(repo_dir):
        for f in files:
            fp = os.path.join(root, f)
            if "split100" in fp.lower() or "split_100" in fp.lower():
                print(f"  Found split file: {fp}")
            if "new" in f.lower() or "price" in f.lower() or "test" in f.lower():
                print(f"  Found test file: {fp}")
            if "train" in f.lower():
                print(f"  Found train file: {fp}")

    return data_base

def process_clean_data(repo_dir):
    """Process CLEAN data into our format."""
    data_base = os.path.join(repo_dir, "app", "data")

    # CLEAN stores data as:
    # - split100.csv (train split with 100% sequence identity threshold)
    # - {id}\t{ec_number} mapping files
    # - FASTA sequence files

    # Let's find the actual data structure
    all_files = []
    for root, dirs, files in os.walk(repo_dir):
        for f in files:
            all_files.append(os.path.join(root, f))

    # Look for key files
    csv_files = [f for f in all_files if f.endswith('.csv')]
    tsv_files = [f for f in all_files if f.endswith('.tsv')]
    fasta_files = [f for f in all_files if f.endswith('.fasta') or f.endswith('.fa')]
    txt_files = [f for f in all_files if f.endswith('.txt')]

    print(f"\nFound {len(csv_files)} CSV files, {len(tsv_files)} TSV files, {len(fasta_files)} FASTA files, {len(txt_files)} TXT files")
    for f in csv_files[:20]:
        print(f"  CSV: {f}")
    for f in tsv_files[:10]:
        print(f"  TSV: {f}")
    for f in fasta_files[:10]:
        print(f"  FASTA: {f}")
    for f in txt_files[:20]:
        print(f"  TXT: {f}")

    return all_files

if __name__ == "__main__":
    print("Step 1: Downloading CLEAN data...")
    repo_dir = download_clean_data()

    print("\nStep 2: Exploring data structure...")
    all_files = process_clean_data(repo_dir)

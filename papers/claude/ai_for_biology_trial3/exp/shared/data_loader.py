"""
Data loading and preprocessing utilities for ProteinGym multi-mutant datasets.
"""
import os
import json
import pickle
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ProteinGym GitHub raw URLs
PROTEINGYM_BASE = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main"

# Reference file mapping assay names to metadata
REFERENCE_URL = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"

# Multi-mutant substitution data hosted on ProteinGym
# The actual DMS files are large and hosted on their own server
DMS_DATA_BASE = "https://marks.hms.harvard.edu/proteingym/DMS_ProteinGym_substitutions.zip"

# Key assays we want (from the plan)
TARGET_ASSAYS = [
    "GB1_Olson_2014",       # pairwise, high epistasis
    "GB1_Wu_2016",          # combinatorial 4-site
    "GFP_Sarkisyan_2016",   # fluorescence, many orders
    "HIS7_Pokusaeva_2019",  # growth fitness
    "PABP_Melamed_2013",    # binding
    "UBE4B_Klevit_2013",    # ubiquitination
    "BRCA1_Findlay_2018",   # clinical relevance
    "TEM1_Firnberg_2014",   # enzyme, beta-lactamase
    "GAL4_Segal_2003",      # transcription factor
    "HSP82_Mishra_2016",    # chaperone
    "SUMO1_Bhatt_2020",     # sumoylation
    "CALM1_Bhatt_2022",     # calcium binding
    "DLG4_Bhatt_2024",      # PDZ domain
    "P53_Giacomelli_2018",  # tumor suppressor
    "ENVZ_Ghose_2023",      # kinase
]


def download_file(url, dest_path, desc=None):
    """Download a file with progress bar."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return dest_path
    print(f"Downloading {desc or url}...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f:
        with tqdm(total=total, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
                pbar.update(len(chunk))
    return dest_path


def download_reference_file():
    """Download the ProteinGym reference CSV."""
    dest = RAW_DIR / "DMS_substitutions.csv"
    if dest.exists():
        return pd.read_csv(dest)
    download_file(REFERENCE_URL, dest, "ProteinGym reference")
    return pd.read_csv(dest)


def download_proteingym_data():
    """Download ProteinGym substitution datasets (zip)."""
    import zipfile
    zip_path = RAW_DIR / "DMS_ProteinGym_substitutions.zip"
    extract_dir = RAW_DIR / "DMS_substitutions"
    if extract_dir.exists() and len(list(extract_dir.glob("*.csv"))) > 0:
        return extract_dir
    download_file(DMS_DATA_BASE, zip_path, "ProteinGym DMS data")
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    return extract_dir


def find_assay_files(extract_dir):
    """Find CSV files matching our target assays."""
    all_csvs = list(Path(extract_dir).rglob("*.csv"))
    matched = {}
    for csv_path in all_csvs:
        name = csv_path.stem
        for target in TARGET_ASSAYS:
            if target.lower() in name.lower() or name.lower().startswith(target.split("_")[0].lower()):
                matched[target] = csv_path
    return matched


def parse_mutant_string(mutant_str):
    """Parse a mutant string like 'A42G:L55V' into components."""
    if pd.isna(mutant_str) or mutant_str == "" or mutant_str == "_wt":
        return [], [], []
    mutations = mutant_str.split(":")
    wt_aas, positions, mut_aas = [], [], []
    for m in mutations:
        if len(m) < 3:
            continue
        wt_aas.append(m[0])
        positions.append(int(m[1:-1]))
        mut_aas.append(m[-1])
    return wt_aas, positions, mut_aas


def compute_epistasis_magnitude(df, assay_name):
    """
    Compute epistasis magnitude for multi-mutant variants.
    Epistasis = observed fitness - sum of individual mutation effects.
    """
    # Get single mutants
    singles = df[df['mutation_order'] == 1].copy()
    single_effects = {}
    for _, row in singles.iterrows():
        mut_str = row['mutant']
        single_effects[mut_str] = row['DMS_score']

    # For multi-mutants, compute additive prediction from singles
    epistasis_scores = []
    for _, row in df.iterrows():
        if row['mutation_order'] <= 1:
            epistasis_scores.append(0.0)
            continue
        muts = row['mutant'].split(":")
        additive_pred = sum(single_effects.get(m, 0.0) for m in muts)
        available = sum(1 for m in muts if m in single_effects)
        if available == len(muts):
            epistasis_scores.append(row['DMS_score'] - additive_pred)
        else:
            epistasis_scores.append(np.nan)

    df['epistasis_score'] = epistasis_scores
    median_abs = df['epistasis_score'].abs().median()
    return median_abs


def load_and_preprocess_assay(csv_path, assay_name):
    """Load a DMS assay CSV and preprocess it."""
    df = pd.read_csv(csv_path)

    # Standardize column names
    if 'mutant' not in df.columns and 'mutated_sequence' in df.columns:
        pass  # Will handle differently

    # Parse mutations
    orders = []
    all_sites = []
    all_mut_aas = []
    for _, row in df.iterrows():
        wt, pos, mut = parse_mutant_string(row.get('mutant', ''))
        orders.append(len(pos))
        all_sites.append(pos)
        all_mut_aas.append(mut)

    df['mutation_order'] = orders
    df['mutation_sites'] = all_sites
    df['mutation_aas'] = all_mut_aas

    return df


def create_cv_folds(df, n_folds=5, seed=42):
    """Create stratified cross-validation folds."""
    np.random.seed(seed)
    n = len(df)
    indices = np.random.permutation(n)
    fold_size = n // n_folds
    folds = np.zeros(n, dtype=int)
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        folds[indices[start:end]] = i
    df['fold_id'] = folds
    return df


def prepare_all_datasets():
    """Main function to download and prepare all datasets."""
    print("=" * 60)
    print("Downloading and preparing ProteinGym datasets")
    print("=" * 60)

    # Download reference
    ref_df = download_reference_file()
    print(f"Reference file: {len(ref_df)} assays")

    # Download DMS data
    extract_dir = download_proteingym_data()

    # Find and process target assays
    assay_files = find_assay_files(extract_dir)
    print(f"Found {len(assay_files)} / {len(TARGET_ASSAYS)} target assays")

    # If we couldn't find all targets, use whatever multi-mutant assays we find
    all_csvs = list(Path(extract_dir).rglob("*.csv"))

    processed_assays = {}
    assay_stats = {}

    for csv_path in sorted(all_csvs):
        assay_name = csv_path.stem
        try:
            df = load_and_preprocess_assay(csv_path, assay_name)
        except Exception as e:
            continue

        # Filter to multi-mutant variants (order >= 2) but also keep singles for reference
        n_multi = (df['mutation_order'] >= 2).sum()
        if n_multi < 100:  # Skip assays with too few multi-mutants
            continue

        # Create CV folds
        df = create_cv_folds(df)

        # Compute epistasis
        epi_mag = compute_epistasis_magnitude(df, assay_name)

        # Save
        out_path = PROCESSED_DIR / f"{assay_name}.parquet"
        df.to_parquet(out_path)

        stats = {
            'assay': assay_name,
            'n_variants': len(df),
            'n_multi': n_multi,
            'n_singles': (df['mutation_order'] == 1).sum(),
            'max_order': df['mutation_order'].max(),
            'n_unique_sites': len(set(s for sites in df['mutation_sites'] for s in sites)),
            'epistasis_magnitude': float(epi_mag) if not np.isnan(epi_mag) else None,
            'fitness_mean': float(df['DMS_score'].mean()),
            'fitness_std': float(df['DMS_score'].std()),
        }
        processed_assays[assay_name] = df
        assay_stats[assay_name] = stats
        print(f"  {assay_name}: {stats['n_variants']} variants, {stats['n_multi']} multi-mutants, order up to {stats['max_order']}")

    # Save stats
    with open(PROCESSED_DIR / "assay_stats.json", 'w') as f:
        json.dump(assay_stats, f, indent=2)

    # Select final 15 assays (by number of multi-mutants)
    sorted_assays = sorted(assay_stats.items(), key=lambda x: x[1]['n_multi'], reverse=True)
    selected = [name for name, _ in sorted_assays[:15]]

    with open(PROCESSED_DIR / "selected_assays.json", 'w') as f:
        json.dump(selected, f, indent=2)

    print(f"\nSelected {len(selected)} assays for evaluation")
    return processed_assays, assay_stats, selected


if __name__ == "__main__":
    prepare_all_datasets()

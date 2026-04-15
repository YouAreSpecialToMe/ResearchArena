"""Step 1: Download and preprocess ProteinGym multi-mutation DMS data."""
import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.config import *

HF_BASE = "https://huggingface.co/datasets/OATML-Markslab/ProteinGym/resolve/main"
HF_API = "https://huggingface.co/api/datasets/OATML-Markslab/ProteinGym/tree/main"


def list_substitution_files():
    """List all substitution CSV files from HuggingFace."""
    r = requests.get(f"{HF_API}/ProteinGym_substitutions", timeout=30)
    r.raise_for_status()
    return [f['path'] for f in r.json() if f['path'].endswith('.csv')]


def download_file(path, dest_dir):
    """Download a single file from HuggingFace."""
    fname = os.path.basename(path)
    dest = os.path.join(dest_dir, fname)
    if os.path.exists(dest):
        return dest
    url = f"{HF_BASE}/{path}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(dest, 'wb') as f:
        f.write(r.content)
    return dest


def download_proteingym_substitutions():
    """Download ProteinGym DMS substitution benchmark data."""
    raw_dir = os.path.join(DATA_DIR, "raw", "substitutions")
    os.makedirs(raw_dir, exist_ok=True)

    # Check if already downloaded
    existing = list(Path(raw_dir).glob("*.csv"))
    if len(existing) > 50:
        print(f"Already downloaded {len(existing)} assay files")
        return raw_dir

    print("Listing ProteinGym substitution files...")
    files = list_substitution_files()
    print(f"Found {len(files)} CSV files to download")

    # Download in parallel
    def dl(path):
        try:
            download_file(path, raw_dir)
            return True
        except Exception as e:
            print(f"  Failed: {os.path.basename(path)}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(dl, files))

    success = sum(results)
    print(f"Downloaded {success}/{len(files)} files")

    # Also download reference file for WT sequences
    try:
        ref_url = f"{HF_BASE}/ProteinGym_reference_file_substitutions.csv"
        r = requests.get(ref_url, timeout=60)
        r.raise_for_status()
        ref_path = os.path.join(DATA_DIR, "raw", "reference_substitutions.csv")
        with open(ref_path, 'wb') as f:
            f.write(r.content)
        print(f"Downloaded reference file")
    except Exception as e:
        print(f"Reference file download failed: {e}")

    return raw_dir


def parse_mutations(mutant_str):
    """Parse mutation string like 'A123B:C456D' into list of (wt, pos, mut) tuples."""
    if pd.isna(mutant_str) or mutant_str == "":
        return []
    mutations = []
    for m in mutant_str.split(":"):
        m = m.strip()
        if len(m) >= 3:
            wt_aa = m[0]
            mut_aa = m[-1]
            pos = int(m[1:-1])
            mutations.append((wt_aa, pos, mut_aa))
    return mutations


def compute_epistasis(df, fitness_col='fitness'):
    """Compute epistasis = observed - sum(single mutation effects)."""
    singles = df[df['num_mutations'] == 1].copy()

    single_effects = {}
    for _, row in singles.iterrows():
        muts = row['mutations_parsed']
        if len(muts) == 1:
            wt, pos, mut = muts[0]
            key = f"{wt}{pos}{mut}"
            single_effects[key] = row[fitness_col]

    epistasis_scores = []
    additive_from_dms = []
    for _, row in df.iterrows():
        if row['num_mutations'] < 2:
            epistasis_scores.append(0.0)
            additive_from_dms.append(row[fitness_col])
            continue

        additive = 0.0
        all_found = True
        for wt, pos, mut in row['mutations_parsed']:
            key = f"{wt}{pos}{mut}"
            if key in single_effects:
                additive += single_effects[key]
            else:
                all_found = False
                break

        if all_found:
            epistasis_scores.append(row[fitness_col] - additive)
            additive_from_dms.append(additive)
        else:
            epistasis_scores.append(np.nan)
            additive_from_dms.append(np.nan)

    df = df.copy()
    df['epistasis_score'] = epistasis_scores
    df['additive_from_dms'] = additive_from_dms
    return df


def create_cv_splits(df, n_folds=N_FOLDS, seed=42):
    """Create stratified CV splits based on mutation count."""
    from sklearn.model_selection import StratifiedKFold

    strat_labels = df['num_mutations'].clip(upper=4).values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_assignments = np.zeros(len(df), dtype=int)

    for fold_idx, (_, test_idx) in enumerate(skf.split(df, strat_labels)):
        fold_assignments[test_idx] = fold_idx

    df = df.copy()
    df['fold'] = fold_assignments
    return df


def main():
    print("=" * 60)
    print("STEP 1: Download and preprocess ProteinGym data")
    print("=" * 60)

    # Download
    raw_dir = download_proteingym_substitutions()

    # Load reference file for WT sequences
    ref_path = os.path.join(DATA_DIR, "raw", "reference_substitutions.csv")
    ref_df = None
    wt_sequences = {}
    if os.path.exists(ref_path):
        ref_df = pd.read_csv(ref_path)
        print(f"Reference file has {len(ref_df)} entries")
        for _, row in ref_df.iterrows():
            dms_id = row.get('DMS_id', '')
            seq = row.get('target_seq', '')
            if dms_id and seq:
                wt_sequences[dms_id] = seq

    # Process assays
    csv_files = sorted(Path(raw_dir).glob("*.csv"))
    print(f"Processing {len(csv_files)} assays...")

    qualifying = []
    all_stats = []

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if 'mutant' not in df.columns:
            continue

        fitness_col = None
        for col in ['DMS_score', 'fitness', 'score']:
            if col in df.columns:
                fitness_col = col
                break
        if fitness_col is None:
            continue

        df['mutations_parsed'] = df['mutant'].apply(parse_mutations)
        df['num_mutations'] = df['mutations_parsed'].apply(len)
        df = df[df['num_mutations'] > 0].copy()
        df = df.dropna(subset=[fitness_col])

        n_single = int((df['num_mutations'] == 1).sum())
        n_double = int((df['num_mutations'] == 2).sum())
        n_triple_plus = int((df['num_mutations'] >= 3).sum())
        n_multi = n_double + n_triple_plus

        protein_name = csv_path.stem
        all_stats.append({
            'protein_name': protein_name,
            'num_total': len(df), 'num_single': n_single,
            'num_double': n_double, 'num_triple_plus': n_triple_plus,
        })

        if n_multi >= MIN_MULTI_MUTANTS:
            qualifying.append((csv_path, df, protein_name, fitness_col))
            print(f"  QUALIFYING: {protein_name}: {n_multi} multi-mutants "
                  f"({n_double} double, {n_triple_plus} triple+)")

    print(f"\n{len(qualifying)} assays qualify (>= {MIN_MULTI_MUTANTS} multi-mutants)")

    if len(qualifying) == 0:
        print("ERROR: No qualifying assays found!")
        sys.exit(1)

    # Save stats
    pd.DataFrame(all_stats).to_csv(os.path.join(DATA_DIR, "dataset_statistics.csv"), index=False)

    processed_info = {}

    for csv_path, df, protein_name, fitness_col in qualifying:
        print(f"\nProcessing {protein_name}...")
        df['fitness'] = df[fitness_col]

        # Get WT sequence
        wt_seq = wt_sequences.get(protein_name, '')
        if not wt_seq:
            # Try to reconstruct from mutations
            positions = {}
            for _, row in df.iterrows():
                for wt, pos, mut in row['mutations_parsed']:
                    positions[pos] = wt
            if positions:
                max_pos = max(positions.keys())
                seq = ['X'] * (max_pos + 1)
                for pos, aa in positions.items():
                    seq[pos] = aa
                wt_seq = ''.join(seq).lstrip('X')

        print(f"  WT sequence length: {len(wt_seq)}")

        # Compute epistasis
        df = compute_epistasis(df, 'fitness')

        # Keep multi-mutants
        multi_df = df[df['num_mutations'] >= 2].copy()
        n_before = len(multi_df)
        multi_df = multi_df.dropna(subset=['epistasis_score'])
        if n_before > len(multi_df):
            print(f"  Dropped {n_before - len(multi_df)} without epistasis scores")

        if len(multi_df) < MIN_MULTI_MUTANTS:
            print(f"  Skipping: only {len(multi_df)} multi-mutants with epistasis")
            continue

        # CV splits
        multi_df = create_cv_splits(multi_df)

        epi = multi_df['epistasis_score']
        info = {
            'protein_name': protein_name,
            'wt_sequence': wt_seq,
            'seq_length': len(wt_seq),
            'num_multi_mutants': len(multi_df),
            'num_double': int((multi_df['num_mutations'] == 2).sum()),
            'num_triple_plus': int((multi_df['num_mutations'] >= 3).sum()),
            'mean_epistasis': float(epi.mean()),
            'std_epistasis': float(epi.std()),
            'mean_abs_epistasis': float(epi.abs().mean()),
            'fitness_range': float(multi_df['fitness'].max() - multi_df['fitness'].min()),
        }
        processed_info[protein_name] = info

        # Save
        save_df = multi_df[['mutant', 'mutations_parsed', 'num_mutations',
                            'fitness', 'epistasis_score', 'additive_from_dms', 'fold']].copy()
        save_df['mutations_parsed'] = save_df['mutations_parsed'].apply(str)
        out_path = os.path.join(DATA_DIR, "processed", f"{protein_name}.parquet")
        save_df.to_parquet(out_path, index=False)
        print(f"  Saved {len(save_df)} variants to {out_path}")

        # Save singles too
        singles_df = df[df['num_mutations'] == 1].copy()
        if len(singles_df) > 0:
            s_save = singles_df[['mutant', 'mutations_parsed', 'fitness']].copy()
            s_save['mutations_parsed'] = s_save['mutations_parsed'].apply(str)
            s_save.to_parquet(os.path.join(DATA_DIR, "processed", f"{protein_name}_singles.parquet"), index=False)

    # Save processed info
    with open(os.path.join(DATA_DIR, "processed_info.json"), 'w') as f:
        json.dump(processed_info, f, indent=2)

    print("\n" + "=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    for name, info in processed_info.items():
        print(f"  {name}: {info['num_multi_mutants']} multi-mutants, "
              f"seq_len={info['seq_length']}, mean|epi|={info['mean_abs_epistasis']:.4f}")


if __name__ == "__main__":
    main()

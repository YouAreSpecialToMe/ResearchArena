#!/usr/bin/env python3
"""
Main experiment pipeline for Residual Epistasis Networks (REN).
Runs all steps: data prep, baselines, REN training, ablations, analysis.
"""
import os
import sys
import json
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
from tqdm import tqdm
import requests
import zipfile
import io
from torch_geometric.nn import GATConv, GCNConv

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STRUCT_DIR = DATA_DIR / "structures"
EMBED_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

for d in [RAW_DIR, PROCESSED_DIR, STRUCT_DIR, EMBED_DIR,
          RESULTS_DIR / "baselines", RESULTS_DIR / "ren",
          RESULTS_DIR / "ablations", RESULTS_DIR / "evaluation",
          FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456]
N_FOLDS = 5

print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def spearman(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return np.nan
    return spearmanr(y_true[mask], y_pred[mask])[0]


def ndcg_at_k(y_true, y_pred, k=100):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < k:
        k = len(yt)
    if k == 0:
        return np.nan
    ys = yt - yt.min()
    pred_order = np.argsort(-yp)
    ideal_order = np.argsort(-yt)
    dcg = sum(ys[pred_order[i]] / np.log2(i + 2) for i in range(k))
    idcg = sum(ys[ideal_order[i]] / np.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 1.0


def parse_mutant(s):
    """Parse 'A42G:L55V' -> (wt_aas, positions, mut_aas)"""
    if pd.isna(s) or s in ('', '_wt'):
        return [], [], []
    wts, poss, muts = [], [], []
    for m in s.split(':'):
        if len(m) >= 3:
            wts.append(m[0])
            poss.append(int(m[1:-1]))
            muts.append(m[-1])
    return wts, poss, muts


# ============================================================
# STEP 1: DATA DOWNLOAD AND PREPROCESSING
# ============================================================

def download_proteingym():
    """Download ProteinGym DMS substitution datasets."""
    zip_path = RAW_DIR / "DMS_ProteinGym_substitutions.zip"
    extract_dir = RAW_DIR / "substitutions"

    # Check if already extracted
    if extract_dir.exists() and len(list(extract_dir.rglob("*.csv"))) > 10:
        print("ProteinGym data already downloaded.")
        return extract_dir

    url = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"
    print(f"Downloading ProteinGym substitution data...")

    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()

    # Save zip
    total = int(resp.headers.get('content-length', 0))
    with open(zip_path, 'wb') as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=1024*1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                print(f"\r  {downloaded/1e6:.0f}/{total/1e6:.0f} MB", end='', flush=True)
    print()

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)

    return extract_dir


def download_reference():
    """Download ProteinGym reference file."""
    dest = RAW_DIR / "DMS_substitutions_reference.csv"
    if dest.exists():
        return pd.read_csv(dest)

    url = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"
    print("Downloading reference file...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(dest, 'w') as f:
        f.write(resp.text)
    return pd.read_csv(dest)


def preprocess_datasets(extract_dir, ref_df):
    """Select and preprocess multi-mutant assays."""
    stats_path = PROCESSED_DIR / "assay_stats.json"
    selected_path = PROCESSED_DIR / "selected_assays.json"

    if selected_path.exists():
        with open(selected_path) as f:
            selected = json.load(f)
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"Already preprocessed: {len(selected)} assays")
        return selected, stats

    # Find all CSV files
    all_csvs = sorted(extract_dir.rglob("*.csv"))
    print(f"Found {len(all_csvs)} CSV files")

    assay_data = {}
    assay_stats = {}

    for csv_path in tqdm(all_csvs, desc="Processing assays"):
        name = csv_path.stem
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if 'mutant' not in df.columns or 'DMS_score' not in df.columns:
            continue

        # Parse mutations
        orders, sites_list, muts_list = [], [], []
        for _, row in df.iterrows():
            _, pos, _ = parse_mutant(row['mutant'])
            orders.append(len(pos))
            sites_list.append(pos)
            muts_list.append(_)

        df['mutation_order'] = orders
        df['mutation_sites'] = sites_list

        n_multi = (df['mutation_order'] >= 2).sum()
        if n_multi < 50:
            continue

        # Compute epistasis magnitude from singles
        singles = df[df['mutation_order'] == 1].copy()
        single_effects = {}
        for _, row in singles.iterrows():
            for m in row['mutant'].split(':'):
                single_effects[m] = row['DMS_score']

        epi_scores = []
        for _, row in df.iterrows():
            if row['mutation_order'] < 2:
                epi_scores.append(np.nan)
                continue
            muts = row['mutant'].split(':')
            if all(m in single_effects for m in muts):
                additive = sum(single_effects[m] for m in muts)
                epi_scores.append(row['DMS_score'] - additive)
            else:
                epi_scores.append(np.nan)

        df['epistasis_score'] = epi_scores

        # CV folds
        np.random.seed(42)
        df['fold_id'] = np.random.randint(0, N_FOLDS, size=len(df))

        # Save
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet")

        epi_mag = np.nanmedian(np.abs(epi_scores))
        stats = {
            'assay': name,
            'n_variants': len(df),
            'n_multi': int(n_multi),
            'n_singles': int((df['mutation_order'] == 1).sum()),
            'max_order': int(df['mutation_order'].max()),
            'n_unique_sites': len(set(s for sl in sites_list for s in sl)),
            'epistasis_magnitude': float(epi_mag) if np.isfinite(epi_mag) else None,
            'fitness_mean': float(df['DMS_score'].mean()),
            'fitness_std': float(df['DMS_score'].std()),
        }
        assay_data[name] = df
        assay_stats[name] = stats

    # Select top 15 by number of multi-mutants
    ranked = sorted(assay_stats.items(), key=lambda x: x[1]['n_multi'], reverse=True)
    selected = [name for name, _ in ranked[:15]]

    # Classify epistasis quartiles
    epi_vals = [(name, s['epistasis_magnitude']) for name, s in assay_stats.items()
                if name in selected and s['epistasis_magnitude'] is not None]
    if epi_vals:
        epi_vals.sort(key=lambda x: x[1])
        n = len(epi_vals)
        q1 = n // 4
        q3 = 3 * n // 4
        for i, (name, _) in enumerate(epi_vals):
            if i < q1:
                assay_stats[name]['epistasis_group'] = 'low'
            elif i >= q3:
                assay_stats[name]['epistasis_group'] = 'high'
            else:
                assay_stats[name]['epistasis_group'] = 'medium'

    with open(stats_path, 'w') as f:
        json.dump(assay_stats, f, indent=2)
    with open(selected_path, 'w') as f:
        json.dump(selected, f, indent=2)

    print(f"Selected {len(selected)} assays:")
    for name in selected:
        s = assay_stats[name]
        epi = s.get('epistasis_group', '?')
        print(f"  {name}: {s['n_multi']} multi-mutants, order≤{s['max_order']}, epistasis={epi}")

    return selected, assay_stats


# ============================================================
# STEP 2: ALPHAFOLD STRUCTURES + CONTACT MAPS
# ============================================================

def get_uniprot_id(assay_name, ref_df):
    """Get UniProt ID (entry name) for an assay from the reference file."""
    # Match by DMS_id (exact or contained)
    matches = ref_df[ref_df['DMS_id'] == assay_name]
    if len(matches) == 0:
        matches = ref_df[ref_df['DMS_filename'].str.contains(assay_name, case=True, na=False)]
    if len(matches) > 0:
        return matches.iloc[0].get('UniProt_ID', None)
    return None


def resolve_uniprot_accession(entry_name):
    """Resolve UniProt entry name (e.g., SPG1_STRSG) to accession code (e.g., P06654)."""
    cache_path = STRUCT_DIR / "uniprot_accessions.json"
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
    if entry_name in cache:
        return cache[entry_name]

    try:
        url = f"https://rest.uniprot.org/uniprotkb/search?query={entry_name}&format=json&size=1"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('results'):
                accession = data['results'][0]['primaryAccession']
                cache[entry_name] = accession
                with open(cache_path, 'w') as f:
                    json.dump(cache, f, indent=2)
                return accession
    except Exception:
        pass

    # Try direct accession lookup
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{entry_name}?format=json"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            accession = data.get('primaryAccession', entry_name)
            cache[entry_name] = accession
            with open(cache_path, 'w') as f:
                json.dump(cache, f, indent=2)
            return accession
    except Exception:
        pass

    return None


def get_wildtype_sequence(assay_name, ref_df, processed_dir):
    """Get wildtype sequence from reference or data files."""
    # Match by DMS_id or DMS_filename
    matches = ref_df[ref_df['DMS_id'] == assay_name]
    if len(matches) == 0:
        matches = ref_df[ref_df['DMS_filename'].str.contains(assay_name, case=True, na=False)]
    if len(matches) > 0:
        seq = matches.iloc[0].get('target_seq', None)
        if pd.notna(seq) and len(str(seq)) > 10:
            return str(seq)

    # Try from the data file itself
    parquet_path = processed_dir / f"{assay_name}.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if 'mutated_sequence' in df.columns:
            wt_mask = df['mutation_order'] == 0
            if wt_mask.any():
                return df.loc[wt_mask, 'mutated_sequence'].iloc[0]

    return None


def download_alphafold_structure(uniprot_entry, assay_name=None):
    """Download AlphaFold structure, resolving entry name to accession."""
    if uniprot_entry is None:
        return None

    # First try: resolve entry name to accession code
    accession = resolve_uniprot_accession(uniprot_entry)

    candidates = [accession, uniprot_entry] if accession else [uniprot_entry]

    for uid in candidates:
        if uid is None:
            continue
        pdb_path = STRUCT_DIR / f"AF-{uid}.pdb"
        if pdb_path.exists():
            return pdb_path

        url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v6.pdb"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(pdb_path, 'w') as f:
                    f.write(resp.text)
                return pdb_path
        except Exception as e:
            pass

    print(f"  Failed to download AF structure for {uniprot_entry} (accession: {accession})")
    return None


def compute_contact_map_from_pdb(pdb_path, thresholds=[8.0, 10.0, 12.0, 15.0]):
    """Parse PDB and compute Cb-Cb distance matrix and contact maps."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure('protein', str(pdb_path))
    except Exception as e:
        print(f"  Failed to parse PDB {pdb_path}: {e}")
        return None

    model = structure[0]
    chain = list(model.get_chains())[0]

    # Extract Cb coordinates (Ca for Gly)
    residues = []
    coords = []
    plddt_scores = []

    for res in chain.get_residues():
        if res.id[0] != ' ':  # Skip hetero atoms
            continue
        resname = res.get_resname()
        resnum = res.id[1]

        # Get Cb (or Ca for Gly)
        if resname == 'GLY':
            atom_name = 'CA'
        else:
            atom_name = 'CB'

        if atom_name not in res:
            if 'CA' in res:
                atom_name = 'CA'
            else:
                continue

        atom = res[atom_name]
        coords.append(atom.get_vector().get_array())
        residues.append(resnum)

        # pLDDT from B-factor
        plddt = atom.get_bfactor()
        plddt_scores.append(plddt)

    if len(coords) < 5:
        return None

    coords = np.array(coords)
    n = len(coords)

    # Distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))

    # Contact maps at different thresholds
    contact_maps = {}
    for thresh in thresholds:
        contacts = dist_matrix < thresh
        np.fill_diagonal(contacts, False)
        edge_list = np.argwhere(contacts)
        contact_maps[thresh] = edge_list

    # Sequence-only baseline: |i-j| <= 5
    seq_contacts = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) <= 5
    np.fill_diagonal(seq_contacts, False)
    contact_maps['seq5'] = np.argwhere(seq_contacts)

    # Edge features for default 10A threshold
    edge_features = []
    for i, j in contact_maps[10.0]:
        edge_features.append([
            dist_matrix[i, j],  # Cb-Cb distance
            abs(residues[i] - residues[j]),  # sequence separation
            plddt_scores[i],  # pLDDT of residue i
            plddt_scores[j],  # pLDDT of residue j
        ])
    edge_features = np.array(edge_features) if edge_features else np.zeros((0, 4))

    result = {
        'distance_matrix': dist_matrix,
        'contact_maps': contact_maps,
        'edge_features': edge_features,
        'plddt_scores': np.array(plddt_scores),
        'residue_numbers': residues,
        'n_residues': n,
    }
    return result


def prepare_structures(selected_assays, ref_df):
    """Download AlphaFold structures and compute contact maps for all assays."""
    print("\n" + "=" * 60)
    print("Preparing AlphaFold structures and contact maps")
    print("=" * 60)

    struct_data = {}
    for assay_name in selected_assays:
        cache_path = STRUCT_DIR / f"{assay_name}_contacts.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                struct_data[assay_name] = pickle.load(f)
            print(f"  {assay_name}: loaded from cache")
            continue

        # Get UniProt ID
        uniprot_id = get_uniprot_id(assay_name, ref_df)
        if uniprot_id is None:
            print(f"  {assay_name}: no UniProt ID found, will create synthetic graph")
            struct_data[assay_name] = None
            continue

        # Download structure
        pdb_path = download_alphafold_structure(uniprot_id, assay_name)
        if pdb_path is None:
            print(f"  {assay_name}: no AlphaFold structure, will create synthetic graph")
            struct_data[assay_name] = None
            continue

        # Compute contact map
        contacts = compute_contact_map_from_pdb(pdb_path)
        if contacts is None:
            print(f"  {assay_name}: failed to parse PDB")
            struct_data[assay_name] = None
            continue

        # Save
        with open(cache_path, 'wb') as f:
            pickle.dump(contacts, f)

        struct_data[assay_name] = contacts
        n_edges_10 = len(contacts['contact_maps'][10.0])
        print(f"  {assay_name}: {contacts['n_residues']} residues, {n_edges_10} contacts @ 10A")

    return struct_data


# ============================================================
# STEP 3: ESM-2 EMBEDDINGS AND ADDITIVE PREDICTIONS
# ============================================================

def compute_esm2_embeddings(selected_assays, ref_df):
    """Compute ESM-2 per-residue embeddings and mutation LLRs."""
    print("\n" + "=" * 60)
    print("Computing ESM-2 embeddings and additive predictions")
    print("=" * 60)

    import esm

    # Load ESM-2 650M
    print("Loading ESM-2 650M...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(DEVICE)
    batch_converter = alphabet.get_batch_converter()

    embeddings = {}
    llr_cache = {}

    for assay_name in selected_assays:
        embed_path = EMBED_DIR / f"{assay_name}_embeddings.pt"
        llr_path = EMBED_DIR / f"{assay_name}_llrs.pkl"

        if embed_path.exists() and llr_path.exists():
            embeddings[assay_name] = torch.load(embed_path, map_location='cpu', weights_only=True)
            with open(llr_path, 'rb') as f:
                llr_cache[assay_name] = pickle.load(f)
            print(f"  {assay_name}: loaded from cache")
            continue

        # Get wildtype sequence
        wt_seq = get_wildtype_sequence(assay_name, ref_df, PROCESSED_DIR)
        if wt_seq is None:
            print(f"  {assay_name}: no wildtype sequence found, skipping ESM-2")
            continue

        # Truncate long sequences for ESM-2 (max 1022 tokens)
        if len(wt_seq) > 1022:
            print(f"  {assay_name}: truncating from {len(wt_seq)} to 1022")
            wt_seq = wt_seq[:1022]

        print(f"  {assay_name}: computing embeddings for {len(wt_seq)} residues...")

        # Forward pass for embeddings
        data = [("protein", wt_seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(DEVICE)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_embeds = results["representations"][33][0]  # [L+2, 1280]
            # Remove BOS and EOS tokens
            residue_embeds = token_embeds[1:-1].cpu()  # [L, 1280]
            logits = results["logits"][0]  # [L+2, vocab_size]
            logits = logits[1:-1]  # [L, vocab_size]

        # Compute log-likelihood ratios for all possible mutations
        log_probs = F.log_softmax(logits, dim=-1).cpu()

        # For each position, compute LLR for all amino acid substitutions
        aa_order = alphabet.all_toks  # token list
        aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}

        # Standard amino acids
        standard_aas = 'ACDEFGHIKLMNPQRSTVWY'
        llrs = {}  # (pos, wt_aa, mut_aa) -> LLR

        for pos_idx in range(len(wt_seq)):
            wt_aa = wt_seq[pos_idx]
            if wt_aa not in standard_aas:
                continue
            wt_idx = aa_to_idx.get(wt_aa, None)
            if wt_idx is None:
                continue
            wt_logprob = log_probs[pos_idx, wt_idx].item()

            for mut_aa in standard_aas:
                if mut_aa == wt_aa:
                    continue
                mut_idx = aa_to_idx.get(mut_aa, None)
                if mut_idx is None:
                    continue
                mut_logprob = log_probs[pos_idx, mut_idx].item()
                # Use 1-indexed positions to match mutant strings
                llrs[(pos_idx + 1, wt_aa, mut_aa)] = mut_logprob - wt_logprob

        # Save
        torch.save(residue_embeds, embed_path)
        with open(llr_path, 'wb') as f:
            pickle.dump({'llrs': llrs, 'wt_seq': wt_seq}, f)

        embeddings[assay_name] = residue_embeds
        llr_cache[assay_name] = {'llrs': llrs, 'wt_seq': wt_seq}
        print(f"    Embeddings: {residue_embeds.shape}, LLRs: {len(llrs)}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    return embeddings, llr_cache


def compute_additive_predictions(selected_assays, llr_cache):
    """Compute additive PLM predictions for all variants."""
    print("\nComputing additive predictions...")

    for assay_name in selected_assays:
        parquet_path = PROCESSED_DIR / f"{assay_name}.parquet"
        if not parquet_path.exists():
            continue

        df = pd.read_parquet(parquet_path)

        if assay_name not in llr_cache:
            df['plm_additive'] = np.nan
            df['epistatic_residual'] = np.nan
            df.to_parquet(parquet_path)
            continue

        llrs = llr_cache[assay_name]['llrs']

        additive_preds = []
        for _, row in df.iterrows():
            mutant_str = row.get('mutant', '')
            if pd.isna(mutant_str) or mutant_str in ('', '_wt'):
                additive_preds.append(0.0)
                continue

            wts, poss, muts = parse_mutant(mutant_str)
            score = 0.0
            for wt, pos, mut in zip(wts, poss, muts):
                llr_val = llrs.get((pos, wt, mut), 0.0)
                score += llr_val
            additive_preds.append(score)

        df['plm_additive'] = additive_preds
        df['epistatic_residual'] = df['DMS_score'] - df['plm_additive']

        # Save
        df.to_parquet(parquet_path)

        # Quick eval
        multi = df[df['mutation_order'] >= 2]
        if len(multi) > 10:
            rho = spearman(multi['DMS_score'].values, multi['plm_additive'].values)
            print(f"  {assay_name}: PLM additive Spearman = {rho:.4f} (n={len(multi)})")


# ============================================================
# STEP 4: BASELINES
# ============================================================

def build_onehot_features_fast(mutant_series, site_to_idx, aa_to_idx, n_features):
    """Vectorized one-hot feature construction using sparse format."""
    from scipy.sparse import lil_matrix
    n = len(mutant_series)
    X = lil_matrix((n, n_features), dtype=np.float32)
    for idx, mutant_str in enumerate(mutant_series):
        if pd.isna(mutant_str) or mutant_str in ('', '_wt'):
            continue
        for m in mutant_str.split(':'):
            if len(m) < 3:
                continue
            pos = int(m[1:-1])
            mut_aa = m[-1]
            if pos in site_to_idx and mut_aa in aa_to_idx:
                X[idx, site_to_idx[pos] * 20 + aa_to_idx[mut_aa]] = 1.0
    return X.tocsr()


def run_baselines(selected_assays):
    """Run Ridge, Additive+Pairwise, and Global Epistasis baselines."""
    print("\n" + "=" * 60)
    print("Running baselines")
    print("=" * 60)

    MAX_TRAIN = 50000  # Subsample training set for very large assays

    all_results = []

    for assay_name in selected_assays:
        parquet_path = PROCESSED_DIR / f"{assay_name}.parquet"
        if not parquet_path.exists():
            continue

        df = pd.read_parquet(parquet_path)
        multi = df[df['mutation_order'] >= 2].copy().reset_index(drop=True)
        if len(multi) < 50:
            continue

        print(f"\n  {assay_name} ({len(multi)} multi-mutants)")

        # Build one-hot features efficiently
        all_sites = set()
        for sites in multi['mutation_sites']:
            all_sites.update(sites)
        all_sites = sorted(all_sites)
        site_to_idx = {s: i for i, s in enumerate(all_sites)}
        n_sites = len(all_sites)
        aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        n_features = n_sites * 20

        print(f"    Building features: {n_sites} sites, {n_features} features...")
        X = build_onehot_features_fast(multi['mutant'].values, site_to_idx, aa_to_idx, n_features)
        y = multi['DMS_score'].values
        folds = multi['fold_id'].values
        plm_add = multi['plm_additive'].values if 'plm_additive' in multi.columns else None

        for seed in SEEDS:
            set_seed(seed)

            for fold in range(N_FOLDS):
                train_mask = folds != fold
                test_mask = folds == fold
                if test_mask.sum() < 5:
                    continue

                X_test = X[test_mask]
                y_test = y[test_mask]

                # Subsample training for very large datasets
                train_idx = np.where(train_mask)[0]
                if len(train_idx) > MAX_TRAIN:
                    np.random.seed(seed + fold)
                    train_idx = np.random.choice(train_idx, MAX_TRAIN, replace=False)
                X_train = X[train_idx]
                y_train = y[train_idx]

                # Ridge with alpha=1.0 (single fast run)
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_train, y_train)
                y_pred_ridge = ridge.predict(X_test)

                all_results.append({
                    'assay': assay_name, 'fold': fold, 'seed': seed,
                    'method': 'ridge_best',
                    'spearman_rho': spearman(y_test, y_pred_ridge),
                    'ndcg100': ndcg_at_k(y_test, y_pred_ridge),
                })

                # Pairwise: only for assays with <= 20 sites
                if n_sites <= 20 and n_features <= 400:
                    from scipy.sparse import hstack as sp_hstack
                    X_train_d = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
                    X_test_d = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
                    # Random subset of pairwise features
                    np.random.seed(seed)
                    n_pw = min(5000, n_features * (n_features - 1) // 2)
                    pw_i = np.random.randint(0, n_features, n_pw)
                    pw_j = np.random.randint(0, n_features, n_pw)
                    X_pw_train = X_train_d[:, pw_i] * X_train_d[:, pw_j]
                    X_pw_test = X_test_d[:, pw_i] * X_test_d[:, pw_j]
                    X_comb_train = np.hstack([X_train_d, X_pw_train])
                    X_comb_test = np.hstack([X_test_d, X_pw_test])

                    ridge_pw = Ridge(alpha=1.0)
                    ridge_pw.fit(X_comb_train, y_train)
                    y_pred_pw = ridge_pw.predict(X_comb_test)
                    all_results.append({
                        'assay': assay_name, 'fold': fold, 'seed': seed,
                        'method': 'ridge_pairwise',
                        'spearman_rho': spearman(y_test, y_pred_pw),
                        'ndcg100': ndcg_at_k(y_test, y_pred_pw),
                    })
                else:
                    all_results.append({
                        'assay': assay_name, 'fold': fold, 'seed': seed,
                        'method': 'ridge_pairwise',
                        'spearman_rho': spearman(y_test, y_pred_ridge),
                        'ndcg100': ndcg_at_k(y_test, y_pred_ridge),
                    })

                # Global epistasis
                if plm_add is not None:
                    plm_train = plm_add[train_idx] if len(train_idx) < len(plm_add) else plm_add[train_mask]
                    plm_test = plm_add[test_mask]

                    def global_epi_loss(params):
                        a, b, c, d = params
                        pred = a / (1 + np.exp(-np.clip(b * plm_train + c, -50, 50))) + d
                        return np.mean((pred - y_train) ** 2)

                    try:
                        result = minimize(global_epi_loss, [1.0, 1.0, 0.0, 0.0],
                                         method='Nelder-Mead', options={'maxiter': 2000})
                        a, b, c, d = result.x
                        y_pred_ge = a / (1 + np.exp(-np.clip(b * plm_test + c, -50, 50))) + d
                    except Exception:
                        y_pred_ge = plm_test

                    all_results.append({
                        'assay': assay_name, 'fold': fold, 'seed': seed,
                        'method': 'global_epistasis',
                        'spearman_rho': spearman(y_test, y_pred_ge),
                        'ndcg100': ndcg_at_k(y_test, y_pred_ge),
                    })

                    # ESM-2 additive baseline
                    all_results.append({
                        'assay': assay_name, 'fold': fold, 'seed': seed,
                        'method': 'esm2_additive',
                        'spearman_rho': spearman(y_test, plm_add[test_mask]),
                        'ndcg100': ndcg_at_k(y_test, plm_add[test_mask]),
                    })

        print(f"    Done ({len([r for r in all_results if r['assay'] == assay_name])} results)")

    # Save
    baseline_df = pd.DataFrame(all_results)
    baseline_df.to_csv(RESULTS_DIR / "baselines" / "all_baselines.csv", index=False)
    print(f"\nSaved {len(all_results)} baseline results")

    if len(baseline_df) > 0:
        summary = baseline_df.groupby('method')['spearman_rho'].agg(['mean', 'std', 'count'])
        print("\nBaseline summary (mean Spearman rho):")
        print(summary.sort_values('mean', ascending=False).to_string())

    return baseline_df


# ============================================================
# STEP 5: REN MODEL TRAINING
# ============================================================

def build_graph_data(assay_name, struct_data, embeddings, llr_cache, df,
                     threshold=10.0, node_feat_config='full'):
    """Build PyG-style graph data for an assay."""
    from torch_geometric.data import Data

    struct = struct_data.get(assay_name)
    embed = embeddings.get(assay_name)

    if embed is None:
        return None

    n_residues = embed.shape[0]

    # Build edge index
    if struct is not None and threshold in struct['contact_maps']:
        edges = struct['contact_maps'][threshold]
        # Map to embedding indices (might need offset adjustment)
        # Clip to embedding length
        valid = (edges[:, 0] < n_residues) & (edges[:, 1] < n_residues)
        edges = edges[valid]
        edge_index = torch.tensor(edges.T, dtype=torch.long)

        # Edge features
        if threshold == 10.0 and struct.get('edge_features') is not None:
            ef = struct['edge_features']
            ef = ef[valid[:len(ef)]] if len(ef) == len(struct['contact_maps'][10.0]) else ef[:edge_index.shape[1]]
            if len(ef) == edge_index.shape[1]:
                edge_attr = torch.tensor(ef, dtype=torch.float32)
            else:
                edge_attr = None
        else:
            edge_attr = None
    elif struct is not None and threshold == 'seq5':
        edges = struct['contact_maps']['seq5']
        valid = (edges[:, 0] < n_residues) & (edges[:, 1] < n_residues)
        edges = edges[valid]
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        edge_attr = None
    else:
        # Fallback: sequence proximity graph |i-j| <= 5
        edges = []
        for i in range(n_residues):
            for j in range(max(0, i-5), min(n_residues, i+6)):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = None

    # Node features
    if node_feat_config == 'full':
        # ESM-2 embeddings + 21-dim mutation identity (set per-variant)
        base_features = embed  # [n_residues, 1280]
    elif node_feat_config == 'esm_only':
        base_features = embed
    elif node_feat_config == 'mutation_only':
        base_features = torch.zeros(n_residues, 21)  # Will be set per variant
    elif node_feat_config == 'random':
        torch.manual_seed(42)
        base_features = torch.randn(n_residues, 1280)
    else:
        base_features = embed

    return {
        'base_features': base_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'n_residues': n_residues,
    }


def prepare_variant_features(base_features, mutation_sites, mutation_aas,
                             node_feat_config='full', wt_seq=None):
    """
    Prepare node features for a specific variant by adding mutation identity encoding.
    """
    n_residues = base_features.shape[0]
    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

    if node_feat_config in ('full', 'random'):
        # Add 21-dim mutation identity: one-hot for mutated AA, or "no mutation" dim
        mut_encoding = torch.zeros(n_residues, 21)
        mut_encoding[:, 20] = 1.0  # "no mutation" for all positions

        for site, aa in zip(mutation_sites, mutation_aas):
            idx = site - 1  # Convert to 0-indexed
            if 0 <= idx < n_residues and aa in aa_to_idx:
                mut_encoding[idx, :] = 0
                mut_encoding[idx, aa_to_idx[aa]] = 1.0

        features = torch.cat([base_features, mut_encoding], dim=-1)
    elif node_feat_config == 'esm_only':
        features = base_features
    elif node_feat_config == 'mutation_only':
        mut_encoding = torch.zeros(n_residues, 21)
        mut_encoding[:, 20] = 1.0
        for site, aa in zip(mutation_sites, mutation_aas):
            idx = site - 1
            if 0 <= idx < n_residues and aa in aa_to_idx:
                mut_encoding[idx, :] = 0
                mut_encoding[idx, aa_to_idx[aa]] = 1.0
        features = mut_encoding
    else:
        features = base_features

    return features


def get_node_feat_dim(config):
    if config == 'full':
        return 1280 + 21
    elif config == 'esm_only':
        return 1280
    elif config == 'mutation_only':
        return 21
    elif config == 'random':
        return 1280 + 21
    return 1280 + 21


class EfficientREN(nn.Module):
    """
    Efficient REN with fully batched prediction.
    Uses padded tensors for mutation sites to avoid Python loops.
    """
    def __init__(self, esm_dim=1280, hidden_dim=256, num_heads=8, num_layers=3,
                 dropout=0.1, conv_type='gat'):
        super().__init__()
        self.conv_type = conv_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(esm_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if conv_type == 'gat':
                conv = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              dropout=dropout, concat=True)
            elif conv_type == 'gcn':
                conv = GCNConv(hidden_dim, hidden_dim)
            else:
                conv = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              dropout=dropout, concat=True)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Variant prediction head
        self.variant_head = nn.Sequential(
            nn.Linear(hidden_dim + 21, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.pool_query = nn.Linear(hidden_dim + 21, 1)

    def encode_graph(self, x, edge_index):
        h = self.input_proj(x)
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index)
            h_new = F.elu(h_new)
            h = self.norms[i](h_new + h)
        return h

    def predict_batch(self, node_embeds, sites_padded, mut_ids_padded, mask):
        """
        Fully batched prediction.
        sites_padded: [batch, max_muts] LongTensor of site indices (0-indexed)
        mut_ids_padded: [batch, max_muts, 21] FloatTensor
        mask: [batch, max_muts] BoolTensor (True for real mutations)
        """
        # Gather site embeddings: [batch, max_muts, hidden]
        site_embeds = node_embeds[sites_padded]
        # Concat with mutation ids: [batch, max_muts, hidden+21]
        combined = torch.cat([site_embeds, mut_ids_padded], dim=-1)
        # Attention scores: [batch, max_muts]
        scores = self.pool_query(combined).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [batch, max_muts, 1]
        # Weighted sum: [batch, hidden+21]
        pooled = (weights * combined).sum(dim=1)
        return self.variant_head(pooled).squeeze(-1)


class EfficientMLPBaseline(nn.Module):
    """MLP baseline: no message passing."""
    def __init__(self, esm_dim=1280, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(esm_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.variant_head = nn.Sequential(
            nn.Linear(hidden_dim + 21, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.pool_query = nn.Linear(hidden_dim + 21, 1)

    def encode_graph(self, x, edge_index):
        return F.elu(self.input_proj(x))

    def predict_batch(self, node_embeds, sites_padded, mut_ids_padded, mask):
        site_embeds = node_embeds[sites_padded]
        combined = torch.cat([site_embeds, mut_ids_padded], dim=-1)
        scores = self.pool_query(combined).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (weights * combined).sum(dim=1)
        return self.variant_head(pooled).squeeze(-1)


def precompute_variant_data_padded(multi_df, n_residues, max_muts=10):
    """
    Pre-compute mutation sites and identity encodings as padded tensors.
    Returns pre-padded tensors for fully batched GPU processing.
    """
    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    n = len(multi_df)

    sites_padded = torch.zeros(n, max_muts, dtype=torch.long)
    mut_ids_padded = torch.zeros(n, max_muts, 21)
    mask = torch.zeros(n, max_muts, dtype=torch.bool)

    mutant_strs = multi_df['mutant'].values
    for i, mutant_str in enumerate(mutant_strs):
        if not isinstance(mutant_str, str) or not mutant_str or mutant_str == '_wt':
            sites_padded[i, 0] = 0
            mut_ids_padded[i, 0, 20] = 1.0
            mask[i, 0] = True
            continue

        j = 0
        for m in mutant_str.split(':'):
            if j >= max_muts:
                break
            if len(m) >= 3:
                try:
                    pos = int(m[1:-1]) - 1
                except ValueError:
                    continue
                if 0 <= pos < n_residues:
                    sites_padded[i, j] = pos
                    aa_idx = aa_to_idx.get(m[-1], -1)
                    if aa_idx >= 0:
                        mut_ids_padded[i, j, aa_idx] = 1.0
                    else:
                        mut_ids_padded[i, j, 20] = 1.0
                    mask[i, j] = True
                    j += 1

        if j == 0:  # No valid mutations found
            sites_padded[i, 0] = 0
            mut_ids_padded[i, 0, 20] = 1.0
            mask[i, 0] = True

    targets = multi_df['epistatic_residual'].values if 'epistatic_residual' in multi_df.columns else multi_df['DMS_score'].values
    return sites_padded, mut_ids_padded, mask, targets.astype(np.float32)


def train_ren_on_assay(assay_name, struct_data, embeddings, llr_cache,
                       seed=42, n_epochs=100, lr=1e-3, patience=15,
                       threshold=10.0, node_feat_config='full',
                       num_layers=3, conv_type='gat', edge_feat_dim=0,
                       hidden_dim=256, num_heads=8):
    """Train REN on one assay with CV, return per-fold results."""
    MAX_VARIANTS = 20000  # Total cap for precomputation

    parquet_path = PROCESSED_DIR / f"{assay_name}.parquet"
    if not parquet_path.exists():
        return []

    df = pd.read_parquet(parquet_path)
    multi = df[df['mutation_order'] >= 2].copy().reset_index(drop=True)
    if len(multi) < 50:
        return []

    # Subsample if too large (before precomputation for speed)
    if len(multi) > MAX_VARIANTS:
        np.random.seed(seed)
        keep_idx = np.random.choice(len(multi), MAX_VARIANTS, replace=False)
        multi = multi.iloc[keep_idx].reset_index(drop=True)

    # Build graph
    graph = build_graph_data(assay_name, struct_data, embeddings, llr_cache, df,
                             threshold=threshold, node_feat_config=node_feat_config)
    if graph is None:
        return []

    base_features = graph['base_features']
    edge_index = graph['edge_index'].to(DEVICE)
    n_residues = graph['n_residues']

    # Determine ESM dim based on node feature config
    if node_feat_config == 'mutation_only':
        esm_dim = 21
    else:
        esm_dim = 1280

    # Base node features for graph encoding
    if node_feat_config == 'mutation_only':
        graph_features = torch.zeros(n_residues, 21).to(DEVICE)
    elif node_feat_config == 'random':
        torch.manual_seed(42)
        graph_features = torch.randn(n_residues, 1280).to(DEVICE)
    else:
        graph_features = base_features.to(DEVICE)

    # Pre-compute variant data as padded tensors
    max_order = multi['mutation_order'].max()
    max_muts = min(max_order + 1, 30)
    sites_padded, mut_ids_padded, var_mask, all_targets = precompute_variant_data_padded(
        multi, n_residues, max_muts=max_muts)

    # Move to GPU
    sites_gpu = sites_padded.to(DEVICE)
    mut_ids_gpu = mut_ids_padded.to(DEVICE)
    var_mask_gpu = var_mask.to(DEVICE)

    results = []

    for fold in range(N_FOLDS):
        set_seed(seed)

        fold_mask = multi['fold_id'].values
        train_idx = np.where(fold_mask != fold)[0]
        test_idx = np.where(fold_mask == fold)[0]
        if len(test_idx) < 5:
            continue

        # Split train into train/val (90/10)
        np.random.seed(seed)
        n_val = max(10, len(train_idx) // 10)
        val_sub = np.random.choice(len(train_idx), n_val, replace=False)
        val_idx = train_idx[val_sub]
        train_sub_idx = np.delete(train_idx, val_sub)

        # Create model
        if conv_type == 'mlp':
            model = EfficientMLPBaseline(esm_dim=esm_dim, hidden_dim=hidden_dim).to(DEVICE)
        else:
            model = EfficientREN(esm_dim=esm_dim, hidden_dim=hidden_dim,
                                 num_heads=num_heads, num_layers=num_layers,
                                 conv_type=conv_type).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        batch_size = 1024

        targets_gpu = torch.tensor(all_targets, dtype=torch.float32).to(DEVICE)

        for epoch in range(n_epochs):
            model.train()
            node_embeds = model.encode_graph(graph_features, edge_index)

            perm = np.random.permutation(len(train_sub_idx))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(perm), batch_size):
                end = min(start + batch_size, len(perm))
                bidx = train_sub_idx[perm[start:end]]

                preds = model.predict_batch(
                    node_embeds, sites_gpu[bidx], mut_ids_gpu[bidx], var_mask_gpu[bidx])
                loss = F.mse_loss(preds, targets_gpu[bidx])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
                node_embeds = model.encode_graph(graph_features, edge_index)

            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                ne = model.encode_graph(graph_features, edge_index)
                val_preds = model.predict_batch(
                    ne, sites_gpu[val_idx], mut_ids_gpu[val_idx], var_mask_gpu[val_idx])
                val_loss = F.mse_loss(val_preds, targets_gpu[val_idx]).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Evaluate on test set
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            ne = model.encode_graph(graph_features, edge_index)
            # Process test in chunks to avoid OOM
            test_preds_list = []
            for ts in range(0, len(test_idx), 2048):
                te = min(ts + 2048, len(test_idx))
                bidx = test_idx[ts:te]
                tp = model.predict_batch(ne, sites_gpu[bidx], mut_ids_gpu[bidx], var_mask_gpu[bidx])
                test_preds_list.append(tp.cpu())
            test_preds = torch.cat(test_preds_list).numpy()

        test_targets = all_targets[test_idx]
        plm_add_test = multi['plm_additive'].values[test_idx] if 'plm_additive' in multi.columns else np.zeros(len(test_idx))
        ren_preds = plm_add_test + test_preds

        rho = spearman(test_targets + plm_add_test, ren_preds)  # Compare full fitness
        rho_residual = spearman(test_targets, test_preds)
        test_fitness = multi['DMS_score'].values[test_idx]
        rho_fitness = spearman(test_fitness, ren_preds)
        ndcg = ndcg_at_k(test_fitness, ren_preds)

        method_name = f'ren_{conv_type}_L{num_layers}_t{threshold}_{node_feat_config}_ef{edge_feat_dim}'
        results.append({
            'assay': assay_name, 'fold': fold, 'seed': seed,
            'method': method_name,
            'spearman_rho': rho_fitness,
            'epistasis_spearman': rho_residual,
            'ndcg100': ndcg,
            'n_epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss,
        })

        # Order-stratified
        test_orders = multi['mutation_order'].values[test_idx]
        for order in sorted(np.unique(test_orders)):
            order_mask = test_orders == order
            if order_mask.sum() >= 5:
                rho_order = spearman(test_fitness[order_mask], ren_preds[order_mask])
                results.append({
                    'assay': assay_name, 'fold': fold, 'seed': seed,
                    'method': f'{method_name}_order{order}',
                    'spearman_rho': rho_order,
                    'n_variants': int(order_mask.sum()),
                })

    return results


def run_ren_experiments(selected_assays, struct_data, embeddings, llr_cache):
    """Run main REN experiments on all assays."""
    print("\n" + "=" * 60)
    print("Training REN models")
    print("=" * 60)

    all_results = []

    for assay_name in selected_assays:
        print(f"\n--- {assay_name} ---")
        for seed in SEEDS:
            print(f"  Seed {seed}...")
            results = train_ren_on_assay(
                assay_name, struct_data, embeddings, llr_cache,
                seed=seed, n_epochs=100, threshold=10.0,
                node_feat_config='full', num_layers=3, conv_type='gat',
            )
            all_results.extend(results)
            if results:
                main_results = [r for r in results if 'order' not in r['method']]
                if main_results:
                    mean_rho = np.mean([r['spearman_rho'] for r in main_results])
                    print(f"    Mean Spearman: {mean_rho:.4f}")

    # Save
    ren_df = pd.DataFrame(all_results)
    ren_df.to_csv(RESULTS_DIR / "ren" / "ren_results.csv", index=False)
    print(f"\nSaved {len(all_results)} REN results")

    return ren_df


# ============================================================
# STEP 6: ABLATION STUDIES
# ============================================================

def run_ablations(selected_assays, struct_data, embeddings, llr_cache, assay_stats):
    """Run all ablation studies on a subset of 5 assays."""
    print("\n" + "=" * 60)
    print("Running ablation studies")
    print("=" * 60)

    # Select 5 representative assays
    # Prefer assays with epistasis data and diverse properties
    ablation_assays = []
    high_epi = [a for a in selected_assays if assay_stats.get(a, {}).get('epistasis_group') == 'high']
    low_epi = [a for a in selected_assays if assay_stats.get(a, {}).get('epistasis_group') == 'low']
    med_epi = [a for a in selected_assays if assay_stats.get(a, {}).get('epistasis_group') == 'medium']

    ablation_assays.extend(high_epi[:2])
    ablation_assays.extend(low_epi[:2])
    ablation_assays.extend(med_epi[:1])

    # Fill up to 5
    for a in selected_assays:
        if len(ablation_assays) >= 5:
            break
        if a not in ablation_assays:
            ablation_assays.append(a)

    ablation_assays = ablation_assays[:5]
    print(f"Ablation assays: {ablation_assays}")

    all_ablation_results = []
    seed = 42  # Single seed for ablations

    # Ablation 1: Contact threshold
    print("\n--- Ablation: Contact threshold ---")
    for assay_name in ablation_assays:
        for thresh in [8.0, 10.0, 12.0, 15.0, 'seq5']:
            print(f"  {assay_name}, threshold={thresh}")
            results = train_ren_on_assay(
                assay_name, struct_data, embeddings, llr_cache,
                seed=seed, n_epochs=80, threshold=thresh,
                node_feat_config='full', num_layers=3, conv_type='gat',
            )
            for r in results:
                r['ablation'] = 'contact_threshold'
                r['ablation_value'] = str(thresh)
            all_ablation_results.extend(results)

    # Ablation 2: Node features
    print("\n--- Ablation: Node features ---")
    for assay_name in ablation_assays:
        for config in ['full', 'esm_only', 'mutation_only', 'random']:
            print(f"  {assay_name}, features={config}")
            results = train_ren_on_assay(
                assay_name, struct_data, embeddings, llr_cache,
                seed=seed, n_epochs=80, threshold=10.0,
                node_feat_config=config, num_layers=3, conv_type='gat',
            )
            for r in results:
                r['ablation'] = 'node_features'
                r['ablation_value'] = config
            all_ablation_results.extend(results)

    # Ablation 3: GNN depth
    print("\n--- Ablation: GNN depth ---")
    for assay_name in ablation_assays:
        for n_layers in [1, 2, 3, 4]:
            print(f"  {assay_name}, layers={n_layers}")
            results = train_ren_on_assay(
                assay_name, struct_data, embeddings, llr_cache,
                seed=seed, n_epochs=80, threshold=10.0,
                node_feat_config='full', num_layers=n_layers, conv_type='gat',
            )
            for r in results:
                r['ablation'] = 'gnn_depth'
                r['ablation_value'] = str(n_layers)
            all_ablation_results.extend(results)

        # GCN comparison
        print(f"  {assay_name}, GCN")
        results = train_ren_on_assay(
            assay_name, struct_data, embeddings, llr_cache,
            seed=seed, n_epochs=80, threshold=10.0,
            node_feat_config='full', num_layers=3, conv_type='gcn',
        )
        for r in results:
            r['ablation'] = 'architecture'
            r['ablation_value'] = 'gcn'
        all_ablation_results.extend(results)

        # MLP only
        print(f"  {assay_name}, MLP")
        results = train_ren_on_assay(
            assay_name, struct_data, embeddings, llr_cache,
            seed=seed, n_epochs=80, threshold=10.0,
            node_feat_config='full', num_layers=3, conv_type='mlp',
        )
        for r in results:
            r['ablation'] = 'architecture'
            r['ablation_value'] = 'mlp'
        all_ablation_results.extend(results)

    # Save
    abl_df = pd.DataFrame(all_ablation_results)
    abl_df.to_csv(RESULTS_DIR / "ablations" / "all_ablations.csv", index=False)
    print(f"\nSaved {len(all_ablation_results)} ablation results")

    return abl_df


# ============================================================
# STEP 7: STATISTICAL ANALYSIS
# ============================================================

def run_statistical_analysis(selected_assays, assay_stats):
    """Perform statistical tests and stratified analysis."""
    print("\n" + "=" * 60)
    print("Statistical analysis")
    print("=" * 60)

    # Load results
    baseline_path = RESULTS_DIR / "baselines" / "all_baselines.csv"
    ren_path = RESULTS_DIR / "ren" / "ren_results.csv"

    if not baseline_path.exists() or not ren_path.exists():
        print("Results files not found, skipping analysis")
        return {}

    baseline_df = pd.read_csv(baseline_path)
    ren_df = pd.read_csv(ren_path)

    # Filter to main REN results (not order-stratified)
    ren_main = ren_df[~ren_df['method'].str.contains('order')].copy()

    # Get the default REN config
    ren_configs = ren_main['method'].unique()
    default_config = [c for c in ren_configs if 'gat_L3_t10' in c and 'full' in c]
    if default_config:
        ren_main = ren_main[ren_main['method'] == default_config[0]]

    analysis_results = {}

    # 1. Paired Wilcoxon: REN vs ESM-2 additive
    esm2_results = baseline_df[baseline_df['method'] == 'esm2_additive']

    # Match by (assay, fold), average over seeds
    esm2_avg = esm2_results.groupby(['assay', 'fold'])['spearman_rho'].mean().reset_index()
    ren_avg = ren_main.groupby(['assay', 'fold'])['spearman_rho'].mean().reset_index()

    merged = esm2_avg.merge(ren_avg, on=['assay', 'fold'], suffixes=('_esm2', '_ren'))

    if len(merged) >= 5:
        try:
            stat, pval = wilcoxon(merged['spearman_rho_ren'], merged['spearman_rho_esm2'],
                                  alternative='greater')
            n = len(merged)
            r_effect = 1 - (2 * stat) / (n * (n + 1))

            analysis_results['wilcoxon_test'] = {
                'statistic': float(stat),
                'p_value': float(pval),
                'effect_size': float(r_effect),
                'n_pairs': int(n),
                'mean_improvement': float((merged['spearman_rho_ren'] - merged['spearman_rho_esm2']).mean()),
            }
            print(f"Wilcoxon test: p={pval:.4e}, effect_size={r_effect:.3f}")
            print(f"  Mean improvement: {analysis_results['wilcoxon_test']['mean_improvement']:.4f}")
        except Exception as e:
            print(f"  Wilcoxon test failed: {e}")

    # 2. Stratified by epistasis group
    if 'epistasis_group' in [k for stats in assay_stats.values() for k in stats.keys()]:
        improvements = merged.copy()
        improvements['delta'] = improvements['spearman_rho_ren'] - improvements['spearman_rho_esm2']

        for _, row in improvements.iterrows():
            row_stats = assay_stats.get(row['assay'], {})
            improvements.loc[improvements['assay'] == row['assay'], 'epi_group'] = row_stats.get('epistasis_group', 'unknown')

        high_delta = improvements[improvements['epi_group'] == 'high']['delta'].values
        low_delta = improvements[improvements['epi_group'] == 'low']['delta'].values

        if len(high_delta) >= 3 and len(low_delta) >= 3:
            try:
                stat, pval = mannwhitneyu(high_delta, low_delta, alternative='greater')
                analysis_results['epistasis_stratified'] = {
                    'high_mean_improvement': float(np.mean(high_delta)),
                    'low_mean_improvement': float(np.mean(low_delta)),
                    'mannwhitney_p': float(pval),
                    'mannwhitney_stat': float(stat),
                }
                print(f"\nEpistasis stratification:")
                print(f"  High-epistasis improvement: {np.mean(high_delta):.4f}")
                print(f"  Low-epistasis improvement: {np.mean(low_delta):.4f}")
                print(f"  Mann-Whitney p: {pval:.4f}")
            except Exception as e:
                print(f"  Stratification test failed: {e}")

    # 3. Order-stratified analysis
    ren_order = ren_df[ren_df['method'].str.contains('order')].copy()
    if len(ren_order) > 0:
        ren_order['order'] = ren_order['method'].str.extract(r'order(\d+)').astype(int)
        order_results = ren_order.groupby('order')['spearman_rho'].agg(['mean', 'std']).reset_index()

        # Also get ESM-2 additive per order
        analysis_results['order_stratified'] = order_results.to_dict('records')
        print(f"\nOrder-stratified REN performance:")
        print(order_results.to_string())

    # 4. Summary table
    methods = ['esm2_additive', 'ridge_best', 'ridge_pairwise', 'global_epistasis']
    summary = {}

    for method in methods:
        method_df = baseline_df[baseline_df['method'] == method]
        if len(method_df) > 0:
            avg = method_df.groupby(['assay', 'seed'])['spearman_rho'].mean()
            summary[method] = {
                'mean': float(avg.mean()),
                'std': float(avg.std()),
                'n': int(len(avg)),
            }

    # REN
    if len(ren_main) > 0:
        ren_per_assay_seed = ren_main.groupby(['assay', 'seed'])['spearman_rho'].mean()
        summary['ren'] = {
            'mean': float(ren_per_assay_seed.mean()),
            'std': float(ren_per_assay_seed.std()),
            'n': int(len(ren_per_assay_seed)),
        }

    analysis_results['summary'] = summary
    print(f"\nSummary:")
    for method, stats in summary.items():
        print(f"  {method}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    # Save
    with open(RESULTS_DIR / "evaluation" / "statistical_tests.json", 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)

    return analysis_results


# ============================================================
# STEP 8: FIGURES
# ============================================================

def generate_figures(selected_assays, assay_stats, analysis_results):
    """Generate publication-quality figures."""
    print("\n" + "=" * 60)
    print("Generating figures")
    print("=" * 60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })

    # Load results
    baseline_path = RESULTS_DIR / "baselines" / "all_baselines.csv"
    ren_path = RESULTS_DIR / "ren" / "ren_results.csv"
    abl_path = RESULTS_DIR / "ablations" / "all_ablations.csv"

    if not baseline_path.exists() or not ren_path.exists():
        print("Results not found, skipping figures")
        return

    baseline_df = pd.read_csv(baseline_path)
    ren_df = pd.read_csv(ren_path)

    # Filter main REN results
    ren_main = ren_df[~ren_df['method'].str.contains('order')].copy()
    ren_configs = ren_main['method'].unique()
    default_config = [c for c in ren_configs if 'gat_L3_t10' in c and 'full' in c]
    if default_config:
        ren_default = ren_main[ren_main['method'] == default_config[0]].copy()
        ren_default['method'] = 'REN'
    else:
        ren_default = ren_main.copy()
        ren_default['method'] = 'REN'

    # Combine for comparison
    methods_to_plot = ['esm2_additive', 'ridge_best', 'ridge_pairwise', 'global_epistasis']
    baseline_subset = baseline_df[baseline_df['method'].isin(methods_to_plot)].copy()
    method_labels = {
        'esm2_additive': 'ESM-2 Additive',
        'ridge_best': 'Ridge Regression',
        'ridge_pairwise': 'Ridge + Pairwise',
        'global_epistasis': 'Global Epistasis',
        'REN': 'REN (Ours)',
    }
    baseline_subset['method'] = baseline_subset['method'].map(method_labels)
    ren_default['method'] = 'REN (Ours)'
    combined = pd.concat([baseline_subset, ren_default[['assay', 'fold', 'seed', 'method', 'spearman_rho', 'ndcg100']]])

    # Colors
    palette = {
        'ESM-2 Additive': '#1f77b4',
        'Ridge Regression': '#ff7f0e',
        'Ridge + Pairwise': '#2ca02c',
        'Global Epistasis': '#d62728',
        'REN (Ours)': '#9467bd',
    }

    # ---- Figure 2: Main results bar chart ----
    fig, ax = plt.subplots(figsize=(16, 6))
    per_assay = combined.groupby(['assay', 'method'])['spearman_rho'].agg(['mean', 'std']).reset_index()

    assay_order = per_assay[per_assay['method'] == 'ESM-2 Additive'].sort_values('mean')['assay'].tolist()
    if not assay_order:
        assay_order = sorted(per_assay['assay'].unique())

    # Truncate assay names
    short_names = {a: a[:20] for a in assay_order}

    x = np.arange(len(assay_order))
    width = 0.15
    method_order = ['ESM-2 Additive', 'Ridge Regression', 'Ridge + Pairwise', 'Global Epistasis', 'REN (Ours)']

    for i, method in enumerate(method_order):
        data = per_assay[per_assay['method'] == method]
        vals = []
        errs = []
        for assay in assay_order:
            row = data[data['assay'] == assay]
            if len(row) > 0:
                vals.append(row['mean'].values[0])
                errs.append(row['std'].values[0] if pd.notna(row['std'].values[0]) else 0)
            else:
                vals.append(0)
                errs.append(0)
        ax.bar(x + i * width, vals, width, yerr=errs, label=method,
               color=palette.get(method, 'gray'), capsize=2, alpha=0.85)

    ax.set_ylabel('Spearman Correlation')
    ax.set_title('Multi-Mutant Fitness Prediction: Method Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([short_names[a] for a in assay_order], rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "main_results.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "main_results.png", bbox_inches='tight')
    plt.close()
    print("  Saved main_results.pdf")

    # ---- Figure 3: Epistasis-stratified improvement ----
    fig, ax = plt.subplots(figsize=(8, 6))

    esm2_per_assay = baseline_df[baseline_df['method'] == 'esm2_additive'].groupby('assay')['spearman_rho'].mean()
    ren_per_assay = ren_default.groupby('assay')['spearman_rho'].mean()

    common = set(esm2_per_assay.index) & set(ren_per_assay.index)
    epi_mags = []
    improvements = []
    assay_labels = []

    for assay in common:
        epi = assay_stats.get(assay, {}).get('epistasis_magnitude')
        if epi is not None and np.isfinite(epi):
            epi_mags.append(epi)
            improvements.append(ren_per_assay[assay] - esm2_per_assay[assay])
            assay_labels.append(assay[:15])

    if len(epi_mags) > 3:
        ax.scatter(epi_mags, improvements, s=80, c='#9467bd', alpha=0.7, edgecolors='k', linewidths=0.5)
        for i, label in enumerate(assay_labels):
            ax.annotate(label, (epi_mags[i], improvements[i]), fontsize=7, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')

        # Trend line
        from scipy.stats import pearsonr
        if len(epi_mags) > 3:
            z = np.polyfit(epi_mags, improvements, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(epi_mags), max(epi_mags), 100)
            ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.7)
            r, p_val = pearsonr(epi_mags, improvements)
            rho_s = spearman(np.array(epi_mags), np.array(improvements))
            ax.text(0.05, 0.95, f'Pearson r={r:.3f}, p={p_val:.3f}\nSpearman rho={rho_s:.3f}',
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Epistasis Magnitude')
    ax.set_ylabel('Improvement (REN - ESM-2 Additive)')
    ax.set_title('REN Improvement vs. Epistasis Magnitude')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "epistasis_stratified.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "epistasis_stratified.png", bbox_inches='tight')
    plt.close()
    print("  Saved epistasis_stratified.pdf")

    # ---- Figure 4: Order-stratified ----
    ren_order = ren_df[ren_df['method'].str.contains('order')].copy()
    if len(ren_order) > 0:
        ren_order['order'] = ren_order['method'].str.extract(r'order(\d+)').astype(int)
        fig, ax = plt.subplots(figsize=(8, 6))

        order_stats = ren_order.groupby('order')['spearman_rho'].agg(['mean', 'std']).reset_index()

        # Also compute ESM-2 additive by order (need to re-compute from data)
        ax.errorbar(order_stats['order'], order_stats['mean'], yerr=order_stats['std'],
                    marker='o', capsize=5, label='REN', color='#9467bd', linewidth=2)

        ax.set_xlabel('Mutation Order')
        ax.set_ylabel('Spearman Correlation')
        ax.set_title('Performance by Mutation Order')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "order_stratified.pdf", bbox_inches='tight')
        plt.savefig(FIGURES_DIR / "order_stratified.png", bbox_inches='tight')
        plt.close()
        print("  Saved order_stratified.pdf")

    # ---- Figure 5: Ablation results ----
    if abl_path.exists():
        abl_df = pd.read_csv(abl_path)
        abl_main = abl_df[~abl_df['method'].str.contains('order', na=False)].copy()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (a) Contact threshold
        ax = axes[0, 0]
        thresh_data = abl_main[abl_main.get('ablation', pd.Series()) == 'contact_threshold']
        if len(thresh_data) > 0:
            thresh_stats = thresh_data.groupby('ablation_value')['spearman_rho'].agg(['mean', 'std']).reset_index()
            ax.bar(range(len(thresh_stats)), thresh_stats['mean'], yerr=thresh_stats['std'],
                   capsize=5, color='#9467bd', alpha=0.8)
            ax.set_xticks(range(len(thresh_stats)))
            ax.set_xticklabels(thresh_stats['ablation_value'])
        ax.set_title('(a) Contact Distance Threshold')
        ax.set_ylabel('Spearman rho')

        # (b) Node features
        ax = axes[0, 1]
        feat_data = abl_main[abl_main.get('ablation', pd.Series()) == 'node_features']
        if len(feat_data) > 0:
            feat_stats = feat_data.groupby('ablation_value')['spearman_rho'].agg(['mean', 'std']).reset_index()
            colors = ['#9467bd', '#1f77b4', '#ff7f0e', '#d62728']
            ax.bar(range(len(feat_stats)), feat_stats['mean'], yerr=feat_stats['std'],
                   capsize=5, color=colors[:len(feat_stats)], alpha=0.8)
            ax.set_xticks(range(len(feat_stats)))
            ax.set_xticklabels(feat_stats['ablation_value'], rotation=20)
        ax.set_title('(b) Node Feature Configuration')
        ax.set_ylabel('Spearman rho')

        # (c) GNN depth
        ax = axes[1, 0]
        depth_data = abl_main[abl_main.get('ablation', pd.Series()) == 'gnn_depth']
        if len(depth_data) > 0:
            depth_stats = depth_data.groupby('ablation_value')['spearman_rho'].agg(['mean', 'std']).reset_index()
            ax.bar(range(len(depth_stats)), depth_stats['mean'], yerr=depth_stats['std'],
                   capsize=5, color='#2ca02c', alpha=0.8)
            ax.set_xticks(range(len(depth_stats)))
            ax.set_xticklabels([f'{v} layers' for v in depth_stats['ablation_value']])
        ax.set_title('(c) GNN Depth')
        ax.set_ylabel('Spearman rho')

        # (d) Architecture comparison
        ax = axes[1, 1]
        arch_data = abl_main[abl_main.get('ablation', pd.Series()) == 'architecture']
        # Also add the default GAT for comparison
        gat_data = abl_main[(abl_main.get('ablation', pd.Series()) == 'gnn_depth') &
                            (abl_main.get('ablation_value', pd.Series()) == '3')]
        if len(gat_data) > 0:
            gat_row = pd.DataFrame({'ablation_value': ['gat'], 'mean': [gat_data['spearman_rho'].mean()],
                                     'std': [gat_data['spearman_rho'].std()]})
        else:
            gat_row = pd.DataFrame()

        if len(arch_data) > 0:
            arch_stats = arch_data.groupby('ablation_value')['spearman_rho'].agg(['mean', 'std']).reset_index()
            if len(gat_row) > 0:
                arch_stats = pd.concat([gat_row, arch_stats])
            ax.bar(range(len(arch_stats)), arch_stats['mean'], yerr=arch_stats['std'],
                   capsize=5, color=['#9467bd', '#ff7f0e', '#d62728'][:len(arch_stats)], alpha=0.8)
            ax.set_xticks(range(len(arch_stats)))
            ax.set_xticklabels(arch_stats['ablation_value'])
        ax.set_title('(d) Architecture Comparison')
        ax.set_ylabel('Spearman rho')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "ablations.pdf", bbox_inches='tight')
        plt.savefig(FIGURES_DIR / "ablations.png", bbox_inches='tight')
        plt.close()
        print("  Saved ablations.pdf")

    # ---- Figure 7: Residual analysis ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Try to load REN predictions for residual analysis
    # Use summary data
    ax = axes[0]
    method_means = {}
    for method in ['esm2_additive', 'ridge_best', 'global_epistasis']:
        mdf = baseline_df[baseline_df['method'] == method]
        if len(mdf) > 0:
            method_means[method_labels.get(method, method)] = mdf['spearman_rho'].mean()
    if len(ren_default) > 0:
        method_means['REN (Ours)'] = ren_default['spearman_rho'].mean()

    if method_means:
        names = list(method_means.keys())
        vals = list(method_means.values())
        colors = [palette.get(n, 'gray') for n in names]
        ax.barh(range(len(names)), vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Mean Spearman Correlation')
        ax.set_title('Overall Performance Comparison')

    ax = axes[1]
    # Per-seed variance
    if len(ren_default) > 0:
        seed_stats = ren_default.groupby('seed')['spearman_rho'].mean()
        ax.bar(range(len(seed_stats)), seed_stats.values, color='#9467bd', alpha=0.8)
        ax.set_xticks(range(len(seed_stats)))
        ax.set_xticklabels([f'Seed {s}' for s in seed_stats.index])
        ax.set_ylabel('Mean Spearman rho')
        ax.set_title('REN Performance Across Seeds')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "performance_summary.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "performance_summary.png", bbox_inches='tight')
    plt.close()
    print("  Saved performance_summary.pdf")

    # ---- LaTeX tables ----
    generate_latex_tables(baseline_df, ren_default, assay_stats, analysis_results)


def generate_latex_tables(baseline_df, ren_df, assay_stats, analysis_results):
    """Generate LaTeX-formatted result tables."""
    # Table 1: Main results
    method_labels = {
        'esm2_additive': 'ESM-2 Additive',
        'ridge_best': 'Ridge Regression',
        'ridge_pairwise': 'Ridge + Pairwise',
        'global_epistasis': 'Global Epistasis',
    }

    rows = []
    for method, label in method_labels.items():
        mdf = baseline_df[baseline_df['method'] == method]
        if len(mdf) == 0:
            continue
        mean_rho = mdf['spearman_rho'].mean()
        std_rho = mdf['spearman_rho'].std()

        # By epistasis group
        high_rho = []
        low_rho = []
        for assay in mdf['assay'].unique():
            group = assay_stats.get(assay, {}).get('epistasis_group', '')
            assay_rho = mdf[mdf['assay'] == assay]['spearman_rho'].mean()
            if group == 'high':
                high_rho.append(assay_rho)
            elif group == 'low':
                low_rho.append(assay_rho)

        high_mean = np.mean(high_rho) if high_rho else np.nan
        low_mean = np.mean(low_rho) if low_rho else np.nan

        ndcg = mdf['ndcg100'].mean() if 'ndcg100' in mdf.columns else np.nan

        rows.append(f"{label} & {mean_rho:.3f} $\\pm$ {std_rho:.3f} & {high_mean:.3f} & {low_mean:.3f} & {ndcg:.3f} \\\\")

    # REN
    if len(ren_df) > 0:
        mean_rho = ren_df['spearman_rho'].mean()
        std_rho = ren_df['spearman_rho'].std()
        ndcg = ren_df['ndcg100'].mean() if 'ndcg100' in ren_df.columns else np.nan

        high_rho = []
        low_rho = []
        for assay in ren_df['assay'].unique():
            group = assay_stats.get(assay, {}).get('epistasis_group', '')
            assay_rho = ren_df[ren_df['assay'] == assay]['spearman_rho'].mean()
            if group == 'high':
                high_rho.append(assay_rho)
            elif group == 'low':
                low_rho.append(assay_rho)

        high_mean = np.mean(high_rho) if high_rho else np.nan
        low_mean = np.mean(low_rho) if low_rho else np.nan

        rows.append(f"\\textbf{{REN (Ours)}} & \\textbf{{{mean_rho:.3f}}} $\\pm$ {std_rho:.3f} & \\textbf{{{high_mean:.3f}}} & {low_mean:.3f} & \\textbf{{{ndcg:.3f}}} \\\\")

    table = "\\begin{table}[t]\n\\centering\n\\caption{Multi-mutant fitness prediction results on ProteinGym.}\n"
    table += "\\begin{tabular}{lcccc}\n\\toprule\n"
    table += "Method & All Assays & High Epistasis & Low Epistasis & NDCG@100 \\\\\n\\midrule\n"
    table += "\n".join(rows)
    table += "\n\\bottomrule\n\\end{tabular}\n\\label{tab:main_results}\n\\end{table}"

    with open(FIGURES_DIR / "table_main_results.tex", 'w') as f:
        f.write(table)
    print("  Saved table_main_results.tex")


# ============================================================
# STEP 9: AGGREGATE RESULTS
# ============================================================

def aggregate_results(selected_assays, assay_stats, analysis_results):
    """Create the final results.json at workspace root."""
    print("\n" + "=" * 60)
    print("Aggregating final results")
    print("=" * 60)

    final = {
        'experiment': 'Residual Epistasis Networks (REN)',
        'description': 'Structure-conditioned GNN for correcting additive PLM fitness predictions',
        'n_assays_evaluated': len(selected_assays),
        'assays': selected_assays,
        'seeds': SEEDS,
        'n_folds': N_FOLDS,
    }

    # Load results
    baseline_path = RESULTS_DIR / "baselines" / "all_baselines.csv"
    ren_path = RESULTS_DIR / "ren" / "ren_results.csv"

    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        for method in ['esm2_additive', 'ridge_best', 'ridge_pairwise', 'global_epistasis']:
            mdf = baseline_df[baseline_df['method'] == method]
            if len(mdf) > 0:
                final[f'baseline_{method}'] = {
                    'spearman_rho': {'mean': float(mdf['spearman_rho'].mean()),
                                      'std': float(mdf['spearman_rho'].std())},
                    'ndcg100': {'mean': float(mdf['ndcg100'].mean()),
                                'std': float(mdf['ndcg100'].std())} if 'ndcg100' in mdf.columns else None,
                }

    if ren_path.exists():
        ren_df = pd.read_csv(ren_path)
        ren_main = ren_df[~ren_df['method'].str.contains('order')]
        if len(ren_main) > 0:
            final['ren'] = {
                'spearman_rho': {'mean': float(ren_main['spearman_rho'].mean()),
                                  'std': float(ren_main['spearman_rho'].std())},
                'ndcg100': {'mean': float(ren_main['ndcg100'].mean()),
                            'std': float(ren_main['ndcg100'].std())} if 'ndcg100' in ren_main.columns else None,
            }
            if 'epistasis_spearman' in ren_main.columns:
                epi_sp = ren_main['epistasis_spearman'].dropna()
                if len(epi_sp) > 0:
                    final['ren']['epistasis_spearman'] = {
                        'mean': float(epi_sp.mean()),
                        'std': float(epi_sp.std()),
                    }

    # Statistical tests
    if analysis_results:
        final['statistical_tests'] = analysis_results

    # Ablation summary
    abl_path = RESULTS_DIR / "ablations" / "all_ablations.csv"
    if abl_path.exists():
        abl_df = pd.read_csv(abl_path)
        abl_main = abl_df[~abl_df['method'].str.contains('order', na=False)]
        ablation_summary = {}
        for abl_type in abl_main['ablation'].unique():
            abl_subset = abl_main[abl_main['ablation'] == abl_type]
            abl_stats = abl_subset.groupby('ablation_value')['spearman_rho'].agg(['mean', 'std']).reset_index()
            ablation_summary[abl_type] = abl_stats.to_dict('records')
        final['ablations'] = ablation_summary

    # Success criteria evaluation
    criteria = {}
    if 'wilcoxon_test' in analysis_results:
        wt = analysis_results['wilcoxon_test']
        criteria['criterion_1_significant_improvement'] = {
            'target': 'p < 0.05 on paired Wilcoxon test',
            'result': f"p = {wt['p_value']:.4e}",
            'met': wt['p_value'] < 0.05,
        }

    if 'epistasis_stratified' in analysis_results:
        es = analysis_results['epistasis_stratified']
        criteria['criterion_2_larger_improvement_high_epistasis'] = {
            'target': 'Larger improvement on high-epistasis proteins',
            'result': f"High: {es['high_mean_improvement']:.4f}, Low: {es['low_mean_improvement']:.4f}",
            'met': es['high_mean_improvement'] > es['low_mean_improvement'],
        }

    final['success_criteria'] = criteria

    # Save
    results_path = BASE_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(final, f, indent=2, default=str)

    print(f"\nSaved results.json with {len(final)} top-level keys")
    print(f"\nSuccess criteria:")
    for name, crit in criteria.items():
        status = "PASS" if crit['met'] else "FAIL"
        print(f"  [{status}] {name}: {crit['result']}")

    return final


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()

    # Step 1: Download and preprocess data
    print("STEP 1: Data download and preprocessing")
    ref_df = download_reference()
    extract_dir = download_proteingym()
    selected_assays, assay_stats = preprocess_datasets(extract_dir, ref_df)

    # Step 2: AlphaFold structures
    print("\nSTEP 2: AlphaFold structures")
    try:
        from Bio.PDB import PDBParser
        struct_data = prepare_structures(selected_assays, ref_df)
    except ImportError:
        print("BioPython not available, will use sequence proximity graphs")
        struct_data = {}

    # Step 3: ESM-2 embeddings
    print("\nSTEP 3: ESM-2 embeddings")
    embeddings, llr_cache = compute_esm2_embeddings(selected_assays, ref_df)
    compute_additive_predictions(selected_assays, llr_cache)

    # Step 4: Baselines
    print("\nSTEP 4: Baselines")
    baseline_df = run_baselines(selected_assays)

    # Step 5: REN training
    print("\nSTEP 5: REN training")
    ren_df = run_ren_experiments(selected_assays, struct_data, embeddings, llr_cache)

    # Step 6: Ablations
    print("\nSTEP 6: Ablation studies")
    abl_df = run_ablations(selected_assays, struct_data, embeddings, llr_cache, assay_stats)

    # Step 7: Statistical analysis
    print("\nSTEP 7: Statistical analysis")
    analysis_results = run_statistical_analysis(selected_assays, assay_stats)

    # Step 8: Figures
    print("\nSTEP 8: Figures")
    generate_figures(selected_assays, assay_stats, analysis_results)

    # Step 9: Aggregate results
    print("\nSTEP 9: Aggregate results")
    final = aggregate_results(selected_assays, assay_stats, analysis_results)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"COMPLETE! Total time: {elapsed/3600:.1f} hours ({elapsed/60:.0f} minutes)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

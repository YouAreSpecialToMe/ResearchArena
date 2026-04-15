"""Step 2: Extract ESM-2 features and coupling matrices."""
import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.config import *

# Select proteins to use (balance diversity, size, seq length)
# Skip very large datasets (SPG1 535K, HIS7 254K) and very long sequences
# to stay within time budget
SELECTED_PROTEINS = None  # Will be set after loading info


def select_proteins(info):
    """Select a diverse subset of proteins for experiments."""
    # Sort by sequence length, prefer shorter for faster ESM inference
    # But want diversity in epistasis magnitude
    candidates = []
    for name, d in info.items():
        candidates.append({
            'name': name,
            'seq_len': d['seq_length'],
            'n_variants': d['num_multi_mutants'],
            'mean_abs_epi': d['mean_abs_epistasis'],
        })

    # For time budget: skip proteins with seq_len > 600 (expensive ESM inference)
    # and limit very large datasets
    selected = []
    for c in candidates:
        if c['seq_len'] > 600:
            print(f"  Skipping {c['name']} (seq_len={c['seq_len']} > 600)")
            continue
        if c['n_variants'] > 100000:
            # Subsample these
            print(f"  Will subsample {c['name']} ({c['n_variants']} variants)")
        selected.append(c['name'])

    print(f"Selected {len(selected)} proteins for experiments")
    return selected


def extract_esm2_features(model, alphabet, batch_converter, wt_seq, protein_name,
                          model_name="650M", embed_dim=1280, num_layers=33,
                          coupling_start=20, device='cuda'):
    """Extract embeddings, masked marginals, and coupling from ESM-2."""
    print(f"  Extracting features for {protein_name} (len={len(wt_seq)})...")
    prefix = f"{protein_name}_{model_name}" if model_name != "650M" else protein_name

    # 1. Forward pass for embeddings and attention
    t0 = time.time()
    data = [("protein", wt_seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[num_layers], need_head_weights=True)

    # Extract embeddings (remove BOS/EOS tokens)
    embeddings = results["representations"][num_layers][0, 1:len(wt_seq)+1].cpu()
    print(f"    Embeddings: {embeddings.shape} ({time.time()-t0:.1f}s)")

    # Extract attention maps
    # Shape: [layers, heads, seq_len+2, seq_len+2]
    attentions = results["attentions"][0]  # [layers, heads, L+2, L+2]
    # Remove BOS/EOS
    attentions = attentions[:, :, 1:len(wt_seq)+1, 1:len(wt_seq)+1].cpu()
    print(f"    Attentions: {attentions.shape}")

    # Save embeddings
    emb_path = os.path.join(FEATURES_DIR, f"{prefix}_wt_embeddings.pt")
    torch.save(embeddings, emb_path)

    # 2. Compute coupling matrix from deep layer attentions
    t0 = time.time()
    # Use layers coupling_start to num_layers (deeper layers encode structure)
    deep_attn = attentions[coupling_start:num_layers]  # [~13, 20, L, L]

    # Symmetrize and average across heads and layers
    C_raw = (deep_attn + deep_attn.transpose(-2, -1)) / 2  # symmetrize
    C_raw = C_raw.mean(dim=(0, 1))  # [L, L]

    # Apply Average Product Correction (APC)
    row_mean = C_raw.mean(dim=1, keepdim=True)
    col_mean = C_raw.mean(dim=0, keepdim=True)
    total_mean = C_raw.mean()
    C_apc = C_raw - (row_mean * col_mean) / total_mean

    # Zero diagonal
    C_apc.fill_diagonal_(0)

    coupling_path = os.path.join(FEATURES_DIR, f"{prefix}_coupling.pt")
    torch.save(C_apc, coupling_path)
    print(f"    Coupling matrix: {C_apc.shape} ({time.time()-t0:.1f}s)")

    # 3. Compute masked marginal scores
    t0 = time.time()
    L = len(wt_seq)
    aa_tokens = alphabet.all_toks
    standard_aas = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in standard_aas}

    masked_marginal = torch.zeros(L, 20)

    # Process in batches of positions to save memory
    batch_size_pos = 50  # positions per batch
    for start in range(0, L, batch_size_pos):
        end = min(start + batch_size_pos, L)
        positions = list(range(start, end))

        # Create masked versions
        masked_tokens = batch_tokens[0].unsqueeze(0).repeat(len(positions), 1).to(device)
        for i, pos in enumerate(positions):
            masked_tokens[i, pos + 1] = alphabet.mask_idx  # +1 for BOS

        with torch.no_grad():
            logits = model(masked_tokens)["logits"]  # [batch, L+2, vocab]

        for i, pos in enumerate(positions):
            log_probs = torch.log_softmax(logits[i, pos + 1], dim=-1)
            wt_aa = wt_seq[pos]
            wt_idx = aa_to_idx.get(wt_aa)
            if wt_idx is None:
                continue
            wt_log_prob = log_probs[wt_idx].item()
            for j, aa in enumerate(standard_aas):
                aa_idx = aa_to_idx[aa]
                masked_marginal[pos, j] = log_probs[aa_idx].item() - wt_log_prob

    mm_path = os.path.join(FEATURES_DIR, f"{prefix}_masked_marginal.pt")
    torch.save(masked_marginal, mm_path)
    print(f"    Masked marginals: {masked_marginal.shape} ({time.time()-t0:.1f}s)")

    # Clean up GPU memory
    del batch_tokens, results, attentions, deep_attn
    torch.cuda.empty_cache()

    return embeddings, C_apc, masked_marginal


def compute_additive_scores(df, masked_marginal, wt_seq):
    """Compute additive ESM-2 scores for all variants."""
    import ast
    standard_aas = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_col = {aa: i for i, aa in enumerate(standard_aas)}

    scores = []
    for _, row in df.iterrows():
        muts = ast.literal_eval(row['mutations_parsed'])
        score = 0.0
        valid = True
        for wt, pos, mut in muts:
            # pos is 1-indexed in mutations, but could be 0-indexed
            # Check which indexing is used
            if pos < masked_marginal.shape[0]:
                idx = pos
            elif pos - 1 < masked_marginal.shape[0]:
                idx = pos - 1
            else:
                valid = False
                break

            mut_col = aa_to_col.get(mut)
            if mut_col is None:
                valid = False
                break
            score += masked_marginal[idx, mut_col].item()

        scores.append(score if valid else np.nan)

    return scores


def main():
    print("=" * 60)
    print("STEP 2: Extract ESM-2 features")
    print("=" * 60)

    # Load processed info
    info_path = os.path.join(DATA_DIR, "processed_info.json")
    with open(info_path) as f:
        info = json.load(f)

    selected = select_proteins(info)

    # Load ESM-2 650M
    print("\nLoading ESM-2 650M model...")
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().cuda()
    print("ESM-2 loaded on GPU")

    feature_manifest = {}

    for protein_name in selected:
        print(f"\n{'='*40}")
        print(f"Processing {protein_name}")
        print(f"{'='*40}")

        pinfo = info[protein_name]
        wt_seq = pinfo['wt_sequence']

        # Check for invalid characters
        wt_seq_clean = ''.join(c for c in wt_seq if c in 'ACDEFGHIKLMNPQRSTVWY')
        if len(wt_seq_clean) != len(wt_seq):
            print(f"  Warning: cleaned WT seq from {len(wt_seq)} to {len(wt_seq_clean)}")
            wt_seq = wt_seq_clean
            pinfo['wt_sequence'] = wt_seq
            pinfo['seq_length'] = len(wt_seq)

        # Extract features
        embeddings, coupling, masked_marginal = extract_esm2_features(
            model, alphabet, batch_converter, wt_seq, protein_name,
            model_name="650M", embed_dim=ESM2_EMBED_DIM,
            num_layers=ESM2_NUM_LAYERS, coupling_start=COUPLING_LAYER_START,
            device='cuda'
        )

        # Compute additive scores for all variants
        df = pd.read_parquet(os.path.join(DATA_DIR, "processed", f"{protein_name}.parquet"))

        # Subsample large datasets
        if len(df) > 50000:
            print(f"  Subsampling from {len(df)} to 50000 variants")
            df = df.sample(n=50000, random_state=42).reset_index(drop=True)
            # Recompute folds
            from sklearn.model_selection import StratifiedKFold
            strat = df['num_mutations'].clip(upper=4).values
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            fold_assignments = np.zeros(len(df), dtype=int)
            for fold_idx, (_, test_idx) in enumerate(skf.split(df, strat)):
                fold_assignments[test_idx] = fold_idx
            df['fold'] = fold_assignments

        print(f"  Computing additive ESM-2 scores for {len(df)} variants...")
        additive_scores = compute_additive_scores(df, masked_marginal, wt_seq)
        df['esm2_additive_score'] = additive_scores

        # Drop variants where additive score couldn't be computed
        n_before = len(df)
        df = df.dropna(subset=['esm2_additive_score'])
        if n_before > len(df):
            print(f"  Dropped {n_before - len(df)} variants without valid additive scores")

        # Save updated dataframe
        out_path = os.path.join(DATA_DIR, "processed", f"{protein_name}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(df)} variants with additive scores")

        feature_manifest[protein_name] = {
            'wt_embeddings': f"{protein_name}_wt_embeddings.pt",
            'coupling': f"{protein_name}_coupling.pt",
            'masked_marginal': f"{protein_name}_masked_marginal.pt",
            'embed_dim': ESM2_EMBED_DIM,
            'seq_length': len(wt_seq),
            'n_variants': len(df),
        }

    # Save manifest
    manifest_path = os.path.join(FEATURES_DIR, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(feature_manifest, f, indent=2)

    # Update processed info with selected proteins
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    # Save selected proteins list
    with open(os.path.join(DATA_DIR, "selected_proteins.json"), 'w') as f:
        json.dump(selected, f, indent=2)

    print(f"\nFeature extraction complete for {len(feature_manifest)} proteins")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()

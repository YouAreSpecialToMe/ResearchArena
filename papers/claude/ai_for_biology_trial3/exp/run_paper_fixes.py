#!/usr/bin/env python3
"""
Fix experiments for paper revision:
1. Ridge+Pairwise with proper pairwise features for ALL assays (bug fix)
2. Cross-protein MLP vs GAT comparison
"""
import os, sys, json, pickle, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
STRUCT_DIR = DATA_DIR / "structures"
EMBED_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "results"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456]
N_FOLDS = 5

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def spearman_corr(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return np.nan
    return spearmanr(y_true[mask], y_pred[mask])[0]

# Import from run_all_v2
sys.path.insert(0, str(Path(__file__).parent))
from run_all_v2 import (RENv2, MLPBaselineV2, build_graph, precompute_variants,
                         load_cached_data, build_onehot)


# ============================================================
# FIX 1: Ridge + Pairwise for all assays
# ============================================================

def run_ridge_pairwise_fixed():
    """
    Fix: compute pairwise features for ALL assays using random feature sampling.
    Previously, pairwise features were only computed for assays with ≤20 sites.
    """
    print("\n" + "=" * 60)
    print("FIX 1: Ridge + Pairwise (fixed for all assays)")
    print("=" * 60)

    selected, stats, struct_data, embeddings, llr_cache = load_cached_data()
    MAX_TRAIN = 50000
    MAX_PW_FEATURES = 10000  # Random pairwise features

    all_results = []
    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

    for assay_name in selected:
        parquet_path = PROCESSED_DIR / f"{assay_name}.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        multi = df[df['mutation_order'] >= 2].copy().reset_index(drop=True)
        if len(multi) < 50:
            continue

        # Build one-hot features
        all_sites = set()
        for sites in multi['mutation_sites']:
            all_sites.update(sites)
        all_sites = sorted(all_sites)
        site_to_idx = {s: i for i, s in enumerate(all_sites)}
        n_sites = len(all_sites)
        n_features = n_sites * 20

        X = build_onehot(multi['mutant'].values, site_to_idx, aa_to_idx, n_features)
        y = multi['DMS_score'].values
        folds = multi['fold_id'].values

        print(f"\n  {assay_name}: {len(multi)} variants, {n_sites} sites, {n_features} features")

        for seed in SEEDS:
            set_seed(seed)
            for fold in range(N_FOLDS):
                train_mask = folds != fold
                test_mask = folds == fold
                if test_mask.sum() < 5:
                    continue

                train_idx = np.where(train_mask)[0]
                if len(train_idx) > MAX_TRAIN:
                    np.random.seed(seed + fold)
                    train_idx = np.random.choice(train_idx, MAX_TRAIN, replace=False)

                X_train = X[train_idx].toarray() if hasattr(X, 'toarray') else X[train_idx]
                X_test = X[test_mask].toarray() if hasattr(X, 'toarray') else X[test_mask]
                y_train = y[train_idx]
                y_test = y[test_mask]

                # Generate random pairwise features (sampled interactions)
                np.random.seed(seed * 1000 + fold)
                n_pw = min(MAX_PW_FEATURES, n_features * (n_features - 1) // 2)
                pw_i = np.random.randint(0, n_features, n_pw)
                pw_j = np.random.randint(0, n_features, n_pw)
                # Ensure i != j
                same = pw_i == pw_j
                pw_j[same] = (pw_j[same] + 1) % n_features

                X_pw_train = X_train[:, pw_i] * X_train[:, pw_j]
                X_pw_test = X_test[:, pw_i] * X_test[:, pw_j]

                X_comb_train = np.hstack([X_train, X_pw_train])
                X_comb_test = np.hstack([X_test, X_pw_test])

                ridge_pw = Ridge(alpha=1.0)
                ridge_pw.fit(X_comb_train, y_train)
                y_pred_pw = ridge_pw.predict(X_comb_test)

                rho = spearman_corr(y_test, y_pred_pw)
                all_results.append({
                    'assay': assay_name, 'fold': fold, 'seed': seed,
                    'method': 'ridge_pairwise_fixed',
                    'spearman_rho': rho,
                })

        assay_res = [r for r in all_results if r['assay'] == assay_name]
        mean_rho = np.mean([r['spearman_rho'] for r in assay_res])
        print(f"    ridge_pairwise_fixed mean rho: {mean_rho:.3f}")

    results_df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / "evaluation" / "ridge_pairwise_fixed.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Summary
    summary = results_df.groupby('assay')['spearman_rho'].mean()
    print(f"\nOverall mean: {summary.mean():.3f} ± {summary.std():.3f}")
    for assay, val in summary.items():
        print(f"  {assay}: {val:.3f}")

    return results_df


# ============================================================
# FIX 2: Cross-protein MLP vs GAT
# ============================================================

def train_model_cross_protein(model, train_data_list, device, epochs=50, lr=1e-3, batch_size=512):
    """Train a model on data from multiple proteins."""
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        np.random.shuffle(train_data_list)

        for data in train_data_list:
            graph = data['graph']
            sites = data['sites'].to(device)
            mut_oh = data['mut_onehot'].to(device)
            mask = data['mask'].to(device)
            targets = data['targets'].to(device)
            plm_add = data['plm_additive'].to(device) if data.get('plm_additive') is not None else None

            base_feat = graph['base_features'].to(device)
            edge_index = graph['edge_index'].to(device)

            node_embeds = model.encode_graph(base_feat, edge_index)

            n_variants = len(targets)
            for start in range(0, n_variants, batch_size):
                end = min(start + batch_size, n_variants)
                b_sites = sites[start:end]
                b_mut = mut_oh[start:end]
                b_mask = mask[start:end]
                b_tgt = targets[start:end]
                b_plm = plm_add[start:end] if plm_add is not None else None

                pred = model.predict_batch(node_embeds, b_sites, b_mut, b_mask, b_plm)
                loss = F.mse_loss(pred, b_tgt)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        scheduler.step()

    return model


def evaluate_model_cross_protein(model, test_data, device):
    """Evaluate model on a single test protein."""
    model.eval()
    with torch.no_grad():
        graph = test_data['graph']
        base_feat = graph['base_features'].to(device)
        edge_index = graph['edge_index'].to(device)
        node_embeds = model.encode_graph(base_feat, edge_index)

        sites = test_data['sites'].to(device)
        mut_oh = test_data['mut_onehot'].to(device)
        mask = test_data['mask'].to(device)
        plm_add = test_data['plm_additive'].to(device) if test_data.get('plm_additive') is not None else None

        preds = []
        n = len(sites)
        for start in range(0, n, 512):
            end = min(start + 512, n)
            pred = model.predict_batch(
                node_embeds, sites[start:end], mut_oh[start:end],
                mask[start:end],
                plm_add[start:end] if plm_add is not None else None
            )
            preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds)
        targets = test_data['targets'].numpy()
        return spearman_corr(targets, preds)


def prepare_protein_data(assay_name, struct_data, embeddings, max_n=5000, seed=42):
    """Prepare graph + variant data for one protein."""
    parquet_path = PROCESSED_DIR / f"{assay_name}.parquet"
    if not parquet_path.exists():
        return None
    df = pd.read_parquet(parquet_path)
    multi = df[df['mutation_order'] >= 2].copy().reset_index(drop=True)
    if len(multi) < 10:
        return None

    if max_n and len(multi) > max_n:
        multi = multi.sample(max_n, random_state=seed).reset_index(drop=True)

    graph = build_graph(assay_name, struct_data, embeddings, threshold=10.0)
    if graph is None:
        return None

    sites, mut_oh, mask = precompute_variants(multi, graph['n_residues'])
    targets = torch.tensor(multi['DMS_score'].values, dtype=torch.float32)
    plm_add = torch.tensor(multi['plm_additive'].values, dtype=torch.float32) if 'plm_additive' in multi.columns else None

    return {
        'graph': graph,
        'sites': sites,
        'mut_onehot': mut_oh,
        'mask': mask,
        'targets': targets,
        'plm_additive': plm_add,
        'assay': assay_name,
    }


def run_cross_protein_mlp_vs_gat():
    """
    Cross-protein generalization comparing MLP vs GAT.
    Leave-3-out: train on ~12 proteins, test on 3 held-out.
    """
    print("\n" + "=" * 60)
    print("FIX 2: Cross-protein MLP vs GAT")
    print("=" * 60)

    selected, stats, struct_data, embeddings, llr_cache = load_cached_data()

    # Filter to assays with both embeddings and structures
    valid_assays = [a for a in selected if a in embeddings and a in struct_data]
    print(f"Valid assays: {len(valid_assays)}")

    # Create leave-3-out folds (same as original)
    np.random.seed(42)
    perm = np.random.permutation(len(valid_assays))
    n_test = 3
    folds = []
    for i in range(0, len(valid_assays), n_test):
        test_idx = perm[i:i+n_test]
        if len(test_idx) < 2:
            continue
        test_assays = [valid_assays[j] for j in test_idx]
        train_assays = [a for a in valid_assays if a not in test_assays]
        folds.append((train_assays, test_assays))
    print(f"Created {len(folds)} folds")

    results = []

    for fold_idx, (train_assays, test_assays) in enumerate(folds):
        print(f"\nFold {fold_idx}: test on {[a.split('_')[0] for a in test_assays]}")

        for seed in [42, 123]:
            set_seed(seed)

            # Prepare training data
            train_data_list = []
            for assay in train_assays:
                data = prepare_protein_data(assay, struct_data, embeddings,
                                            max_n=5000, seed=seed)
                if data is not None:
                    train_data_list.append(data)

            if not train_data_list:
                continue

            total_train = sum(len(d['targets']) for d in train_data_list)
            print(f"  Seed {seed}: {total_train} train variants from {len(train_data_list)} proteins")

            esm_dim = train_data_list[0]['graph']['esm_dim']

            # Train GAT model
            gat_model = RENv2(esm_dim=esm_dim, hidden_dim=128, num_heads=4,
                              num_layers=2, dropout=0.1, conv_type='gat',
                              use_plm_input=True)
            gat_model = train_model_cross_protein(gat_model, train_data_list,
                                                   DEVICE, epochs=30, lr=5e-4)

            # Train MLP model
            mlp_model = MLPBaselineV2(esm_dim=esm_dim, hidden_dim=128,
                                      dropout=0.1, use_plm_input=True)
            mlp_model = train_model_cross_protein(mlp_model, train_data_list,
                                                   DEVICE, epochs=30, lr=5e-4)

            # Also train with random features
            # Build random-feature graphs for train data
            train_data_rand = []
            for data in train_data_list:
                rand_data = {k: v for k, v in data.items()}
                rand_graph = dict(data['graph'])
                n_res = rand_graph['n_residues']
                torch.manual_seed(seed + 999)
                rand_graph['base_features'] = torch.randn(n_res, esm_dim)
                rand_data['graph'] = rand_graph
                train_data_rand.append(rand_data)

            mlp_rand = MLPBaselineV2(esm_dim=esm_dim, hidden_dim=128,
                                     dropout=0.1, use_plm_input=True)
            mlp_rand = train_model_cross_protein(mlp_rand, train_data_rand,
                                                  DEVICE, epochs=30, lr=5e-4)

            # Evaluate on test proteins
            for test_assay in test_assays:
                test_data = prepare_protein_data(test_assay, struct_data, embeddings,
                                                  max_n=10000, seed=seed)
                if test_data is None:
                    continue

                # GAT with ESM-2
                rho_gat = evaluate_model_cross_protein(gat_model, test_data, DEVICE)

                # MLP with ESM-2
                rho_mlp = evaluate_model_cross_protein(mlp_model, test_data, DEVICE)

                # MLP with random features
                rand_test = {k: v for k, v in test_data.items()}
                rand_graph = dict(test_data['graph'])
                n_res = rand_graph['n_residues']
                torch.manual_seed(seed + 999)
                rand_graph['base_features'] = torch.randn(n_res, esm_dim)
                rand_test['graph'] = rand_graph
                rho_mlp_rand = evaluate_model_cross_protein(mlp_rand, rand_test, DEVICE)

                # ESM-2 additive baseline
                plm_add = test_data['plm_additive'].numpy() if test_data['plm_additive'] is not None else None
                rho_additive = spearman_corr(test_data['targets'].numpy(), plm_add) if plm_add is not None else np.nan

                print(f"    {test_assay}: GAT={rho_gat:.3f}, MLP={rho_mlp:.3f}, MLP_rand={rho_mlp_rand:.3f}, Additive={rho_additive:.3f}")

                for method, rho in [('cross_gat_esm2', rho_gat),
                                     ('cross_mlp_esm2', rho_mlp),
                                     ('cross_mlp_random', rho_mlp_rand),
                                     ('cross_additive', rho_additive)]:
                    results.append({
                        'fold': fold_idx, 'seed': seed, 'test_assay': test_assay,
                        'method': method, 'spearman_rho': rho,
                    })

    results_df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "evaluation" / "cross_protein_mlp_vs_gat.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Summary
    if len(results_df) > 0:
        summary = results_df.groupby('method')['spearman_rho'].agg(['mean', 'std', 'count'])
        print("\nCross-protein MLP vs GAT summary:")
        print(summary.sort_values('mean', ascending=False))

    return results_df


if __name__ == '__main__':
    t0 = time.time()

    print("Running paper fixes...")
    pw_df = run_ridge_pairwise_fixed()
    cp_df = run_cross_protein_mlp_vs_gat()

    # Save combined summary
    summary = {
        'ridge_pairwise_fixed': {},
        'cross_protein_mlp_vs_gat': {},
    }

    if pw_df is not None and len(pw_df) > 0:
        pw_summary = pw_df.groupby('assay')['spearman_rho'].mean()
        summary['ridge_pairwise_fixed'] = {
            'mean': float(pw_summary.mean()),
            'std': float(pw_summary.std()),
            'per_assay': {k: float(v) for k, v in pw_summary.items()},
        }

    if cp_df is not None and len(cp_df) > 0:
        cp_summary = cp_df.groupby('method')['spearman_rho'].agg(['mean', 'std', 'count'])
        summary['cross_protein_mlp_vs_gat'] = {
            method: {'mean': float(row['mean']), 'std': float(row['std']), 'n': int(row['count'])}
            for method, row in cp_summary.iterrows()
        }

    with open(RESULTS_DIR / "evaluation" / "paper_fixes_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal time: {time.time() - t0:.1f}s")

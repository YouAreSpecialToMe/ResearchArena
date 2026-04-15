#!/usr/bin/env python3
"""
Additional experiments requested by self-review:
1. Cross-protein generalization (leave-3-out: train on 12, test on 3)
2. GB1 order-generalization (train on singles+doubles, predict triples+quadruples)
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

# Import model classes from run_all_v2
sys.path.insert(0, str(Path(__file__).parent))
from run_all_v2 import (RENv2, MLPBaselineV2, build_graph, precompute_variants,
                         load_cached_data, build_onehot)


# ============================================================
# EXPERIMENT 1: Cross-protein generalization
# ============================================================

def run_cross_protein_experiment():
    """
    Leave-3-out cross-protein generalization.
    Train on 12 proteins, test on 3 held-out proteins.
    Tests whether ESM-2 features provide benefit over random features
    in the transfer setting.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Cross-protein generalization")
    print("=" * 60)

    selected, stats, struct_data, embeddings, llr_cache = load_cached_data()

    # Only use assays with both embeddings and structures
    valid_assays = [a for a in selected
                    if a in embeddings and a in struct_data]
    print(f"Valid assays for cross-protein: {len(valid_assays)}")

    # Create 5 folds of 3 test proteins each
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
    print(f"Created {len(folds)} cross-protein folds")

    results = []
    MAX_TRAIN_PER_ASSAY = 5000
    MAX_TEST = 10000

    for fold_idx, (train_assays, test_assays) in enumerate(folds):
        print(f"\nFold {fold_idx}: test on {test_assays}")

        for seed in [42, 123]:
            set_seed(seed)

            # Collect training data from all train proteins
            train_sites_all, train_mut_all, train_mask_all = [], [], []
            train_targets_all = []
            train_plm_all = []

            # For cross-protein, we need a unified graph approach
            # Use protein-level training: for each train protein, get node embeddings
            # and variant predictions, then combine
            # Simpler approach: use ESM-2 embeddings at mutation sites as features (no graph)
            # This is the MLP approach - which is the right test for cross-protein

            # Build feature vectors: for each variant, extract ESM-2 embeddings at mutation sites
            # and average them, concatenated with mutation identity
            def extract_variant_features(assay_name, max_n=None, feat_type='esm'):
                """Extract per-variant features for cross-protein learning."""
                parquet_path = PROCESSED_DIR / f"{assay_name}.parquet"
                df = pd.read_parquet(parquet_path)
                multi = df[df['mutation_order'] >= 2].copy().reset_index(drop=True)
                if max_n and len(multi) > max_n:
                    multi = multi.sample(max_n, random_state=seed).reset_index(drop=True)

                embed = embeddings.get(assay_name)
                if embed is None:
                    return None, None, None
                n_res = embed.shape[0]

                aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
                features = []
                targets = []
                plm_adds = []

                for _, row in multi.iterrows():
                    mutant_str = row['mutant']
                    if not isinstance(mutant_str, str) or not mutant_str:
                        continue

                    site_embeds = []
                    mut_onehots = []
                    for m in mutant_str.split(':'):
                        if len(m) < 3:
                            continue
                        try:
                            pos = int(m[1:-1]) - 1  # 0-indexed
                        except ValueError:
                            continue
                        if 0 <= pos < n_res:
                            if feat_type == 'esm':
                                site_embeds.append(embed[pos].numpy())
                            else:  # random
                                np.random.seed(pos * 31 + 7)  # deterministic per-position random
                                site_embeds.append(np.random.randn(1280).astype(np.float32))
                            oh = np.zeros(20, dtype=np.float32)
                            aa_idx = aa_to_idx.get(m[-1], -1)
                            if aa_idx >= 0:
                                oh[aa_idx] = 1.0
                            mut_onehots.append(oh)

                    if site_embeds:
                        # Average ESM-2 embeddings + average mutation identity
                        avg_emb = np.mean(site_embeds, axis=0)
                        avg_mut = np.mean(mut_onehots, axis=0)
                        feat = np.concatenate([avg_emb, avg_mut])
                        features.append(feat)
                        targets.append(row['DMS_score'])
                        plm_adds.append(row.get('plm_additive', 0.0))

                if not features:
                    return None, None, None
                return np.array(features), np.array(targets), np.array(plm_adds)

            # Build train set
            X_train_list, y_train_list = [], []
            for assay in train_assays:
                X, y, _ = extract_variant_features(assay, max_n=MAX_TRAIN_PER_ASSAY, feat_type='esm')
                if X is not None:
                    X_train_list.append(X)
                    y_train_list.append(y)

            if not X_train_list:
                continue
            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
            print(f"  Train: {X_train.shape[0]} variants from {len(train_assays)} proteins")

            # Also build random-feature train set
            X_train_rand_list, y_train_rand_list = [], []
            for assay in train_assays:
                X, y, _ = extract_variant_features(assay, max_n=MAX_TRAIN_PER_ASSAY, feat_type='random')
                if X is not None:
                    X_train_rand_list.append(X)
                    y_train_rand_list.append(y)
            X_train_rand = np.vstack(X_train_rand_list)
            y_train_rand = np.concatenate(y_train_rand_list)

            # Test on each held-out protein
            for test_assay in test_assays:
                X_test_esm, y_test, plm_test = extract_variant_features(
                    test_assay, max_n=MAX_TEST, feat_type='esm')
                X_test_rand, _, _ = extract_variant_features(
                    test_assay, max_n=MAX_TEST, feat_type='random')
                if X_test_esm is None:
                    continue

                # Ridge with ESM-2 features
                ridge_esm = Ridge(alpha=1.0)
                ridge_esm.fit(X_train, y_train)
                pred_esm = ridge_esm.predict(X_test_esm)
                rho_esm = spearman_corr(y_test, pred_esm)

                # Ridge with random features
                ridge_rand = Ridge(alpha=1.0)
                ridge_rand.fit(X_train_rand, y_train_rand)
                pred_rand = ridge_rand.predict(X_test_rand)
                rho_rand = spearman_corr(y_test, pred_rand)

                # ESM-2 additive baseline
                rho_additive = spearman_corr(y_test, plm_test)

                print(f"  Test {test_assay}: ESM-2={rho_esm:.3f}, Random={rho_rand:.3f}, Additive={rho_additive:.3f}")

                results.append({
                    'fold': fold_idx, 'seed': seed, 'test_assay': test_assay,
                    'method': 'cross_protein_esm2', 'spearman_rho': rho_esm,
                })
                results.append({
                    'fold': fold_idx, 'seed': seed, 'test_assay': test_assay,
                    'method': 'cross_protein_random', 'spearman_rho': rho_rand,
                })
                results.append({
                    'fold': fold_idx, 'seed': seed, 'test_assay': test_assay,
                    'method': 'cross_protein_additive', 'spearman_rho': rho_additive,
                })

    results_df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "evaluation" / "cross_protein_results.csv"
    results_df.to_csv(out_path, index=False)

    # Summary
    if len(results_df) > 0:
        summary = results_df.groupby('method')['spearman_rho'].agg(['mean', 'std', 'count'])
        print("\n\nCross-protein generalization summary:")
        print(summary.sort_values('mean', ascending=False))
    return results_df


# ============================================================
# EXPERIMENT 2: GB1 order-generalization
# ============================================================

def run_gb1_order_generalization():
    """
    Train on GB1 Wu 2016 singles+doubles, predict triples+quadruples.
    Tests whether the model can extrapolate to higher mutation orders.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: GB1 order-generalization")
    print("=" * 60)

    selected, stats, struct_data, embeddings, llr_cache = load_cached_data()

    assay_name = 'SPG1_STRSG_Wu_2016'
    parquet_path = PROCESSED_DIR / f"{assay_name}.parquet"
    df = pd.read_parquet(parquet_path)

    # Split by order
    singles = df[df['mutation_order'] == 1].reset_index(drop=True)
    doubles = df[df['mutation_order'] == 2].reset_index(drop=True)
    triples = df[df['mutation_order'] == 3].reset_index(drop=True)
    quads = df[df['mutation_order'] == 4].reset_index(drop=True)

    print(f"Singles: {len(singles)}, Doubles: {len(doubles)}, Triples: {len(triples)}, Quadruples: {len(quads)}")

    # Train set: singles + doubles
    train_df = pd.concat([singles, doubles], ignore_index=True)
    # Test sets: triples and quadruples separately

    results = []

    # --- Method 1: Ridge regression ---
    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    all_sites = sorted(set(s for sites in df['mutation_sites'] for s in sites))
    site_to_idx = {s: i for i, s in enumerate(all_sites)}
    n_features = len(all_sites) * 20

    X_train = build_onehot(train_df['mutant'].values, site_to_idx, aa_to_idx, n_features)
    y_train = train_df['DMS_score'].values

    for seed in [42, 123, 456]:
        set_seed(seed)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)

        for order, test_df in [(3, triples), (4, quads)]:
            X_test = build_onehot(test_df['mutant'].values, site_to_idx, aa_to_idx, n_features)
            y_test = test_df['DMS_score'].values
            y_pred = ridge.predict(X_test)
            rho = spearman_corr(y_test, y_pred)
            print(f"  Ridge seed={seed} order={order}: rho={rho:.4f}")
            results.append({
                'seed': seed, 'method': 'ridge', 'test_order': order,
                'spearman_rho': rho, 'n_test': len(test_df),
            })

        # ESM-2 additive
        for order, test_df in [(3, triples), (4, quads)]:
            if 'plm_additive' in test_df.columns:
                rho = spearman_corr(test_df['DMS_score'].values, test_df['plm_additive'].values)
                results.append({
                    'seed': seed, 'method': 'esm2_additive', 'test_order': order,
                    'spearman_rho': rho, 'n_test': len(test_df),
                })

    # --- Method 2: REN-E2E (GAT) ---
    graph = build_graph(assay_name, struct_data, embeddings, threshold=10.0, node_feat_config='full')
    if graph is None:
        print("ERROR: No graph for GB1 Wu 2016")
        return pd.DataFrame(results)

    n_residues = graph['n_residues']
    esm_dim = graph['esm_dim']
    edge_index = graph['edge_index'].to(DEVICE)
    graph_features = graph['base_features'].to(DEVICE)

    # Precompute variants for train and test
    max_muts = 5
    train_sites, train_mut, train_mask = precompute_variants(train_df, n_residues, max_muts=max_muts)
    triple_sites, triple_mut, triple_mask = precompute_variants(triples, n_residues, max_muts=max_muts)
    quad_sites, quad_mut, quad_mask = precompute_variants(quads, n_residues, max_muts=max_muts)

    train_targets = train_df['DMS_score'].values.astype(np.float32)
    t_mean, t_std = np.mean(train_targets), max(np.std(train_targets), 1e-6)
    train_targets_norm = (train_targets - t_mean) / t_std

    train_plm = train_df['plm_additive'].values.astype(np.float32) if 'plm_additive' in train_df.columns else np.zeros(len(train_df), dtype=np.float32)

    for seed in [42, 123, 456]:
        set_seed(seed)

        # Train REN-E2E
        model = RENv2(esm_dim=esm_dim, hidden_dim=256, num_heads=8, num_layers=3,
                      conv_type='gat', use_plm_input=True).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)

        sites_gpu = train_sites.to(DEVICE)
        mut_gpu = train_mut.to(DEVICE)
        mask_gpu = train_mask.to(DEVICE)
        targets_gpu = torch.tensor(train_targets_norm, dtype=torch.float32).to(DEVICE)
        plm_gpu = torch.tensor(train_plm, dtype=torch.float32).to(DEVICE)

        # Val split from train
        n_val = max(10, len(train_df) // 10)
        val_idx = np.random.choice(len(train_df), n_val, replace=False)
        train_sub = np.array([i for i in range(len(train_df)) if i not in val_idx])

        best_val_loss = float('inf')
        best_state = None
        patience = 15
        patience_counter = 0

        for epoch in range(100):
            model.train()
            ne = model.encode_graph(graph_features, edge_index)
            perm = np.random.permutation(len(train_sub))
            epoch_loss = 0.0
            n_b = 0
            for start in range(0, len(perm), 1024):
                end = min(start + 1024, len(perm))
                bi = train_sub[perm[start:end]]
                preds = model.predict_batch(ne, sites_gpu[bi], mut_gpu[bi], mask_gpu[bi],
                                           plm_additive=plm_gpu[bi])
                loss = F.mse_loss(preds, targets_gpu[bi])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ne = model.encode_graph(graph_features, edge_index)
                epoch_loss += loss.item()
                n_b += 1
            scheduler.step()

            model.eval()
            with torch.no_grad():
                ne = model.encode_graph(graph_features, edge_index)
                vp = model.predict_batch(ne, sites_gpu[val_idx], mut_gpu[val_idx],
                                        mask_gpu[val_idx], plm_additive=plm_gpu[val_idx])
                vl = F.mse_loss(vp, targets_gpu[val_idx]).item()
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        # Evaluate on triples and quadruples
        for order, test_df, t_sites, t_mut, t_mask in [
            (3, triples, triple_sites, triple_mut, triple_mask),
            (4, quads, quad_sites, quad_mut, quad_mask)
        ]:
            test_plm = test_df['plm_additive'].values.astype(np.float32) if 'plm_additive' in test_df.columns else np.zeros(len(test_df), dtype=np.float32)
            ts_gpu = t_sites.to(DEVICE)
            tm_gpu = t_mut.to(DEVICE)
            tmask_gpu = t_mask.to(DEVICE)
            plm_t_gpu = torch.tensor(test_plm, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                ne = model.encode_graph(graph_features, edge_index)
                all_preds = []
                for s in range(0, len(test_df), 2048):
                    e = min(s + 2048, len(test_df))
                    p = model.predict_batch(ne, ts_gpu[s:e], tm_gpu[s:e], tmask_gpu[s:e],
                                           plm_additive=plm_t_gpu[s:e])
                    all_preds.append(p.cpu())
                preds = torch.cat(all_preds).numpy() * t_std + t_mean

            y_test = test_df['DMS_score'].values
            rho = spearman_corr(y_test, preds)
            print(f"  REN-E2E seed={seed} order={order}: rho={rho:.4f}")
            results.append({
                'seed': seed, 'method': 'ren_e2e', 'test_order': order,
                'spearman_rho': rho, 'n_test': len(test_df),
            })

        # Also MLP baseline
        set_seed(seed)
        mlp_model = MLPBaselineV2(esm_dim=esm_dim, hidden_dim=256, use_plm_input=True).to(DEVICE)
        optimizer = AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(100):
            mlp_model.train()
            ne = mlp_model.encode_graph(graph_features, edge_index)
            perm = np.random.permutation(len(train_sub))
            for start in range(0, len(perm), 1024):
                end = min(start + 1024, len(perm))
                bi = train_sub[perm[start:end]]
                preds = mlp_model.predict_batch(ne, sites_gpu[bi], mut_gpu[bi], mask_gpu[bi],
                                               plm_additive=plm_gpu[bi])
                loss = F.mse_loss(preds, targets_gpu[bi])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), 1.0)
                optimizer.step()
                ne = mlp_model.encode_graph(graph_features, edge_index)
            scheduler.step()

            mlp_model.eval()
            with torch.no_grad():
                ne = mlp_model.encode_graph(graph_features, edge_index)
                vp = mlp_model.predict_batch(ne, sites_gpu[val_idx], mut_gpu[val_idx],
                                            mask_gpu[val_idx], plm_additive=plm_gpu[val_idx])
                vl = F.mse_loss(vp, targets_gpu[val_idx]).item()
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.cpu().clone() for k, v in mlp_model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state:
            mlp_model.load_state_dict(best_state)
        mlp_model.eval()

        for order, test_df, t_sites, t_mut, t_mask in [
            (3, triples, triple_sites, triple_mut, triple_mask),
            (4, quads, quad_sites, quad_mut, quad_mask)
        ]:
            test_plm = test_df['plm_additive'].values.astype(np.float32) if 'plm_additive' in test_df.columns else np.zeros(len(test_df), dtype=np.float32)
            ts_gpu = t_sites.to(DEVICE)
            tm_gpu = t_mut.to(DEVICE)
            tmask_gpu = t_mask.to(DEVICE)
            plm_t_gpu = torch.tensor(test_plm, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                ne = mlp_model.encode_graph(graph_features, edge_index)
                all_preds = []
                for s in range(0, len(test_df), 2048):
                    e = min(s + 2048, len(test_df))
                    p = mlp_model.predict_batch(ne, ts_gpu[s:e], tm_gpu[s:e], tmask_gpu[s:e],
                                               plm_additive=plm_t_gpu[s:e])
                    all_preds.append(p.cpu())
                preds = torch.cat(all_preds).numpy() * t_std + t_mean

            y_test = test_df['DMS_score'].values
            rho = spearman_corr(y_test, preds)
            print(f"  MLP seed={seed} order={order}: rho={rho:.4f}")
            results.append({
                'seed': seed, 'method': 'mlp', 'test_order': order,
                'spearman_rho': rho, 'n_test': len(test_df),
            })

    results_df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "evaluation" / "gb1_order_generalization.csv"
    results_df.to_csv(out_path, index=False)

    # Summary
    if len(results_df) > 0:
        summary = results_df.groupby(['method', 'test_order'])['spearman_rho'].agg(['mean', 'std'])
        print("\n\nGB1 Order Generalization Summary:")
        print(summary)
    return results_df


if __name__ == "__main__":
    t0 = time.time()
    cp_results = run_cross_protein_experiment()
    gb1_results = run_gb1_order_generalization()

    # Save combined summary
    summary = {
        'cross_protein': {},
        'gb1_order_gen': {},
    }
    if len(cp_results) > 0:
        for method in cp_results['method'].unique():
            subset = cp_results[cp_results['method'] == method]
            summary['cross_protein'][method] = {
                'mean': float(subset['spearman_rho'].mean()),
                'std': float(subset['spearman_rho'].std()),
                'n': int(len(subset)),
            }
    if len(gb1_results) > 0:
        for method in gb1_results['method'].unique():
            for order in gb1_results['test_order'].unique():
                subset = gb1_results[(gb1_results['method'] == method) & (gb1_results['test_order'] == order)]
                if len(subset) > 0:
                    key = f"{method}_order{int(order)}"
                    summary['gb1_order_gen'][key] = {
                        'mean': float(subset['spearman_rho'].mean()),
                        'std': float(subset['spearman_rho'].std()),
                        'n': int(len(subset)),
                    }

    with open(RESULTS_DIR / "evaluation" / "additional_experiments.json", 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print("Done!")

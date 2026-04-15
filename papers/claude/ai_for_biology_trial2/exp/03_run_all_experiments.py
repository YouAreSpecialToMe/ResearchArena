"""Step 3: Run all experiments - heavily optimized for speed."""
import os
import sys
import json
import time
import ast
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
from copy import deepcopy

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.config import *
from shared.models import EpiGNN, MLPBaseline
from shared.metrics import compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Speed settings
N_FOLDS_RUN = 3
GNN_MAX_EPOCHS = 80
GNN_PATIENCE = 10
MLP_MAX_EPOCHS = 60
MLP_PATIENCE = 8
BATCH_SZ = 128
MAX_VARIANTS = 15000  # subsample large datasets

AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


def build_all_graphs(df, wt_emb, coupling, mm, use_random_coupling=False, seed=42):
    """Build PyG graphs for all variants."""
    if use_random_coupling:
        rng = np.random.RandomState(seed)
    L, D = wt_emb.shape
    graphs = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        muts = ast.literal_eval(row['mutations_parsed']) if isinstance(row['mutations_parsed'], str) else row['mutations_parsed']
        k = len(muts)
        positions = []
        node_feats = []

        for wt, pos, mut in muts:
            if pos >= L: pos -= 1
            pos = max(0, min(pos, L - 1))
            positions.append(pos)
            feat = wt_emb[pos].clone()
            mut_idx = AA_TO_IDX.get(mut, 0)
            feat = feat * (1.0 + 0.1 * mm[pos, mut_idx].item())
            node_feats.append(feat)

        x = torch.stack(node_feats)

        if k >= 2:
            src, dst, ef = [], [], []
            for i in range(k):
                for j in range(k):
                    if i == j: continue
                    src.append(i); dst.append(j)
                    pi, pj = positions[i], positions[j]
                    ss = abs(pi - pj)
                    c = rng.uniform(0, 1) if use_random_coupling else coupling[pi, pj].item()
                    ef.append([c, ss / 100.0, 1.0 / (1.0 + ss)])
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(ef, dtype=torch.float)
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float)

        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        g.fitness = torch.tensor([row['fitness']], dtype=torch.float)
        g.epistasis = torch.tensor([row['epistasis_score']], dtype=torch.float)
        g.additive_score = torch.tensor([row['esm2_additive_score']], dtype=torch.float)
        g.num_mut = k
        graphs.append(g)

    return graphs


def split_data(df, seed):
    """Get 3-fold CV splits."""
    strat = df['num_mutations'].clip(upper=4).values
    skf = StratifiedKFold(n_splits=N_FOLDS_RUN, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(df, strat)]


# ======================================================================
# Training functions
# ======================================================================
def train_gnn(train_g, val_g, test_g, num_layers=2, target='epistasis',
              seed=42, input_dim=ESM2_EMBED_DIM):
    torch.manual_seed(seed)
    tl = DataLoader(train_g, batch_size=BATCH_SZ, shuffle=True, num_workers=0)
    vl = DataLoader(val_g, batch_size=BATCH_SZ*2, num_workers=0)
    tel = DataLoader(test_g, batch_size=BATCH_SZ*2, num_workers=0)

    model = EpiGNN(input_dim=input_dim, hidden_dim=GNN_HIDDEN_DIM,
                   num_heads=GNN_NUM_HEADS, num_layers=num_layers,
                   edge_dim=GNN_EDGE_DIM, dropout=GNN_DROPOUT).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, mode='max')

    best_rho, best_state, wait = -1e9, None, 0
    for ep in range(GNN_MAX_EPOCHS):
        model.train()
        for b in tl:
            b = b.to(device)
            ep_pred = model(b)
            if target == 'epistasis':
                loss = nn.MSELoss()(ep_pred, b.epistasis.squeeze())
            else:
                loss = nn.MSELoss()(b.additive_score.squeeze() + ep_pred, b.fitness.squeeze())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for b in vl:
                b = b.to(device)
                p = b.additive_score.squeeze() + model(b)
                vp.extend(p.cpu().tolist()); vt.extend(b.fitness.squeeze().cpu().tolist())
        rho = spearmanr(vt, vp).statistic
        if np.isnan(rho): rho = -1.0
        sched.step(rho)
        if rho > best_rho:
            best_rho, best_state, wait = rho, deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= GNN_PATIENCE: break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    ps, fs, ads, eps_, nms = [], [], [], [], []
    with torch.no_grad():
        for b in tel:
            b = b.to(device)
            p = b.additive_score.squeeze() + model(b)
            ps.extend(p.cpu().tolist())
            fs.extend(b.fitness.squeeze().cpu().tolist())
            ads.extend(b.additive_score.squeeze().cpu().tolist())
            eps_.extend(b.epistasis.squeeze().cpu().tolist())
            for i in range(b.num_graphs):
                nms.append(int((b.batch == i).sum().item()))
    return compute_metrics(fs, ps, additive_scores=ads, epistasis_true=eps_, num_mutations=nms)


def train_mlp(train_g, val_g, test_g, seed=42):
    torch.manual_seed(seed)
    tl = DataLoader(train_g, batch_size=BATCH_SZ, shuffle=True, num_workers=0)
    vl = DataLoader(val_g, batch_size=BATCH_SZ*2, num_workers=0)
    tel = DataLoader(test_g, batch_size=BATCH_SZ*2, num_workers=0)

    model = MLPBaseline(input_dim=ESM2_EMBED_DIM, hidden_dims=MLP_HIDDEN_DIMS,
                        extra_features=4, dropout=GNN_DROPOUT).to(device)
    opt = optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=WEIGHT_DECAY)

    def get_extra(b):
        bs = b.num_graphs
        extra = torch.zeros(bs, 4, device=device)
        # Vectorized: use scatter for efficiency
        extra[:, 0] = b.additive_score.squeeze()
        # Count nodes per graph
        ones = torch.ones(b.x.size(0), device=device)
        counts = torch.zeros(bs, device=device)
        counts.scatter_add_(0, b.batch, ones)
        extra[:, 1] = counts
        # Mean coupling per graph
        if b.edge_attr.size(0) > 0:
            edge_batch = b.batch[b.edge_index[0]]
            coup = b.edge_attr[:, 0]
            coup_sum = torch.zeros(bs, device=device).scatter_add_(0, edge_batch, coup)
            coup_cnt = torch.zeros(bs, device=device).scatter_add_(0, edge_batch, torch.ones_like(coup))
            coup_cnt = coup_cnt.clamp(min=1)
            extra[:, 2] = coup_sum / coup_cnt
            # Max coupling - use scatter_reduce
            extra[:, 3] = torch.zeros(bs, device=device).scatter_reduce_(0, edge_batch, coup, reduce='amax', include_self=False)
        return extra

    best_rho, best_state, wait = -1e9, None, 0
    for ep in range(MLP_MAX_EPOCHS):
        model.train()
        for b in tl:
            b = b.to(device)
            extra = get_extra(b)
            ep_pred = model(b.x, extra, b.batch)
            loss = nn.MSELoss()(ep_pred, b.epistasis.squeeze())
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for b in vl:
                b = b.to(device)
                extra = get_extra(b)
                p = b.additive_score.squeeze() + model(b.x, extra, b.batch)
                vp.extend(p.cpu().tolist()); vt.extend(b.fitness.squeeze().cpu().tolist())
        rho = spearmanr(vt, vp).statistic
        if np.isnan(rho): rho = -1.0
        if rho > best_rho:
            best_rho, best_state, wait = rho, deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= MLP_PATIENCE: break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    ps, fs, ads, eps_, nms = [], [], [], [], []
    with torch.no_grad():
        for b in tel:
            b = b.to(device)
            extra = get_extra(b)
            p = b.additive_score.squeeze() + model(b.x, extra, b.batch)
            ps.extend(p.cpu().tolist())
            fs.extend(b.fitness.squeeze().cpu().tolist())
            ads.extend(b.additive_score.squeeze().cpu().tolist())
            eps_.extend(b.epistasis.squeeze().cpu().tolist())
            for i in range(b.num_graphs):
                nms.append(int((b.batch == i).sum().item()))
    return compute_metrics(fs, ps, additive_scores=ads, epistasis_true=eps_, num_mutations=nms)


def run_ridge(df, wt_emb, coupling, train_idx, test_idx):
    L = wt_emb.shape[0]
    def feats(indices):
        rows = []
        for i in indices:
            r = df.iloc[i]
            muts = ast.literal_eval(r['mutations_parsed']) if isinstance(r['mutations_parsed'], str) else r['mutations_parsed']
            poss = [max(0, min(p - 1 if p >= L else p, L - 1)) for _, p, _ in muts]
            es = sum(wt_emb[p].numpy() for p in poss)
            cs = [coupling[poss[a], poss[b]].item() for a in range(len(poss)) for b in range(a+1, len(poss))]
            rows.append(np.concatenate([[r['esm2_additive_score'], r['num_mutations'],
                                         np.mean(cs) if cs else 0, max(cs) if cs else 0], es]))
        return np.array(rows, dtype=np.float32)

    Xtr, Xte = feats(train_idx), feats(test_idx)
    nc = min(32, Xtr.shape[0]-1, Xtr.shape[1]-4)
    pca = PCA(n_components=nc)
    Xtr2 = np.hstack([Xtr[:,:4], pca.fit_transform(Xtr[:,4:])])
    Xte2 = np.hstack([Xte[:,:4], pca.transform(Xte[:,4:])])
    sc = StandardScaler()
    Xtr2 = sc.fit_transform(Xtr2); Xte2 = sc.transform(Xte2)

    m = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
    m.fit(Xtr2, df.iloc[train_idx]['fitness'].values)
    yp = m.predict(Xte2)

    return compute_metrics(
        df.iloc[test_idx]['fitness'].values, yp,
        additive_scores=df.iloc[test_idx]['esm2_additive_score'].values,
        epistasis_true=df.iloc[test_idx]['epistasis_score'].values,
        num_mutations=df.iloc[test_idx]['num_mutations'].values)


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("=" * 60, flush=True)
    print("RUNNING ALL EXPERIMENTS", flush=True)
    print("=" * 60, flush=True)

    with open(os.path.join(DATA_DIR, "selected_proteins.json")) as f:
        proteins = json.load(f)

    methods = ['additive_esm2', 'ridge', 'mlp', 'epignn',
               'ablation_random_edges', 'ablation_1layer', 'ablation_3layer',
               'ablation_fitness_target']
    R = {m: {} for m in methods}
    total_t = time.time()

    for pname in proteins:
        pt = time.time()
        print(f"\nPROTEIN: {pname}", flush=True)

        df = pd.read_parquet(os.path.join(DATA_DIR, "processed", f"{pname}.parquet"))
        wt_emb = torch.load(os.path.join(FEATURES_DIR, f"{pname}_wt_embeddings.pt"), weights_only=True)
        coup = torch.load(os.path.join(FEATURES_DIR, f"{pname}_coupling.pt"), weights_only=True)
        mm = torch.load(os.path.join(FEATURES_DIR, f"{pname}_masked_marginal.pt"), weights_only=True)

        if len(df) > MAX_VARIANTS:
            df = df.sample(n=MAX_VARIANTS, random_state=42).reset_index(drop=True)

        print(f"  n={len(df)}, L={wt_emb.shape[0]}", flush=True)

        # Build graphs once
        t0 = time.time()
        graphs = build_all_graphs(df, wt_emb, coup, mm)
        graphs_rand = build_all_graphs(df, wt_emb, coup, mm, use_random_coupling=True, seed=42)
        print(f"  Graphs built in {time.time()-t0:.1f}s", flush=True)

        for m in methods:
            R[m][pname] = {}

        for seed in SEEDS:
            for m in methods:
                R[m][pname][str(seed)] = {}

            folds = split_data(df, seed)
            for fi, (train_idx, test_idx) in enumerate(folds):
                t0 = time.time()
                rng = np.random.RandomState(seed + fi)
                nv = max(1, len(train_idx) // 5)
                perm = rng.permutation(len(train_idx))
                val_i = train_idx[perm[:nv]]
                tr_i = train_idx[perm[nv:]]

                tg = [graphs[i] for i in tr_i]
                vg = [graphs[i] for i in val_i]
                teg = [graphs[i] for i in test_idx]
                tgr = [graphs_rand[i] for i in tr_i]
                vgr = [graphs_rand[i] for i in val_i]
                tegr = [graphs_rand[i] for i in test_idx]

                # Additive
                fitness = [graphs[i].fitness.item() for i in test_idx]
                additive = [graphs[i].additive_score.item() for i in test_idx]
                epi = [graphs[i].epistasis.item() for i in test_idx]
                nmut = [graphs[i].num_mut for i in test_idx]
                R['additive_esm2'][pname][str(seed)][str(fi)] = compute_metrics(
                    fitness, additive, additive_scores=additive, epistasis_true=epi, num_mutations=nmut)

                # Ridge
                R['ridge'][pname][str(seed)][str(fi)] = run_ridge(df, wt_emb, coup, train_idx, test_idx)

                # MLP
                R['mlp'][pname][str(seed)][str(fi)] = train_mlp(tg, vg, teg, seed=seed)

                # EpiGNN variants
                R['epignn'][pname][str(seed)][str(fi)] = train_gnn(tg, vg, teg, num_layers=2, target='epistasis', seed=seed)
                R['ablation_random_edges'][pname][str(seed)][str(fi)] = train_gnn(tgr, vgr, tegr, num_layers=2, target='epistasis', seed=seed)
                R['ablation_1layer'][pname][str(seed)][str(fi)] = train_gnn(tg, vg, teg, num_layers=1, target='epistasis', seed=seed)
                R['ablation_3layer'][pname][str(seed)][str(fi)] = train_gnn(tg, vg, teg, num_layers=3, target='epistasis', seed=seed)
                R['ablation_fitness_target'][pname][str(seed)][str(fi)] = train_gnn(tg, vg, teg, num_layers=2, target='fitness', seed=seed)

                el = time.time() - t0
                ar = R['additive_esm2'][pname][str(seed)][str(fi)].get('spearman', 0)
                er = R['epignn'][pname][str(seed)][str(fi)].get('spearman', 0)
                rr = R['ridge'][pname][str(seed)][str(fi)].get('spearman', 0)
                mr = R['mlp'][pname][str(seed)][str(fi)].get('spearman', 0)
                print(f"  s{seed} f{fi}: add={ar:.3f} ridge={rr:.3f} mlp={mr:.3f} epignn={er:.3f} ({el:.0f}s)", flush=True)

        # Save intermediate
        for m in methods:
            with open(os.path.join(RESULTS_DIR, f"{m}.json"), 'w') as f:
                json.dump(R[m], f, indent=2)

        print(f"  Done in {(time.time()-pt)/60:.1f}m, total {(time.time()-total_t)/60:.1f}m", flush=True)

    print(f"\nALL DONE in {(time.time()-total_t)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

"""Additional ablations with fitness target: random edges, MLP, layer variants."""
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
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
from copy import deepcopy

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.config import *
from shared.models import EpiGNN, MLPBaseline
from shared.metrics import compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_FOLDS_RUN = 3
GNN_MAX_EPOCHS = 80
GNN_PATIENCE = 10
MLP_MAX_EPOCHS = 60
MLP_PATIENCE = 8
BATCH_SZ = 128
MAX_VARIANTS = 15000
AA_TO_IDX = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}


def build_all_graphs(df, wt_emb, coupling, mm, use_random_coupling=False, seed=42):
    if use_random_coupling:
        rng = np.random.RandomState(seed)
    L, D = wt_emb.shape
    graphs = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        muts = ast.literal_eval(row['mutations_parsed']) if isinstance(row['mutations_parsed'], str) else row['mutations_parsed']
        k = len(muts)
        positions, node_feats = [], []
        for wt, pos, mut in muts:
            if pos >= L: pos -= 1
            pos = max(0, min(pos, L - 1))
            positions.append(pos)
            feat = wt_emb[pos].clone()
            feat = feat * (1.0 + 0.1 * mm[pos, AA_TO_IDX.get(mut, 0)].item())
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


def train_gnn(train_g, val_g, test_g, num_layers=2, target='fitness', seed=42):
    torch.manual_seed(seed)
    tl = DataLoader(train_g, batch_size=BATCH_SZ, shuffle=True, num_workers=0)
    vl = DataLoader(val_g, batch_size=BATCH_SZ*2, num_workers=0)
    tel = DataLoader(test_g, batch_size=BATCH_SZ*2, num_workers=0)
    model = EpiGNN(input_dim=ESM2_EMBED_DIM, hidden_dim=GNN_HIDDEN_DIM,
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
            if target == 'fitness':
                loss = nn.MSELoss()(b.additive_score.squeeze() + ep_pred, b.fitness.squeeze())
            else:
                loss = nn.MSELoss()(ep_pred, b.epistasis.squeeze())
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
            ps.extend(p.cpu().tolist()); fs.extend(b.fitness.squeeze().cpu().tolist())
            ads.extend(b.additive_score.squeeze().cpu().tolist())
            eps_.extend(b.epistasis.squeeze().cpu().tolist())
            for i in range(b.num_graphs):
                nms.append(int((b.batch == i).sum().item()))
    return compute_metrics(fs, ps, additive_scores=ads, epistasis_true=eps_, num_mutations=nms)


def train_mlp_fitness(train_g, val_g, test_g, seed=42):
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
        extra[:, 0] = b.additive_score.squeeze()
        ones = torch.ones(b.x.size(0), device=device)
        counts = torch.zeros(bs, device=device).scatter_add_(0, b.batch, ones)
        extra[:, 1] = counts
        if b.edge_attr.size(0) > 0:
            eb = b.batch[b.edge_index[0]]
            c = b.edge_attr[:, 0]
            cs = torch.zeros(bs, device=device).scatter_add_(0, eb, c)
            cc = torch.zeros(bs, device=device).scatter_add_(0, eb, torch.ones_like(c)).clamp(min=1)
            extra[:, 2] = cs / cc
            extra[:, 3] = torch.zeros(bs, device=device).scatter_reduce_(0, eb, c, reduce='amax', include_self=False)
        return extra

    best_rho, best_state, wait = -1e9, None, 0
    for ep in range(MLP_MAX_EPOCHS):
        model.train()
        for b in tl:
            b = b.to(device)
            extra = get_extra(b)
            ep_pred = model(b.x, extra, b.batch)
            # Fitness target
            pred_fitness = b.additive_score.squeeze() + ep_pred
            loss = nn.MSELoss()(pred_fitness, b.fitness.squeeze())
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for b in vl:
                b = b.to(device)
                p = b.additive_score.squeeze() + model(b.x, get_extra(b), b.batch)
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
            p = b.additive_score.squeeze() + model(b.x, get_extra(b), b.batch)
            ps.extend(p.cpu().tolist()); fs.extend(b.fitness.squeeze().cpu().tolist())
            ads.extend(b.additive_score.squeeze().cpu().tolist())
            eps_.extend(b.epistasis.squeeze().cpu().tolist())
            for i in range(b.num_graphs):
                nms.append(int((b.batch == i).sum().item()))
    return compute_metrics(fs, ps, additive_scores=ads, epistasis_true=eps_, num_mutations=nms)


def main():
    print("=" * 60, flush=True)
    print("RUNNING FITNESS-TARGET ABLATIONS", flush=True)
    print("=" * 60, flush=True)

    with open(os.path.join(DATA_DIR, "selected_proteins.json")) as f:
        proteins = json.load(f)

    ablations = {
        'ft_random_edges': {},
        'ft_mlp': {},
        'ft_1layer': {},
        'ft_3layer': {},
    }
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

        graphs = build_all_graphs(df, wt_emb, coup, mm)
        graphs_rand = build_all_graphs(df, wt_emb, coup, mm, use_random_coupling=True, seed=42)

        for a in ablations:
            ablations[a][pname] = {}

        for seed in SEEDS:
            for a in ablations:
                ablations[a][pname][str(seed)] = {}

            strat = df['num_mutations'].clip(upper=4).values
            skf = StratifiedKFold(n_splits=N_FOLDS_RUN, shuffle=True, random_state=seed)
            folds = [(tr, te) for tr, te in skf.split(df, strat)]

            for fi, (train_idx, test_idx) in enumerate(folds):
                t0 = time.time()
                rng = np.random.RandomState(seed + fi)
                nv = max(1, len(train_idx) // 5)
                perm = rng.permutation(len(train_idx))
                val_i, tr_i = train_idx[perm[:nv]], train_idx[perm[nv:]]

                tg = [graphs[i] for i in tr_i]
                vg = [graphs[i] for i in val_i]
                teg = [graphs[i] for i in test_idx]
                tgr = [graphs_rand[i] for i in tr_i]
                vgr = [graphs_rand[i] for i in val_i]
                tegr = [graphs_rand[i] for i in test_idx]

                # Random edges + fitness target
                ablations['ft_random_edges'][pname][str(seed)][str(fi)] = train_gnn(
                    tgr, vgr, tegr, num_layers=2, target='fitness', seed=seed)

                # MLP + fitness target
                ablations['ft_mlp'][pname][str(seed)][str(fi)] = train_mlp_fitness(
                    tg, vg, teg, seed=seed)

                # 1-layer + fitness target
                ablations['ft_1layer'][pname][str(seed)][str(fi)] = train_gnn(
                    tg, vg, teg, num_layers=1, target='fitness', seed=seed)

                # 3-layer + fitness target
                ablations['ft_3layer'][pname][str(seed)][str(fi)] = train_gnn(
                    tg, vg, teg, num_layers=3, target='fitness', seed=seed)

                el = time.time() - t0
                e1 = ablations['ft_random_edges'][pname][str(seed)][str(fi)].get('spearman', 0)
                e2 = ablations['ft_mlp'][pname][str(seed)][str(fi)].get('spearman', 0)
                print(f"  s{seed} f{fi}: rand={e1:.3f} mlp={e2:.3f} ({el:.0f}s)", flush=True)

        for a in ablations:
            with open(os.path.join(RESULTS_DIR, f"{a}.json"), 'w') as f:
                json.dump(ablations[a], f, indent=2)

        print(f"  Done in {(time.time()-pt)/60:.1f}m", flush=True)

    print(f"\nAll ablations done in {(time.time()-total_t)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

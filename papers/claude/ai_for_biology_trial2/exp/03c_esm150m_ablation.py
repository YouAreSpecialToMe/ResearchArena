"""ESM-2 150M ablation: test with smaller backbone."""
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
from shared.models import EpiGNN
from shared.metrics import compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_FOLDS_RUN = 3
GNN_MAX_EPOCHS = 80
GNN_PATIENCE = 10
BATCH_SZ = 128
MAX_VARIANTS = 15000
AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


def extract_150m_features(proteins, info):
    """Extract ESM-2 150M features."""
    import esm as esm_module
    print("Loading ESM-2 150M...", flush=True)
    model, alphabet = esm_module.pretrained.esm2_t30_150M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().cuda()

    for pname in proteins:
        pinfo = info[pname]
        wt_seq = pinfo['wt_sequence']
        wt_seq = ''.join(c for c in wt_seq if c in AA_LIST)
        prefix = f"{pname}_150M"

        if os.path.exists(os.path.join(FEATURES_DIR, f"{prefix}_wt_embeddings.pt")):
            print(f"  {pname}: already extracted", flush=True)
            continue

        print(f"  Extracting 150M features for {pname} (L={len(wt_seq)})...", flush=True)
        data = [("protein", wt_seq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[30], need_head_weights=True)

        emb = results["representations"][30][0, 1:len(wt_seq)+1].cpu()
        torch.save(emb, os.path.join(FEATURES_DIR, f"{prefix}_wt_embeddings.pt"))

        attn = results["attentions"][0][:, :, 1:len(wt_seq)+1, 1:len(wt_seq)+1].cpu()
        deep_attn = attn[ESM2_150M_COUPLING_LAYER_START:30]
        C_raw = (deep_attn + deep_attn.transpose(-2, -1)) / 2
        C_raw = C_raw.mean(dim=(0, 1))
        rm = C_raw.mean(dim=1, keepdim=True)
        cm = C_raw.mean(dim=0, keepdim=True)
        C_apc = C_raw - (rm * cm) / C_raw.mean()
        C_apc.fill_diagonal_(0)
        torch.save(C_apc, os.path.join(FEATURES_DIR, f"{prefix}_coupling.pt"))

        # Masked marginals
        L = len(wt_seq)
        mm = torch.zeros(L, 20)
        aa_to_tok = {aa: alphabet.get_idx(aa) for aa in AA_LIST}
        for start in range(0, L, 50):
            end = min(start + 50, L)
            masked = tokens[0].unsqueeze(0).repeat(end - start, 1).to(device)
            for i, pos in enumerate(range(start, end)):
                masked[i, pos + 1] = alphabet.mask_idx
            with torch.no_grad():
                logits = model(masked)["logits"]
            for i, pos in enumerate(range(start, end)):
                lp = torch.log_softmax(logits[i, pos + 1], dim=-1)
                wt_aa = wt_seq[pos]
                wt_lp = lp[aa_to_tok.get(wt_aa, 0)].item()
                for j, aa in enumerate(AA_LIST):
                    mm[pos, j] = lp[aa_to_tok[aa]].item() - wt_lp
        torch.save(mm, os.path.join(FEATURES_DIR, f"{prefix}_masked_marginal.pt"))

        del tokens, results, attn
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()


def build_graphs_150m(df, wt_emb, coupling, mm):
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
                    ef.append([coupling[pi, pj].item(), ss / 100.0, 1.0 / (1.0 + ss)])
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


def train_eval_gnn_150m(train_g, val_g, test_g, seed=42):
    torch.manual_seed(seed)
    tl = DataLoader(train_g, batch_size=BATCH_SZ, shuffle=True, num_workers=0)
    vl = DataLoader(val_g, batch_size=BATCH_SZ*2, num_workers=0)
    tel = DataLoader(test_g, batch_size=BATCH_SZ*2, num_workers=0)
    model = EpiGNN(input_dim=ESM2_150M_EMBED_DIM, hidden_dim=GNN_HIDDEN_DIM,
                   num_heads=GNN_NUM_HEADS, num_layers=2,
                   edge_dim=GNN_EDGE_DIM, dropout=GNN_DROPOUT).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, mode='max')
    best_rho, best_state, wait = -1e9, None, 0
    for ep in range(GNN_MAX_EPOCHS):
        model.train()
        for b in tl:
            b = b.to(device)
            ep_pred = model(b)
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
            ps.extend(p.cpu().tolist()); fs.extend(b.fitness.squeeze().cpu().tolist())
            ads.extend(b.additive_score.squeeze().cpu().tolist())
            eps_.extend(b.epistasis.squeeze().cpu().tolist())
            for i in range(b.num_graphs):
                nms.append(int((b.batch == i).sum().item()))
    return compute_metrics(fs, ps, additive_scores=ads, epistasis_true=eps_, num_mutations=nms)


def main():
    print("=" * 60, flush=True)
    print("ESM-2 150M ABLATION", flush=True)
    print("=" * 60, flush=True)

    with open(os.path.join(DATA_DIR, "selected_proteins.json")) as f:
        proteins = json.load(f)
    with open(os.path.join(DATA_DIR, "processed_info.json")) as f:
        info = json.load(f)

    # Extract 150M features
    extract_150m_features(proteins, info)

    results = {}
    total_t = time.time()

    for pname in proteins:
        print(f"\nPROTEIN: {pname}", flush=True)
        df = pd.read_parquet(os.path.join(DATA_DIR, "processed", f"{pname}.parquet"))
        if len(df) > MAX_VARIANTS:
            df = df.sample(n=MAX_VARIANTS, random_state=42).reset_index(drop=True)

        prefix = f"{pname}_150M"
        wt_emb = torch.load(os.path.join(FEATURES_DIR, f"{prefix}_wt_embeddings.pt"), weights_only=True)
        coup = torch.load(os.path.join(FEATURES_DIR, f"{prefix}_coupling.pt"), weights_only=True)
        mm = torch.load(os.path.join(FEATURES_DIR, f"{prefix}_masked_marginal.pt"), weights_only=True)

        graphs = build_graphs_150m(df, wt_emb, coup, mm)
        results[pname] = {}

        for seed in SEEDS:
            results[pname][str(seed)] = {}
            strat = df['num_mutations'].clip(upper=4).values
            skf = StratifiedKFold(n_splits=N_FOLDS_RUN, shuffle=True, random_state=seed)
            folds = list(skf.split(df, strat))

            for fi, (train_idx, test_idx) in enumerate(folds):
                rng = np.random.RandomState(seed + fi)
                nv = max(1, len(train_idx) // 5)
                perm = rng.permutation(len(train_idx))
                val_i, tr_i = train_idx[perm[:nv]], train_idx[perm[nv:]]
                tg = [graphs[i] for i in tr_i]
                vg = [graphs[i] for i in val_i]
                teg = [graphs[i] for i in test_idx]

                res = train_eval_gnn_150m(tg, vg, teg, seed=seed)
                results[pname][str(seed)][str(fi)] = res
                print(f"  s{seed} f{fi}: rho={res['spearman']:.3f}", flush=True)

        with open(os.path.join(RESULTS_DIR, "ft_esm150m.json"), 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nDone in {(time.time()-total_t)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

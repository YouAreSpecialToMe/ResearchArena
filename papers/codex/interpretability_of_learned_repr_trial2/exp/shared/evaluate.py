from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score

from .shapes3d import FACTOR_NAMES, FACTOR_TO_ID


def normalized_recon_error(x, x_hat):
    return ((x - x_hat).norm(dim=1) / (x.norm(dim=1) + 1e-8)).mean().item()


def train_factor_probes(train_x, train_y, val_x, val_y, seed: int):
    probes = {}
    stats = {}
    for idx, name in enumerate(FACTOR_NAMES):
        best = None
        best_acc = -1.0
        best_cfg = None
        for alpha in [1.0]:
            probe = RidgeClassifier(alpha=alpha, random_state=seed)
            probe.fit(train_x, train_y[:, idx])
            pred = probe.predict(val_x)
            acc = accuracy_score(val_y[:, idx], pred)
            if acc > best_acc:
                best_acc = acc
                best = probe
                best_cfg = {"probe_type": "ridge", "alpha": float(alpha)}
        for c in [1.0]:
            probe = LogisticRegression(C=c, max_iter=4000, solver="lbfgs", random_state=seed)
            probe.fit(train_x, train_y[:, idx])
            pred = probe.predict(val_x)
            acc = accuracy_score(val_y[:, idx], pred)
            if acc > best_acc:
                best_acc = acc
                best = probe
                best_cfg = {"probe_type": "logistic_regression", "C": float(c)}
        probes[name] = best
        stats[name] = {"val_acc": float(best_acc), **best_cfg}
    return probes, stats


def probe_predict(probes, x):
    return {name: probe.predict(x) for name, probe in probes.items()}


def compute_support90(delta):
    sorted_mass = np.sort(np.abs(delta), axis=1)[:, ::-1]
    cumsum = np.cumsum(sorted_mass, axis=1)
    total = cumsum[:, -1:] + 1e-8
    return np.argmax(cumsum >= 0.9 * total, axis=1) + 1


def build_rule_a(z_src, z_tgt, factor_ids, k_top=8):
    edits = {}
    delta = np.abs(z_tgt - z_src)
    signed = z_tgt - z_src
    for factor_name in FACTOR_NAMES:
        factor_id = FACTOR_TO_ID[factor_name]
        mask = factor_ids == factor_id
        pos = delta[mask].mean(axis=0)
        neg = delta[~mask].mean(axis=0)
        score = pos - neg
        top_idx = np.argsort(score)[-k_top:]
        shift = np.zeros(z_src.shape[1], dtype=np.float32)
        shift[top_idx] = signed[mask][:, top_idx].mean(axis=0)
        edits[factor_name] = shift
    return edits


def build_rule_b(z_src, z_tgt, factor_ids):
    edits = {}
    signed = z_tgt - z_src
    for factor_name in FACTOR_NAMES:
        factor_id = FACTOR_TO_ID[factor_name]
        mask = factor_ids == factor_id
        edits[factor_name] = signed[mask].mean(axis=0).astype(np.float32)
    return edits


def _edit_features(model, x_src, shift, alpha, device):
    with torch.no_grad():
        z = model.encode(x_src.to(device), topk=getattr(model, "eval_topk", None))
        z = z + alpha * torch.tensor(np.array(shift, copy=True), device=device, dtype=z.dtype).unsqueeze(0)
        return model.decode(z).cpu().numpy()


def choose_alpha(model, x_all, val_pairs, edits, probes, alpha_grid, device):
    best_alpha = {}
    x_all_np = x_all.cpu().numpy()
    base_preds = probe_predict(probes, x_all_np)
    for factor_name in FACTOR_NAMES:
        factor_rows = val_pairs[val_pairs["changed_factor"] == factor_name]
        idx = factor_rows["source_pos"].to_numpy(dtype=np.int64)
        tgt_value = factor_rows[f"target_{factor_name}"].to_numpy(dtype=np.int64)
        best_score = -1e9
        best = alpha_grid[0]
        for alpha in alpha_grid:
            h_edit = _edit_features(model, x_all[idx], edits[factor_name], alpha, device)
            pred = probes[factor_name].predict(h_edit)
            success = float((pred == tgt_value).mean())
            off_pres = []
            for other in FACTOR_NAMES:
                if other == factor_name:
                    continue
                src_pred = base_preds[other][idx]
                edit_pred = probes[other].predict(h_edit)
                off_pres.append(float((src_pred == edit_pred).mean()))
            off_pres_mean = float(np.mean(off_pres))
            selective = success - (1.0 - off_pres_mean)
            if selective > best_score:
                best_score = selective
                best = alpha
        best_alpha[factor_name] = best
    return best_alpha


def evaluate_edits(model, x_all, test_pairs, edits, alphas, probes, candidate_targets, device):
    x_all_np = x_all.cpu().numpy()
    base_preds = probe_predict(probes, x_all_np)
    metrics = defaultdict(list)
    for _, row in test_pairs.iterrows():
        factor_name = row["changed_factor"]
        source_pos = int(row["source_pos"])
        target_pos = int(row["target_pos"])
        h_edit = _edit_features(model, x_all[source_pos:source_pos + 1], edits[factor_name], alphas[factor_name], device)
        target_pred = probes[factor_name].predict(h_edit)[0]
        metrics["target_factor_success"].append(float(target_pred == int(row[f"target_{factor_name}"])))
        off_pres = []
        for other in FACTOR_NAMES:
            if other == factor_name:
                continue
            base = base_preds[other][source_pos]
            now = probes[other].predict(h_edit)[0]
            off_pres.append(float(base == now))
        off_pres_mean = float(np.mean(off_pres))
        metrics["off_target_preservation"].append(off_pres_mean)
        metrics["selective_intervention_score"].append(
            metrics["target_factor_success"][-1] - (1.0 - off_pres_mean)
        )

        h_tgt = x_all[target_pos:target_pos + 1].cpu().numpy()
        true_sim = float(np.dot(h_edit[0], h_tgt[0]) / (np.linalg.norm(h_edit[0]) * np.linalg.norm(h_tgt[0]) + 1e-8))
        metrics["cosine_distance_to_true_cf"].append(1.0 - true_sim)

        same_ids = candidate_targets[factor_name]["same"]
        other_ids = candidate_targets[factor_name]["other"]
        if len(same_ids):
            same_pool = x_all[same_ids].cpu().numpy()
            same_sims = same_pool @ h_edit[0] / (np.linalg.norm(same_pool, axis=1) * np.linalg.norm(h_edit[0]) + 1e-8)
        else:
            same_sims = np.asarray([], dtype=np.float32)
        if len(other_ids):
            other_pool = x_all[other_ids].cpu().numpy()
            other_sims = other_pool @ h_edit[0] / (np.linalg.norm(other_pool, axis=1) * np.linalg.norm(h_edit[0]) + 1e-8)
        else:
            other_sims = np.asarray([], dtype=np.float32)

        full_sims = np.concatenate([same_sims, other_sims], axis=0)
        rank = 1 + int((full_sims > true_sim).sum())
        metrics["retrieval_rank"].append(rank)

        other_best = float(other_sims.max()) if len(other_sims) else -1.0
        metrics["consistency_rate"].append(float(true_sim > other_best))
        metrics["best_other_factor_cosine"].append(other_best)
        metrics["same_factor_pool_best_cosine"].append(float(same_sims.max()) if len(same_sims) else -1.0)
    return {k: float(np.mean(v)) for k, v in metrics.items()}, metrics


def bootstrap_ci(values, seed: int, n_boot=1000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"low": None, "high": None}
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(float(sample.mean()))
    return {"low": float(np.percentile(boots, 2.5)), "high": float(np.percentile(boots, 97.5))}

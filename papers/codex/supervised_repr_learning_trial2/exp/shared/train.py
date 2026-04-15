import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import DataLoader, Dataset

from exp.shared.metrics import metrics_for_dataset
from exp.shared.models import Adapter, CosineClassifier, LinearProbe, PrototypeHead
from exp.shared.utils import Timer, append_csv_row, count_parameters, ensure_dir, save_json, set_seed


class CachedFeaturesDataset(Dataset):
    def __init__(self, base, aug1=None, aug2=None):
        self.base = base["features"].float()
        self.labels = base["labels"].long()
        self.groups = base["groups"].long()
        self.ids = base["ids"]
        self.aug1 = aug1["features"].float() if aug1 is not None else None
        self.aug2 = aug2["features"].float() if aug2 is not None else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = {
            "x": self.base[idx],
            "y": self.labels[idx],
            "group": self.groups[idx],
            "id": self.ids[idx],
        }
        if self.aug1 is not None:
            row["aug1"] = self.aug1[idx]
            row["aug2"] = self.aug2[idx]
        return row


def supcon_loss(v1, v2, y, temperature=0.07):
    z = torch.cat([v1, v2], dim=0)
    labels = torch.cat([y, y], dim=0)
    sim = z @ z.t() / temperature
    logits_mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(~logits_mask, -1e9)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    same = same & logits_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positives = (same.float() * log_prob).sum(dim=1) / same.float().sum(dim=1).clamp_min(1.0)
    return -positives.mean()


def spread_loss(z, y, trace_targets):
    loss = z.new_tensor(0.0)
    seen = 0
    for cls in torch.unique(y):
        mask = y == cls
        if mask.sum() < 2:
            continue
        class_z = z[mask]
        centered = class_z - class_z.mean(dim=0, keepdim=True)
        trace = centered.pow(2).sum(dim=1).mean()
        loss = loss + (trace - trace_targets[int(cls)]) ** 2
        seen += 1
    return loss / max(seen, 1)


def prototype_losses(head, z, y, lambda_align=0.25, lambda_occ=0.05):
    q, true_sims = head.true_class_assignments(z, y)
    align = (q * (1 - true_sims)).sum(dim=1).mean()
    occ_loss = z.new_tensor(0.0)
    classes_seen = 0
    occupancies = {}
    for cls in torch.unique(y):
        mask = y == cls
        if mask.sum() == 0:
            continue
        q_cls = q[mask]
        occ = q_cls.mean(dim=0)
        occupancies[int(cls)] = occ.detach().cpu().numpy().tolist()
        k = head.active_k[int(cls)]
        target = torch.full((k,), 1.0 / k, device=z.device)
        occ_loss = occ_loss + F.mse_loss(occ[:k], target)
        classes_seen += 1
    occ_loss = occ_loss / max(classes_seen, 1)
    return lambda_align * align, lambda_occ * occ_loss, occupancies


def _metric_name(dataset_name):
    return "worst_group_accuracy" if dataset_name == "waterbirds" else "accuracy"


def _build_run_config(dataset_name, method_name, seed, sensitivity_margin, batch_size, num_classes, max_epochs, patience):
    config = {
        "dataset": dataset_name,
        "method": method_name,
        "seed": seed,
        "sensitivity_margin": sensitivity_margin,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "scheduler": {"name": "cosine", "t_max": max_epochs},
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "selection_metric": _metric_name(dataset_name),
        "num_classes": num_classes,
    }
    if method_name == "linear_probe":
        config["model"] = {"type": "linear_probe", "in_dim": 512, "out_dim": num_classes}
        config["optimizer_groups"] = [{"target": "linear", "lr": 1e-3}]
        config["losses"] = {"cross_entropy": 1.0}
        return config

    config["adapter"] = {
        "in_dim": 512,
        "hidden_dim": 512,
        "out_dim": 128,
        "dropout": 0.1,
        "normalization": "l2",
        "activation": "GELU",
        "layer_norm": True,
    }
    config["optimizer_groups"] = [
        {"target": "adapter", "lr": 3e-4},
        {"target": "classifier", "lr": 3e-4 if method_name in {"ce_adapter", "contrastive_adapter", "pb_spread"} else 1e-3},
    ]
    if method_name in {"ce_adapter", "contrastive_adapter", "pb_spread"}:
        config["classifier"] = {"type": "cosine", "in_dim": 128, "num_classes": num_classes}
    else:
        config["classifier"] = {
            "type": "prototype",
            "in_dim": 128,
            "num_classes": num_classes,
            "max_k": 3 if dataset_name == "cub" else 2,
            "tau_p": 0.07,
            "initial_active_k": 1 if method_name in {"adaptive_vmf", "ablation_no_vmf", "ablation_no_occ"} else 2,
        }

    losses = {"cross_entropy": 1.0}
    if method_name in {"contrastive_adapter", "fixed_k_contrastive", "adaptive_vmf", "ablation_no_adapt", "ablation_no_vmf", "ablation_no_occ", "pb_spread"}:
        losses["supcon"] = {"weight": 0.5, "temperature": 0.07}
    if method_name in {"fixed_k_contrastive", "fixed_k_noncontrastive", "adaptive_vmf", "ablation_no_adapt", "ablation_no_vmf", "ablation_no_occ"}:
        losses["align"] = 0.25
        losses["occupancy"] = 0.0 if method_name == "ablation_no_occ" else 0.05
    if method_name == "pb_spread":
        losses["spread"] = 0.1
    config["losses"] = losses

    if method_name in {"adaptive_vmf", "ablation_no_vmf", "ablation_no_occ"}:
        config["structure_updates"] = {
            "warmup_epochs": 5,
            "start_epoch": 10,
            "period_epochs": 5,
            "selection_rule": "grow" if method_name == "ablation_no_vmf" else "vmf",
            "waterbirds_max_k": 2,
            "cub_max_k": 3,
            "waterbirds_child_floor": {"abs": 10, "frac": 0.10},
            "cub_child_floor": {"abs": 12, "frac": 0.08},
            "waterbirds_low_val_fallback": {
                "max_val_examples": 39,
                "bootstrap_samples": 5,
                "agreement_required": 4,
                "mean_improvement_margin": sensitivity_margin,
            },
        }
    return config


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint.pt"


def evaluate(dataset_name, model_bundle, loader, device):
    adapter = model_bundle.get("adapter")
    classifier = model_bundle["classifier"]
    linear = model_bundle.get("linear")
    rows = []
    y_true, y_pred, groups, subclass_pred = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            group = batch["group"].cpu().numpy()
            if linear is not None:
                logits = linear(x)
                preds = logits.argmax(dim=-1)
            else:
                z = adapter(x)
                if isinstance(classifier, PrototypeHead):
                    logits, _ = classifier.class_logits(z)
                    preds = logits.argmax(dim=-1)
                    subclass_pred.extend(classifier.predict_subclasses(z, y))
                else:
                    logits = classifier(z)
                    preds = logits.argmax(dim=-1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            groups.extend(group.tolist())
            for i in range(len(batch["id"])):
                rows.append(
                    {
                        "id": batch["id"][i],
                        "y_true": int(y[i].cpu()),
                        "y_pred": int(preds[i].cpu()),
                        "group": int(group[i]),
                    }
                )
    metrics = metrics_for_dataset(dataset_name, np.asarray(y_true), np.asarray(y_pred), np.asarray(groups))
    return metrics, rows, subclass_pred


def fit_spherical_kmeans(points, k):
    if len(points) < k:
        return None
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(points)
    centers = km.cluster_centers_
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-8)
    return centers


def estimate_kappa(points, mu):
    if len(points) < 2:
        return 5.0
    r_bar = np.clip(np.mean(points @ mu), 1e-3, 0.999)
    dim = points.shape[1]
    kappa = (r_bar * dim - r_bar**3) / max(1e-6, 1 - r_bar**2)
    return float(np.clip(kappa, 5.0, 200.0))


def _normalize_rows(array):
    return array / np.linalg.norm(array, axis=1, keepdims=True).clip(min=1e-8)


def estimate_class_kappa(points):
    mu = _normalize_rows(points.mean(axis=0, keepdims=True))[0]
    return estimate_kappa(points, mu)


def penalized_vmf_score(points, centers, kappa):
    if len(points) == 0:
        return -1e9
    centers = _normalize_rows(centers)
    sims = points @ centers.T
    total = float(np.logaddexp.reduce(kappa * sims, axis=1).sum())
    dim = points.shape[1]
    penalty = 0.5 * (len(centers) * dim) * math.log(len(points))
    return total - penalty


def _class_embeddings(adapter, features, labels, target_class, device, batch_size=1024):
    adapter.eval()
    xs = features[labels == target_class]
    outs = []
    with torch.inference_mode():
        for start in range(0, len(xs), batch_size):
            outs.append(adapter(xs[start : start + batch_size].to(device)).cpu())
    return torch.cat(outs, dim=0).numpy()


def _class_groups(groups, labels, target_class):
    return groups[labels == target_class].cpu().numpy()


def _child_floor(dataset_name, n_points):
    if dataset_name == "waterbirds":
        return max(10, int(0.10 * n_points))
    return max(12, int(0.08 * n_points))


def _counts_for_centers(points, centers):
    assign = (points @ centers.T).argmax(axis=1)
    counts = np.bincount(assign, minlength=len(centers))
    return assign, counts


def _most_dispersed_prototype(points, centers):
    assign, _ = _counts_for_centers(points, centers)
    dispersions = []
    for idx in range(len(centers)):
        subset = points[assign == idx]
        if len(subset) == 0:
            dispersions.append(float("-inf"))
            continue
        cosine = np.clip(subset @ centers[idx], -1.0, 1.0)
        dispersions.append(float(np.arccos(cosine).mean()))
    return int(np.argmax(dispersions)), assign, dispersions


def _refine_vmf_split(points, init_centers, kappa, n_steps=3):
    centers = _normalize_rows(init_centers.copy())
    for _ in range(n_steps):
        sims = points @ centers.T
        responsibilities = np.exp(kappa * sims - np.max(kappa * sims, axis=1, keepdims=True))
        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True).clip(min=1e-8)
        new_centers = responsibilities.T @ points
        centers = _normalize_rows(new_centers)
    return centers


def _replace_split_centers(current_centers, split_idx, refined_children):
    updated = []
    for idx, center in enumerate(current_centers):
        if idx == split_idx:
            updated.extend(refined_children.tolist())
        else:
            updated.append(center.tolist())
    return _normalize_rows(np.asarray(updated, dtype=np.float32))


def _merge_pair(current_centers, pair):
    keep = [idx for idx in range(len(current_centers)) if idx not in pair]
    merged = current_centers[list(pair)].mean(axis=0, keepdims=True)
    merged = _normalize_rows(merged)
    updated = [current_centers[idx].tolist() for idx in keep]
    updated.extend(merged.tolist())
    return _normalize_rows(np.asarray(updated, dtype=np.float32))


def _closest_pair(centers):
    sims = centers @ centers.T
    np.fill_diagonal(sims, -np.inf)
    pair = np.unravel_index(np.argmax(sims), sims.shape)
    return tuple(sorted((int(pair[0]), int(pair[1]))))


def _bootstrap_grouped_indices(groups, rng):
    sampled = []
    for group in np.unique(groups):
        group_idx = np.flatnonzero(groups == group)
        sampled.append(rng.choice(group_idx, size=len(group_idx), replace=True))
    if not sampled:
        return np.arange(len(groups))
    return np.concatenate(sampled)


def maybe_initialize_fixed_prototypes(adapter, head, train_base, device):
    labels = train_base["labels"].numpy()
    for cls in range(head.num_classes):
        points = _class_embeddings(adapter, train_base["features"].float(), train_base["labels"], cls, device)
        centers = fit_spherical_kmeans(points, head.active_k[cls])
        if centers is not None:
            head.prototypes.data[cls, : head.active_k[cls]] = torch.tensor(centers, dtype=torch.float32, device=device)


def adaptive_structure_update(adapter, head, train_base, val_base, dataset_name, diagnostics, device, epoch, rule="vmf", margin=0.0):
    train_feats = train_base["features"].float()
    train_labels = train_base["labels"]
    train_groups = train_base["groups"]
    val_feats = val_base["features"].float()
    val_labels = val_base["labels"]
    max_cap = 2 if dataset_name == "waterbirds" else 3
    epoch_scores = {"epoch": int(epoch), "before": {}, "after": {}}
    for cls in range(head.num_classes):
        current_k = head.active_k[cls]
        if current_k >= max_cap:
            train_points = _class_embeddings(adapter, train_feats, train_labels, cls, device)
            val_points = _class_embeddings(adapter, val_feats, val_labels, cls, device)
            if len(train_points) and len(val_points):
                current_centers = F.normalize(head.prototypes.data[cls, :current_k], dim=-1).detach().cpu().numpy()
                epoch_scores["before"][str(cls)] = float(
                    penalized_vmf_score(val_points, current_centers, estimate_class_kappa(train_points))
                )
                epoch_scores["after"][str(cls)] = epoch_scores["before"][str(cls)]
            continue
        train_points = _class_embeddings(adapter, train_feats, train_labels, cls, device)
        val_points = _class_embeddings(adapter, val_feats, val_labels, cls, device)
        class_groups = _class_groups(train_groups, train_labels, cls)
        if len(train_points) < 24 or len(val_points) == 0:
            continue
        current_centers = F.normalize(head.prototypes.data[cls, :current_k], dim=-1).detach().cpu().numpy()
        class_kappa = estimate_class_kappa(train_points)
        current_score = penalized_vmf_score(val_points, current_centers, class_kappa)
        epoch_scores["before"][str(cls)] = float(current_score)
        accept = False
        proposal_score = current_score
        reason = "rejected"
        gain = 0.0
        target_idx = None
        split_counts = None
        new_centers = None
        if rule == "grow":
            child_floor = _child_floor(dataset_name, len(train_points))
            if len(train_points) >= (current_k + 1) * child_floor:
                new_centers = fit_spherical_kmeans(train_points, current_k + 1)
                if new_centers is not None:
                    _, split_counts = _counts_for_centers(train_points, new_centers)
                    proposal_score = penalized_vmf_score(val_points, new_centers, class_kappa)
                    accept = bool(split_counts.min() >= child_floor)
                    gain = float(proposal_score - current_score)
                    reason = "grow"
        else:
            child_floor = _child_floor(dataset_name, len(train_points))
            target_idx, current_assign, dispersions = _most_dispersed_prototype(train_points, current_centers)
            target_points = train_points[current_assign == target_idx]
            if len(target_points) >= 2 * child_floor:
                init_children = fit_spherical_kmeans(target_points, 2)
                if init_children is not None:
                    refined_children = _refine_vmf_split(target_points, init_children, class_kappa, n_steps=3)
                    new_centers = _replace_split_centers(current_centers, target_idx, refined_children)
                    _, split_counts = _counts_for_centers(train_points, new_centers)
                    child_counts = sorted(split_counts[[target_idx, target_idx + 1]].tolist())
                    proposal_score = penalized_vmf_score(val_points, new_centers, class_kappa)
                    gain = float(proposal_score - current_score)
                    if dataset_name == "waterbirds" and len(val_points) < 40:
                        gains = []
                        agrees = 0
                        rng = np.random.default_rng(epoch * 1000 + cls)
                        for _ in range(5):
                            boot_idx = _bootstrap_grouped_indices(class_groups, rng)
                            boot = train_points[boot_idx]
                            boot_current = current_centers
                            boot_candidate = new_centers
                            boot_gain = (
                                penalized_vmf_score(boot, boot_candidate, class_kappa)
                                - penalized_vmf_score(boot, boot_current, class_kappa)
                            ) / len(boot)
                            gains.append(float(boot_gain))
                            agrees += int(boot_gain > margin)
                        accept = float(np.mean(gains)) > margin and agrees >= 4 and min(child_counts) >= child_floor
                        reason = "bootstrap"
                    else:
                        accept = gain > 0.0 and min(child_counts) >= child_floor
                        reason = "vmf"
        diagnostics["proposal_history"].append(
            {
                "epoch": int(epoch),
                "class": cls,
                "kind": "split",
                "current_k": current_k,
                "proposed_k": current_k + 1,
                "current_score": float(current_score),
                "proposal_score": float(proposal_score),
                "score_gain": float(gain),
                "accepted": bool(accept),
                "rule": reason,
                "target_prototype": None if target_idx is None else int(target_idx),
                "split_counts": None if split_counts is None else [int(v) for v in split_counts],
            }
        )
        if accept:
            head.active_k[cls] = current_k + 1
            head.prototypes.data[cls, : current_k + 1] = torch.tensor(new_centers, dtype=torch.float32, device=device)
            diagnostics["accepted_splits"] += 1
            diagnostics["accepted_operation_gains"].append(float(gain))
        epoch_scores["after"][str(cls)] = float(proposal_score if accept else current_score)

    for cls in range(head.num_classes):
        current_k = head.active_k[cls]
        if current_k <= 1:
            continue
        train_points = _class_embeddings(adapter, train_feats, train_labels, cls, device)
        val_points = _class_embeddings(adapter, val_feats, val_labels, cls, device)
        current_centers = F.normalize(head.prototypes.data[cls, :current_k], dim=-1).detach().cpu().numpy()
        class_kappa = estimate_class_kappa(train_points)
        current_score = penalized_vmf_score(val_points, current_centers, class_kappa)
        pair = _closest_pair(current_centers)
        _, current_counts = _counts_for_centers(train_points, current_centers)
        child_floor = _child_floor(dataset_name, len(train_points))
        merged_centers = _merge_pair(current_centers, pair)
        merged_score = penalized_vmf_score(val_points, merged_centers, class_kappa)
        floor_violation = bool(current_counts[pair[0]] < child_floor or current_counts[pair[1]] < child_floor)
        accept_merge = bool(merged_score > current_score or floor_violation)
        merge_gain = float(merged_score - current_score)
        diagnostics["proposal_history"].append(
            {
                "epoch": int(epoch),
                "class": cls,
                "kind": "merge",
                "current_k": current_k,
                "proposed_k": current_k - 1,
                "current_score": float(current_score),
                "proposal_score": float(merged_score),
                "score_gain": float(merge_gain),
                "accepted": accept_merge,
                "rule": "floor_violation" if floor_violation and merged_score <= current_score else "vmf",
                "merge_pair": [int(pair[0]), int(pair[1])],
                "merge_pair_counts": [int(current_counts[pair[0]]), int(current_counts[pair[1]])],
            }
        )
        if accept_merge:
            head.active_k[cls] = current_k - 1
            head.prototypes.data[cls, : current_k - 1] = torch.tensor(merged_centers, dtype=torch.float32, device=device)
            diagnostics["accepted_merges"] += 1
            diagnostics["accepted_operation_gains"].append(float(merge_gain))
            epoch_scores["after"][str(cls)] = float(merged_score)
        else:
            epoch_scores["after"].setdefault(str(cls), float(current_score))
    diagnostics["validation_score_trajectories"].append(epoch_scores)


def train_method(dataset_name, method_name, seed, cache_payloads, output_dir, sensitivity_margin=0.0):
    set_seed(seed)
    ensure_dir(output_dir)
    for stale_name in ["train_log.csv", "metrics.json", "runtime.json", "predictions.parquet", "checkpoint.pt"]:
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    timer = Timer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats(device=device) if device == "cuda" else None

    train_ds = CachedFeaturesDataset(cache_payloads["train_base"], cache_payloads["train_aug1"], cache_payloads["train_aug2"])
    val_ds = CachedFeaturesDataset(cache_payloads["val"])
    test_ds = CachedFeaturesDataset(cache_payloads["test"])
    batch_size = 512 if method_name in {"linear_probe", "ce_adapter", "contrastive_adapter", "fixed_k_contrastive", "fixed_k_noncontrastive", "pb_spread"} else 256
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_train_loader = DataLoader(CachedFeaturesDataset(cache_payloads["train_base"]), batch_size=512, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    num_classes = int(cache_payloads["train_base"]["labels"].max().item() + 1)
    diagnostics = {"proposal_history": [], "accepted_splits": 0, "accepted_merges": 0, "accepted_operation_gains": [], "validation_score_trajectories": []}
    if method_name == "linear_probe":
        linear = LinearProbe(512, num_classes).to(device)
        optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-3, weight_decay=1e-4)
        adapter = None
        classifier = None
        max_epochs = 60
    else:
        adapter = Adapter().to(device)
        max_k = 3 if dataset_name == "cub" else 2
        linear = None
        if method_name in {"ce_adapter", "contrastive_adapter", "pb_spread"}:
            classifier = CosineClassifier(128, num_classes).to(device)
        else:
            classifier = PrototypeHead(num_classes, 128, max_k=max_k).to(device)
            if method_name in {"fixed_k_contrastive", "fixed_k_noncontrastive", "ablation_no_adapt"}:
                classifier.set_active_k({c: 2 for c in range(num_classes)})
        params = list(adapter.parameters()) + list(classifier.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": adapter.parameters(), "lr": 3e-4},
                {"params": classifier.parameters(), "lr": 1e-3 if isinstance(classifier, PrototypeHead) else 3e-4},
            ],
            weight_decay=1e-4,
        )
        max_epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    trace_targets = None
    if method_name == "pb_spread":
        trace_targets = {}
        base_feats = cache_payloads["train_base"]["features"].float()
        labels = cache_payloads["train_base"]["labels"]
        for cls in range(num_classes):
            pts = base_feats[labels == cls]
            centered = pts - pts.mean(dim=0, keepdim=True)
            trace_targets[cls] = centered.pow(2).sum(dim=1).mean().item() / 512.0

    best_metric = -1e9
    best_epoch = 0
    best_state = None
    best_diagnostics = deepcopy(diagnostics)
    patience = 8
    bad_epochs = 0
    config = _build_run_config(dataset_name, method_name, seed, sensitivity_margin, batch_size, num_classes, max_epochs, patience)
    save_json(output_dir / "config.json", config)

    for epoch in range(1, max_epochs + 1):
        if adapter is not None:
            adapter.train()
            classifier.train()
            if isinstance(classifier, PrototypeHead) and epoch == 6 and method_name in {"fixed_k_contrastive", "fixed_k_noncontrastive", "ablation_no_adapt"}:
                maybe_initialize_fixed_prototypes(adapter, classifier, cache_payloads["train_base"], device)
        else:
            linear.train()

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            if linear is not None:
                logits = linear(x)
                loss = F.cross_entropy(logits, y)
            else:
                z = adapter(x)
                if isinstance(classifier, PrototypeHead):
                    logits, _ = classifier.class_logits(z)
                    loss = F.cross_entropy(logits, y)
                    align_loss, occ_loss, occupancies = prototype_losses(
                        classifier,
                        z,
                        y,
                        lambda_align=0.25,
                        lambda_occ=0.0 if method_name == "ablation_no_occ" else 0.05,
                    )
                    loss = loss + align_loss + occ_loss
                    if method_name not in {"fixed_k_noncontrastive"}:
                        v1 = adapter(batch["aug1"].to(device))
                        v2 = adapter(batch["aug2"].to(device))
                        loss = loss + 0.5 * supcon_loss(v1, v2, y)
                else:
                    logits = classifier(z)
                    loss = F.cross_entropy(logits, y)
                    if method_name in {"contrastive_adapter", "pb_spread"}:
                        v1 = adapter(batch["aug1"].to(device))
                        v2 = adapter(batch["aug2"].to(device))
                        loss = loss + 0.5 * supcon_loss(v1, v2, y)
                    if method_name == "pb_spread":
                        loss = loss + 0.1 * spread_loss(z, y, trace_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                linear.parameters() if linear is not None else list(adapter.parameters()) + list(classifier.parameters()), 1.0
            )
            optimizer.step()
        scheduler.step()

        if isinstance(classifier, PrototypeHead) and method_name in {"adaptive_vmf", "ablation_no_vmf", "ablation_no_occ"} and epoch >= 10 and epoch % 5 == 0:
            adaptive_structure_update(
                adapter,
                classifier,
                cache_payloads["train_base"],
                cache_payloads["val"],
                dataset_name,
                diagnostics,
                device,
                epoch,
                rule="grow" if method_name == "ablation_no_vmf" else "vmf",
                margin=sensitivity_margin,
            )

        model_bundle = {"adapter": adapter, "classifier": classifier, "linear": linear}
        val_metrics, _, val_subclasses = evaluate(dataset_name, model_bundle, val_loader, device)
        metric_name = _metric_name(dataset_name)
        append_csv_row(
            output_dir / "train_log.csv",
            ["epoch", metric_name, "accuracy", "macro_f1"],
            {"epoch": epoch, metric_name: val_metrics.get(metric_name), "accuracy": val_metrics["accuracy"], "macro_f1": val_metrics["macro_f1"]},
        )
        if val_metrics[metric_name] > best_metric:
            best_metric = val_metrics[metric_name]
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                "adapter": deepcopy(adapter.state_dict()) if adapter is not None else None,
                "classifier": deepcopy(classifier.state_dict()) if classifier is not None else None,
                "linear": deepcopy(linear.state_dict()) if linear is not None else None,
                "active_k": deepcopy(classifier.active_k) if isinstance(classifier, PrototypeHead) else None,
            }
            best_diagnostics = deepcopy(diagnostics)
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state["linear"] is not None:
        linear.load_state_dict(best_state["linear"])
    else:
        adapter.load_state_dict(best_state["adapter"])
        classifier.load_state_dict(best_state["classifier"])
        if isinstance(classifier, PrototypeHead):
            classifier.set_active_k(best_state["active_k"])
    torch.save(
        {
            "dataset": dataset_name,
            "method": method_name,
            "seed": seed,
            "selected_epoch": best_epoch,
            "selected_val_metric": float(best_metric),
            "state": best_state,
        },
        _checkpoint_path(output_dir),
    )

    model_bundle = {"adapter": adapter, "classifier": classifier, "linear": linear}
    test_metrics, prediction_rows, test_subclasses = evaluate(dataset_name, model_bundle, test_loader, device)
    metrics = dict(test_metrics)

    if isinstance(classifier, PrototypeHead):
        train_metrics, _, train_subclasses = evaluate(dataset_name, model_bundle, eval_train_loader, device)
        diagnostics = best_diagnostics
        active_counts = classifier.active_k
        occupancy_entropy = {}
        occupancies = {}
        with torch.inference_mode():
            for cls in range(num_classes):
                z = adapter(cache_payloads["train_base"]["features"].float()[cache_payloads["train_base"]["labels"] == cls].to(device))
                y = torch.full((z.shape[0],), cls, device=device, dtype=torch.long)
                q, _ = classifier.true_class_assignments(z, y)
                occ = q.mean(dim=0)[: classifier.active_k[cls]]
                occupancies[str(cls)] = occ.cpu().tolist()
                entropy = -(occ * (occ + 1e-8).log()).sum().item()
                occupancy_entropy[str(cls)] = entropy
        diagnostics["active_counts"] = {str(k): int(v) for k, v in active_counts.items()}
        diagnostics["occupancies"] = occupancies
        diagnostics["occupancy_entropy"] = occupancy_entropy
        metrics["active_classes_with_k_gt_1"] = int(sum(v > 1 for v in active_counts.values()))
        metrics["median_active_k"] = float(np.median(list(active_counts.values())))
        metrics["accepted_splits"] = int(diagnostics["accepted_splits"])
        metrics["accepted_merges"] = int(diagnostics["accepted_merges"])
        metrics["occupancy_entropy_mean"] = float(np.mean(list(occupancy_entropy.values())))
        metrics["mean_penalized_vmf_gain_per_accepted_operation"] = float(
            np.mean(diagnostics["accepted_operation_gains"]) if diagnostics["accepted_operation_gains"] else 0.0
        )
        if dataset_name == "waterbirds":
            train_groups = cache_payloads["train_base"]["groups"].numpy()
            subclass_arr = np.asarray(train_subclasses)
            if len(subclass_arr) == len(train_groups):
                metrics["subclass_group_nmi"] = float(normalized_mutual_info_score(train_groups, subclass_arr))
        diagnostics["selected_epoch"] = int(best_epoch)
        save_json(output_dir / "diagnostics" / "structure.json", diagnostics)

    param_count = count_parameters(linear if linear is not None else nn.ModuleList([adapter, classifier]))
    metrics["trainable_params"] = int(param_count)
    metrics["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated() / 1024**2) if device == "cuda" else 0.0
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(prediction_rows)), output_dir / "predictions.parquet")
    save_json(output_dir / "metrics.json", metrics)
    save_json(
        output_dir / "runtime.json",
        {
            "epochs_completed": int(epoch),
            "selected_epoch": int(best_epoch),
            "selected_val_metric": float(best_metric),
            "selected_checkpoint": str(_checkpoint_path(output_dir).relative_to(output_dir.parents[2])),
            "wall_clock_minutes": float(timer.minutes()),
        },
    )
    return metrics

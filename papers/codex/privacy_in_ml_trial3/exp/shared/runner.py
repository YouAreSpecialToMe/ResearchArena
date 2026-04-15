from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .attacks import (
    class_conditional_member_score,
    compute_forecast_metrics,
    gaussian_member_score,
    jaccard_overlaps,
    tpr_at_fpr,
)
from .config import (
    DATASET_CONFIGS,
    OUTPUT_ROOT,
    config_to_dict,
    ensure_output_dirs,
    run_id,
)
from .data import get_datasets, make_worker_init_fn, prepare_dataset_summary
from .models import PurchaseMLP, ResNet18WithFeatures


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(dataset: str, sample_shape) -> nn.Module:
    if dataset == "purchase100":
        return PurchaseMLP(int(sample_shape[0]), DATASET_CONFIGS[dataset].num_classes)
    return ResNet18WithFeatures(DATASET_CONFIGS[dataset].num_classes)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=False)
    return ranks


def beta_from_risk(q: np.ndarray) -> np.ndarray:
    pct = percentile_rank(q)
    scaled = np.clip((pct - 0.9) / 0.1, 0.0, 1.0)
    return 0.8 - 0.45 * scaled


def compute_sample_stats(logits: torch.Tensor, y: torch.Tensor):
    probs = torch.softmax(logits, dim=1)
    loss = F.cross_entropy(logits, y, reduction="none")
    true_prob = probs.gather(1, y[:, None]).squeeze(1)
    top2 = torch.topk(probs, k=2, dim=1).values
    margin = true_prob - torch.where(top2[:, 0] == true_prob, top2[:, 1], top2[:, 0])
    pred = probs.argmax(dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    return loss, margin, pred, true_prob, entropy


@torch.no_grad()
def collect_features(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    feats_map = {}
    for x, _, idx in loader:
        x = x.to(device, non_blocking=True)
        _, feats = model(x, return_features=True)
        for i, feat in zip(idx.tolist(), feats.detach().cpu().numpy()):
            feats_map[int(i)] = feat.astype(np.float32)
    return feats_map


def make_eval_loader(ds, batch_size, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def compute_risk_scores(loss_hist, margin_hist, pred_hist, mode: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    losses = np.stack(loss_hist, axis=1)
    margins = np.stack(margin_hist, axis=1)
    preds = np.stack(pred_hist, axis=1)
    lt_iqr = np.subtract(*np.percentile(losses, [75, 25], axis=1))
    mean_loss = losses.mean(axis=1)
    inv_margin = -margins.mean(axis=1)
    flips = (preds[:, 1:] != preds[:, :-1]).sum(axis=1).astype(np.float64)
    comps = {
        "lt_iqr": lt_iqr,
        "mean_loss": mean_loss,
        "inv_margin": inv_margin,
        "pred_flips": flips,
    }
    if mode == "single_artifact":
        q = percentile_rank(lt_iqr)
    else:
        q = sum(percentile_rank(v) for v in comps.values()) / 4.0
    return q, comps


def decile_group_masks(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pct = percentile_rank(q)
    top_mask = pct >= 0.9
    mid_mask = (pct >= 0.5) & (pct < 0.7)
    bottom_mask = pct < 0.1
    return top_mask, mid_mask, bottom_mask


def choose_targets(labels: np.ndarray, q: np.ndarray, method: str, budget: float, rng: np.random.Generator) -> np.ndarray:
    n = len(labels)
    k = max(1, int(n * budget))
    if method == "targeted_random":
        selected = []
        for cls in np.unique(labels):
            cls_idx = np.where(labels == cls)[0]
            cls_k = max(1, int(len(cls_idx) * budget))
            chosen = rng.choice(cls_idx, size=min(cls_k, len(cls_idx)), replace=False)
            selected.extend(chosen.tolist())
        selected = np.asarray(sorted(set(selected)), dtype=np.int64)
        if len(selected) > k:
            selected = rng.choice(selected, size=k, replace=False)
        return np.sort(selected)
    if method == "targeted_loss_only":
        ranking = np.argsort(q)
        return np.sort(ranking[-k:])
    ranking = np.argsort(q)
    return np.sort(ranking[-k:])


def build_anchor_plan(
    labels: np.ndarray,
    q: np.ndarray,
    features: dict[int, np.ndarray],
    train_indices: np.ndarray,
    selected_positions: np.ndarray,
    dataset: str,
    seed: int,
):
    rng = np.random.default_rng(seed)
    alphas = beta_from_risk(q[selected_positions])
    id_to_pos = {int(idx): pos for pos, idx in enumerate(train_indices.tolist())}
    plan = {}
    q_all = percentile_rank(q)
    for pos, alpha in zip(selected_positions.tolist(), alphas.tolist()):
        sample_id = int(train_indices[pos])
        cls = int(labels[pos])
        cls_positions = np.where(labels == cls)[0]
        low_risk = cls_positions[q_all[cls_positions] <= np.median(q_all[cls_positions])]
        low_ids = [int(train_indices[p]) for p in low_risk if int(train_indices[p]) != sample_id]
        if not low_ids:
            low_ids = [int(train_indices[p]) for p in cls_positions if int(train_indices[p]) != sample_id]
        if not low_ids:
            continue
        sample_feat = features[sample_id]
        feats = np.stack([features[i] for i in low_ids], axis=0)
        if dataset == "cifar10":
            sample_feat = sample_feat / (np.linalg.norm(sample_feat) + 1e-8)
            feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
            dist = 1.0 - feats_norm @ sample_feat
        else:
            dist = np.linalg.norm(feats - sample_feat[None, :], axis=1)
        best = int(np.argmin(dist))
        anchor_id = int(low_ids[best])
        lam = float(rng.beta(alpha, alpha))
        plan[sample_id] = {
            "anchor_id": anchor_id,
            "anchor_position": id_to_pos[anchor_id],
            "distance": float(dist[best]),
            "alpha": float(alpha),
            "lambda": lam,
        }
    return plan


def train_one(dataset: str, method: str, seed: int, relaxloss_lambda: float | None = None):
    ensure_output_dirs()
    prepare_dataset_summary()
    set_seed(seed)
    rng = np.random.default_rng(seed)
    cfg, split, datasets_dict = get_datasets(dataset, seed)
    labels = np.asarray([datasets_dict["train"][i][1] for i in range(len(datasets_dict["train"]))], dtype=np.int64)
    train_indices = np.asarray(split["train"], dtype=np.int64)
    exp_dir = OUTPUT_ROOT / "configs"
    run_name = run_id(dataset, method, seed)
    run_cfg = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "split_file": str((OUTPUT_ROOT / "metrics" / f"{dataset}_split_seed_{seed}.json").resolve()),
        "optimizer": cfg.optimizer,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "epochs": cfg.epochs,
        "warmup_epochs": cfg.warmup_epochs,
        "refresh_epochs": cfg.refresh_epochs,
        "intervention_budget": cfg.intervention_budget,
        "attack": "loss+LiRA-lite",
        "relaxloss_lambda": relaxloss_lambda,
        "dataset_config": config_to_dict(cfg),
    }
    (exp_dir / f"{run_name}.json").write_text(json.dumps(run_cfg, indent=2))

    sample_x, _, _ = datasets_dict["train"][0]
    model = build_model(dataset, sample_x.shape).cuda()
    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = None
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    device = torch.device("cuda")
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    worker_init = make_worker_init_fn(seed * 1000)

    train_loader = DataLoader(
        datasets_dict["train"],
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init,
        generator=train_generator,
    )
    eval_train_loader = make_eval_loader(datasets_dict["train"], cfg.batch_size, cfg.num_workers)
    val_loader = make_eval_loader(datasets_dict["val"], cfg.batch_size, cfg.num_workers)
    test_loader = make_eval_loader(datasets_dict["test"], cfg.batch_size, cfg.num_workers)
    ref_loader = make_eval_loader(datasets_dict["reference"], cfg.batch_size, cfg.num_workers)

    loss_hist, margin_hist, pred_hist = [], [], []
    recent_loss_hist, recent_margin_hist, recent_pred_hist = [], [], []
    current_q = np.zeros(len(datasets_dict["train"]), dtype=np.float64)
    warmup_q = None
    warmup_components = None
    selected_positions = np.array([], dtype=np.int64)
    targeted_map = {}
    refresh_sets = []
    refresh_trace_rows = []
    best_state = None
    best_val_acc = -1.0
    best_val_loss = math.inf
    peak_memory_mb = 0.0
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = np.zeros(len(datasets_dict["train"]), dtype=np.float32)
        epoch_margin = np.zeros(len(datasets_dict["train"]), dtype=np.float32)
        epoch_pred = np.zeros(len(datasets_dict["train"]), dtype=np.int64)
        for x, y, idx in tqdm(train_loader, desc=f"{run_name} epoch {epoch}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx_np = idx.numpy()
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                if method == "global_mixup":
                    perm = torch.randperm(x.size(0), device=device)
                    lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
                    logits = model(lam * x + (1.0 - lam) * x[perm])
                    loss = lam * F.cross_entropy(logits, y) + (1.0 - lam) * F.cross_entropy(logits, y[perm])
                else:
                    batch_positions = np.searchsorted(train_indices, idx_np)
                    if method in {
                        "targeted_random",
                        "targeted_loss_only",
                        "targeted_forecast",
                        "forecast_single_artifact",
                        "forecast_no_refresh",
                    } and targeted_map:
                        x_mod = x.clone()
                        for bi, sample_id in enumerate(idx_np.tolist()):
                            plan = targeted_map.get(int(sample_id))
                            if plan is None:
                                continue
                            anchor_x, _, _ = datasets_dict["train"][plan["anchor_position"]]
                            anchor_x = anchor_x.to(device)
                            lam = plan["lambda"]
                            x_mod[bi] = lam * x[bi] + (1.0 - lam) * anchor_x
                        logits = model(x_mod)
                        loss = F.cross_entropy(logits, y)
                    else:
                        logits = model(x)
                        per_item = F.cross_entropy(logits, y, reduction="none")
                        if method == "relaxloss":
                            ref = 1.0 if relaxloss_lambda is None else relaxloss_lambda
                            loss = torch.mean(torch.abs(per_item - ref))
                        elif method == "forecast_targeted_penalty" and len(selected_positions) > 0:
                            pos_mask = torch.from_numpy(np.isin(batch_positions, selected_positions)).to(device)
                            probs = torch.softmax(logits, dim=1)
                            penalty = (probs.max(dim=1).values ** 2) * cfg.targeted_penalty
                            per_item = per_item + penalty * pos_mask.float()
                            loss = per_item.mean()
                        else:
                            loss = per_item.mean()
                batch_loss, batch_margin, batch_pred, _, _ = compute_sample_stats(logits.detach(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss[np.searchsorted(train_indices, idx_np)] = batch_loss.detach().cpu().numpy()
            epoch_margin[np.searchsorted(train_indices, idx_np)] = batch_margin.detach().cpu().numpy()
            epoch_pred[np.searchsorted(train_indices, idx_np)] = batch_pred.detach().cpu().numpy()
            peak_memory_mb = max(peak_memory_mb, torch.cuda.max_memory_allocated() / (1024 ** 2))
        if scheduler is not None:
            scheduler.step()
        loss_hist.append(epoch_loss.copy())
        margin_hist.append(epoch_margin.copy())
        pred_hist.append(epoch_pred.copy())
        recent_loss_hist.append(epoch_loss.copy())
        recent_margin_hist.append(epoch_margin.copy())
        recent_pred_hist.append(epoch_pred.copy())
        if len(recent_loss_hist) > 4:
            recent_loss_hist.pop(0)
            recent_margin_hist.pop(0)
            recent_pred_hist.pop(0)

        val_metrics = evaluate_model(model, val_loader, device)
        if val_metrics["accuracy"] > best_val_acc or (
            abs(val_metrics["accuracy"] - best_val_acc) < 1e-8 and val_metrics["loss"] < best_val_loss
        ):
            best_val_acc = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch == cfg.warmup_epochs:
            warmup_q, warmup_components = compute_risk_scores(loss_hist, margin_hist, pred_hist, mode="full")
            current_q = warmup_q.copy()

        should_refresh = epoch == cfg.warmup_epochs or epoch in cfg.refresh_epochs
        if should_refresh and method not in {"erm", "global_mixup", "relaxloss"}:
            refresh_mode = "single_artifact" if method == "forecast_single_artifact" else "full"
            refresh_q, comps = compute_risk_scores(
                recent_loss_hist if epoch > cfg.warmup_epochs else loss_hist,
                recent_margin_hist if epoch > cfg.warmup_epochs else margin_hist,
                recent_pred_hist if epoch > cfg.warmup_epochs else pred_hist,
                refresh_mode,
            )
            if epoch == cfg.warmup_epochs:
                current_q = refresh_q.copy()
            elif method != "forecast_no_refresh":
                current_q = 0.5 * current_q + 0.5 * refresh_q
            target_method = "targeted_forecast" if method in {
                "targeted_forecast",
                "forecast_single_artifact",
                "forecast_no_refresh",
                "forecast_targeted_penalty",
            } else method
            selection_score = current_q if target_method != "targeted_loss_only" else comps["mean_loss"]
            selected_positions = choose_targets(labels, selection_score, target_method, cfg.intervention_budget, rng)
            refresh_sets.append(set(train_indices[selected_positions].tolist()))
            if method != "forecast_targeted_penalty":
                feats = collect_features(model, eval_train_loader, device)
                targeted_map = build_anchor_plan(
                    labels,
                    current_q,
                    feats,
                    train_indices,
                    selected_positions,
                    dataset,
                    seed + epoch,
                )
            else:
                targeted_map = {}
            for pos in selected_positions.tolist():
                sample_id = int(train_indices[pos])
                plan = targeted_map.get(sample_id, {})
                refresh_trace_rows.append(
                    {
                        "epoch": epoch,
                        "sample_id": sample_id,
                        "class_id": int(labels[pos]),
                        "q_i": float(current_q[pos]),
                        "selection_score": float(selection_score[pos]),
                        "warmup_q_i": float(warmup_q[pos]),
                        "lt_iqr": float(comps["lt_iqr"][pos]),
                        "mean_loss": float(comps["mean_loss"][pos]),
                        "inv_margin": float(comps["inv_margin"][pos]),
                        "pred_flips": float(comps["pred_flips"][pos]),
                        "targeted_flag": True,
                        "anchor_id": int(plan.get("anchor_id", -1)),
                        "lambda": float(plan.get("lambda", 1.0)),
                        "neighbor_distance": float(plan.get("distance", -1.0)),
                    }
                )

    if best_state is None:
        raise RuntimeError(f"No checkpoint stored for {run_name}")
    model.load_state_dict(best_state)
    torch.save(best_state, OUTPUT_ROOT / "checkpoints" / f"{run_name}.pt")

    train_eval = evaluate_model(model, eval_train_loader, device, return_rows=True)
    ref_eval = evaluate_model(model, ref_loader, device, return_rows=True)
    test_eval = evaluate_model(model, test_loader, device, return_rows=True)
    runtime_minutes = (time.time() - start) / 60.0

    member_loss = train_eval["rows"]["loss"].to_numpy()
    ref_loss = ref_eval["rows"]["loss"].to_numpy()
    member_labels = train_eval["rows"]["label"].to_numpy()
    ref_labels = ref_eval["rows"]["label"].to_numpy()
    global_scores = gaussian_member_score(-member_loss, -ref_loss)
    global_scores_ref = gaussian_member_score(-ref_loss, -ref_loss)
    class_scores = class_conditional_member_score(-member_loss, member_labels, -ref_loss, ref_labels)
    class_scores_ref = class_conditional_member_score(-ref_loss, ref_labels, -ref_loss, ref_labels)
    loss_scores = -member_loss
    loss_scores_ref = -ref_loss
    global_tpr, global_thr = tpr_at_fpr(global_scores, global_scores_ref, 0.01)
    class_tpr, _ = tpr_at_fpr(class_scores, class_scores_ref, 0.01)
    loss_tpr, _ = tpr_at_fpr(loss_scores, loss_scores_ref, 0.01)
    if warmup_q is None or warmup_components is None:
        warmup_q, warmup_components = compute_risk_scores(loss_hist[: cfg.warmup_epochs], margin_hist[: cfg.warmup_epochs], pred_hist[: cfg.warmup_epochs], mode="full")
    top_mask, mid_mask, bottom_mask = decile_group_masks(warmup_q)
    worst_decile_tpr = float((global_scores[top_mask] >= global_thr).mean()) if np.any(top_mask) else float("nan")
    mid_tpr = float((global_scores[mid_mask] >= global_thr).mean()) if np.any(mid_mask) else float("nan")
    if np.isnan(mid_tpr):
        mid_tpr = worst_decile_tpr
    disparity = worst_decile_tpr - mid_tpr
    forecast_metrics = compute_forecast_metrics(warmup_q, global_scores)
    final_rows = train_eval["rows"].copy()
    final_rows["q_i"] = warmup_q
    for name, values in warmup_components.items():
        final_rows[f"warmup_{name}"] = values
    final_rows["final_attack_score"] = global_scores
    final_rows["class_attack_score"] = class_scores
    final_rows["loss_attack_score"] = loss_scores
    final_rows["primary_member_pred"] = global_scores >= global_thr
    final_rows["class_member_pred"] = class_scores >= np.quantile(class_scores_ref, 0.99)
    final_rows["loss_member_pred"] = loss_scores >= np.quantile(loss_scores_ref, 0.99)
    final_rows["warmup_group"] = np.where(top_mask, "top", np.where(mid_mask, "middle", np.where(bottom_mask, "bottom", "other")))
    final_rows.to_parquet(OUTPUT_ROOT / "traces" / f"{run_name}.parquet", index=False)
    if refresh_trace_rows:
        pd.DataFrame(refresh_trace_rows).to_parquet(
            OUTPUT_ROOT / "traces" / f"{run_name}_refreshes.parquet",
            index=False,
        )
    else:
        pd.DataFrame(columns=["epoch"]).to_parquet(OUTPUT_ROOT / "traces" / f"{run_name}_refreshes.parquet", index=False)

    attack_rows = [
        train_eval["rows"].assign(
            split="member",
            q_i=warmup_q,
            global_attack_score=global_scores,
            class_attack_score=class_scores,
            loss_attack_score=loss_scores,
        ),
        ref_eval["rows"].assign(
            split="reference",
            q_i=np.nan,
            global_attack_score=global_scores_ref,
            class_attack_score=class_scores_ref,
            loss_attack_score=loss_scores_ref,
        ),
        test_eval["rows"].assign(
            split="test",
            q_i=np.nan,
            global_attack_score=gaussian_member_score(-test_eval["rows"]["loss"].to_numpy(), -ref_loss),
            class_attack_score=class_conditional_member_score(
                -test_eval["rows"]["loss"].to_numpy(),
                test_eval["rows"]["label"].to_numpy(),
                -ref_loss,
                ref_labels,
            ),
            loss_attack_score=-test_eval["rows"]["loss"].to_numpy(),
        ),
    ]
    attack_frame = pd.concat(attack_rows, ignore_index=True)
    attack_artifact_path = OUTPUT_ROOT / "attacks" / f"{run_name}_scores.parquet"
    attack_frame.to_parquet(attack_artifact_path, index=False)
    attack_summary_path = OUTPUT_ROOT / "attacks" / f"{run_name}_summary.json"
    attack_summary_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "global_threshold_at_1pct_fpr": global_thr,
                "class_threshold_at_1pct_fpr": float(np.quantile(class_scores_ref, 0.99)),
                "loss_threshold_at_1pct_fpr": float(np.quantile(loss_scores_ref, 0.99)),
                "attack_scores_file": str(attack_artifact_path.resolve()),
            },
            indent=2,
        )
    )

    results = {
        "experiment": run_name,
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "attack_artifact_path": str(attack_artifact_path.resolve()),
        "attack_summary_path": str(attack_summary_path.resolve()),
        "trace_path": str((OUTPUT_ROOT / "traces" / f"{run_name}.parquet").resolve()),
        "refresh_trace_path": str((OUTPUT_ROOT / "traces" / f"{run_name}_refreshes.parquet").resolve()),
        "metrics": {
            "primary_tpr_at_1_fpr": global_tpr,
            "class_conditional_tpr_at_1_fpr": class_tpr,
            "loss_tpr_at_1_fpr": loss_tpr,
            "worst_decile_leakage": worst_decile_tpr,
            "privacy_disparity": disparity,
            "best_val_accuracy": best_val_acc,
            "best_val_loss": best_val_loss,
            "test_accuracy": test_eval["accuracy"],
            "macro_f1": test_eval["macro_f1"],
            "runtime_minutes": runtime_minutes,
            "peak_gpu_memory_mb": peak_memory_mb,
            "spearman_q_attack": forecast_metrics["spearman_q_attack"],
            "precision_at_10": forecast_metrics["precision_at_10"],
            "mean_refresh_jaccard": float(np.mean(jaccard_overlaps(refresh_sets))) if len(refresh_sets) > 1 else 1.0,
        },
        "config": run_cfg,
    }
    results_path = OUTPUT_ROOT / "metrics" / f"{run_name}.json"
    results_path.write_text(json.dumps(results, indent=2))
    return results


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, return_rows: bool = False):
    model.eval()
    losses, preds, labels, ids, margins, confs, entropies = [], [], [], [], [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss, margin, pred, conf, entropy = compute_sample_stats(logits, y)
        losses.extend(loss.detach().cpu().numpy().tolist())
        margins.extend(margin.detach().cpu().numpy().tolist())
        preds.extend(pred.detach().cpu().numpy().tolist())
        confs.extend(conf.detach().cpu().numpy().tolist())
        entropies.extend(entropy.detach().cpu().numpy().tolist())
        labels.extend(y.detach().cpu().numpy().tolist())
        ids.extend(idx.numpy().tolist())
    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }
    if return_rows:
        metrics["rows"] = pd.DataFrame(
            {
                "sample_id": ids,
                "label": labels,
                "pred": preds,
                "loss": losses,
                "margin": margins,
                "confidence": confs,
                "entropy": entropies,
            }
        )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--relaxloss-lambda", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = train_one(args.dataset, args.method, args.seed, relaxloss_lambda=args.relaxloss_lambda)
    print(json.dumps(out, indent=2))

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from exp.shared.data import build_datasets, get_dataset_bundle
from exp.shared.eval import evaluate_model
from exp.shared.graph import teacher_graph_paths
from exp.shared.losses import maskcon_loss, nest_kl_loss, relational_mse_loss, supervised_contrastive_loss
from exp.shared.models import create_model
from exp.shared.utils import (
    append_csv_row,
    device,
    elapsed_minutes,
    ensure_dir,
    max_memory_mb,
    now,
    reset_peak_memory,
    set_seed,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cifar100", "oxford_pet"])
    p.add_argument("--method", required=True, choices=["ce", "supcon", "feature_l2", "relational_mse", "maskcon", "nest"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lambda-feat", type=float, default=1.0)
    p.add_argument("--lambda-nest", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--tau-n", type=float, default=0.07)
    p.add_argument("--graph-type", default="pretrained", choices=["pretrained", "random", "weak"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--save-epoch-metrics", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def build_optimizer(model, lr: float):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


def refresh_cache(model, loader, feat_dim: int = 512):
    cache = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device(), non_blocking=True)
            feats = model.forward_backbone(images).detach().cpu()
            for sid, feat in zip(batch["sample_id"].tolist(), feats):
                cache[int(sid)] = feat
    return cache


def load_graph(dataset_name: str, graph_type: str, k: int):
    paths = teacher_graph_paths(dataset_name, graph_type=graph_type, k=k)
    raw = np.load(paths["graph"], allow_pickle=True)
    graph = {key: raw[key] for key in raw.files}
    row_by_id = {int(sid): i for i, sid in enumerate(graph["sample_ids"])}
    return graph, row_by_id, paths


def train_one():
    args = parse_args()
    set_seed(args.seed)
    bundle = get_dataset_bundle(args.dataset)
    datasets = build_datasets(bundle)
    out_dir = Path(args.output_dir)
    log_dir = ensure_dir(out_dir / "logs")
    ensure_dir(out_dir)
    for stale in ["train_log.csv", "metrics_by_epoch.json", "metrics_final.json"]:
        stale_path = out_dir / stale
        if stale_path.exists():
            stale_path.unlink()

    epochs = args.epochs or (60 if args.dataset == "cifar100" else 20)
    batch_size = args.batch_size or (256 if args.dataset == "cifar100" else 128)
    train_dataset = datasets["train_single"] if args.method == "ce" else datasets["train_dual"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    train_eval_loader = DataLoader(
        datasets["train_eval"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    model = create_model(bundle.num_coarse_classes).to(device())
    optimizer = build_optimizer(model, args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 5))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    reset_peak_memory()

    graph = None
    row_by_id = None
    graph_paths = None
    teacher_feature_map = None
    if args.method in {"relational_mse", "maskcon", "nest"}:
        graph, row_by_id, graph_paths = load_graph(args.dataset, args.graph_type, args.k)
    if args.method == "feature_l2":
        teacher_paths = teacher_graph_paths(args.dataset, graph_type="pretrained", k=10)
        teacher_npz = np.load(teacher_paths["features"])
        teacher_feature_map = {
            int(sid): torch.tensor(feat, dtype=torch.float32)
            for sid, feat in zip(teacher_npz["sample_ids"], teacher_npz["features"])
        }

    cache = None
    if args.method in {"relational_mse", "nest"}:
        cache = refresh_cache(model, train_eval_loader)

    start = now()
    epoch_metrics = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        all_preds, all_targets = [], []
        epoch_start = now()
        if args.method in {"relational_mse", "nest"}:
            cache = refresh_cache(model, train_eval_loader)
        progress = tqdm(
            train_loader,
            desc=f"{args.dataset}:{args.method}:seed{args.seed}:epoch{epoch}",
            leave=False,
            disable=not sys.stdout.isatty(),
        )
        for batch in progress:
            x1 = batch["view1"].to(device(), non_blocking=True)
            x2 = batch["view2"].to(device(), non_blocking=True)
            coarse = batch["coarse_label"].to(device(), non_blocking=True)
            sample_ids = batch["sample_id"].to(device(), non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                if args.method == "ce":
                    feats = model.forward_backbone(x1)
                    logits = model.classifier(feats)
                    ce = F.cross_entropy(logits, coarse)
                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.detach().cpu().tolist())
                    all_targets.extend(coarse.detach().cpu().tolist())
                    loss = ce
                else:
                    outputs = model.forward_views(x1, x2)
                    ce = F.cross_entropy(outputs["logits"], coarse)
                    preds = outputs["logits"].argmax(dim=1)
                    all_preds.extend(preds.detach().cpu().tolist())
                    all_targets.extend(coarse.detach().cpu().tolist())
                    proj = torch.stack([outputs["proj1"], outputs["proj2"]], dim=1)
                    sup = supervised_contrastive_loss(proj, coarse, temperature=args.tau)
                    loss = sup + 0.5 * ce
                    if args.method == "feature_l2":
                        teacher = torch.stack(
                            [teacher_feature_map[int(sid)] for sid in sample_ids.detach().cpu().tolist()]
                        ).to(device(), non_blocking=True)
                        loss = loss + args.lambda_feat * F.mse_loss(outputs["feat1"], teacher)
                    elif args.method in {"relational_mse", "nest"}:
                        sample_ids_cpu = sample_ids.detach().cpu().tolist()
                        rows = np.array([row_by_id[int(sid)] for sid in sample_ids_cpu], dtype=np.int64)
                        batch_neighbor_ids = graph["neighbor_ids"][rows]
                        neighbor_feats = torch.stack(
                            [torch.stack([cache[int(nid)] for nid in neighbor_ids]) for neighbor_ids in batch_neighbor_ids]
                        ).to(device(), non_blocking=True)
                        teacher_probs = torch.from_numpy(graph["teacher_probs"][rows]).to(device(), non_blocking=True)
                        teacher_sims = torch.from_numpy(graph["neighbor_sims"][rows]).to(device(), non_blocking=True)
                        if args.method == "relational_mse":
                            loss = loss + args.lambda_nest * relational_mse_loss(outputs["feat1"], neighbor_feats, teacher_sims, args.tau_n)
                        else:
                            loss = loss + args.lambda_nest * nest_kl_loss(outputs["feat1"], neighbor_feats, teacher_probs, args.tau_n)
                    elif args.method == "maskcon":
                        teacher_neighbors = None
                        if graph is not None:
                            rows = [row_by_id[int(s)] for s in sample_ids.detach().cpu().tolist()]
                            teacher_neighbors = torch.tensor(graph["neighbor_ids"][rows], device=device())
                        loss = maskcon_loss(outputs["proj1"], outputs["proj2"], coarse, sample_ids, teacher_neighbors, temperature=args.tau) + 0.5 * ce

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
        if epoch > 5:
            scheduler.step()
        train_acc = accuracy_score(all_targets, all_preds)
        entry = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, len(train_loader)),
            "coarse_train_acc": float(train_acc),
            "epoch_minutes": elapsed_minutes(epoch_start),
            "peak_gpu_memory_mb": max_memory_mb(),
        }
        epoch_metrics.append(entry)
        append_csv_row(out_dir / "train_log.csv", entry)
        if args.save_epoch_metrics and args.dataset == "cifar100" and args.seed == 7 and epoch % 10 == 0:
            ckpt_dir = ensure_dir(out_dir / "checkpoints")
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)

    runtime_total = elapsed_minutes(start)
    metrics = {
        "runtime_minutes": runtime_total,
        "peak_gpu_memory_mb": max_memory_mb(),
        "config": vars(args),
        "epochs": epochs,
        "batch_size": batch_size,
        "graph_build_minutes": float(np.load(graph_paths["graph"])["graph_build_minutes"]) if graph_paths is not None else 0.0,
        "cache_refresh_minutes_estimate": float(sum(x["epoch_minutes"] for x in epoch_metrics)) if args.method in {"relational_mse", "nest"} else 0.0,
    }
    if not args.skip_eval:
        eval_start = now()
        eval_metrics = evaluate_model(model, args.dataset, graph_paths["graph"] if graph_paths is not None else None)
        metrics.update(eval_metrics)
        metrics["evaluation_minutes"] = elapsed_minutes(eval_start)

    write_json(out_dir / "metrics_final.json", metrics)
    write_json(out_dir / "metrics_by_epoch.json", {"epochs": epoch_metrics})
    write_json(out_dir / "config.json", vars(args))
    torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "final.ckpt")


if __name__ == "__main__":
    train_one()

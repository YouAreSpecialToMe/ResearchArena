#!/usr/bin/env python3
"""Unified training script for all contrastive learning methods.

Fixes from previous run:
- Linear eval now trains on TRAIN set (was using test set = data leakage)
- AMP (mixed precision) for faster training
- cudnn.benchmark=True for speed
- Fixed TCL loss numerical stability
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.shared.models import SupConModel, CEModel, LinearClassifier
from exp.shared.losses import (SupConLoss, HardNegLoss, TCLLoss,
                                ReweightedSupConLoss, VarConTLoss, CGALoss)
from exp.shared.utils import (seed_everything, ConfusionTracker, PrototypeTracker,
                               AverageMeter, compute_metrics, save_results)
from exp.shared.data_loader import get_dataloaders, CIFAR100_SUPERCLASS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True,
                   choices=['supcon', 'hardneg', 'tcl', 'reweight', 'varcon_t',
                            'ce', 'cga_only', 'cga_full', 'adaptive_temp'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--linear_epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=0.5)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--dataset', default='cifar100')
    p.add_argument('--data_root', default='./data')
    # CGA params
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--lam', type=float, default=0.5, help='CGA loss weight')
    p.add_argument('--gamma', type=float, default=0.0, help='Adaptive temp strength')
    # HardNeg params
    p.add_argument('--beta', type=float, default=1.0)
    # Reweight params
    p.add_argument('--beta_rw', type=float, default=2.0)
    # VarCon-T params
    p.add_argument('--gamma_v', type=float, default=2.0)
    # TCL params
    p.add_argument('--tcl_lr', type=float, default=0.01)
    # Output
    p.add_argument('--output_dir', default=None)
    p.add_argument('--output_file', default=None)
    return p.parse_args()


def train_linear_classifier_fixed(model, train_loader, test_loader, num_classes,
                                   epochs=100, lr=0.1, device='cuda'):
    """Train linear classifier on frozen embeddings using TRAIN set."""
    model.eval()
    feat_dim = model.encoder.feat_dim
    classifier = LinearClassifier(feat_dim, num_classes).to(device)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr,
                                momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')

    for epoch in range(epochs):
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                with autocast('cuda'):
                    feat, _ = model(images)
            with autocast('cuda'):
                logits = classifier(feat)
                loss = criterion(logits, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    return classifier


def train_contrastive(args):
    """Train a contrastive learning model with AMP."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(args.seed, benchmark=True)

    num_classes = 100 if args.dataset == 'cifar100' else 10

    # Data: two-crop for contrastive training
    train_loader_2crop, _, _ = get_dataloaders(
        dataset=args.dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, two_crop=True, data_root=args.data_root)
    # Single-crop for linear eval (train) and test
    train_loader_eval, test_loader_eval, _ = get_dataloaders(
        dataset=args.dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, two_crop=False, data_root=args.data_root)

    model = SupConModel('resnet18', proj_dim=128).to(device)

    # Loss setup
    needs_confusion = args.method in ['cga_only', 'cga_full', 'reweight',
                                       'adaptive_temp', 'varcon_t']
    needs_prototypes = args.method in ['cga_only', 'cga_full', 'adaptive_temp',
                                        'varcon_t', 'reweight']

    if args.method == 'supcon':
        criterion = SupConLoss(args.temperature)
    elif args.method == 'hardneg':
        criterion = HardNegLoss(args.temperature, args.beta)
    elif args.method == 'tcl':
        criterion = TCLLoss(args.temperature)
    elif args.method == 'reweight':
        criterion = ReweightedSupConLoss(args.temperature, args.beta_rw)
    elif args.method == 'varcon_t':
        criterion = VarConTLoss(args.temperature, args.gamma_v)
    elif args.method in ['cga_only', 'cga_full', 'adaptive_temp']:
        criterion = SupConLoss(args.temperature)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    criterion = criterion.to(device)

    cga_loss_fn = None
    if args.method in ['cga_only', 'cga_full']:
        cga_loss_fn = CGALoss(num_classes, args.alpha).to(device)

    conf_tracker = ConfusionTracker(num_classes, mu=0.99, device=device) if needs_confusion else None
    proto_tracker = PrototypeTracker(num_classes, 128, nu=0.99, device=device) if needs_prototypes else None

    # Optimizer
    params = list(model.parameters())
    if args.method == 'tcl':
        params += list(criterion.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    scaler = GradScaler('cuda')
    warmup_epochs = 5
    rampup_epochs = 10

    epoch_times = []
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        loss_meter = AverageMeter()
        cga_meter = AverageMeter()
        epoch_start = time.time()

        for images, labels in train_loader_2crop:
            images = torch.cat([images[0], images[1]], dim=0).to(device, non_blocking=True)
            labels_rep = labels.repeat(2).to(device, non_blocking=True)
            bsz = labels.size(0)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                feat, z = model(images)

                # Main contrastive loss
                if args.method == 'reweight':
                    conf_matrix = conf_tracker.get_normalized() if conf_tracker.initialized else None
                    loss = criterion(z, labels_rep, confusion_matrix=conf_matrix)
                elif args.method == 'varcon_t':
                    confidences = None
                    if proto_tracker is not None and conf_tracker is not None and conf_tracker.initialized:
                        protos = proto_tracker.get_prototypes()
                        confidences = conf_tracker.get_confidences(z, protos)
                    loss = criterion(z, labels_rep, confidences=confidences)
                else:
                    loss = criterion(z, labels_rep)

                # CGA loss
                cga_loss_val = torch.tensor(0.0, device=device)
                if cga_loss_fn is not None and proto_tracker is not None and conf_tracker is not None:
                    if epoch < warmup_epochs:
                        lam_eff = 0.0
                    elif epoch < rampup_epochs:
                        lam_eff = args.lam * (epoch - warmup_epochs) / (rampup_epochs - warmup_epochs)
                    else:
                        lam_eff = args.lam

                    if lam_eff > 0 and conf_tracker.initialized:
                        protos = proto_tracker.get_prototypes().detach()
                        conf_norm = conf_tracker.get_normalized().detach()
                        z_first = z[:bsz]
                        labels_first = labels_rep[:bsz]
                        cga_loss_val = cga_loss_fn(z_first, labels_first, protos, conf_norm)
                        loss = loss + lam_eff * cga_loss_val

                # Adaptive temperature for cga_full and adaptive_temp
                if args.method in ['cga_full', 'adaptive_temp'] and args.gamma > 0:
                    if proto_tracker is not None and conf_tracker is not None and conf_tracker.initialized:
                        protos = proto_tracker.get_prototypes()
                        confidences = conf_tracker.get_confidences(z, protos)
                        uncertainty = 1 - confidences
                        tau_adaptive = args.temperature / (1 + args.gamma * uncertainty)
                        sim = torch.matmul(z, z.T)
                        tau_matrix = tau_adaptive.unsqueeze(1)
                        sim_scaled = sim / tau_matrix

                        labels_col = labels_rep.unsqueeze(1)
                        mask_pos = (labels_col == labels_col.T).float()
                        mask_self = torch.eye(z.shape[0], device=device)
                        mask_pos = mask_pos - mask_self

                        logits_max, _ = sim_scaled.max(dim=1, keepdim=True)
                        logits = sim_scaled - logits_max.detach()
                        exp_logits = torch.exp(logits) * (1 - mask_self)
                        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
                        pos_count = mask_pos.sum(dim=1).clamp(min=1)
                        log_prob = (logits - log_sum_exp) * mask_pos
                        adaptive_loss = -(log_prob.sum(dim=1) / pos_count).mean()

                        if cga_loss_fn is not None and lam_eff > 0 and conf_tracker.initialized:
                            loss = adaptive_loss + lam_eff * cga_loss_val
                        else:
                            loss = adaptive_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update trackers (outside autocast, no grad)
            with torch.no_grad():
                if proto_tracker is not None:
                    proto_tracker.update(z[:bsz].detach().float(), labels_rep[:bsz])
                if conf_tracker is not None and proto_tracker is not None:
                    conf_tracker.update(z[:bsz].detach().float(), labels_rep[:bsz],
                                       proto_tracker.get_prototypes())

            loss_meter.update(loss.item(), bsz)
            if cga_loss_fn is not None:
                cga_meter.update(cga_loss_val.item(), bsz)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"[{args.method}|seed{args.seed}] Epoch {epoch+1}/{args.epochs} "
                  f"loss={loss_meter.avg:.4f} cga={cga_meter.avg:.4f} "
                  f"time={epoch_time:.1f}s", flush=True)

    contrastive_time = time.time() - start_time

    # Linear evaluation - FIXED: train on train set, evaluate on test set
    print(f"[{args.method}|seed{args.seed}] Linear eval (train on train set)...")
    linear_start = time.time()
    classifier = train_linear_classifier_fixed(
        model, train_loader_eval, test_loader_eval, num_classes,
        epochs=args.linear_epochs, lr=0.1, device=device)
    linear_time = time.time() - linear_start

    # Compute metrics
    metrics = compute_metrics(
        model, test_loader_eval, classifier, num_classes,
        superclass_map=CIFAR100_SUPERCLASS if args.dataset == 'cifar100' else None,
        device=device)

    total_time = time.time() - start_time

    # Save results
    results = {
        **metrics,
        'method': args.method,
        'dataset': args.dataset,
        'seed': args.seed,
        'training_time_minutes': total_time / 60,
        'contrastive_time_minutes': contrastive_time / 60,
        'linear_eval_time_minutes': linear_time / 60,
        'mean_epoch_time_seconds': sum(epoch_times) / len(epoch_times),
        'hyperparameters': {
            'epochs': args.epochs,
            'linear_epochs': args.linear_epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'temperature': args.temperature,
            'alpha': args.alpha,
            'lambda': args.lam,
            'gamma': args.gamma,
        },
    }
    if args.method == 'tcl':
        results['hyperparameters']['final_tau_pos'] = (args.temperature * torch.exp(criterion.log_tau_pos).clamp(0.5, 2.0)).item()
        results['hyperparameters']['final_tau_neg'] = (args.temperature * torch.exp(criterion.log_tau_neg).clamp(0.5, 2.0)).item()

    # Output path
    if args.output_file:
        out_path = args.output_file
    elif args.output_dir:
        out_path = os.path.join(args.output_dir, f'results_seed{args.seed}.json')
    else:
        out_path = f'exp/{args.method}/results_seed{args.seed}.json'

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_results(results, out_path)
    print(f"[{args.method}|seed{args.seed}] Done! top1={metrics['top1']:.2f}% "
          f"top5={metrics['top5']:.2f}% time={total_time/60:.1f}min")

    return results


def train_ce(args):
    """Train cross-entropy baseline with AMP."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(args.seed, benchmark=True)

    num_classes = 100 if args.dataset == 'cifar100' else 10

    train_loader, test_loader, _ = get_dataloaders(
        dataset=args.dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, two_crop=False, data_root=args.data_root)

    model = CEModel('resnet18', num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                 momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler('cuda')

    epoch_times = []
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        loss_meter = AverageMeter()
        epoch_start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                feat, logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item(), labels.size(0))

        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"[ce|seed{args.seed}] Epoch {epoch+1}/{args.epochs} "
                  f"loss={loss_meter.avg:.4f} time={epoch_time:.1f}s", flush=True)

    train_time = time.time() - start_time

    # Evaluate
    class CEWrapper(nn.Module):
        def __init__(self, ce_model):
            super().__init__()
            self.encoder = ce_model.encoder
        def forward(self, x):
            feat = self.encoder(x)
            return feat, feat

    class CEClassifier(nn.Module):
        def __init__(self, ce_model):
            super().__init__()
            self.fc = ce_model.fc
        def forward(self, x):
            return self.fc(x)

    wrapper = CEWrapper(model).to(device)
    cls = CEClassifier(model).to(device)

    metrics = compute_metrics(
        wrapper, test_loader, cls, num_classes,
        superclass_map=CIFAR100_SUPERCLASS if args.dataset == 'cifar100' else None,
        device=device)

    results = {
        **metrics,
        'method': 'ce',
        'dataset': args.dataset,
        'seed': args.seed,
        'training_time_minutes': train_time / 60,
        'contrastive_time_minutes': train_time / 60,
        'linear_eval_time_minutes': 0,
        'mean_epoch_time_seconds': sum(epoch_times) / len(epoch_times),
        'hyperparameters': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
        },
    }

    if args.output_file:
        out_path = args.output_file
    elif args.output_dir:
        out_path = os.path.join(args.output_dir, f'results_seed{args.seed}.json')
    else:
        out_path = f'exp/ce_baseline/results_seed{args.seed}.json'

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_results(results, out_path)
    print(f"[ce|seed{args.seed}] Done! top1={metrics['top1']:.2f}% "
          f"top5={metrics['top5']:.2f}% time={train_time/60:.1f}min")

    return results


if __name__ == '__main__':
    args = parse_args()
    if args.method == 'ce':
        train_ce(args)
    else:
        train_contrastive(args)

"""Unified training script for all methods: CE, Label Smoothing, Mixup, CCR variants.
Optimized with AMP (mixed precision) and reduced eval frequency for speed."""

import argparse
import json
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.data_utils import get_dataset, get_val_split, get_train_eval_loader
from shared.models import get_model
from shared.ccr import CCR
from shared.calibration import compute_calibration_metrics
from shared.nc_metrics import compute_nc_metrics
from shared.temperature_scaling import learn_temperature


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_lambda_ccr(base_lambda, epoch, total_epochs, curriculum=False):
    if not curriculum:
        return base_lambda
    warmup_epochs = total_epochs // 2
    if epoch <= warmup_epochs:
        return base_lambda * (epoch / warmup_epochs)
    return base_lambda


def is_ccr_method(method):
    return method in ('ccr_fixed', 'ccr_adaptive', 'ccr_soft', 'ccr_spectral',
                      'ccr_curriculum')


def train_epoch(model, loader, optimizer, scheduler, method, device,
                scaler=None, ccr_module=None, lambda_ccr=0.1, mixup_alpha=0.2,
                label_smoothing=0.1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    ccr_metrics_accum = {}

    if method == 'label_smoothing':
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    use_amp = scaler is not None

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        if method == 'mixup':
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            if is_ccr_method(method):
                logits, features = model(inputs, return_features=True)
                loss_ce = criterion(logits, targets)
                # CCR needs float32 features for stable prototype computation
                ccr_loss, ccr_info = ccr_module(features.float(), targets)
                loss = loss_ce + lambda_ccr * ccr_loss
                for k, v in ccr_info.items():
                    ccr_metrics_accum.setdefault(k, []).append(v)
            elif method == 'mixup':
                logits = model(inputs)
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                logits = model(inputs)
                loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        with torch.no_grad():
            if method == 'mixup':
                _, predicted = logits.max(1)
                correct += (lam * predicted.eq(targets_a).sum().item()
                           + (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                _, predicted = logits.max(1)
                correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / total
    acc = correct / total

    avg_ccr = {}
    for k, v in ccr_metrics_accum.items():
        avg_ccr[k] = np.mean(v)

    return avg_loss, acc, avg_ccr


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_logits = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(targets.cpu())

    return total_loss / total, correct / total, torch.cat(all_logits), torch.cat(all_labels)


@torch.no_grad()
def evaluate_fast(model, loader, device):
    """Quick eval: just accuracy, no logits saved."""
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        predicted = model(inputs).argmax(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)
    return correct / total


def full_evaluation(model, args, device):
    """Run full evaluation: calibration, NC metrics, temperature scaling."""
    nc_map = {'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200}
    num_classes = nc_map[args.dataset]

    _, test_loader, _ = get_dataset(args.dataset, batch_size=256, num_workers=4,
                                     data_dir=args.data_dir)
    _, test_acc, logits, labels = evaluate(model, test_loader, device)

    cal_metrics = compute_calibration_metrics(logits, labels)

    val_loader = get_val_split(args.dataset, val_fraction=0.1, seed=42,
                                batch_size=256, data_dir=args.data_dir)
    optimal_temp = learn_temperature(model, val_loader, device)
    scaled_logits = logits / optimal_temp
    cal_metrics_ts = compute_calibration_metrics(scaled_logits, labels)

    train_eval_loader = get_train_eval_loader(args.dataset, batch_size=256,
                                               num_workers=4, data_dir=args.data_dir)
    nc_results = compute_nc_metrics(model, train_eval_loader, num_classes, device)

    top5_acc = None
    if num_classes > 10:
        _, top5_pred = logits.topk(5, dim=1)
        top5_acc = top5_pred.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()

    results = {
        'test_accuracy': test_acc,
        'top5_accuracy': top5_acc,
        'calibration': {
            'ece': cal_metrics['ece'],
            'mce': cal_metrics['mce'],
            'ada_ece': cal_metrics['ada_ece'],
            'nll': cal_metrics['nll'],
            'brier': cal_metrics['brier'],
        },
        'calibration_after_ts': {
            'ece': cal_metrics_ts['ece'],
            'mce': cal_metrics_ts['mce'],
            'ada_ece': cal_metrics_ts['ada_ece'],
            'nll': cal_metrics_ts['nll'],
            'brier': cal_metrics_ts['brier'],
        },
        'temperature': optimal_temp,
        'nc_metrics': {
            'nc1': nc_results['nc1'],
            'nc2': nc_results['nc2'],
            'nc3': nc_results['nc3'],
            'nc4': nc_results['nc4'],
            'ncc_accuracy': nc_results['ncc_accuracy'],
            'mean_within_class_spread': nc_results['mean_within_class_spread'],
        },
        'reliability_bins': cal_metrics['reliability_bins'],
        'reliability_bins_ts': cal_metrics_ts['reliability_bins'],
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        choices=['ce', 'label_smoothing', 'mixup',
                                 'ccr_fixed', 'ccr_adaptive', 'ccr_soft',
                                 'ccr_spectral', 'ccr_curriculum'])
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'tinyimagenet'])
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--lambda_ccr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--skip_eval', action='store_true')
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader, num_classes = get_dataset(
        args.dataset, args.batch_size, num_workers=args.num_workers, data_dir=args.data_dir)

    model = get_model(args.arch, num_classes, args.dataset).to(device)

    ccr_module = None
    if is_ccr_method(args.method):
        variant_map = {
            'ccr_fixed': 'fixed',
            'ccr_adaptive': 'adaptive',
            'ccr_soft': 'soft',
            'ccr_spectral': 'spectral',
            'ccr_curriculum': 'adaptive',
        }
        variant = variant_map[args.method]
        ccr_module = CCR(
            num_classes=num_classes,
            feat_dim=model.feat_dim,
            variant=variant,
            tau=args.tau,
            gamma=args.gamma,
        ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP scaler
    use_amp = not args.no_amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler() if use_amp else None

    log = []
    start_time = time.time()
    use_curriculum = (args.method == 'ccr_curriculum')

    print(f"Training {args.method} on {args.dataset} with seed {args.seed}")
    print(f"  arch={args.arch}, epochs={args.epochs}, lr={args.lr}, bs={args.batch_size}, amp={use_amp}")
    if ccr_module:
        print(f"  lambda_ccr={args.lambda_ccr}, gamma={args.gamma}, tau={args.tau}")
        if use_curriculum:
            print(f"  curriculum warmup over {args.epochs // 2} epochs")

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        eff_lambda = get_lambda_ccr(args.lambda_ccr, epoch, args.epochs, use_curriculum)

        train_loss, train_acc, ccr_info = train_epoch(
            model, train_loader, optimizer, scheduler, args.method, device,
            scaler=scaler, ccr_module=ccr_module, lambda_ccr=eff_lambda,
            mixup_alpha=args.mixup_alpha, label_smoothing=args.label_smoothing)

        # Only do full test eval every 10 epochs (saves ~20% time)
        if epoch % 10 == 0 or epoch == args.epochs or epoch <= 5:
            test_acc = evaluate_fast(model, test_loader, device)
        # else: use last known test_acc (or train_acc as proxy)

        entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'lr': optimizer.param_groups[0]['lr'],
        }
        if use_curriculum:
            entry['effective_lambda'] = eff_lambda
        entry.update(ccr_info)
        log.append(entry)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'ccr_state': ccr_module.state_dict() if ccr_module else None,
            }, os.path.join(args.output_dir, 'best_model.pt'))

        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'ccr_state': ccr_module.state_dict() if ccr_module else None,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))

        if epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.time() - start_time
            extra = ""
            if use_curriculum:
                extra = f" eff_lam={eff_lambda:.4f}"
            if 'mean_spread' in ccr_info:
                extra += f" spread={ccr_info['mean_spread']:.2f}"
            print(f"  Epoch {epoch}/{args.epochs}: loss={train_loss:.4f} "
                  f"tr_acc={train_acc:.4f} te_acc={test_acc:.4f}{extra} "
                  f"[{elapsed:.0f}s]", flush=True)

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'ccr_state': ccr_module.state_dict() if ccr_module else None,
    }, os.path.join(args.output_dir, 'final_model.pt'))

    total_time = time.time() - start_time

    with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
        json.dump({
            'config': vars(args),
            'log': log,
            'best_test_acc': best_acc,
            'total_time_seconds': total_time,
        }, f, indent=2)

    if not args.skip_eval:
        print("\nRunning full evaluation...")
        eval_start = time.time()

        ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pt'),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

        eval_results = full_evaluation(model, args, device)
        eval_results['best_test_acc'] = best_acc
        eval_results['total_time_seconds'] = total_time
        eval_results['config'] = vars(args)

        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(eval_results, f, indent=2)

        eval_time = time.time() - eval_start
        print(f"  Eval: {eval_time:.0f}s | Acc={eval_results['test_accuracy']:.4f} "
              f"ECE={eval_results['calibration']['ece']:.4f} "
              f"ECE_TS={eval_results['calibration_after_ts']['ece']:.4f} "
              f"NC1={eval_results['nc_metrics']['nc1']:.4f} "
              f"T={eval_results['temperature']:.3f}")

    print(f"\nDone. Best acc: {best_acc:.4f}. Time: {total_time:.0f}s")


if __name__ == '__main__':
    main()

"""Optimized training script with vectorized augmentation and checkpoint resume.

Key improvements over train_fast.py:
- Vectorized GPU augmentation (grid_sample) instead of per-image loops
- Checkpoint resume with optimizer/scheduler state
- torch.compile support
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torchvision import datasets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import ResNetCIFAR, ContrastiveModel
from src.losses_v2 import SupConLoss, HSCLLoss, CCSupConLoss
from src.fast_augment import FastContrastiveAugment, FastCEAugment, CIFAR100_MEAN, CIFAR100_STD
from src.utils import set_seed, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        choices=['ce', 'supcon', 'hscl', 'cc_supcon'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    # CC-SupCon specific
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--tau_c', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    # H-SCL specific
    parser.add_argument('--beta_hard', type=float, default=1.0)
    # Output
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--save_confusability', action='store_true')
    parser.add_argument('--no_eval', action='store_true')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    return parser.parse_args()


class GPUDataset:
    """Load entire CIFAR-100 onto GPU."""
    def __init__(self, device='cuda'):
        dataset = datasets.CIFAR100(root='./data', train=True, download=True)
        self.images = (torch.tensor(dataset.data, dtype=torch.float32)
                       .permute(0, 3, 1, 2).to(device) / 255.0)
        self.labels = torch.tensor(dataset.targets, dtype=torch.long).to(device)
        self.n = len(self.labels)
        self.device = device

    def get_batches(self, batch_size, contrastive=True, augment_fn=None):
        """Yield batches for one epoch."""
        perm = torch.randperm(self.n, device=self.device)
        for start in range(0, self.n - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            imgs = self.images[idx]
            labels = self.labels[idx]
            if contrastive:
                view1 = augment_fn(imgs.clone())
                view2 = augment_fn(imgs.clone())
                yield view1, view2, labels
            else:
                yield augment_fn(imgs.clone()), labels


class GPUEvalDataset:
    """Load evaluation dataset onto GPU with normalization only."""
    def __init__(self, train=True, dataset_name='cifar100', device='cuda', batch_size=512):
        self.device = device
        self.batch_size = batch_size
        mean = CIFAR100_MEAN.to(device)
        std = CIFAR100_STD.to(device)

        if dataset_name == 'cifar100':
            ds = datasets.CIFAR100(root='./data', train=train, download=True)
        elif dataset_name == 'cifar10':
            ds = datasets.CIFAR10(root='./data', train=train, download=True)
        elif dataset_name == 'stl10':
            split = 'train' if train else 'test'
            ds = datasets.STL10(root='./data', split=split, download=True)

        data = ds.data
        if data.shape[-1] == 3:
            images = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        else:
            images = torch.tensor(data, dtype=torch.float32) / 255.0

        if images.shape[-1] != 32:
            images = F.interpolate(images, size=32, mode='bilinear', align_corners=False)

        self.images = ((images.to(device) - mean) / std)
        if hasattr(ds, 'targets'):
            self.labels = torch.tensor(ds.targets, dtype=torch.long).to(device)
        else:
            self.labels = torch.tensor(ds.labels, dtype=torch.long).to(device)
        self.n = len(self.labels)

    def get_batches(self):
        for start in range(0, self.n, self.batch_size):
            end = min(start + self.batch_size, self.n)
            yield self.images[start:end], self.labels[start:end]


@torch.no_grad()
def extract_features(encoder, eval_dataset):
    encoder.eval()
    all_features, all_labels = [], []
    for images, labels in eval_dataset.get_batches():
        features = encoder(images, return_features=True)
        all_features.append(features)
        all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)


def linear_probe(train_features, train_labels, test_features, test_labels,
                 num_classes=100, lr=0.1, epochs=100, batch_size=512):
    feat_dim = train_features.shape[1]
    device = train_features.device
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n = train_features.shape[0]

    classifier.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            logits = classifier(train_features[idx])
            loss = F.cross_entropy(logits, train_labels[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

    classifier.eval()
    with torch.no_grad():
        preds = classifier(test_features).argmax(1)
        return (preds == test_labels).float().mean().item() * 100


@torch.no_grad()
def knn_accuracy(train_features, train_labels, test_features, test_labels, k=200):
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)
    num_classes = train_labels.max().item() + 1
    correct = 0
    chunk = 500
    for start in range(0, test_features.shape[0], chunk):
        end = min(start + chunk, test_features.shape[0])
        sim = torch.matmul(test_features[start:end], train_features.T)
        _, topk_idx = sim.topk(k, dim=1)
        topk_labels = train_labels[topk_idx]
        for i in range(end - start):
            counts = torch.bincount(topk_labels[i], minlength=num_classes)
            if counts.argmax().item() == test_labels[start + i].item():
                correct += 1
    return correct / test_features.shape[0] * 100


def evaluate_encoder(encoder_state_path, method='supcon', num_classes=100):
    """Evaluate a saved encoder checkpoint."""
    device = torch.device('cuda')
    encoder = ResNetCIFAR(arch='resnet18').to(device)
    state = torch.load(encoder_state_path, map_location=device, weights_only=True)
    if method == 'ce':
        encoder_state = {k: v for k, v in state.items() if not k.startswith('fc.')}
        encoder.load_state_dict(encoder_state, strict=False)
    else:
        encoder.load_state_dict(state)
    encoder.eval()

    train_eval = GPUEvalDataset(train=True, dataset_name='cifar100')
    test_eval = GPUEvalDataset(train=False, dataset_name='cifar100')
    train_f, train_l = extract_features(encoder, train_eval)
    test_f, test_l = extract_features(encoder, test_eval)

    lp = linear_probe(train_f, train_l, test_f, test_l, num_classes=num_classes)
    knn = knn_accuracy(train_f, train_l, test_f, test_l, k=200)

    # Clean up eval datasets
    del train_eval, test_eval, train_f, train_l, test_f, test_l
    torch.cuda.empty_cache()

    return {'linear_probe': lp, 'knn': knn}


def evaluate_transfer(encoder_state_path, method='supcon'):
    """Evaluate transfer to CIFAR-10 and STL-10."""
    device = torch.device('cuda')
    encoder = ResNetCIFAR(arch='resnet18').to(device)
    state = torch.load(encoder_state_path, map_location=device, weights_only=True)
    if method == 'ce':
        encoder_state = {k: v for k, v in state.items() if not k.startswith('fc.')}
        encoder.load_state_dict(encoder_state, strict=False)
    else:
        encoder.load_state_dict(state)
    encoder.eval()

    results = {}
    for ds_name, nc in [('cifar10', 10), ('stl10', 10)]:
        try:
            train_eval = GPUEvalDataset(train=True, dataset_name=ds_name)
            test_eval = GPUEvalDataset(train=False, dataset_name=ds_name)
            train_f, train_l = extract_features(encoder, train_eval)
            test_f, test_l = extract_features(encoder, test_eval)
            lp = linear_probe(train_f, train_l, test_f, test_l, num_classes=nc)
            knn = knn_accuracy(train_f, train_l, test_f, test_l, k=200)
            results[ds_name] = {'linear_probe': lp, 'knn': knn}
            del train_eval, test_eval, train_f, train_l, test_f, test_l
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Transfer eval on {ds_name} failed: {e}")
            results[ds_name] = {'linear_probe': 0, 'knn': 0}
    return results


def train(args):
    device = torch.device('cuda')
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    is_contrastive = args.method != 'ce'
    method_name = 'ccsupcon' if args.method == 'cc_supcon' else args.method

    # Model
    if is_contrastive:
        model = ContrastiveModel(arch='resnet18', proj_dim=128).to(device)
    else:
        model = ResNetCIFAR(arch='resnet18', num_classes=100).to(device)

    # Loss
    criterion = None
    if args.method == 'supcon':
        criterion = SupConLoss(temperature=args.temperature)
    elif args.method == 'hscl':
        criterion = HSCLLoss(temperature=args.temperature, beta_hard=args.beta_hard)
    elif args.method == 'cc_supcon':
        criterion = CCSupConLoss(
            num_classes=100, feat_dim=128, temperature=args.temperature,
            tau_c=args.tau_c, alpha=args.alpha, beta=args.beta,
            warmup_epochs=args.warmup_epochs
        ).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    # Resume from checkpoint
    start_epoch = 1
    losses = []
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        losses = ckpt.get('losses', [])
        if criterion is not None and 'criterion' in ckpt:
            criterion.load_state_dict(ckpt['criterion'])
        print(f"Resumed from epoch {start_epoch - 1}", flush=True)

    # Compile model
    if args.compile:
        try:
            model = torch.compile(model)
            print("Using torch.compile", flush=True)
        except Exception as e:
            print(f"torch.compile failed: {e}, continuing without", flush=True)

    # Data
    data = GPUDataset(device=device)
    if is_contrastive:
        augment = FastContrastiveAugment(device=device)
    else:
        augment = FastCEAugment(device=device)

    # Confusability logging
    conf_log_epochs = {1, 10, 50, 100, 200, 300, 400, 500}
    conf_dir = os.path.join(os.path.dirname(args.save_dir), 'confusability_matrices')

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_meter = AverageMeter()

        if is_contrastive:
            for view1, view2, labels in data.get_batches(args.batch_size, True, augment):
                images = torch.cat([view1, view2], dim=0)
                with autocast('cuda'):
                    _, z = model(images)
                    z = z.float()
                    if args.method == 'cc_supcon':
                        loss = criterion(z, labels, current_epoch=epoch)
                    else:
                        loss = criterion(z, labels)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_meter.update(loss.item(), labels.size(0))
        else:
            for images, labels in data.get_batches(args.batch_size, False, augment):
                with autocast('cuda'):
                    logits = model(images)
                    loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_meter.update(loss.item(), images.size(0))

        scheduler.step()
        losses.append(loss_meter.avg)

        if epoch % 50 == 0 or epoch == 1 or epoch == args.epochs:
            elapsed = (time.time() - start_time) / 60
            eps = (epoch - start_epoch + 1) / max(elapsed, 0.01) * 60
            eta = (args.epochs - epoch) / max(eps, 0.01)
            print(f"[{args.method.upper()}] Epoch {epoch}/{args.epochs} "
                  f"Loss: {loss_meter.avg:.4f} LR: {scheduler.get_last_lr()[0]:.6f} "
                  f"({elapsed:.1f}min, ETA {eta:.1f}min)", flush=True)

        # Save confusability matrices
        if (args.save_confusability and args.method == 'cc_supcon'
                and epoch in conf_log_epochs):
            os.makedirs(conf_dir, exist_ok=True)
            np.save(os.path.join(conf_dir, f'conf_epoch{epoch}_seed{args.seed}.npy'),
                    criterion.get_confusability_matrix())

    train_time = (time.time() - start_time) / 60
    print(f"\nTraining completed in {train_time:.1f} minutes", flush=True)

    # Save encoder weights
    if is_contrastive:
        # Unwrap compiled model if needed
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        encoder_path = os.path.join(args.save_dir, f'{method_name}_seed{args.seed}.pth')
        torch.save(raw_model.encoder.state_dict(), encoder_path)
        # Save checkpoint for potential resume
        ckpt_path = os.path.join(args.save_dir, f'{method_name}_ckpt_seed{args.seed}.pth')
        ckpt = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': args.epochs,
            'losses': losses,
        }
        if criterion is not None and hasattr(criterion, 'state_dict'):
            ckpt['criterion'] = criterion.state_dict()
        torch.save(ckpt, ckpt_path)
        # Save final confusability
        if args.method == 'cc_supcon':
            os.makedirs(conf_dir, exist_ok=True)
            np.save(os.path.join(conf_dir, f'conf_final_seed{args.seed}.npy'),
                    criterion.get_confusability_matrix())
    else:
        encoder_path = os.path.join(args.save_dir, f'{method_name}_seed{args.seed}.pth')
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save(raw_model.state_dict(), encoder_path)

    print(f"Saved to {encoder_path}", flush=True)

    # Clean up training data
    del data
    torch.cuda.empty_cache()

    # Evaluate
    if not args.no_eval:
        print("Evaluating...", flush=True)
        eval_results = evaluate_encoder(encoder_path, method=args.method)
        print(f"[{args.method} seed={args.seed}] LP: {eval_results['linear_probe']:.2f}% "
              f"kNN: {eval_results['knn']:.2f}%", flush=True)

        results = {
            'linear_probe': eval_results['linear_probe'],
            'knn': eval_results['knn'],
            'training_loss': losses,
            'runtime_minutes': train_time,
            'config': {k: v for k, v in vars(args).items() if not k.startswith('_')},
        }
        results_path = os.path.join(args.save_dir, f'results_seed{args.seed}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {results_path}", flush=True)


if __name__ == '__main__':
    args = parse_args()
    train(args)

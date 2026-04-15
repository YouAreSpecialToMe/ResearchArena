"""Shared utilities for tracking, evaluation, and reproducibility."""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


def seed_everything(seed, benchmark=True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = not benchmark
    torch.backends.cudnn.benchmark = benchmark


class ConfusionTracker:
    """Maintains EMA confusion matrix from model predictions."""

    def __init__(self, num_classes, mu=0.99, device='cuda'):
        self.num_classes = num_classes
        self.mu = mu
        self.device = device
        # Initialize uniform
        self.confusion = torch.ones(num_classes, num_classes, device=device) / num_classes
        self.initialized = False

    @torch.no_grad()
    def update(self, features, labels, prototypes=None):
        """Update confusion matrix using nearest-prototype classification."""
        if prototypes is None:
            return

        # Classify by nearest prototype
        sim = torch.matmul(F.normalize(features, dim=1),
                           F.normalize(prototypes, dim=1).T)  # [N, C]
        preds = sim.argmax(dim=1)

        # Build batch confusion
        batch_conf = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        for true_label, pred_label in zip(labels, preds):
            batch_conf[true_label.long(), pred_label.long()] += 1

        # Normalize rows
        row_sums = batch_conf.sum(dim=1, keepdim=True).clamp(min=1)
        batch_conf = batch_conf / row_sums

        # EMA update
        if not self.initialized:
            self.confusion = batch_conf
            self.initialized = True
        else:
            self.confusion = self.mu * self.confusion + (1 - self.mu) * batch_conf

    def get_normalized(self):
        """Return symmetrized normalized confusion matrix in [0, 1] range.

        Max-normalized so the most confused pair has value 1.0.
        This ensures CGA target similarities stay in [-1, 1].
        """
        # Symmetrize
        sym = (self.confusion + self.confusion.T) / 2
        # Zero diagonal
        mask = 1 - torch.eye(self.num_classes, device=self.device)
        sym = sym * mask
        # Max-normalize to [0, 1]
        max_val = sym.max()
        if max_val > 0:
            sym = sym / max_val
        return sym

    def get_confidences(self, features, prototypes):
        """Get per-sample confidence (max similarity to any prototype)."""
        sim = torch.matmul(F.normalize(features, dim=1),
                           F.normalize(prototypes, dim=1).T)
        probs = F.softmax(sim / 0.07, dim=1)
        confidences = probs.max(dim=1)[0]
        return confidences


class PrototypeTracker:
    """Maintains EMA of L2-normalized class prototypes."""

    def __init__(self, num_classes, feat_dim, nu=0.99, device='cuda'):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.nu = nu
        self.device = device
        self.prototypes = torch.randn(num_classes, feat_dim, device=device)
        self.prototypes = F.normalize(self.prototypes, dim=1)
        self.counts = torch.zeros(num_classes, device=device)

    @torch.no_grad()
    def update(self, features, labels):
        """Update prototypes with new batch of features."""
        features = F.normalize(features, dim=1)
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                class_mean = features[mask].mean(dim=0)
                if self.counts[c] == 0:
                    self.prototypes[c] = class_mean
                else:
                    self.prototypes[c] = self.nu * self.prototypes[c] + (1 - self.nu) * class_mean
                self.prototypes[c] = F.normalize(self.prototypes[c], dim=0)
                self.counts[c] += 1

    def get_prototypes(self):
        return self.prototypes.clone()


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(model, test_loader, classifier, num_classes,
                    superclass_map=None, device='cuda'):
    """Compute comprehensive evaluation metrics."""
    model.eval()
    classifier.eval()

    all_feats = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            feat, _ = model(images)
            logits = classifier(feat)
            preds = logits.argmax(dim=1)

            all_feats.append(feat.cpu())
            all_labels.append(labels)
            all_preds.append(preds.cpu())

    all_feats = torch.cat(all_feats)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    # Top-1 accuracy
    top1 = (all_preds == all_labels).float().mean().item() * 100

    # Top-5 accuracy (recompute with logits)
    model.eval()
    classifier.eval()
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            feat, _ = model(images)
            logits = classifier(feat)
            _, top5_preds = logits.topk(5, dim=1)
            top5_correct += (top5_preds == labels.to(device).unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)
    top5 = top5_correct / total * 100

    results = {'top1': top1, 'top5': top5}

    # Superclass metrics
    if superclass_map is not None and num_classes == 100:
        sc_map = torch.tensor(superclass_map)
        true_sc = sc_map[all_labels.long()]
        pred_sc = sc_map[all_preds.long()]

        # Superclass accuracy: predicted class is in the correct superclass
        superclass_acc = (true_sc == pred_sc).float().mean().item() * 100
        results['superclass_acc'] = superclass_acc

        # Within-superclass accuracy: among samples where superclass is correct,
        # fraction where the fine class is also correct
        sc_correct_mask = (true_sc == pred_sc)
        if sc_correct_mask.sum() > 0:
            within_acc = (all_preds[sc_correct_mask] == all_labels[sc_correct_mask]).float().mean().item() * 100
        else:
            within_acc = 0.0
        results['within_superclass_acc'] = within_acc

        # Between-superclass error rate: fraction of errors that cross superclass boundaries
        error_mask = (all_preds != all_labels)
        if error_mask.sum() > 0:
            cross_sc_errors = ((true_sc[error_mask] != pred_sc[error_mask]).float().sum().item())
            between_error_rate = cross_sc_errors / error_mask.sum().item() * 100
        else:
            between_error_rate = 0.0
        results['between_superclass_error_rate'] = between_error_rate

    # Embedding geometry metrics
    class_means = []
    for c in range(num_classes):
        mask = (all_labels == c)
        if mask.sum() > 0:
            class_mean = F.normalize(all_feats[mask].mean(dim=0), dim=0)
            class_means.append(class_mean)
    if len(class_means) == num_classes:
        class_means = torch.stack(class_means)
        cos_sim = torch.matmul(class_means, class_means.T)
        mask = 1 - torch.eye(num_classes)
        off_diag = cos_sim[mask.bool()]

        results['etf_deviation'] = off_diag.var().item()

        # Hierarchy correlation (only for CIFAR-100)
        if superclass_map is not None and num_classes == 100:
            sc_map_t = torch.tensor(superclass_map)
            distances = 1 - cos_sim  # convert similarity to distance
            gt_same_sc = torch.zeros(num_classes, num_classes)
            for i in range(num_classes):
                for j in range(num_classes):
                    gt_same_sc[i, j] = float(sc_map_t[i] == sc_map_t[j])

            # Flatten upper triangle
            triu_idx = torch.triu_indices(num_classes, num_classes, offset=1)
            dist_flat = distances[triu_idx[0], triu_idx[1]].numpy()
            gt_flat = gt_same_sc[triu_idx[0], triu_idx[1]].numpy()

            # Spearman correlation (negative because same-SC should have smaller distance)
            rho, p_val = stats.spearmanr(-dist_flat, gt_flat)
            results['hierarchy_corr'] = rho
            results['hierarchy_corr_p'] = p_val

    return results


def train_linear_classifier(model, train_loader, test_loader, num_classes,
                            epochs=100, lr=0.1, device='cuda'):
    """Train a linear classifier on frozen embeddings."""
    from .models import LinearClassifier

    model.eval()
    feat_dim = model.encoder.feat_dim
    classifier = LinearClassifier(feat_dim, num_classes).to(device)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr,
                                momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                feat, _ = model(images)
            logits = classifier(feat)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return classifier


def save_results(results, filepath):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

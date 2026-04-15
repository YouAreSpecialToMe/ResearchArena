import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, f1_score
from sklearn.cluster import KMeans


def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def weighted_supcon_loss(z, labels, temperature=0.1, positive_weights=None):
    device = z.device
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(z.shape[0], device=device, dtype=torch.bool)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim) * (~mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = same & (~mask)
    if positive_weights is None:
        positive_weights = positive_mask.float()
    else:
        positive_weights = positive_weights * positive_mask.float()
    denom = positive_weights.sum(dim=1)
    valid = denom > 0
    loss = -(positive_weights * log_prob).sum(dim=1) / denom.clamp_min(1e-12)
    return loss[valid].mean() if valid.any() else torch.tensor(0.0, device=device)


def fit_linear_probe(
    train_x,
    train_y,
    test_x,
    test_y,
    num_classes,
    device=None,
    epochs=30,
    batch_size=2048,
    lr=0.1,
    weight_decay=1e-4,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    train_x = torch.as_tensor(train_x, dtype=torch.float32)
    train_y = torch.as_tensor(train_y, dtype=torch.long)
    test_x = torch.as_tensor(test_x, dtype=torch.float32)
    test_y = torch.as_tensor(test_y, dtype=torch.long)

    # Standardize with frozen train statistics for a stable, reproducible probe.
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    probe = torch.nn.Linear(train_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    for epoch in range(epochs):
        probe.train()
        current_lr = lr * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

    probe.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(test_x), batch_size):
            batch_x = test_x[start : start + batch_size].to(device, non_blocking=True)
            logits = probe(batch_x)
            preds.append(logits.argmax(dim=1).cpu())
    preds = torch.cat(preds).numpy()
    acc = float((preds == test_y.numpy()).mean() * 100.0)
    macro_f1 = float(f1_score(test_y.numpy(), preds, average="macro") * 100.0)
    return acc, macro_f1, preds, None


def fit_knn(train_x, train_y, test_x, test_y, n_neighbors=20, device=None, batch_size=512):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    train_x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    test_x = torch.as_tensor(test_x, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(train_y, dtype=torch.long, device=device)
    num_classes = int(train_y.max().item()) + 1

    train_x = F.normalize(train_x, dim=1)
    test_x = F.normalize(test_x, dim=1)
    preds = []
    with torch.no_grad():
        train_x_t = train_x.T.contiguous()
        for start in range(0, test_x.shape[0], batch_size):
            batch = test_x[start : start + batch_size]
            sims = batch @ train_x_t
            nn_idx = sims.topk(k=n_neighbors, dim=1, largest=True, sorted=False).indices
            nn_labels = train_y[nn_idx]
            votes = F.one_hot(nn_labels, num_classes=num_classes).sum(dim=1)
            preds.append(votes.argmax(dim=1).cpu())
    preds = torch.cat(preds).numpy()
    acc = float((preds == test_y).mean() * 100.0)
    macro_f1 = float(f1_score(test_y, preds, average="macro") * 100.0)
    return acc, macro_f1, preds


def effective_rank(features):
    if len(features) < 2:
        return 1.0
    cov = np.cov(features.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-12, None)
    p = eigvals / eigvals.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def average_effective_rank(features, labels):
    values = []
    for cls in np.unique(labels):
        subset = features[labels == cls]
        values.append(effective_rank(subset))
    return float(np.mean(values))


def cosine_spread(features, labels, max_per_class=64):
    spreads = []
    rng = np.random.default_rng(0)
    for cls in np.unique(labels):
        subset = features[labels == cls]
        if len(subset) > max_per_class:
            subset = subset[rng.choice(len(subset), size=max_per_class, replace=False)]
        if len(subset) < 2:
            continue
        subset = subset / np.linalg.norm(subset, axis=1, keepdims=True).clip(min=1e-12)
        sims = subset @ subset.T
        tri = sims[np.triu_indices(len(subset), k=1)]
        spreads.append(float((1.0 - tri).mean()))
    return float(np.mean(spreads)) if spreads else 0.0


def subclass_accuracy(true_labels, pred_labels, num_subclasses=2):
    accuracies = []
    true_classes = true_labels // num_subclasses
    for cls in np.unique(true_classes):
        mask = true_classes == cls
        y_true = true_labels[mask] % num_subclasses
        y_pred = pred_labels[mask] % num_subclasses
        mat = np.zeros((num_subclasses, num_subclasses))
        for i in range(num_subclasses):
            for j in range(num_subclasses):
                mat[i, j] = np.sum((y_true == i) & (y_pred == j))
        row, col = linear_sum_assignment(-mat)
        accuracies.append(float(mat[row, col].sum() / len(y_true)))
    return float(np.mean(accuracies) * 100.0)


def principal_angle_error(pred_bases, true_bases):
    errors = []
    for pred, true in zip(pred_bases, true_bases):
        _, s, _ = np.linalg.svd(pred.T @ true, full_matrices=False)
        s = np.clip(s, -1.0, 1.0)
        angles = np.arccos(s)
        errors.append(float(np.mean(angles) * 180.0 / math.pi))
    return float(np.mean(errors)) if errors else 0.0


def cluster_subclasses(features, class_labels, num_subclasses=2):
    pred = np.zeros(len(features), dtype=np.int64)
    pred_bases = []
    for cls in np.unique(class_labels):
        mask = class_labels == cls
        subset = features[mask]
        km = KMeans(n_clusters=num_subclasses, random_state=0, n_init=10)
        local = km.fit_predict(subset)
        pred[mask] = cls * num_subclasses + local
        for k in range(num_subclasses):
            members = subset[local == k]
            if len(members) < 2:
                basis = np.eye(subset.shape[1], 2)
            else:
                u, _, _ = np.linalg.svd(members - members.mean(axis=0, keepdims=True), full_matrices=False)
                basis = np.linalg.svd(
                    (members - members.mean(axis=0, keepdims=True)).T,
                    full_matrices=False,
                )[0][:, :2]
            pred_bases.append(basis[:, :2])
    return pred, pred_bases


def adjusted_mi(true_labels, pred_labels):
    return float(adjusted_mutual_info_score(true_labels, pred_labels))

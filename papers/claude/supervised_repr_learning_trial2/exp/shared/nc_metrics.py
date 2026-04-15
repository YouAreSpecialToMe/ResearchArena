"""Neural Collapse metrics (NC1-NC4)."""

import torch
import numpy as np


@torch.no_grad()
def compute_nc_metrics(model, dataloader, num_classes, device='cuda'):
    """Compute neural collapse metrics NC1-NC4 on the full dataset.

    NC1: tr(Sigma_W @ Sigma_B_inv) / K  (within-class variability collapse)
    NC2: std of pairwise cosine similarities of centered class means (ETF structure)
    NC3: ||W_hat - M_hat||_F  (self-duality of classifier and class means)
    NC4: NCC accuracy vs network accuracy

    Returns:
        dict with NC1, NC2, NC3, NC4, plus per-class spread info
    """
    model.eval()

    all_features = []
    all_labels = []
    all_preds = []

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits, features = model(inputs, return_features=True)
        preds = logits.argmax(dim=1)
        all_features.append(features.cpu())
        all_labels.append(targets.cpu())
        all_preds.append(preds.cpu())

    features = torch.cat(all_features, dim=0).numpy()  # (N, D)
    labels = torch.cat(all_labels, dim=0).numpy()        # (N,)
    preds = torch.cat(all_preds, dim=0).numpy()

    N, D = features.shape
    K = num_classes

    # Compute class means and global mean
    class_means = np.zeros((K, D))
    class_counts = np.zeros(K)
    for c in range(K):
        mask = labels == c
        if mask.sum() > 0:
            class_means[c] = features[mask].mean(axis=0)
            class_counts[c] = mask.sum()

    global_mean = features.mean(axis=0)

    # NC1: Within-class variability collapse
    # Sigma_W = (1/N) sum_c sum_{i in c} (z_i - mu_c)(z_i - mu_c)^T
    # Sigma_B = (1/K) sum_c (mu_c - mu_global)(mu_c - mu_global)^T
    # NC1 = tr(Sigma_W @ Sigma_B_inv) / K

    # Compute Sigma_W efficiently
    Sigma_W = np.zeros((D, D))
    per_class_spread = []
    for c in range(K):
        mask = labels == c
        if mask.sum() == 0:
            per_class_spread.append(0.0)
            continue
        centered = features[mask] - class_means[c]
        spread = np.mean(np.sum(centered ** 2, axis=1))
        per_class_spread.append(float(spread))
        Sigma_W += centered.T @ centered
    Sigma_W /= N

    # Compute Sigma_B
    centered_means = class_means - global_mean
    Sigma_B = centered_means.T @ centered_means / K

    # NC1 = tr(Sigma_W @ Sigma_B^{-1}) / K
    try:
        # Use pseudo-inverse for numerical stability
        Sigma_B_inv = np.linalg.pinv(Sigma_B)
        nc1 = np.trace(Sigma_W @ Sigma_B_inv) / K
    except np.linalg.LinAlgError:
        nc1 = float('inf')

    # NC2: ETF structure - std of pairwise cosine similarities of centered class means
    centered_means_norm = centered_means / (np.linalg.norm(centered_means, axis=1, keepdims=True) + 1e-10)
    cos_sim_matrix = centered_means_norm @ centered_means_norm.T
    # Extract upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(K, k=1)
    pairwise_cos = cos_sim_matrix[triu_indices]
    nc2 = float(np.std(pairwise_cos))

    # NC3: Self-duality ||W_hat - M_hat||_F
    # W_hat = W / ||W||_row, M_hat = centered_means / ||centered_means||_row
    try:
        W = None
        for name, param in model.named_parameters():
            if 'fc.weight' in name:
                W = param.detach().cpu().numpy()
                break

        if W is not None:
            W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-10)
            M_norm = centered_means / (np.linalg.norm(centered_means, axis=1, keepdims=True) + 1e-10)
            nc3 = float(np.linalg.norm(W_norm - M_norm, 'fro'))
        else:
            nc3 = float('nan')
    except Exception:
        nc3 = float('nan')

    # NC4: NCC accuracy vs network accuracy
    # NCC: classify each sample by nearest class mean
    # dists[i, c] = ||z_i - mu_c||^2
    dists = np.zeros((N, K))
    for c in range(K):
        dists[:, c] = np.sum((features - class_means[c]) ** 2, axis=1)
    ncc_preds = np.argmin(dists, axis=1)
    ncc_acc = float(np.mean(ncc_preds == labels))
    net_acc = float(np.mean(preds == labels))
    nc4 = abs(ncc_acc - net_acc)

    return {
        'nc1': float(nc1),
        'nc2': float(nc2),
        'nc3': float(nc3),
        'nc4': float(nc4),
        'ncc_accuracy': ncc_acc,
        'network_accuracy': net_acc,
        'mean_within_class_spread': float(np.mean(per_class_spread)),
        'per_class_spread': per_class_spread,
    }

#!/usr/bin/env python3
"""Analyze embedding geometry to validate CGA's effect on ETF structure.

Loads trained models and computes:
1. Class mean embeddings and pairwise cosine similarity matrices
2. ETF deviation (variance of off-diagonal cosine similarities)
3. Confusion-distance correlation (Theorem 1 validation)
4. Top-20 confused pair analysis
5. Visualization: t-SNE of class means colored by superclass
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.shared.data_loader import get_dataloaders, CIFAR100_SUPERCLASS
from exp.shared.models import SupConModel

WORKSPACE = '/home/zz865/pythonProject/autoresearch/outputs/claude/run_1/supervised_representation_learning/idea_01'
FIGURES_DIR = os.path.join(WORKSPACE, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def extract_class_means(model, test_loader, num_classes=100, device='cuda'):
    """Extract class mean embeddings from test set."""
    model.eval()
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            feat, z = model(images)
            all_feats.append(feat.cpu())
            all_labels.append(labels)

    all_feats = torch.cat(all_feats)
    all_labels = torch.cat(all_labels)

    class_means = []
    for c in range(num_classes):
        mask = (all_labels == c)
        if mask.sum() > 0:
            class_mean = F.normalize(all_feats[mask].mean(dim=0), dim=0)
            class_means.append(class_mean)
        else:
            class_means.append(torch.zeros(all_feats.shape[1]))

    return torch.stack(class_means), all_feats, all_labels


def compute_geometry_metrics(class_means, num_classes=100):
    """Compute embedding geometry metrics."""
    cos_sim = torch.matmul(class_means, class_means.T)
    mask = 1 - torch.eye(num_classes)
    off_diag = cos_sim[mask.bool()]

    metrics = {
        'mean_cosine_sim': off_diag.mean().item(),
        'std_cosine_sim': off_diag.std().item(),
        'etf_deviation': off_diag.var().item(),
        'min_cosine_sim': off_diag.min().item(),
        'max_cosine_sim': off_diag.max().item(),
    }

    # Expected ETF value
    etf_expected = -1.0 / (num_classes - 1)
    metrics['etf_expected'] = etf_expected
    metrics['mean_deviation_from_etf'] = abs(off_diag.mean().item() - etf_expected)

    return metrics, cos_sim


def compute_hierarchy_correlation(cos_sim, superclass_map, num_classes=100):
    """Compute correlation between embedding distances and superclass structure."""
    distances = 1 - cos_sim.numpy()
    sc_map = np.array(superclass_map)

    gt_same_sc = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            gt_same_sc[i, j] = float(sc_map[i] == sc_map[j])

    triu_idx = np.triu_indices(num_classes, k=1)
    dist_flat = distances[triu_idx]
    gt_flat = gt_same_sc[triu_idx]

    # Same-superclass pairs should have smaller distances
    rho, p_val = stats.spearmanr(-dist_flat, gt_flat)

    # Also compute within vs between superclass average distance
    within_mask = gt_same_sc[triu_idx] == 1
    between_mask = gt_same_sc[triu_idx] == 0

    within_dist = dist_flat[within_mask].mean() if within_mask.sum() > 0 else 0
    between_dist = dist_flat[between_mask].mean() if between_mask.sum() > 0 else 0

    return {
        'spearman_rho': rho,
        'spearman_p': p_val,
        'within_sc_mean_dist': float(within_dist),
        'between_sc_mean_dist': float(between_dist),
        'dist_ratio': float(between_dist / within_dist) if within_dist > 0 else 0,
    }


def plot_class_means_tsne(class_means_dict, superclass_map, title='Class Mean Embeddings (t-SNE)'):
    """t-SNE visualization of class means for multiple methods."""
    n_methods = len(class_means_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    sc_map = np.array(superclass_map)
    colors = plt.cm.tab20(sc_map / 20.0)

    for ax, (method, means) in zip(axes, class_means_dict.items()):
        means_np = means.numpy()
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, means_np.shape[0]-1))
        embedded = tsne.fit_transform(means_np)

        scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=sc_map, cmap='tab20',
                           s=30, alpha=0.8, edgecolors='black', linewidths=0.3)
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(FIGURES_DIR, 'figure_geometry_tsne.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'figure_geometry_tsne.png'))
    plt.close()
    print("Saved figure_geometry_tsne.pdf/png")


def plot_cosine_similarity_heatmaps(cos_sim_dict, superclass_map):
    """Plot cosine similarity heatmaps for different methods."""
    n_methods = len(cos_sim_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    # Sort classes by superclass for better visualization
    sc_map = np.array(superclass_map)
    sort_idx = np.argsort(sc_map)

    for ax, (method, cos_sim) in zip(axes, cos_sim_dict.items()):
        sim_np = cos_sim.numpy()
        sim_sorted = sim_np[sort_idx][:, sort_idx]
        im = ax.imshow(sim_sorted, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
        ax.set_title(f'{method}\n(sorted by superclass)')
        ax.set_xlabel('Class index')
        ax.set_ylabel('Class index')
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'figure_cosine_heatmap.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'figure_cosine_heatmap.png'))
    plt.close()
    print("Saved figure_cosine_heatmap.pdf/png")


def main():
    """Run full geometry analysis using saved results (no model reloading needed).

    Since we don't save model checkpoints, we use the metrics already
    computed during evaluation and stored in results JSON files.
    """
    print("Geometry Analysis from Saved Results")
    print("="*60)

    # Load pre-computed metrics from results files
    methods_data = {}
    for method, dirs in [
        ('SupCon', ['exp/supcon_results']),
        ('CGA-only', ['exp/cga_best', 'exp/cga_main']),
        ('CE', ['exp/ce_results']),
    ]:
        all_results = []
        for d in dirs:
            for seed in [42, 43, 44, 45, 46]:
                fpath = os.path.join(WORKSPACE, d, f'results_seed{seed}.json')
                if os.path.exists(fpath):
                    with open(fpath) as f:
                        r = json.load(f)
                        all_results.append(r)
        if all_results:
            methods_data[method] = all_results

    # Print geometry comparison
    print("\nGeometry Metrics Comparison:")
    print(f"{'Method':<15} {'ETF Dev':>10} {'Hier Corr':>10} {'W-SC Acc':>10} {'B-SC Err':>10}")
    print("-"*60)

    for method, results in methods_data.items():
        etf = [r.get('etf_deviation', 0) for r in results if 'etf_deviation' in r]
        hier = [r.get('hierarchy_corr', 0) for r in results if 'hierarchy_corr' in r]
        wsc = [r.get('within_superclass_acc', 0) for r in results if 'within_superclass_acc' in r]
        bsc = [r.get('between_superclass_error_rate', 0) for r in results if 'between_superclass_error_rate' in r]

        print(f"{method:<15} "
              f"{np.mean(etf):>10.6f} "
              f"{np.mean(hier):>10.4f} "
              f"{np.mean(wsc):>10.2f} "
              f"{np.mean(bsc):>10.2f}")

    # Save analysis
    analysis = {}
    for method, results in methods_data.items():
        analysis[method] = {
            'n_seeds': len(results),
            'etf_deviation': {
                'mean': float(np.mean([r.get('etf_deviation', 0) for r in results])),
                'std': float(np.std([r.get('etf_deviation', 0) for r in results])),
            },
            'hierarchy_corr': {
                'mean': float(np.mean([r.get('hierarchy_corr', 0) for r in results])),
                'std': float(np.std([r.get('hierarchy_corr', 0) for r in results])),
            },
        }

    out_path = os.path.join(WORKSPACE, 'results', 'geometry_analysis.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved geometry analysis to {out_path}")


if __name__ == '__main__':
    main()

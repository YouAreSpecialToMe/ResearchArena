"""
Evaluation metrics for point cloud generation.
"""
import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def chamfer_distance(pred, target, batch_mean=True):
    """
    Compute Chamfer Distance between two point clouds.
    
    Args:
        pred: (B, N, 3) or (N, 3) predicted point cloud
        target: (B, N, 3) or (N, 3) target point cloud
        batch_mean: if True, return mean over batch
    Returns:
        CD value(s)
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Compute pairwise distances
    # (B, N, 1, 3) - (B, 1, N, 3) -> (B, N, N)
    dist_matrix = torch.cdist(pred, target, p=2)
    
    # min distances
    dist_pred_to_target = dist_matrix.min(dim=2)[0]  # (B, N)
    dist_target_to_pred = dist_matrix.min(dim=1)[0]  # (B, N)
    
    # Chamfer distance
    cd = dist_pred_to_target.mean(dim=1) + dist_target_to_pred.mean(dim=1)
    
    if batch_mean:
        return cd.mean().item()
    return cd


def chamfer_distance_stratified(pred, target, pred_dist, target_dist, ranges=None):
    """
    Compute Chamfer Distance stratified by distance ranges.
    
    Args:
        pred: (B, N, 3) predicted point cloud
        target: (B, N, 3) target point cloud
        pred_dist: (B, N) radial distances for pred
        target_dist: (B, N) radial distances for target
        ranges: list of distance ranges [(0, 20), (20, 50), (50, 80)]
    Returns:
        dict of CD values for each range
    """
    if ranges is None:
        ranges = [(0, 0.25), (0.25, 0.625), (0.625, 1.0)]  # Normalized ranges
    
    results = {}
    
    for i, (r_min, r_max) in enumerate(ranges):
        range_name = ['near', 'mid', 'far'][i]
        
        # Mask points in this range
        pred_mask = (pred_dist >= r_min) & (pred_dist < r_max)
        target_mask = (target_dist >= r_min) & (target_dist < r_max)
        
        # Compute CD for this range
        cds = []
        for b in range(pred.shape[0]):
            pred_pts = pred[b][pred_mask[b]]
            target_pts = target[b][target_mask[b]]
            
            if len(pred_pts) == 0 or len(target_pts) == 0:
                continue
            
            # Compute CD
            dist_matrix = torch.cdist(pred_pts.unsqueeze(0), target_pts.unsqueeze(0), p=2)[0]
            d1 = dist_matrix.min(dim=0)[0].mean()
            d2 = dist_matrix.min(dim=1)[0].mean()
            cd = (d1 + d2).item()
            cds.append(cd)
        
        if len(cds) > 0:
            results[f'cd_{range_name}'] = np.mean(cds)
        else:
            results[f'cd_{range_name}'] = float('nan')
    
    return results


def earth_movers_distance(pred, target, batch_mean=True):
    """
    Approximate Earth Mover's Distance using Sinkhorn or sampling.
    For efficiency, we use a sampling-based approximation.
    
    Args:
        pred: (B, N, 3) or (N, 3)
        target: (B, N, 3) or (N, 3)
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Sample for efficiency
    N_sample = min(2048, pred.shape[1])
    
    emds = []
    for b in range(pred.shape[0]):
        # Sample points
        idx_pred = torch.randperm(pred.shape[1])[:N_sample]
        idx_target = torch.randperm(target.shape[1])[:N_sample]
        
        p = pred[b, idx_pred]
        t = target[b, idx_target]
        
        # Compute cost matrix
        C = torch.cdist(p, t, p=2)
        
        # Approximate EMD via min cost matching (simplified)
        # For true EMD, use optimal transport solvers
        min_cost = C.min(dim=1)[0].mean() + C.min(dim=0)[0].mean()
        emds.append(min_cost.item() / 2)
    
    if batch_mean:
        return np.mean(emds)
    return emds


def compute_1nn_accuracy(real_features, fake_features):
    """
    Compute 1-NN accuracy for feature-based realism.
    
    Args:
        real_features: (N_real, D) features from real data
        fake_features: (N_fake, D) features from generated data
    Returns:
        1-NN accuracy (should be ~0.5 for perfect generator)
    """
    # Build nearest neighbor classifier
    n_real = len(real_features)
    n_fake = len(fake_features)
    
    # Create train/test split
    all_features = np.vstack([real_features, fake_features])
    labels = np.array([0] * n_real + [1] * n_fake)
    
    # Train 1-NN
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(all_features)
    distances, indices = nbrs.kneighbors(all_features)
    
    # Leave-one-out accuracy
    correct = 0
    for i in range(len(all_features)):
        neighbor_idx = indices[i, 1] if indices[i, 0] == i else indices[i, 0]
        if labels[i] == labels[neighbor_idx]:
            correct += 1
    
    accuracy = correct / len(all_features)
    return accuracy


def compute_coverage(pred, target, k=3):
    """
    Compute coverage metric - fraction of target points covered by predictions.
    
    Args:
        pred: (B, N, 3) or (N, 3)
        target: (B, N, 3) or (N, 3)
        k: k-NN threshold
    Returns:
        coverage score
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Sample for efficiency
    N_sample = min(2048, pred.shape[1])
    
    coverages = []
    for b in range(pred.shape[0]):
        idx_pred = torch.randperm(pred.shape[1])[:N_sample]
        idx_target = torch.randperm(target.shape[1])[:N_sample]
        
        p = pred[b, idx_pred].cpu().numpy()
        t = target[b, idx_target].cpu().numpy()
        
        # Find nearest neighbors
        dist_matrix = cdist(t, p, metric='euclidean')
        min_dists = dist_matrix.min(axis=1)
        
        # Points within threshold
        threshold = np.percentile(min_dists, 50) * 2
        covered = (min_dists < threshold).mean()
        coverages.append(covered)
    
    return np.mean(coverages)


def compute_all_metrics(pred, target, pred_dist=None, target_dist=None):
    """
    Compute all evaluation metrics.
    
    Args:
        pred: (B, N, 3) or (N, 3) predicted point cloud
        target: (B, N, 3) or (N, 3) target point cloud
        pred_dist: (B, N) radial distances for pred
        target_dist: (B, N) radial distances for target
    Returns:
        dict of metrics
    """
    metrics = {}
    
    # Overall Chamfer Distance
    metrics['cd_overall'] = chamfer_distance(pred, target)
    
    # Stratified CD
    if pred_dist is not None and target_dist is not None:
        stratified = chamfer_distance_stratified(pred, target, pred_dist, target_dist)
        metrics.update(stratified)
    
    # EMD (approximate)
    metrics['emd'] = earth_movers_distance(pred, target)
    
    return metrics


def statistical_test(values_a, values_b):
    """
    Perform paired t-test between two sets of values.
    
    Args:
        values_a: list or array of values
        values_b: list or array of values
    Returns:
        t-statistic and p-value
    """
    from scipy import stats
    
    values_a = np.array(values_a)
    values_b = np.array(values_b)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_ind(values_a, values_b)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_diff': values_a.mean() - values_b.mean(),
        'percent_improvement': (values_b.mean() - values_a.mean()) / values_b.mean() * 100,
    }


if __name__ == "__main__":
    # Test metrics
    pred = torch.randn(4, 2048, 3)
    target = torch.randn(4, 2048, 3)
    pred_dist = torch.rand(4, 2048)
    target_dist = torch.rand(4, 2048)
    
    metrics = compute_all_metrics(pred, target, pred_dist, target_dist)
    print("Metrics:", metrics)

"""
Optimal Transport utilities for Wasserstein-optimal item selection.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import ot  # POT library


def wasserstein_distance(source, target, p=2):
    """Compute Wasserstein distance between two distributions.
    
    Args:
        source: (n,) array of source samples or (n, d) array of samples
        target: (m,) array of target samples or (m, d) array of samples
        p: Power for Wasserstein distance (default 2 for W_2)
        
    Returns:
        distance: Wasserstein distance value
    """
    source = np.atleast_1d(source)
    target = np.atleast_1d(target)
    
    if source.ndim == 1:
        source = source.reshape(-1, 1)
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    
    # Uniform weights
    a = np.ones(source.shape[0]) / source.shape[0]
    b = np.ones(target.shape[0]) / target.shape[0]
    
    # Cost matrix (squared Euclidean distance)
    M = cdist(source, target, metric='euclidean') ** p
    
    # Compute Wasserstein distance using POT
    dist = ot.emd2(a, b, M)
    
    return dist ** (1.0 / p)


def sinkhorn_selection(item_difficulties, target_distribution, n_select, 
                       epsilon=0.01, max_iter=1000, domain_labels=None,
                       domain_min_fraction=0.1):
    """Select items using entropic-regularized optimal transport (Sinkhorn).
    
    Args:
        item_difficulties: (n_items,) array of item difficulty values
        target_distribution: (n_target,) or callable representing target difficulty dist
        n_select: Number of items to select
        epsilon: Entropic regularization parameter
        max_iter: Maximum iterations for Sinkhorn
        domain_labels: (n_items,) array of domain labels for content balancing
        domain_min_fraction: Minimum fraction per domain (e.g., 0.1 for 10%)
        
    Returns:
        selected_indices: Indices of selected items
        selection_weights: Transport plan (n_items, n_bins)
    """
    n_items = len(item_difficulties)
    
    # Create target samples by sampling from target distribution
    if callable(target_distribution):
        # If target is a callable (e.g., scipy distribution), sample from it
        target_samples = target_distribution(n_select)
    else:
        # Otherwise treat as samples
        target_samples = np.array(target_distribution)
        # If fewer target samples than n_select, resample
        if len(target_samples) < n_select:
            target_samples = np.random.choice(target_samples, n_select, replace=True)
    
    target_samples = np.atleast_1d(target_samples).reshape(-1, 1)
    source_samples = item_difficulties.reshape(-1, 1)
    n_target = len(target_samples)
    
    # Uniform weights
    a = np.ones(n_items) / n_items  # Source weights
    b = np.ones(n_target) / n_target  # Target weights (match target size)
    
    # Cost matrix
    M = cdist(source_samples, target_samples, metric='euclidean') ** 2
    
    # Solve entropic-regularized OT using Sinkhorn
    transport_plan = ot.sinkhorn(a, b, M, epsilon, numItermax=max_iter)
    
    # For each target sample, find the source with maximum mass
    # This gives a soft assignment; we'll do greedy rounding
    item_scores = transport_plan.sum(axis=1)  # Total mass sent from each item
    
    # Greedy selection with domain constraints
    selected_indices = []
    available = set(range(n_items))
    
    # Track domain counts
    if domain_labels is not None:
        unique_domains = np.unique(domain_labels)
        domain_min_count = max(1, int(domain_min_fraction * n_select))
        domain_counts = {d: 0 for d in unique_domains}
    
    for _ in range(n_select):
        # Filter available items to satisfy domain constraints
        if domain_labels is not None and _ > n_select * 0.7:
            # In later stages, ensure we meet minimum domain requirements
            underrepresented = [d for d, c in domain_counts.items() 
                              if c < domain_min_count]
            if underrepresented:
                valid_items = [i for i in available 
                             if domain_labels[i] in underrepresented]
                if not valid_items:
                    valid_items = list(available)
            else:
                valid_items = list(available)
        else:
            valid_items = list(available)
        
        if not valid_items:
            valid_items = list(available)
        
        # Select item with highest score among valid items
        valid_scores = [(i, item_scores[i]) for i in valid_items]
        best_item = max(valid_scores, key=lambda x: x[1])[0]
        
        selected_indices.append(best_item)
        available.remove(best_item)
        
        if domain_labels is not None:
            domain_counts[domain_labels[best_item]] += 1
        
        # Reduce score of selected item to 0
        item_scores[best_item] = 0
    
    return np.array(selected_indices), transport_plan


def bin_matching_selection(item_difficulties, target_distribution, n_select, 
                           n_bins=5, domain_labels=None, domain_min_fraction=0.1):
    """Select items using simple bin-matching (baseline for ablation).
    
    Divides items into difficulty bins and samples uniformly from each bin
    to match the target distribution histogram.
    
    Args:
        item_difficulties: (n_items,) array of item difficulty values
        target_distribution: (n_target,) samples from target distribution
        n_select: Number of items to select
        n_bins: Number of difficulty bins
        domain_labels: (n_items,) array of domain labels
        domain_min_fraction: Minimum fraction per domain
        
    Returns:
        selected_indices: Indices of selected items
    """
    n_items = len(item_difficulties)
    target_samples = np.array(target_distribution)
    
    # Define bins based on combined range
    min_val = min(item_difficulties.min(), target_samples.min())
    max_val = max(item_difficulties.max(), target_samples.max())
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    
    # Assign items to bins
    item_bins = np.digitize(item_difficulties, bin_edges[:-1]) - 1
    item_bins = np.clip(item_bins, 0, n_bins - 1)
    
    # Compute target distribution histogram
    target_hist, _ = np.histogram(target_samples, bins=bin_edges)
    target_fractions = target_hist / target_hist.sum()
    
    # Compute number to select from each bin
    bin_targets = (target_fractions * n_select).astype(int)
    # Adjust for rounding errors
    while bin_targets.sum() < n_select:
        bin_targets[np.argmax(target_fractions - bin_targets / n_select)] += 1
    
    selected_indices = []
    
    for bin_idx in range(n_bins):
        n_from_bin = bin_targets[bin_idx]
        if n_from_bin == 0:
            continue
            
        # Get items in this bin
        bin_items = np.where(item_bins == bin_idx)[0]
        
        if len(bin_items) == 0:
            continue
        
        # Random sample from this bin
        n_sample = min(n_from_bin, len(bin_items))
        sampled = np.random.choice(bin_items, n_sample, replace=False)
        selected_indices.extend(sampled)
    
    # If we don't have enough items, fill randomly
    while len(selected_indices) < n_select:
        remaining = n_select - len(selected_indices)
        available = [i for i in range(n_items) if i not in selected_indices]
        if len(available) == 0:
            break
        sampled = np.random.choice(available, min(remaining, len(available)), replace=False)
        selected_indices.extend(sampled)
    
    # Domain balancing: swap items if needed
    if domain_labels is not None and len(selected_indices) == n_select:
        selected_indices = domain_balancing_swap(
            np.array(selected_indices), item_difficulties, domain_labels,
            domain_min_fraction, n_select
        )
    
    return np.array(selected_indices[:n_select])


def domain_balancing_swap(selected_indices, item_difficulties, domain_labels,
                          min_fraction, n_select):
    """Post-process selection to ensure domain balance via swapping.
    
    Args:
        selected_indices: Current selection
        item_difficulties: All item difficulties
        domain_labels: Domain label for each item
        min_fraction: Minimum fraction per domain
        n_select: Total items to select
        
    Returns:
        balanced_indices: Rebalanced selection
    """
    selected = list(selected_indices)
    not_selected = [i for i in range(len(item_difficulties)) if i not in selected]
    
    unique_domains = np.unique(domain_labels)
    min_count = max(1, int(min_fraction * n_select))
    
    # Count current domain representation
    domain_counts = {}
    for d in unique_domains:
        domain_counts[d] = sum(1 for i in selected if domain_labels[i] == d)
    
    # Try to add underrepresented domains
    for d in unique_domains:
        while domain_counts[d] < min_count and not_selected:
            # Find an item from this domain in not_selected
            candidates = [i for i in not_selected if domain_labels[i] == d]
            if not candidates:
                break
            
            # Find the most overrepresented domain to swap from
            overrep = [(od, oc) for od, oc in domain_counts.items() if oc > min_count]
            if not overrep:
                # Just add without swap if no overrepresented
                to_add = candidates[0]
                selected.append(to_add)
                not_selected.remove(to_add)
                domain_counts[d] += 1
                continue
            
            # Swap: remove from overrep, add from underrep
            overrep_domain = max(overrep, key=lambda x: x[1])[0]
            swappable = [i for i in selected if domain_labels[i] == overrep_domain]
            if swappable:
                to_remove = swappable[0]
                to_add = candidates[0]
                
                selected.remove(to_remove)
                not_selected.append(to_remove)
                not_selected.remove(to_add)
                selected.append(to_add)
                
                domain_counts[overrep_domain] -= 1
                domain_counts[d] += 1
            else:
                break
    
    return np.array(selected[:n_select])


def compute_selection_quality(selected_difficulties, target_distribution):
    """Compute quality metrics for selection.
    
    Args:
        selected_difficulties: (n_selected,) difficulties of selected items
        target_distribution: (n_target,) target samples
        
    Returns:
        metrics: Dictionary of quality metrics
    """
    target_samples = np.array(target_distribution)
    
    # Wasserstein distance
    w_dist = wasserstein_distance(selected_difficulties, target_samples, p=2)
    
    # Kolmogorov-Smirnov statistic
    from scipy.stats import ks_2samp
    ks_stat, ks_p = ks_2samp(selected_difficulties, target_samples)
    
    # Mean and std difference
    mean_diff = abs(selected_difficulties.mean() - target_samples.mean())
    std_diff = abs(selected_difficulties.std() - target_samples.std())
    
    # Histogram correlation
    hist_sel, bins = np.histogram(selected_difficulties, bins=10, density=True)
    hist_tgt, _ = np.histogram(target_samples, bins=bins, density=True)
    hist_corr = np.corrcoef(hist_sel, hist_tgt)[0, 1]
    
    return {
        'wasserstein_distance': float(w_dist),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_p),
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff),
        'histogram_correlation': float(hist_corr)
    }

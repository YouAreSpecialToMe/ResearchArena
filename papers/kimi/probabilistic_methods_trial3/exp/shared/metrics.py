"""
Metrics for evaluating MCMC sampler performance.
"""
import numpy as np
from scipy import stats


def compute_ess_ar1(samples):
    """
    Compute Effective Sample Size using AR(1) approximation.
    ESS = n / (1 + 2 * sum_k rho_k) where rho_k is autocorrelation at lag k.
    
    Uses the AR(1) approximation: ESS ≈ n * (1 - r) / (1 + r)
    where r is the lag-1 autocorrelation.
    
    Args:
        samples: Array of shape (n_samples, dim)
    
    Returns:
        ESS per dimension and mean ESS
    """
    samples = np.asarray(samples)
    n_samples = samples.shape[0]
    
    if n_samples < 10:
        return np.zeros(samples.shape[1] if len(samples.shape) > 1 else 1), 0.0
    
    # Compute per-dimension ESS
    if len(samples.shape) == 1:
        samples = samples.reshape(-1, 1)
    
    ess_per_dim = []
    for d in range(samples.shape[1]):
        x = samples[:, d]
        
        # Center
        x_centered = x - np.mean(x)
        
        # Lag-1 autocorrelation
        c0 = np.mean(x_centered ** 2)
        if c0 < 1e-10:
            ess_per_dim.append(n_samples)
            continue
            
        c1 = np.mean(x_centered[1:] * x_centered[:-1])
        rho = c1 / c0
        
        # Clip to avoid numerical issues
        rho = np.clip(rho, -0.99, 0.99)
        
        # AR(1) ESS approximation
        ess = n_samples * (1 - rho) / (1 + rho)
        ess = max(1, min(ess, n_samples))  # Clip to valid range
        ess_per_dim.append(ess)
    
    ess_per_dim = np.array(ess_per_dim)
    return ess_per_dim, np.mean(ess_per_dim)


def compute_ess_spectral(samples, max_lag=None):
    """
    Compute ESS using spectral density method (more accurate).
    
    Args:
        samples: Array of shape (n_samples, dim)
        max_lag: Maximum lag for autocorrelation sum
    
    Returns:
        ESS per dimension and mean ESS
    """
    samples = np.asarray(samples)
    n_samples = samples.shape[0]
    
    if n_samples < 10:
        return np.zeros(samples.shape[1] if len(samples.shape) > 1 else 1), 0.0
    
    if len(samples.shape) == 1:
        samples = samples.reshape(-1, 1)
    
    if max_lag is None:
        max_lag = min(n_samples // 3, 100)
    
    ess_per_dim = []
    for d in range(samples.shape[1]):
        x = samples[:, d]
        x_centered = x - np.mean(x)
        
        # Compute autocorrelations up to max_lag
        c0 = np.mean(x_centered ** 2)
        if c0 < 1e-10:
            ess_per_dim.append(n_samples)
            continue
        
        rho_sum = 0
        for k in range(1, max_lag + 1):
            ck = np.mean(x_centered[k:] * x_centered[:-k])
            rho = ck / c0
            if rho < 0.05:  # Truncate when autocorr becomes small
                break
            rho_sum += rho
        
        ess = n_samples / (1 + 2 * rho_sum)
        ess = max(1, min(ess, n_samples))
        ess_per_dim.append(ess)
    
    ess_per_dim = np.array(ess_per_dim)
    return ess_per_dim, np.mean(ess_per_dim)


def compute_ess_bulk(samples):
    """
    Compute minimum ESS across dimensions (bulk ESS).
    This is the most conservative measure.
    
    Args:
        samples: Array of shape (n_samples, dim)
    
    Returns:
        Bulk ESS
    """
    ess_per_dim, _ = compute_ess_spectral(samples)
    return np.min(ess_per_dim)


def compute_r_hat(chains):
    """
    Compute Gelman-Rubin R-hat statistic for convergence diagnostics.
    R-hat ≈ 1 indicates convergence.
    
    Args:
        chains: List of sample arrays, each shape (n_samples, dim)
    
    Returns:
        R-hat per dimension
    """
    chains = [np.asarray(c) for c in chains]
    m = len(chains)  # Number of chains
    n = chains[0].shape[0]  # Samples per chain
    
    if len(chains[0].shape) == 1:
        chains = [c.reshape(-1, 1) for c in chains]
    
    d = chains[0].shape[1]
    
    r_hats = []
    for i in range(d):
        # Get samples for dimension i
        x = np.array([c[:, i] for c in chains])  # Shape: (m, n)
        
        # Between-chain variance
        chain_means = np.mean(x, axis=1)
        overall_mean = np.mean(chain_means)
        B = n * np.var(chain_means, ddof=1)
        
        # Within-chain variance
        W = np.mean([np.var(x[j], ddof=1) for j in range(m)])
        
        # Pooled variance
        var_plus = (n - 1) / n * W + B / n
        
        # R-hat
        if var_plus < 1e-10:
            r_hat = 1.0
        else:
            r_hat = np.sqrt(var_plus / W)
        
        r_hats.append(r_hat)
    
    return np.array(r_hats)


def compute_convergence_stats(samples, n_splits=4):
    """
    Compute various convergence statistics.
    
    Args:
        samples: Array of shape (n_samples, dim)
        n_splits: Number of splits for R-hat
    
    Returns:
        Dict with convergence statistics
    """
    samples = np.asarray(samples)
    n_samples = samples.shape[0]
    
    # Split into chains for R-hat
    chain_size = n_samples // n_splits
    chains = [samples[i*chain_size:(i+1)*chain_size] for i in range(n_splits)]
    
    r_hats = compute_r_hat(chains)
    
    return {
        'r_hat_mean': np.mean(r_hats),
        'r_hat_max': np.max(r_hats),
        'r_hat_per_dim': r_hats.tolist()
    }


def compute_mode_discovery(samples, true_modes, threshold=0.1):
    """
    Count how many modes have been discovered in samples.
    
    Args:
        samples: Array of shape (n_samples, dim)
        true_modes: List of mode centers
        threshold: Distance threshold to consider a mode discovered
    
    Returns:
        Number of modes discovered
    """
    if len(true_modes) == 0:
        return 0
    
    samples = np.asarray(samples)
    true_modes = np.array(true_modes)
    
    # Compute minimum distance from each sample to each mode
    discovered = set()
    for i, mode in enumerate(true_modes):
        distances = np.linalg.norm(samples - mode, axis=1)
        if np.min(distances) < threshold * len(mode):
            discovered.add(i)
    
    return len(discovered)


def compute_mode_switching_rate(samples, window=10):
    """
    Compute rate of mode switching using clustering.
    
    Args:
        samples: Array of shape (n_samples, dim)
        window: Window size for detecting switches
    
    Returns:
        Mode switching rate
    """
    from sklearn.cluster import KMeans
    
    samples = np.asarray(samples)
    n_samples = samples.shape[0]
    
    if n_samples < 2 * window:
        return 0.0
    
    # Use k-means to identify modes
    n_clusters = min(10, n_samples // 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
    labels = kmeans.fit_predict(samples)
    
    # Count transitions
    transitions = 0
    for i in range(len(labels) - 1):
        if labels[i] != labels[i + 1]:
            transitions += 1
    
    return transitions / (len(labels) - 1)


def compute_cohens_d(group1, group2):
    """
    Compute Cohen's d effect size.
    
    Args:
        group1: Array of measurements
        group2: Array of measurements
    
    Returns:
        Cohen's d
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std < 1e-10:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def mann_whitney_test(group1, group2):
    """
    Perform Mann-Whitney U test.
    
    Returns:
        statistic, p-value
    """
    try:
        statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        return statistic, pvalue
    except:
        return 0, 1.0


def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
    
    Returns:
        Corrected alpha, list of rejected null hypotheses
    """
    m = len(p_values)
    alpha_corrected = alpha / m
    rejected = [p < alpha_corrected for p in p_values]
    return alpha_corrected, rejected


def compute_summary_statistics(samples, method_name="", runtime=None):
    """
    Compute comprehensive summary statistics for a sampler run.
    
    Args:
        samples: Array of samples
        method_name: Name of the method
        runtime: Runtime in seconds
    
    Returns:
        Dict of statistics
    """
    samples = np.asarray(samples)
    n_samples, dim = samples.shape[0], samples.shape[1] if len(samples.shape) > 1 else 1
    
    # ESS
    ess_per_dim, mean_ess = compute_ess_spectral(samples)
    bulk_ess = compute_ess_bulk(samples)
    
    # Convergence
    conv_stats = compute_convergence_stats(samples)
    
    stats_dict = {
        'method': method_name,
        'n_samples': n_samples,
        'dim': dim,
        'ess_mean': float(mean_ess),
        'ess_bulk': float(bulk_ess),
        'ess_per_dim': ess_per_dim.tolist(),
        'r_hat_mean': conv_stats['r_hat_mean'],
        'r_hat_max': conv_stats['r_hat_max'],
    }
    
    if runtime is not None:
        stats_dict['runtime_seconds'] = runtime
        stats_dict['ess_per_second'] = float(mean_ess / runtime) if runtime > 0 else 0
    
    return stats_dict


def aggregate_results(results_list):
    """
    Aggregate results across multiple seeds.
    
    Args:
        results_list: List of result dicts from multiple seeds
    
    Returns:
        Aggregated statistics
    """
    if not results_list:
        return {}
    
    # Extract numeric fields
    numeric_fields = ['ess_mean', 'ess_bulk', 'ess_per_second', 'runtime_seconds']
    
    aggregated = {}
    for field in numeric_fields:
        values = [r[field] for r in results_list if field in r]
        if values:
            aggregated[f'{field}_mean'] = float(np.mean(values))
            aggregated[f'{field}_std'] = float(np.std(values, ddof=1))
            aggregated[f'{field}_min'] = float(np.min(values))
            aggregated[f'{field}_max'] = float(np.max(values))
    
    # Coefficient of variation
    if 'ess_mean_mean' in aggregated and 'ess_mean_std' in aggregated:
        if aggregated['ess_mean_mean'] > 0:
            aggregated['ess_cv'] = aggregated['ess_mean_std'] / aggregated['ess_mean_mean']
    
    aggregated['n_seeds'] = len(results_list)
    return aggregated

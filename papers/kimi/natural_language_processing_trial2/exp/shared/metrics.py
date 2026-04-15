"""
Evaluation metrics for ACT-DRS experiments.
"""
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import json

def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, and confidence interval."""
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "ci_95_low": 0.0, "ci_95_high": 0.0}
    
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    
    # 95% confidence interval using bootstrap
    if len(values) > 1:
        ci_low, ci_high = bootstrap_ci(values)
    else:
        ci_low = ci_high = mean
    
    return {
        "mean": float(mean),
        "std": float(std),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "n": len(values)
    }

def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(data) < 2:
        return data[0] if data else 0.0, data[0] if data else 0.0
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = (1 - ci) / 2
    ci_low = np.percentile(bootstrap_means, alpha * 100)
    ci_high = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return ci_low, ci_high

def paired_t_test(group1: List[float], group2: List[float]) -> Dict[str, float]:
    """Perform paired t-test and compute Cohen's d."""
    if len(group1) != len(group2) or len(group1) == 0:
        return {"t_statistic": 0.0, "p_value": 1.0, "cohens_d": 0.0, "significant": False}
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(group1, group2)
    
    # Cohen's d for paired samples
    diff = np.array(group1) - np.array(group2)
    cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant": p_value < 0.05,
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1))
    }

def independent_t_test(group1: List[float], group2: List[float]) -> Dict[str, float]:
    """Perform independent t-test."""
    if len(group1) == 0 or len(group2) == 0:
        return {"t_statistic": 0.0, "p_value": 1.0, "cohens_d": 0.0, "significant": False}
    
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    # Cohen's d
    pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
    cohens_d = (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-8)
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant": p_value < 0.05
    }

def compute_cross_lingual_gap(english_score: float, lrl_score: float) -> float:
    """Compute cross-lingual performance gap."""
    if english_score == 0:
        return 0.0
    return (english_score - lrl_score) / english_score

def compute_transfer_effectiveness(baseline: float, steered: float, english: float) -> float:
    """Compute transfer effectiveness: (steered - baseline) / (english - baseline)."""
    if abs(english - baseline) < 1e-8:
        return 0.0
    return (steered - baseline) / (english - baseline)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def aggregate_results(results_list: List[Dict], metric_key: str = 'accuracy') -> Dict:
    """Aggregate results across multiple seeds."""
    values = [r[metric_key] for r in results_list if metric_key in r]
    return compute_statistics(values)

class ResultsTracker:
    """Track and aggregate experiment results."""
    
    def __init__(self):
        self.results = {}
    
    def add(self, experiment: str, seed: int, metrics: Dict):
        """Add results for an experiment with a specific seed."""
        if experiment not in self.results:
            self.results[experiment] = {}
        self.results[experiment][seed] = metrics
    
    def get_aggregated(self, experiment: str) -> Dict:
        """Get aggregated statistics for an experiment."""
        if experiment not in self.results:
            return {}
        
        seeds_data = self.results[experiment]
        aggregated = {}
        
        # Collect all metric keys
        all_keys = set()
        for metrics in seeds_data.values():
            all_keys.update(metrics.keys())
        
        # Aggregate each metric
        for key in all_keys:
            values = [m[key] for m in seeds_data.values() if key in m and isinstance(m[key], (int, float))]
            if values:
                aggregated[key] = compute_statistics(values)
        
        return aggregated
    
    def save(self, path: str):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load(self, path: str):
        """Load results from JSON file."""
        with open(path, 'r') as f:
            self.results = json.load(f)

if __name__ == "__main__":
    # Test metrics
    print("Testing metrics module...")
    
    # Test statistics
    values = [0.75, 0.78, 0.72]
    stats = compute_statistics(values)
    print(f"Statistics for {values}: {stats}")
    
    # Test t-test
    group1 = [0.75, 0.78, 0.72]
    group2 = [0.65, 0.68, 0.62]
    ttest = paired_t_test(group1, group2)
    print(f"Paired t-test: {ttest}")

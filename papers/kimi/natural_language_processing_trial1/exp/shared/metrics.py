"""
Evaluation metrics for QA and retrieval experiments.
"""
import json
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def calculate_retrieval_precision(retrieved_docs: List[str], relevant_docs: List[str], k: int = 3) -> float:
    """Calculate precision@k for retrieval."""
    if not retrieved_docs:
        return 0.0
    
    retrieved_k = retrieved_docs[:k]
    relevant_set = set(relevant_docs)
    
    hits = sum(1 for doc in retrieved_k if doc in relevant_set)
    return hits / len(retrieved_k)


def calculate_retrieval_recall(retrieved_docs: List[str], relevant_docs: List[str], k: int = 3) -> float:
    """Calculate recall@k for retrieval."""
    if not relevant_docs:
        return 0.0
    
    retrieved_k = retrieved_docs[:k]
    retrieved_set = set(retrieved_k)
    relevant_set = set(relevant_docs)
    
    hits = len(retrieved_set & relevant_set)
    return hits / len(relevant_set)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, min, max for a list of values."""
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr))
    }


def aggregate_results(results_list: List[Dict]) -> Dict:
    """Aggregate results from multiple seeds."""
    metrics = defaultdict(list)
    
    for result in results_list:
        for metric_name, metric_value in result.items():
            if isinstance(metric_value, (int, float)):
                metrics[metric_name].append(metric_value)
    
    aggregated = {}
    for metric_name, values in metrics.items():
        aggregated[metric_name] = compute_statistics(values)
    
    return aggregated


def paired_t_test(values_a: List[float], values_b: List[float]) -> Tuple[float, float]:
    """Perform paired t-test and return (t-statistic, p-value)."""
    from scipy import stats
    
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return (0.0, 1.0)
    
    t_stat, p_value = stats.ttest_rel(values_a, values_b)
    return (float(t_stat), float(p_value))


def cohens_d(values_a: List[float], values_b: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    if len(values_a) < 2 or len(values_b) < 2:
        return 0.0
    
    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)
    std_a = np.std(values_a, ddof=1)
    std_b = np.std(values_b, ddof=1)
    
    n_a = len(values_a)
    n_b = len(values_b)
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean_a - mean_b) / pooled_std


def confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval."""
    from scipy import stats
    
    if len(values) < 2:
        return (0.0, 0.0)
    
    arr = np.array(values)
    mean = np.mean(arr)
    sem = stats.sem(arr)
    interval = stats.t.interval(confidence, len(arr) - 1, loc=mean, scale=sem)
    
    return (float(interval[0]), float(interval[1]))


class ResultsAggregator:
    """Helper class to aggregate results across seeds and experiments."""
    
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(list))
    
    def add_result(self, experiment: str, seed: int, metrics: Dict):
        """Add a single result."""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.results[experiment][metric_name].append(value)
    
    def get_aggregated(self, experiment: str) -> Dict:
        """Get aggregated statistics for an experiment."""
        if experiment not in self.results:
            return {}
        
        agg = {}
        for metric_name, values in self.results[experiment].items():
            agg[metric_name] = compute_statistics(values)
        return agg
    
    def compare_experiments(self, exp_a: str, exp_b: str, metric: str) -> Dict:
        """Compare two experiments on a specific metric."""
        values_a = self.results[exp_a].get(metric, [])
        values_b = self.results[exp_b].get(metric, [])
        
        if not values_a or not values_b:
            return {}
        
        t_stat, p_value = paired_t_test(values_a, values_b)
        effect_size = cohens_d(values_a, values_b)
        ci_a = confidence_interval(values_a)
        ci_b = confidence_interval(values_b)
        
        return {
            'metric': metric,
            'exp_a': exp_a,
            'exp_b': exp_b,
            'mean_a': float(np.mean(values_a)),
            'mean_b': float(np.mean(values_b)),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': effect_size,
            'ci_a': ci_a,
            'ci_b': ci_b,
            'significant': p_value < 0.05
        }
    
    def save(self, filepath: str):
        """Save aggregated results to JSON."""
        output = {}
        for experiment in self.results:
            output[experiment] = self.get_aggregated(experiment)
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
    
    def load(self, filepath: str):
        """Load results from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for experiment, metrics in data.items():
            for metric_name, stats in metrics.items():
                # Extract mean as single value representation
                if 'mean' in stats:
                    self.results[experiment][metric_name] = [stats['mean']]


if __name__ == '__main__':
    # Test metrics
    test_values = [0.5, 0.6, 0.55, 0.7, 0.65]
    stats = compute_statistics(test_values)
    print(f"Statistics: {stats}")
    
    # Test aggregator
    agg = ResultsAggregator()
    for seed in [42, 43, 44]:
        agg.add_result('test_exp', seed, {'em': 0.5 + seed/1000, 'f1': 0.6 + seed/1000})
    
    print(f"Aggregated: {agg.get_aggregated('test_exp')}")

"""Shared utilities for DynaScale experiments."""

from .irt_models import TwoPLModel, ThreePLModel, OnePLModel
from .optimal_transport import (
    wasserstein_distance, 
    sinkhorn_selection, 
    bin_matching_selection,
    compute_selection_quality
)
from .metrics import (
    kendalls_tau,
    pairwise_ranking_accuracy,
    top_k_ranking_correlation,
    expected_fisher_information,
    compute_discriminative_power,
    compute_all_metrics,
    statistical_significance_test,
    ranking_stability
)
from .data_generation import (
    generate_math_problems,
    generate_code_problems,
    generate_science_problems,
    simulate_responses,
    create_model_population,
    simulate_temporal_evolution
)

__all__ = [
    'TwoPLModel', 'ThreePLModel', 'OnePLModel',
    'wasserstein_distance', 'sinkhorn_selection', 'bin_matching_selection',
    'compute_selection_quality',
    'kendalls_tau', 'pairwise_ranking_accuracy', 'top_k_ranking_correlation',
    'expected_fisher_information', 'compute_discriminative_power',
    'compute_all_metrics', 'statistical_significance_test', 'ranking_stability',
    'generate_math_problems', 'generate_code_problems', 'generate_science_problems',
    'simulate_responses', 'create_model_population', 'simulate_temporal_evolution'
]

"""Shared utilities for SAE experiments."""
from .utils import set_seed, save_results, load_results
from .models import get_model_and_sae, load_gpt2_small, load_sae_gpt2_small
from .data_loader import prepare_all_datasets, load_prepared_datasets
from .metrics import compute_side_effect_score, compute_perplexity
from .ifs import compute_ifs_for_features, compute_ifs
from .steering import (
    select_features_by_activation,
    select_features_by_output_score,
    select_features_by_ifs,
    evaluate_steering_on_prompts
)

__all__ = [
    'set_seed',
    'save_results',
    'load_results',
    'get_model_and_sae',
    'load_gpt2_small',
    'load_sae_gpt2_small',
    'prepare_all_datasets',
    'load_prepared_datasets',
    'compute_side_effect_score',
    'compute_perplexity',
    'compute_ifs_for_features',
    'compute_ifs',
    'select_features_by_activation',
    'select_features_by_output_score',
    'select_features_by_ifs',
    'evaluate_steering_on_prompts'
]

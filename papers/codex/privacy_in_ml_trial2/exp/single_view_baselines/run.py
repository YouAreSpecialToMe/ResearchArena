from __future__ import annotations

from exp.shared.config import DATASET_CONFIGS, EPSILON_TARGETS, SEEDS
from exp.shared.pipeline import evaluate_all_methods, score_checkpoint


if __name__ == "__main__":
    for dataset in DATASET_CONFIGS:
        for epsilon in EPSILON_TARGETS:
            for seed in SEEDS:
                score_checkpoint(dataset, epsilon, seed, k_values=[2, 4])
                evaluate_all_methods(dataset, epsilon, seed)

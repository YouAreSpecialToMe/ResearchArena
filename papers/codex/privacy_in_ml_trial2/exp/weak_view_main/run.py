from __future__ import annotations

from exp.shared.config import DATASET_CONFIGS, EPSILON_TARGETS, SEEDS
from exp.shared.pipeline import proxy_lambda_selection, score_checkpoint


if __name__ == "__main__":
    for dataset in DATASET_CONFIGS:
        for epsilon in EPSILON_TARGETS:
            for seed in SEEDS:
                score_checkpoint(dataset, epsilon, seed, k_values=[2, 4])
                proxy_lambda_selection(dataset, epsilon, seed)

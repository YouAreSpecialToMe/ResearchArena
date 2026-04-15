from __future__ import annotations

from exp.shared.config import DATASET_CONFIGS, EPSILON_TARGETS, SEEDS
from exp.shared.pipeline import evaluate_all_methods


if __name__ == "__main__":
    for dataset in DATASET_CONFIGS:
        for epsilon in EPSILON_TARGETS:
            for seed in SEEDS:
                evaluate_all_methods(dataset, epsilon, seed)

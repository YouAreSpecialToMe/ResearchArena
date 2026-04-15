from __future__ import annotations

from exp.shared.config import DATASET_CONFIGS, EPSILON_TARGETS, SEEDS
from exp.shared.pipeline import train_dp_target


if __name__ == "__main__":
    for dataset in DATASET_CONFIGS:
        for epsilon in EPSILON_TARGETS:
            for seed in SEEDS:
                train_dp_target(dataset, epsilon, seed)

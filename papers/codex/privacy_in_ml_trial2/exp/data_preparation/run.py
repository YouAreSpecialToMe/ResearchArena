from __future__ import annotations

from exp.shared.config import DATASET_CONFIGS, SEEDS
from exp.shared.pipeline import build_audit_pool, screen_transforms, split_indices


if __name__ == "__main__":
    for dataset in DATASET_CONFIGS:
        for seed in SEEDS:
            split_indices(dataset, seed)
            build_audit_pool(dataset, seed)
        screen_transforms(dataset)

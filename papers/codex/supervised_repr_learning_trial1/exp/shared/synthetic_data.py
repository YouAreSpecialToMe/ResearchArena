from pathlib import Path

import numpy as np

from exp.shared.utils import ensure_dir, json_dump


def _orthonormal(rng, dim, rank):
    q, _ = np.linalg.qr(rng.normal(size=(dim, rank)))
    return q[:, :rank]


def generate_synthetic_dataset(root, seed, regime, num_classes=10, num_subclasses=2, dim=32, rank=2):
    root = Path(root)
    ensure_dir(root)
    rng = np.random.default_rng(seed)
    train_x = []
    train_y = []
    train_sub = []
    test_x = []
    test_y = []
    test_sub = []
    true_bases = []
    subclass_means = []

    if regime == "directional_low_rank":
        mean_scale = 0.15
        signal_scale = 2.0
        noise_scale = 0.2
        subclass_offset_scale = 0.0
        isotropic_mode = False
    else:
        mean_scale = 0.05
        signal_scale = 0.05
        noise_scale = 0.35
        subclass_offset_scale = 1.35
        isotropic_mode = True

    for cls in range(num_classes):
        class_center = rng.normal(scale=0.8, size=dim)
        subclass_axis = _orthonormal(rng, dim, 1)[:, 0]
        for sub in range(num_subclasses):
            if isotropic_mode:
                direction = -1.0 if sub == 0 else 1.0
                mean = class_center + direction * subclass_offset_scale * subclass_axis + rng.normal(scale=mean_scale, size=dim)
            else:
                mean = class_center + rng.normal(scale=mean_scale, size=dim)
            basis = _orthonormal(rng, dim, rank)
            true_bases.append(basis)
            subclass_means.append(mean)
            cov_latent = np.diag(np.linspace(signal_scale, max(signal_scale * 0.7, 1e-4), rank))
            for split, count in [("train", 600), ("test", 200)]:
                latent = rng.multivariate_normal(np.zeros(rank), cov_latent, size=count)
                isotropic = rng.normal(scale=noise_scale, size=(count, dim))
                if isotropic_mode:
                    points = mean[None, :] + isotropic
                else:
                    points = mean[None, :] + latent @ basis.T + isotropic
                points = points / np.linalg.norm(points, axis=1, keepdims=True).clip(min=1e-12)
                labels = np.full(count, cls, dtype=np.int64)
                sublabels = np.full(count, cls * num_subclasses + sub, dtype=np.int64)
                if split == "train":
                    train_x.append(points)
                    train_y.append(labels)
                    train_sub.append(sublabels)
                else:
                    test_x.append(points)
                    test_y.append(labels)
                    test_sub.append(sublabels)
    data = {
        "train_x": np.concatenate(train_x).astype(np.float32),
        "train_y": np.concatenate(train_y).astype(np.int64),
        "train_sub": np.concatenate(train_sub).astype(np.int64),
        "test_x": np.concatenate(test_x).astype(np.float32),
        "test_y": np.concatenate(test_y).astype(np.int64),
        "test_sub": np.concatenate(test_sub).astype(np.int64),
        "true_bases": np.stack(true_bases).astype(np.float32),
        "subclass_means": np.stack(subclass_means).astype(np.float32),
    }
    np.savez(root / f"{regime}_seed{seed}.npz", **data)
    metadata = {
        "seed": seed,
        "regime": regime,
        "num_classes": num_classes,
        "num_subclasses": num_subclasses,
        "dim": dim,
        "rank": rank,
    }
    json_dump(metadata, root / f"{regime}_seed{seed}.json")
    return data

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
FEATURE_ROOT = ROOT / "features"
PAIR_ROOT = ROOT / "pairs"
CHECKPOINT_ROOT = ROOT / "checkpoints"
RESULT_ROOT = ROOT / "results"
FIGURE_ROOT = ROOT / "figures"
ARTIFACT_ROOT = ROOT / "artifacts"
ENV_ROOT = ARTIFACT_ROOT / "environment"
META_ROOT = ARTIFACT_ROOT / "metadata"

SEEDS = [11, 17, 23]
SENSITIVITY_SEED = 29
SPLIT_SEED = 2026

DATASETS = {
    "dsprites": {
        "factors": ["shape", "scale", "rotation", "x_position", "y_position"],
        "factor_sizes": [3, 6, 40, 32, 32],
        "download_url": "https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        "raw_name": "dsprites.npz",
        "channels": 3,
        "image_size": 224,
    },
    "shapes3d": {
        "factors": ["object_hue", "wall_hue", "floor_hue", "scale", "shape", "orientation"],
        "factor_sizes": [10, 10, 10, 8, 4, 15],
        "download_url": "https://storage.googleapis.com/3d-shapes/3dshapes.h5",
        "raw_name": "3dshapes.h5",
        "channels": 3,
        "image_size": 224,
    },
}

BACKBONES = {
    "dinov2_vits14": {
        "kind": "timm",
        "model_name": "vit_small_patch14_dinov2.lvd142m",
        "image_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "batch_size": 256,
        "sensitivity_only": False,
    },
    "openclip_vit_b32": {
        "kind": "open_clip",
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "image_size": 224,
        "mean": (0.48145466, 0.4578275, 0.40821073),
        "std": (0.26862954, 0.26130258, 0.27577711),
        "batch_size": 256,
        "sensitivity_only": True,
    },
}

PAIR_CAPS = {
    "train": {"counterfactual": 120000, "nuisance": 60000},
    "val": {"counterfactual": 15000, "nuisance": 10000},
    "test": {"counterfactual": 15000, "nuisance": 10000},
}

PILOT_GRID = {
    "lambda_nuis": [0.1, 0.3],
    "lambda_cf": [0.3, 1.0],
    "ra_relax": [0.01, 0.05],
}

TRAINING = {
    "latent_multiplier": 4,
    "batch_size": 2048,
    "max_epochs": 20,
    "patience": 3,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "margin": 0.1,
    "singleton_fraction": 0.5,
    "counterfactual_fraction": 0.25,
    "nuisance_fraction": 0.25,
    "eval_batch_size": 4096,
    "bootstrap_samples": 200,
    "permutation_trials": 20,
}

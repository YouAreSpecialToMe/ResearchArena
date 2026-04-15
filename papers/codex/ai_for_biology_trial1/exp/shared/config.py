from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL = ROOT / "data" / "external"
FIGURES_DIR = ROOT / "figures"

SEEDS = [11, 23, 37]
PREP_CACHE_VERSION = "v3_full_target_svd_strict_json_metadata"
DATASETS = {
    "Adamson": DATA_RAW / "Adamson.h5ad",
    "Replogle_K562essential": DATA_RAW / "Replogle_K562essential.h5ad",
}

CONTROL_LABELS = {"control", "ctrl"}
MIN_PERT_CELLS = 20
MIN_TRAIN_CONTROL_CELLS = 100
MAX_HVGS = 2000
FORCED_GENE_CAP = 2200
STRING_SCORE_THRESHOLD = 700
STRING_EMBED_DIM = 128
RESIDUAL_PCA_DIM = 64
FULL_PCA_DIM = 64
LINEAR_EMBED_DIM = 32
TRAIN_VAL_FRAC = 0.8
TRAIN_TEST_FRAC = 0.8
RIDGE_ALPHA_GRID = [0.1, 1.0, 10.0, 100.0]
RETRIEVAL_RIDGE_ALPHA_GRID = [1e-5, 1e-4, 1e-3]
PLS_COMPONENTS_GRID = [8, 16, 32]
RETRIEVAL_K_GRID = [5, 10, 20]
MLP_HIDDEN_DIM = 256
DROPOUT = 0.1
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
BOOTSTRAP_DRAWS = 1000

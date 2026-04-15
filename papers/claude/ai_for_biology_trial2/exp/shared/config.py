"""Shared configuration for all experiments."""
import os

# Random seeds for reproducibility
SEEDS = [42, 123, 456]
N_FOLDS = 5

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.join(ROOT, "exp")
DATA_DIR = os.path.join(EXP_DIR, "data")
FEATURES_DIR = os.path.join(EXP_DIR, "features")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

# ESM-2 config
ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"
ESM2_EMBED_DIM = 1280
ESM2_NUM_LAYERS = 33
ESM2_NUM_HEADS = 20
# Use deeper layers for coupling (layers 20-33 encode structural info)
COUPLING_LAYER_START = 20
COUPLING_LAYER_END = 33

# ESM-2 150M config (for ablation)
ESM2_150M_MODEL_NAME = "esm2_t30_150M_UR50D"
ESM2_150M_EMBED_DIM = 640
ESM2_150M_NUM_LAYERS = 30
ESM2_150M_COUPLING_LAYER_START = 15

# GNN hyperparameters
GNN_HIDDEN_DIM = 128
GNN_NUM_HEADS = 4
GNN_NUM_LAYERS = 2
GNN_DROPOUT = 0.1
GNN_EDGE_DIM = 3

# Training hyperparameters
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 30
LR_PATIENCE = 10
LR_FACTOR = 0.5

# MLP hyperparameters
MLP_HIDDEN_DIMS = [256, 128]
MLP_LR = 1e-3
MLP_MAX_EPOCHS = 200
MLP_PATIENCE = 20

# Minimum multi-mutant threshold
MIN_MULTI_MUTANTS = 200

"""Shared configuration for all experiments."""

# Model
MODEL_NAME = "EleutherAI/pythia-160m"
MODEL_LAYERS = 12
HIDDEN_DIM = 768

# Layers to study
LAYERS = [2, 6, 10]

# SAE Training
N_SEEDS_PRIMARY = 8
RANDOM_SEEDS = [42, 137, 256, 512, 1024, 2048, 4096, 8192]
DICT_SIZE = 16384
TOPK_K = 50
N_TRAINING_TOKENS = 20_000_000  # 20M tokens (reduced for time budget)
CONTEXT_SIZE = 128
LR = 3e-4
BATCH_SIZE = 4096

# Feature Matching
MATCHING_THRESHOLD = 0.7

# Consensus Tiers
CONSENSUS_HIGH = 0.75  # >= 6/8 seeds
CONSENSUS_LOW = 0.25   # <= 2/8 seeds

# Evaluation
EVAL_N_SEQUENCES = 5000
EVAL_BATCH_SIZE = 32

# Dataset
DATASET_PATH = "monology/pile-uncopyrighted"

# Colors for plotting
COLORS = {
    "consensus": "#2196F3",
    "partial": "#FF9800",
    "singleton": "#F44336",
    "random": "#9E9E9E",
    "frequency": "#4CAF50",
}

# Paths
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
FIGURES_DIR = os.path.join(WORKSPACE_DIR, "figures")

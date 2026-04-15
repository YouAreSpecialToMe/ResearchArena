"""Configuration for SAE-GUIDE experiments."""

import torch
import random
import numpy as np

# Random seeds for reproducibility
RANDOM_SEEDS = [42, 123, 456]

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
HIDDEN_SIZE = 3584  # Qwen2.5-7B hidden size

# SAE configuration
SAE_EXPANSION_FACTOR = 8
SAE_HIDDEN_DIM = HIDDEN_SIZE * SAE_EXPANSION_FACTOR  # 28672
SAE_TOP_K = 32
SAE_LAYERS = list(range(12, 21))  # Layers 12-20 (0-indexed)
SAE_LAYER = 16  # Primary layer for extraction

# Training configuration
SAE_LR = 1e-3
SAE_BATCH_SIZE = 256
SAE_EPOCHS = 100
SAE_PATIENCE = 10

PROBE_LR = 1e-4
PROBE_BATCH_SIZE = 64
PROBE_EPOCHS = 20
PROBE_HIDDEN_DIM = 1024

# Retrieval configuration
BM25_K1 = 1.5
BM25_B = 0.75
RETRIEVAL_TOP_K = 3
MAX_RETRIEVAL_ROUNDS = 5
CUMULATIVE_DECAY = 0.9

# Dataset configuration
HOTPOTQA_SAMPLE_SIZE = 500
WIKI_SAMPLE_SIZE = 300
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# IRCOT configuration
IRCOT_INTERVAL = 50  # tokens

# Paths
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

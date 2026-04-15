"""Configuration for AACD experiments."""

RANDOM_SEEDS = list(range(10))
TUNING_SEEDS = list(range(10000, 10030))
DATA_TYPES = ['HLG', 'HLNG', 'HNL', 'HM', 'SH', 'NU']
P_VALUES_FULL = [10, 20]
P_VALUES_TARGETED = [50]
EXPECTED_DEGREE = 3
SAMPLE_SIZES = [500, 1000, 5000]
ALGORITHM_TIMEOUT_SEC = 300
N_BOOTSTRAP = 10

# Sigmoid grid search parameters
SIGMOID_T_VALUES = [0.3, 0.5, 0.7]
SIGMOID_S_VALUES = [5, 10]
TAU_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]

# Diagnostic confidence thresholds (initial, calibrated later)
N_STAR = {
    'D1': 300,  # linearity
    'D2': 200,  # non-Gaussianity
    'D3': 500,  # ANM
    'D4': 500,  # faithfulness
    'D5': 300,  # homoscedasticity
}

# Paths
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

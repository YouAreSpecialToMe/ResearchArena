from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "exp"
FIGURES_ROOT = ROOT / "figures"
TABLES_ROOT = ROOT / "tables"
LOGS_ROOT = ROOT / "logs"

DATASET_SEEDS = [11, 23, 37]
BOOTSTRAP_SEEDS = [101, 131]
SUBSET_BANK_SEED = 509

P = 20
EXPECTED_DEGREE = 2.0
EDGE_PROB = EXPECTED_DEGREE / (P - 1)

GRAPH_FAMILIES = ["erdos_renyi", "scale_free"]
REGIMES = [
    "linear_gaussian",
    "nonlinear_anm",
    "near_unfaithful_linear",
    "mild_misspecification",
]
SAMPLE_SIZES = [200, 1000]
HARD_REGIMES = {"near_unfaithful_linear", "mild_misspecification"}

SUBSET_BANK = {"M": 20, "k": 8, "B": 2, "alpha": 0.01}
SUBSET_SENSITIVITY_KS = [6, 10]
ALPHA_GRID = [0.001, 0.01, 0.05, 0.1]

CPU_WORKERS = 2
GPU_COUNT = 0
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}

DAGBAG_BOOTSTRAPS = 10
DAGBAG_FALLBACK_BOOTSTRAPS = 5

NOTEARS_LAMBDA1 = 0.1
NOTEARS_MAX_ITER = 100
NOTEARS_EDGE_THRESHOLD = 0.3


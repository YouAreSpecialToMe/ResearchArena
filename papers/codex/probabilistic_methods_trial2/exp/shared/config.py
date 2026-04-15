from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS_MODELS = ROOT / "artifacts" / "models"
ARTIFACTS_POSTERIORS = ROOT / "artifacts" / "posteriors"
ARTIFACTS_RUNTIME = ROOT / "artifacts" / "runtime"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
EXP_DIR = ROOT / "exp"

SEEDS = [11, 22, 33]
ALPHA_MAIN = 0.10
ALPHA_SECONDARY = 0.05
MAX_PARALLEL_JOBS = 2

DATASETS = ["synthetic", "anuran", "mice"]
PILOT_DATASETS = ["synthetic", "anuran"]
ABLATION_DATASETS = ["synthetic", "anuran"]

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

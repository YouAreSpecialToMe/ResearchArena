from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from exp.shared.config import RESULTS_DIR, THREAD_ENV
from exp.shared.data import prepare_all_datasets
from exp.shared.io import write_json


def write_requirements() -> None:
    import importlib

    packages = ["numpy", "scipy", "pandas", "sklearn", "matplotlib", "seaborn"]
    lines = []
    for name in packages:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        pkg_name = "scikit-learn" if name == "sklearn" else name
        lines.append(f"{pkg_name}=={version}")
    (ROOT / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    os.environ.update(THREAD_ENV)
    prepare_all_datasets()
    write_requirements()
    write_json(
        RESULTS_DIR / "runtime_budget.json",
        {
            "cpu_cores": 2,
            "ram_gb": 128,
            "gpus": 0,
            "max_parallel_jobs": 2,
            "alpha_main": 0.10,
            "alpha_secondary": 0.05,
        },
    )


if __name__ == "__main__":
    main()

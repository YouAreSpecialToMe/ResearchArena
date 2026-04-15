from __future__ import annotations

from pathlib import Path

from exp.shared.runner import run_exact_validation


if __name__ == "__main__":
    run_exact_validation(Path(__file__).resolve().parent)


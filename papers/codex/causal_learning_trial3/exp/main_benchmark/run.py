from __future__ import annotations

from pathlib import Path

from exp.shared.runner import run_main_benchmark


if __name__ == "__main__":
    run_main_benchmark(Path(__file__).resolve().parent)


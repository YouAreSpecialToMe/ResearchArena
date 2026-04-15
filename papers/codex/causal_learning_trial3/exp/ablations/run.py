from __future__ import annotations

from pathlib import Path

from exp.shared.runner import run_ablations


if __name__ == "__main__":
    run_ablations(Path(__file__).resolve().parent)


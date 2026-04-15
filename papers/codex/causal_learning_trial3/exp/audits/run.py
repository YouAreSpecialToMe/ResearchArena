from __future__ import annotations

from pathlib import Path

from exp.shared.runner import run_audits


if __name__ == "__main__":
    run_audits(Path(__file__).resolve().parent)


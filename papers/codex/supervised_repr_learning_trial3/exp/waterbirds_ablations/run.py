import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from exp.shared.core import run_suite


if __name__ == "__main__":
    run_suite("waterbirds_ablations")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.runner import run_experiment_suite


if __name__ == "__main__":
    run_experiment_suite("static_consensus")

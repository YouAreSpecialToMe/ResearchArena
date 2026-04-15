import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from run_experiments import aggregate_results


if __name__ == "__main__":
    aggregate_results([], [])

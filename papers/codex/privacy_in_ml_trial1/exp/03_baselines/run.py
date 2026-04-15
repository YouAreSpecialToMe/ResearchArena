import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from run_experiments import train_shadows, train_targets_and_retrains


if __name__ == "__main__":
    train_targets_and_retrains()
    train_shadows()

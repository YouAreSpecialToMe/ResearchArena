from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.pipeline import train_ablations

if __name__ == "__main__":
    train_ablations()

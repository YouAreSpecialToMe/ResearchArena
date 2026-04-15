from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.pipeline import generate_data
from exp.shared.utils import ROOT


if __name__ == "__main__":
    generate_data(ROOT / "exp" / "02_data")

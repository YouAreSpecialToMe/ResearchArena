from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.pipeline import run_primary
from exp.shared.utils import ROOT


if __name__ == "__main__":
    run_primary(ROOT / "exp" / "03_primary")

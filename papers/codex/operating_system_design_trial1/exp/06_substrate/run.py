from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.pipeline import write_skipped_substrate
from exp.shared.utils import ROOT


if __name__ == "__main__":
    write_skipped_substrate(ROOT / "exp" / "06_substrate")

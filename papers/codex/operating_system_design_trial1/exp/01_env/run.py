from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.pipeline import write_env
from exp.shared.utils import ROOT


if __name__ == "__main__":
    write_env(ROOT / "exp" / "01_env")

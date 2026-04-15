from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.pipeline import run_ablations, run_exact_windows, run_primary, summarize, write_skipped_substrate
from exp.shared.utils import ROOT


if __name__ == "__main__":
    primary = run_primary(ROOT / "exp" / "03_primary")
    exact = run_exact_windows(ROOT / "exp" / "04_exact_windows")
    ablations = run_ablations(ROOT / "exp" / "05_ablations", primary)
    substrate = write_skipped_substrate(ROOT / "exp" / "06_substrate")
    summarize(primary, exact, ablations, substrate, ROOT / "exp" / "07_analysis")

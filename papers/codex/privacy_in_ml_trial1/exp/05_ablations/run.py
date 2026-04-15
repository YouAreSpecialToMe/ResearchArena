import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from run_experiments import compute_forgetting_band, fit_attacks, run_ablations


if __name__ == "__main__":
    attacks = fit_attacks()
    bands = compute_forgetting_band()
    run_ablations(attacks, bands)

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from exp.shared.core import plot_ablation, plot_main_comparison


if __name__ == "__main__":
    summary_path = Path(__file__).resolve().parents[2] / "exp" / "06_evaluation" / "summary_table.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        plot_main_comparison(df[df["method"].isin(["base_ft", "gu_global", "orthograd", "aspire"])])
        plot_ablation(df[df["method"].str.startswith("aspire_")])

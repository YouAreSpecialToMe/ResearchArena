import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from exp.shared.runner import run_condition

if __name__ == "__main__":
    run_condition("imdb_erm_smoke_test")

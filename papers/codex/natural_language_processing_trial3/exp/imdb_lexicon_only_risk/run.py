import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from exp.shared.runner import run_condition

run_condition("imdb_lexicon_only_risk")


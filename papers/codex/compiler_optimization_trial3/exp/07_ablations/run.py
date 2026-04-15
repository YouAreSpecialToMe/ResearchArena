import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.pipeline_study import run_step

run_step("07_ablations")

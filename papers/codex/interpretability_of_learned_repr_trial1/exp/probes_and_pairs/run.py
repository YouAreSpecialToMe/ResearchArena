from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.pipeline import probes_and_pairs

if __name__ == "__main__":
    probes_and_pairs()

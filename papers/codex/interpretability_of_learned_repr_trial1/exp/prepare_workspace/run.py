from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.pipeline import prepare_workspace

if __name__ == "__main__":
    prepare_workspace()

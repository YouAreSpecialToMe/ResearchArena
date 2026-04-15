from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from exp.shared.pipeline import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "preprocess"]
    main()


from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from exp.shared.runner import main


if __name__ == "__main__":
    main()

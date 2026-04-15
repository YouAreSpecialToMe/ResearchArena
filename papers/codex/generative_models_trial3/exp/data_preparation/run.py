import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.dataset import write_dataset_artifacts
from exp.shared.common import EXP_DIR, write_json


def main() -> None:
    audit = write_dataset_artifacts()
    write_json(EXP_DIR / "data_preparation" / "results.json", audit)


if __name__ == "__main__":
    main()

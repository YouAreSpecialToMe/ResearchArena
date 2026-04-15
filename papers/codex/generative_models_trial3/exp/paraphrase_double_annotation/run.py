import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.common import EXP_DIR, write_json


def main() -> None:
    report = {
        "experiment": "paraphrase_double_annotation",
        "status": "skipped",
        "planned_pairs": 48,
        "completed_pairs": 0,
        "cohens_kappa": None,
        "reason": "The preregistered double-annotation audit requires independent human annotations, which are not available inside this coding-only workspace.",
        "revision_status": "formally_revised_to_infeasible",
    }
    write_json(EXP_DIR / "paraphrase_double_annotation" / "results.json", report)


if __name__ == "__main__":
    main()

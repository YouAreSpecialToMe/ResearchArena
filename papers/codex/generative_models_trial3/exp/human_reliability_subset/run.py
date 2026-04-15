import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.common import EXP_DIR, write_json


def main() -> None:
    report = {
        "experiment": "human_reliability_subset",
        "status": "skipped",
        "planned_cases": 72,
        "completed_cases": 0,
        "reason": "The preregistered human reliability subset requires external human annotators, which are not available inside this coding-only workspace.",
        "revision_status": "formally_revised_to_infeasible",
    }
    write_json(EXP_DIR / "human_reliability_subset" / "results.json", report)


if __name__ == "__main__":
    main()

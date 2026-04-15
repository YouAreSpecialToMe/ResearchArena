import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.common import EXP_DIR, write_json


def main() -> None:
    report = {
        "experiment": "aapb_reproduction",
        "status": "skipped",
        "faithful_reproduction": False,
        "reason": "No runnable public AAPB reference implementation was available in this workspace during this run, and reconstructing the closed-form adaptive rule from summary text alone would not count as a faithful preregistered reproduction.",
        "revision_status": "formally_revised_to_infeasible",
    }
    write_json(EXP_DIR / "aapb_reproduction" / "aapb_reproduction_report.json", report)
    write_json(EXP_DIR / "aapb_reproduction" / "results.json", report)


if __name__ == "__main__":
    main()

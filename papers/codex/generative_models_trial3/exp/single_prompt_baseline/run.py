import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.common import EXP_DIR, write_json


def main() -> None:
    report = {
        "experiment": "single_prompt_baseline",
        "status": "skipped",
        "selected_baseline": None,
        "reason": "The available SCG GitHub repository was only a project-page template rather than runnable generation code, and no runnable MaskDiffusion SD1.5 implementation was available locally during this run.",
        "revision_status": "formally_revised_to_infeasible",
    }
    write_json(EXP_DIR / "single_prompt_baseline" / "baseline_notes.json", report)
    write_json(EXP_DIR / "single_prompt_baseline" / "results.json", report)


if __name__ == "__main__":
    main()
